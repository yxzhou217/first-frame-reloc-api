"""reloc_server.py — 首帧定位 API 服务(seed_expand 管线,2026-08-17)

把 eval_heldout.py 验证过的 seed_expand 管线包成 HTTP 服务:
  DINO 检索 → VLM 按相似度顺序验证(第一个 Yes = 主帧) → 主帧邻居凑窗口
  (默认空间近邻:位置+朝向筛选,不依赖时间信息;--window_mode temporal 可切回时间邻居)
  → lingbot 窗口联合重建 → Sim(3) 对齐 → 返回查询帧在地图坐标系中的位姿

接口:
  GET  /health   — 服务状态(模型/地图是否加载完成)
  POST /localize — 上传一张照片,返回位姿。支持两种格式:
                   ① multipart/form-data 文件字段 file(推荐,curl -F)
                   ② application/json {"image_base64": "..."}

启动(map 环境):
  PYTHONPATH=/path/to/lingbot-map \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python reloc_server.py --port 8100

注意:返回的坐标是"地图单位",不是米(米制标定未完成)。
"""

import argparse
import base64
import io
import os
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 须在 import torch 前
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from PIL import Image, ImageOps

import cv2
import relocalize as R
from build_map_db import extract_descriptors, IMAGENET_MEAN, IMAGENET_STD
from vlm_rerank import VLMVerifier
from eval_heldout import spatial_subsample
from image_quality_gate import ImageQualityGate, QualityConfig


# =============================================================================
# 图像预处理(必须与建图时完全一致)
# =============================================================================

def preprocess_query(pil_img: Image.Image, rotate_clockwise_90: bool,
                     image_size: int = 518, patch_size: int = 14) -> np.ndarray:
    """单张照片 → (3,518,518) fp16 [0,1],与建图时 load_and_preprocess_images
    (mode="crop") 逐像素对齐:
      EXIF 方向校正 → (可选)顺时针 90° → 宽缩放到 518(等比,高对齐 patch 倍数)
      → 高 >518 时中心裁剪到 518。
    """
    img = ImageOps.exif_transpose(pil_img)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    if rotate_clockwise_90:
        img = img.transpose(Image.ROTATE_270)  # 与 my_demo --rotate_clockwise_90 相同

    w, h = img.size
    new_w = image_size
    new_h = round(h * (new_w / w) / patch_size) * patch_size
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0       # (H,W,3)
    arr = arr.transpose(2, 0, 1)                           # (3,H,W)
    if new_h > image_size:                                 # crop 模式:中心裁高
        y0 = (new_h - image_size) // 2
        arr = arr[:, y0:y0 + image_size, :]
    return arr.astype(np.float16)


# =============================================================================
# 定位器(启动时加载一次,全程复用)
# =============================================================================

class Localizer:
    def __init__(self, args):
        import torch
        self.torch = torch
        self.args = args
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.lock = threading.Lock()          # GPU 串行化

        print(f"[启动] 加载数据库 {args.db} ...")
        db = np.load(args.db)
        self.desc_all = db["descriptors"].astype(np.float32)
        self.ext_all = R.w2c_to_c2w(db["extrinsic"])      # (N,4,4) c2w

        print(f"[启动] 加载地图 {args.map_npz} ...")
        map_data = np.load(args.map_npz)
        self.imgs_all = map_data["images"][0]             # (N,3,518,518) fp16
        self.n = self.desc_all.shape[0]
        assert self.ext_all.shape[0] == self.n and self.imgs_all.shape[0] == self.n

        # 空间子采样 DB(与评估同参数,target_ratio=0.46);
        # --db_stride N 时改为隔 N 取一(关键帧地图推荐: VidMap 关键帧本身已按运动间距选出)
        if getattr(args, "db_stride", None):
            self.db_idx = np.arange(0, self.n, args.db_stride)
        else:
            self.db_idx = spatial_subsample(self.ext_all[:, :3, 3],
                                            target_ratio=args.target_db_ratio)
        self.desc_db = self.desc_all[self.db_idx]
        ext_db = self.ext_all[self.db_idx]
        self.baseline = float(np.linalg.norm(
            np.diff(ext_db[:, :3, 3], axis=0), axis=1).mean())
        print(f"[启动] 地图 {self.n} 帧,DB {len(self.db_idx)} 帧,基线 {self.baseline:.4f}")

        print("[启动] 加载 lingbot-map 模型 ...")
        self.model = R.load_lingbot_model(
            SimpleNamespace(model_path=args.model_path), self.device)

        print(f"[启动] 加载描述子模型 {args.dino_path} ...")
        from transformers import AutoModel
        self.dino = AutoModel.from_pretrained(args.dino_path).to(self.device).eval()

        self.vlm = None
        if not args.no_vlm:
            self.vlm = VLMVerifier(base_url=args.vlm_url, model=args.vlm_model)
            print(f"[启动] VLM 验证: {args.vlm_url} ({args.vlm_model})")

        # patch 互匹配重排(默认开): 预计算库图 patch 特征
        self.db_patches = None
        if not args.no_patch_rerank:
            print("[启动] 预计算库图 patch 特征(DINO 去 CLS, L2 归一化) ...")
            _t0 = time.perf_counter()
            self.db_patches = self._patch_feats(self.imgs_all)
            print(f"[启动] patch 特征 {self.db_patches.shape}, 耗时 {time.perf_counter()-_t0:.1f}s")

        # CPU 图像质检门禁(--min_quality 0 关闭)
        self.quality_gate = None
        if args.min_quality > 0:
            self.quality_gate = ImageQualityGate(
                QualityConfig(retry_threshold=args.min_quality))
            print(f"[启动] 图像质检: 阈值 {args.min_quality}")
        print("[启动] 完成,服务就绪")

    # ------------------------------------------------------------------
    def _patch_feats(self, imgs_np, batch_size=16):
        """(N,3,H,W) [0,1] → (N,P,D) fp16, DINO patch 特征(去 CLS, L2 归一化)。

        预处理与 extract_descriptors 一致(ImageNet 归一化), 只改取 patch token。"""
        torch = self.torch
        mean = torch.from_numpy(IMAGENET_MEAN).view(1, 3, 1, 1).to(self.device)
        std = torch.from_numpy(IMAGENET_STD).view(1, 3, 1, 1).to(self.device)
        feats = []
        for s in range(0, len(imgs_np), batch_size):
            b = torch.from_numpy(imgs_np[s:s + batch_size]).float().to(self.device)
            b = (b - mean) / std
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                out = self.dino(pixel_values=b)
            p = out.last_hidden_state[:, 1:].float()          # 去 CLS
            p = p / p.norm(dim=-1, keepdim=True)
            feats.append(p.half().cpu().numpy())
        return np.concatenate(feats, axis=0)

    def _rerank_by_patches(self, qpatch, cand_idx):
        """原始 MNN 重排(与 ab_patch_eval.py 的 raw 方法一致):
        查询图与每个候选做 patch 互最近邻, 按互认对数降序。"""
        torch = self.torch
        q = torch.from_numpy(qpatch).float().to(self.device)
        scores = []
        for c in cand_idx:
            cp = torch.from_numpy(self.db_patches[c]).float().to(self.device)
            sim = cp @ q.T                                   # (Pc,Pq)
            c2q = sim.argmax(dim=1)
            q2c = sim.argmax(dim=0)
            ok = q2c[c2q] == torch.arange(cp.shape[0], device=self.device)
            scores.append(int(ok.sum()))
        return cand_idx[np.argsort(-np.array(scores))]

    # ------------------------------------------------------------------
    def localize(self, pil_img: Image.Image, rotate: bool | None = None) -> dict:
        """单张苏醒照片 → 位姿。全程持锁(GPU 串行)。

        rotate: 是否顺时针转 90°;None = 用服务启动时的默认值。
        真实机器人照片应与建图视频同处理方式(默认转);
        若调用方传的是已旋转/地图同规格的图,应传 False。
        """
        with self.lock:
            return self._localize_locked(pil_img, rotate)

    def _localize_locked(self, pil_img: Image.Image, rotate: bool | None) -> dict:
        torch = self.torch
        t0 = time.perf_counter()
        warnings = []

        # 0. CPU 图像质检门禁(先于一切 GPU 工作;低于阈值直接拒绝, 不执行定位)
        if self.quality_gate is not None:
            bgr = cv2.cvtColor(
                np.asarray(ImageOps.exif_transpose(pil_img).convert("RGB")),
                cv2.COLOR_RGB2BGR)
            qres = self.quality_gate.evaluate(bgr)
            if qres["action"] == "turn_and_retake":
                return {
                    "success": False,
                    "error": f"画面信息不足(质量分 {qres['score']:.0f} < 阈值 {self.args.min_quality:.0f}),未执行定位",
                    "quality": qres,
                    "confidence": {"residual": None, "residual_x_baseline": None,
                                   "inliers": "0/0"},
                    "debug": {"seed_frame": None,
                              "latency_ms": (time.perf_counter() - t0) * 1000},
                    "warnings": [],
                }

        # 1. 预处理 + 描述子
        if rotate is None:
            rotate = self.args.rotate_clockwise_90
        qimg = preprocess_query(pil_img, rotate)  # (3,518,518) fp16
        qdesc = extract_descriptors(qimg[None], self.dino, self.device,
                                    batch_size=1)[0].astype(np.float32)

        # 2. DINO 检索(相似度只决定问 VLM 的顺序)
        fetch = min(self.args.vlm_fetch * 2, len(self.desc_db))
        cand = R.retrieve(qdesc, self.desc_db, k=fetch)

        # 2.5 patch 互匹配重排(默认开; 把视野重叠度高的候选提前)
        if self.db_patches is not None:
            qpatch = self._patch_feats(qimg[None])[0]
            cand = self._rerank_by_patches(qpatch, cand)

        # 3. VLM 按顺序验证,第一个 Yes = 主帧
        seed_frame = None
        vlm_calls = 0
        vlm_yes_rank = -1
        if self.vlm is not None:
            for rank, i in enumerate(cand):
                cand_img = self.imgs_all[self.db_idx[i]]
                try:
                    vres = self.vlm.verify_pair(qimg, cand_img)
                except Exception as e:
                    warnings.append(f"VLM 调用失败({e}),退回 DINO top-1 作主帧")
                    break
                vlm_calls += 1
                if vres.is_match:
                    seed_frame = int(self.db_idx[i])
                    vlm_yes_rank = rank
                    break
            if seed_frame is None and not warnings:
                # VLM 活着但全部拒绝 = 查询场景大概率不在地图里(2026-08-17 修复:
                # 之前退回 DINO top-1,此时锚点全是同一错处的相邻帧,几何自洽,
                # 残差自检拦不住,会返回一个"看起来可信"的错误位姿)
                return {
                    "success": False,
                    "error": f"VLM 拒绝了相似度前 {fetch} 个候选,查询场景可能不在地图中,请重拍或确认位置",
                    "confidence": {"residual": None, "residual_x_baseline": None, "inliers": "0/0"},
                    "debug": {"seed_frame": None, "vlm_yes_rank": -1,
                              "vlm_calls": vlm_calls,
                              "latency_ms": (time.perf_counter() - t0) * 1000},
                    "warnings": [],
                }
        else:
            warnings.append("VLM 未启用(--no_vlm),直接用 DINO top-1 作主帧")
        if seed_frame is None:
            seed_frame = int(self.db_idx[cand[0]])

        # 4. 凑满 k 个锚点,两种窗口模式(--window_mode):
        #    temporal: 主帧时间邻居(±1,±2,...,按 |Δ| 从小到大)——要求帧号沿轨迹有序
        #    spatial:  空间近邻(位置门+朝向门+逐步放宽,不依赖时间信息,2026-09-17)
        k = self.args.k
        if self.args.window_mode == "spatial":
            neighbors = R.spatial_neighbors(
                self.ext_all, seed_frame, k, self.baseline,
                diversity=self.args.spatial_diversity)
        else:
            neighbors = R.temporal_neighbors(self.n, seed_frame, k)
        anchor_frames = np.array([seed_frame] + neighbors)

        # 5. 窗口联合重建 + Sim(3) 对齐
        window = torch.from_numpy(
            np.stack([qimg] + [self.imgs_all[f] for f in anchor_frames]))
        c2w_fresh = R.run_window_c2w(self.model, window, self.device, self.dtype)
        s, Rm, t, res, inl = R.align_sim3_rotcons(
            c2w_fresh[1:], self.ext_all[anchor_frames])
        R_est = Rm @ c2w_fresh[0, :3, :3]
        t_est = s * (Rm @ c2w_fresh[0, :3, 3]) + t

        quat = R.rot_to_quat(R_est)  # (w,x,y,z)
        res_x = res / self.baseline
        success = bool(res_x <= 2.0)
        if not success:
            warnings.append(f"对齐残差 {res_x:.1f}x基线 超过 2x 阈值,结果可疑")

        return {
            "success": success,
            "position": [float(v) for v in t_est],           # 地图单位,非米
            "quaternion_wxyz": [float(v) for v in quat],
            "rotation_matrix": R_est.astype(float).tolist(),
            "scale": float(s),                                # 应 ≈1
            "confidence": {
                "residual": float(res),
                "residual_x_baseline": float(res_x),
                "inliers": f"{int(inl.sum())}/{k}",
            },
            "debug": {
                "seed_frame": seed_frame,
                "window_mode": self.args.window_mode,
                "vlm_yes_rank": vlm_yes_rank,                 # -1 = VLM 未确认(回退)
                "anchor_frames": anchor_frames.tolist(),
                "vlm_calls": vlm_calls,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
            "warnings": warnings,
        }


# =============================================================================
# FastAPI 服务
# =============================================================================

def create_app(loc: Localizer):
    from fastapi import FastAPI, File, Request, UploadFile
    from fastapi.responses import JSONResponse

    app = FastAPI(title="首帧定位 API", version="1.0 (seed_expand)")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "pipeline": f"seed_expand (质检 + DINO检索 + patch重排 + VLM主帧 + {loc.args.window_mode}邻居窗口 + Sim3对齐)",
            "map_frames": loc.n,
            "db_frames": int(len(loc.db_idx)),
            "k": loc.args.k,
            "window_mode": loc.args.window_mode,
            "vlm_enabled": loc.vlm is not None,
            "patch_rerank": loc.db_patches is not None,
            "quality_gate": loc.args.min_quality,
        }

    @app.post("/localize")
    async def localize(request: Request, file: UploadFile = File(None)):
        # 两种输入:multipart 文件字段 file;或 JSON {"image_base64": "..."}
        # 可选旋转覆盖: JSON 字段 "rotate_clockwise_90": true/false
        #               或 multipart 查询参数 ?rotate=false
        raw = None
        rotate = None
        q_rotate = request.query_params.get("rotate")
        if q_rotate is not None:
            rotate = q_rotate.lower() not in ("false", "0", "no")
        if file is not None:
            raw = await file.read()
        elif "application/json" in request.headers.get("content-type", ""):
            body = await request.json()
            if "rotate_clockwise_90" in body:
                rotate = bool(body["rotate_clockwise_90"])
            b64 = body.get("image_base64")
            if b64:
                if "," in b64:  # 允许 data URL 前缀
                    b64 = b64.split(",", 1)[1]
                try:
                    raw = base64.b64decode(b64)
                except Exception as e:
                    return JSONResponse({"success": False,
                                         "error": f"base64 解码失败: {e}"},
                                        status_code=400)
        if raw is None:
            return JSONResponse(
                {"success": False,
                 "error": "需要 multipart 文件字段 file,或 JSON {\"image_base64\": \"...\"}"},
                status_code=400)
        try:
            pil_img = Image.open(io.BytesIO(raw))
            pil_img.load()
        except Exception as e:
            return JSONResponse({"success": False,
                                 "error": f"图片解码失败: {e}"}, status_code=400)
        try:
            return loc.localize(pil_img, rotate)
        except Exception as e:
            return JSONResponse({"success": False,
                                 "error": f"定位失败: {type(e).__name__}: {e}"},
                                status_code=500)

    return app


def main():
    parser = argparse.ArgumentParser(description="首帧定位 API 服务(seed_expand)")
    parser.add_argument("--map_npz", default="output/short_stream_full.npz")
    parser.add_argument("--db", default="output/short_stream_full_db.npz")
    parser.add_argument("--model_path",
                        default="checkpoints/lingbot-map-long.pt")
    parser.add_argument("--dino_path", default="facebook/dinov2-small")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--window_mode", choices=["temporal", "spatial"],
                        default="spatial",
                        help="锚点窗口选取: spatial=空间近邻(位置+朝向筛选,默认,不依赖时间信息); "
                             "temporal=主帧时间邻居(仅当地图帧号沿轨迹有序时可用)")
    parser.add_argument("--spatial_diversity", action="store_true",
                        help="spatial 模式加贪心方位多样性(防锚点挤同一侧,稀疏地图建议开)")
    parser.add_argument("--target_db_ratio", type=float, default=0.46)
    parser.add_argument("--db_stride", type=int, default=None,
                        help="隔 N 帧取 1 帧作检索库(替代空间子采样;关键帧地图建议 2)")
    parser.add_argument("--vlm_url", default="http://127.0.0.1:8000")
    parser.add_argument("--vlm_model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--vlm_fetch", type=int, default=5)
    parser.add_argument("--no_vlm", action="store_true",
                        help="不用 VLM,直接 DINO top-1 作主帧(仅调试用)")
    parser.add_argument("--no_patch_rerank", action="store_true",
                        help="关闭 patch 互匹配重排(默认开: DINO top-N 后按视野重叠度重排)")
    parser.add_argument("--min_quality", type=float, default=40.0,
                        help="CPU 图像质检阈值(0~100), 低于则拒绝定位; 0=关闭质检")
    parser.add_argument("--rotate_clockwise_90", action="store_true", default=True,
                        help="查询照片预处理时顺时针旋转90°(与 short.mp4 建图一致,默认开)")
    parser.add_argument("--no_rotate", dest="rotate_clockwise_90", action="store_false")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    loc = Localizer(args)
    app = create_app(loc)

    import uvicorn
    print(f"[服务] 监听 http://{args.host}:{args.port}  (GET /health, POST /localize)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
