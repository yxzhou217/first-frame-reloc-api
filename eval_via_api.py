"""eval_via_api.py — 通过首帧定位 HTTP API 做留出法评估(2026-08-20)

与 eval_heldout.py 的区别:不走离线管线,而是把留出查询帧当照片 POST 给
正在运行的 reloc_server.py,完全按线上行为评估(含 VLM 全拒→success=false)。

用法(map 环境,先启动 reloc_server.py):
    python eval_via_api.py --num_queries 100 --url http://127.0.0.1:8100
"""

import argparse
import io
import json
import time

import numpy as np
import requests
from PIL import Image

import relocalize as R
from eval_heldout import spatial_subsample


def frame_to_png_bytes(img_chw: np.ndarray) -> bytes:
    """(3,H,W) [0,1] fp16 → PNG 字节流(与真实上传照片等价)"""
    arr = (np.asarray(img_chw.transpose(1, 2, 0), dtype=np.float32) * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--map_npz", default="output/short_stream_full.npz")
    ap.add_argument("--db", default="output/short_stream_full_db.npz")
    ap.add_argument("--num_queries", type=int, default=100)
    ap.add_argument("--target_db_ratio", type=float, default=0.46)
    ap.add_argument("--db_stride", type=int, default=None,
                        help="隔 N 取一作检索库(须与服务启动参数一致;关键帧地图建议 2)")
    ap.add_argument("--save_json", default="eval_via_api_results.json")
    args = ap.parse_args()

    # 与线上服务一致:空间子采样 DB;查询池 = 不在 DB 里的帧
    db = np.load(args.db)
    ext_all = R.w2c_to_c2w(db["extrinsic"])          # (N,4,4) c2w 伪真值
    map_data = np.load(args.map_npz)
    imgs_all = map_data["images"][0]
    n = ext_all.shape[0]
    if args.db_stride:
        db_idx = np.arange(0, n, args.db_stride)
    else:
        db_idx = spatial_subsample(ext_all[:, :3, 3], target_ratio=args.target_db_ratio)
    db_set = set(db_idx.tolist())
    q_pool = np.array([i for i in range(n) if i not in db_set])
    baseline = float(np.linalg.norm(
        np.diff(ext_all[db_idx][:, :3, 3], axis=0), axis=1).mean())
    qids = np.unique(np.linspace(0, len(q_pool) - 1,
                                 min(args.num_queries, len(q_pool))).astype(int))
    print(f"地图 {n} 帧,DB {len(db_idx)} 帧,查询池 {len(q_pool)},抽 {len(qids)} 例,基线 {baseline:.4f}")
    print(f"API: {args.url}  (?rotate=false,查询帧已是建图规格)\n")

    # 健康检查
    h = requests.get(f"{args.url}/health", timeout=10).json()
    assert h["status"] == "ok" and h["map_frames"] == n, f"服务状态异常: {h}"
    print(f"服务正常: VLM={'开' if h['vlm_enabled'] else '关'}\n")

    results = []
    for cnt, qi in enumerate(qids, 1):
        q_frame = int(q_pool[qi])
        png = frame_to_png_bytes(imgs_all[q_frame])
        t0 = time.perf_counter()
        try:
            resp = requests.post(f"{args.url}/localize?rotate=false",
                                 files={"file": ("q.png", png, "image/png")},
                                 timeout=300)
            r = resp.json()
        except Exception as e:
            r = {"success": False, "error": f"HTTP异常: {e}"}
        dt = time.perf_counter() - t0

        gt = ext_all[q_frame]
        row = {"query": q_frame, "latency_s": round(dt, 2), "response": r}
        if r.get("success"):
            t_est = np.array(r["position"])
            R_est = np.array(r["rotation_matrix"])
            row["pos_err"] = float(np.linalg.norm(t_est - gt[:3, 3]))
            row["rot_err"] = float(R.rot_err_deg(R_est, gt[:3, :3]))
            row["res_x_base"] = r["confidence"]["residual_x_baseline"]
            row["seed"] = r["debug"]["seed_frame"]
            print(f"[{cnt:3d}/{len(qids)}] 帧{q_frame:4d} | 主帧{row['seed']:4d} | "
                  f"位置 {row['pos_err']:.4f} ({row['pos_err']/baseline:5.2f}x基线) | "
                  f"旋转 {row['rot_err']:6.2f}° | 残差 {row['res_x_base']:.2f}x | {dt:.1f}s")
        else:
            row["pos_err"] = None
            reason = r.get("error") or "; ".join(r.get("warnings", [])) or "未知"
            print(f"[{cnt:3d}/{len(qids)}] 帧{q_frame:4d} | 拒绝/失败: "
                  f"{reason[:60]} | {dt:.1f}s")
        results.append(row)

    with open(args.save_json, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # ---- 汇总 ----
    ok_rows = [r for r in results if r["pos_err"] is not None]
    rej_rows = [r for r in results if r["pos_err"] is None]
    pe = np.array([r["pos_err"] for r in ok_rows])
    re_ = np.array([r["rot_err"] for r in ok_rows])
    pex = pe / baseline
    good = (pex <= 1.0).sum(); accept = ((pex > 1) & (pex <= 3)).sum(); bad = (pex > 3).sum()

    print(f"\n===== API 留出法评估汇总({len(results)} 例)=====")
    print(f"返回位姿 {len(ok_rows)} 例,未返回 {len(rej_rows)} 例 (VLM全拒或残差超标, success=false)")
    if len(ok_rows):
        print(f"返回位姿的 {len(ok_rows)} 例:")
        print(f"  位置误差: 中位 {np.median(pe):.4f} ({np.median(pex):.2f}x基线), "
              f"90分位 {np.percentile(pex, 90):.2f}x")
        print(f"  旋转误差: 中位 {np.median(re_):.2f}°, 90分位 {np.percentile(re_, 90):.2f}°")
        print(f"  优(≤1x基线) {good} | 可接受(1~3x) {accept} | 错误(>3x) {bad}")
        if bad:
            print(f"\n  错误位姿案例(API 返回了 success=true 但位置错误,最危险):")
            for r in sorted([r for r in ok_rows if r["pos_err"] / baseline > 3],
                            key=lambda r: -r["pos_err"]):
                print(f"    查询{r['query']:4d} 主帧{r['seed']:4d} "
                      f"位置{r['pos_err']/baseline:5.1f}x 旋转{r['rot_err']:6.1f}° "
                      f"残差{r['res_x_base']:.2f}x")
    if rej_rows:
        print(f"\n  被拒绝的查询帧: {[r['query'] for r in rej_rows]}")
    print(f"\n详细结果已存 {args.save_json}")


if __name__ == "__main__":
    main()
