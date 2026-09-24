"""eval_heldout.py — 留出法"苏醒帧"测试(真实视频,无外部数据集)

思路:
  建图 npz 包含全部 N 帧的位姿(同一坐标系)。把偶数帧当"数据库"
  (假装是建图成果),奇数帧当"机器人苏醒时拍的新照片"(数据库里没有),
  用奇数帧在同一次建图里的位姿当参照(伪真值,自洽)。
  和 7-Scenes 评估的差别:没有米制尺度,但坐标系统一、无需全局对齐,
  且数据是机器人平稳巡场视频 —— 最接近真实部署条件。

前置:
  python my_demo.py --video_path example/short.mp4 --fps 6 ... --save_output output/short_full.npz --slim_output --no_view
  python build_map_db.py --map_npz output/short_full.npz --out short_full_db.npz

用法(map 环境,任意目录):
  PYTHONPATH=/path/to/lingbot-map python eval_heldout.py \
      --map_npz output/short_full.npz --db short_full_db.npz --num_queries 30
"""

import argparse
import os
import time

import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import relocalize as R
from build_map_db import extract_descriptors, IMAGENET_MEAN, IMAGENET_STD
from vlm_rerank import VLMVerifier  # VLM 图片对验证(2026-08-12)


def spatial_subsample(centers, interval=None, target_ratio=None):
    """按等路径距离选关键帧,保证空间均匀覆盖(2026-08-05)。

    解决问题:匀速时间采样(fps=10)在机器人速度变化 9.5× 时,
    慢速区段帧冗余堆积、快速区段帧稀疏缺失。

    两种调用方式(普适性改进):
    - interval=数值: 直接指定路径间距(地图单位,需了解地图尺度)
    - target_ratio=比例: 自动计算 interval = 总路径长度 / (N * ratio),
      适配任意地图尺度,无需手调

    centers: (N,3) 相机位置序列
    返回: 选中的帧索引数组
    """
    if interval is None and target_ratio is not None:
        # 自动计算:总路径长度 / 期望DB帧数
        diffs = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        total_path = diffs.sum()
        target_count = max(1, int(len(centers) * target_ratio))
        interval = total_path / target_count
    elif interval is None:
        interval = 0.06  # fallback(不推荐,应优先用 target_ratio)

    selected = [0]
    accumulated = 0.0
    for i in range(1, len(centers)):
        d = np.linalg.norm(centers[i] - centers[selected[-1]])
        accumulated += d
        if accumulated >= interval:
            selected.append(i)
            accumulated = 0.0
    return np.array(selected)


def extract_patches(images, model, device, batch_size=8, skip=1):
    """提取 DINO patch tokens(用于 patch-level 重排)。

    与 extract_descriptors 的区别:返回所有 patch 的特征(不只是 CLS)。
    images: (N,3,H,W) float16 [0,1] → (N, P, D) L2归一化 patch 特征
    skip: 头部特殊 token 数(dinov2=1 仅 CLS;dinov3=5 CLS+4 registers)
    """
    import torch
    n = images.shape[0]
    all_patches = []
    mean = torch.from_numpy(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.from_numpy(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    for start in range(0, n, batch_size):
        batch = torch.from_numpy(images[start:start + batch_size]).float().to(device)
        batch = (batch - mean) / std
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(pixel_values=batch)
        patches = out.last_hidden_state[:, skip:]  # 丢掉特殊 token,保留 patch tokens
        patches = patches.float()
        patches = patches / patches.norm(dim=-1, keepdim=True)  # L2 归一化
        all_patches.append(patches.cpu().numpy())

    return np.concatenate(all_patches, axis=0)


def main():
    parser = argparse.ArgumentParser(description="留出法苏醒帧测试(真实视频)")
    parser.add_argument("--map_npz", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--model_path", default="checkpoints/lingbot-map-long.pt")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--num_queries", type=int, default=30)
    parser.add_argument("--dino_path", default="facebook/dinov2-small",
                        help="描述子/patch 模型路径(本地 dinov3 目录或 HF id)")
    parser.add_argument("--dino_skip", type=int, default=1,
                        help="patch 提取时跳过的特殊 token 数(dinov2=1, dinov3=5)")
    parser.add_argument("--exclude_radius", type=int, default=30,
                        help="检索时排除查询帧 ±R 帧(消除同视频'插值'乐观偏差,2026-07-30 评审 P1)")
    parser.add_argument("--adaptive", action="store_true",
                        help="启用自适应检索(相似度阈值+空间聚类,2026-08-05)")
    parser.add_argument("--sim_ratio", type=float, default=0.7,
                        help="自适应:保留相似度>=top1*ratio 的候选")
    parser.add_argument("--spatial_radius_mult", type=float, default=5.0,
                        help="自适应:聚类半径=该值×基线")
    parser.add_argument("--spatial_sort", action="store_true",
                        help="自适应:按到top-1空间距离排序取最近k个(替代半径过滤)")
    parser.add_argument("--scatter_thresh", type=float, default=10.0,
                        help="混合策略:top-k最大两两距离>该值×基线才启用spatial_sort(0=禁用)")
    parser.add_argument("--spatial_db", action="store_true",
                        help="用空间子采样选DB(按等路径距离选关键帧,替代偶数帧)")
    parser.add_argument("--spatial_interval", type=float, default=None,
                        help="空间子采样间距(地图单位,需了解地图尺度)")
    parser.add_argument("--target_db_ratio", type=float, default=0.46,
                        help="自动计算间距:DB占总帧的比例(默认0.46,普适,无需了解地图尺度)")
    parser.add_argument("--rerank", action="store_true",
                        help="启用 DINOv2 patch-level 重排(抗视觉混淆,2026-08-05)")
    parser.add_argument("--multi_hypothesis", action="store_true",
                        help="多假设检验:同时跑 patch重排 和 patch重排+空间聚类,"
                             "用 Sim(3) 残差选更优假设(2026-08-06)")
    parser.add_argument("--rerank_cluster", action="store_true",
                        help="patch重排 + 空间聚类串行组合:先用 patch重排得到更好的 top-1,"
                             "再围绕重排 top-1 做空间聚类(2026-08-06)")
    # VLM 验证(2026-08-12): 用 VLM 做检索后语义过滤
    parser.add_argument("--vlm_verify", action="store_true",
                        help="启用 VLM 图片对验证过滤检索候选(需 Qwen3-VL 服务运行)")
    parser.add_argument("--vlm_url", default="http://127.0.0.1:8000",
                        help="VLM 服务地址")
    parser.add_argument("--vlm_model", default="Qwen/Qwen3-VL-8B-Instruct",
                        help="VLM 模型名")
    parser.add_argument("--vlm_fetch", type=int, default=5,
                        help="送 VLM 验证的 DINOv2 top-K 候选数(建议 3~8)")
    # 主帧+时间邻居扩展(2026-08-17): VLM 确认的第一个 Yes 作主帧,
    # 其余锚点用主帧的时间邻居(±1,±2,...)凑满,不再从检索结果里凑
    parser.add_argument("--seed_expand", action="store_true",
                        help="主帧扩展模式: VLM 按相似度顺序验证,第一个 Yes 作主帧,"
                             "用主帧的邻居凑满窗口(隐含启用 VLM)")
    parser.add_argument("--window_mode", choices=["temporal", "spatial"],
                        default="spatial",
                        help="seed_expand 的窗口选取: spatial=空间近邻(位置+朝向筛选,默认,"
                             "不依赖时间信息); temporal=时间邻居(需帧号沿轨迹有序)")
    parser.add_argument("--spatial_diversity", action="store_true",
                        help="spatial 模式加贪心方位多样性(防锚点挤同一侧)")
    parser.add_argument("--queries", type=int, nargs="*", default=None,
                        help="只测这些全局帧号(必须在留出查询池里),如 --queries 97 98 100")
    args = parser.parse_args()

    import torch

    db = np.load(args.db)
    desc_all = db["descriptors"].astype(np.float32)
    ext_all = R.w2c_to_c2w(db["extrinsic"])         # (N,4,4) c2w(2026-07-30 约定修正)
    map_data = np.load(args.map_npz)
    imgs_all = map_data["images"][0]            # (N,3,H,W) fp16
    n = desc_all.shape[0]
    assert ext_all.shape[0] == n and imgs_all.shape[0] == n

    # 偶数帧 = 数据库;奇数帧 = 留出查询
    if args.spatial_db:
        # 空间子采样:按等路径距离选DB关键帧(消除速度变化导致的基线不均)
        all_centers = ext_all[:, :3, 3]
        db_idx = spatial_subsample(all_centers, interval=args.spatial_interval,
                                   target_ratio=args.target_db_ratio)
        db_set = set(db_idx.tolist())
        q_pool = np.array([i for i in range(n) if i not in db_set])
        print(f"空间子采样 DB: 选中 {len(db_idx)}/{n} 帧, 查询池 {len(q_pool)} 帧")
    else:
        db_idx = np.arange(0, n, 2)
        q_pool = np.arange(1, n, 2)
    desc_db = desc_all[db_idx]
    ext_db = ext_all[db_idx]

    # 基线:数据库相邻帧平均间距(用于归一化误差)
    centers_db = ext_db[:, :3, 3]
    baseline = np.linalg.norm(np.diff(centers_db, axis=0), axis=1).mean()
    print(f"总帧数 {n}:数据库 {len(db_idx)} 帧(偶),留出查询池 {len(q_pool)} 帧(奇)")
    print(f"数据库相邻帧平均间距(基线): {baseline:.4f} 地图单位\n")

    device = torch.device("cuda")
    model = R.load_lingbot_model(args, device)
    from transformers import AutoModel
    dino_path = getattr(args, "dino_path", "facebook/dinov2-small")
    print(f"描述子/重排模型: {dino_path}")
    dino = AutoModel.from_pretrained(dino_path).to(device).eval()
    dtype = torch.bfloat16

    # VLM 验证器初始化(2026-08-12);seed_expand 模式隐含需要 VLM
    vlm: VLMVerifier | None = None
    if args.vlm_verify or args.seed_expand:
        vlm = VLMVerifier(base_url=args.vlm_url, model=args.vlm_model)
        print(f"VLM 验证: {args.vlm_url} (模型={args.vlm_model}), fetch={args.vlm_fetch}")
        # 启动时做一次健康检查
        try:
            health_url = f"{args.vlm_url.rstrip('/')}/health"
            hdrs = {"Authorization": f"Bearer {vlm.api_key}"} if vlm.api_key else {}
            resp = __import__("requests").get(health_url, headers=hdrs, timeout=5)
            if resp.ok:
                h = resp.json()
                print(f"  VLM 服务状态: {h.get('status','?')}, "
                      f"模型={h.get('qwenModelVersion','?')}, "
                      f"后端={'✓' if h.get('backend',{}).get('status')=='ok' else '?'}")
            else:
                print(f"  ⚠ VLM 服务返回 {resp.status_code},将继续尝试调用")
        except Exception as e:
            print(f"  ⚠ 无法连接 VLM 服务 ({e}),后续调用将报错")

    qids = np.unique(np.linspace(0, len(q_pool) - 1, min(args.num_queries, len(q_pool))).astype(int))
    if args.queries is not None:
        # 只测指定帧号:校验它们在留出查询池里
        q_set = set(q_pool.tolist())
        bad = [f for f in args.queries if f not in q_set]
        if bad:
            raise ValueError(f"帧 {bad} 不在留出查询池(在 DB 里或越界),无法作查询")
        qids = np.array([np.where(q_pool == f)[0][0] for f in args.queries])
    mode_str = "自适应(相似度阈值+空间聚类)" if args.adaptive else "固定top-k"
    if args.seed_expand:
        mode_str = f"主帧+{args.window_mode}邻居扩展(seed_expand)"
    elif args.vlm_verify:
        mode_str += "+VLM验证"
    print(f"抽 {len(qids)} 个查询帧,k={args.k}, 检索模式: {mode_str}\n")

    pos_errs, rot_errs, residuals, anchor_dists = [], [], [], []
    neighbor_hits = 0
    for qi in qids:
        q_frame = q_pool[qi]                     # 全局帧号(奇)
        t0 = time.time()
        qimg = torch.from_numpy(imgs_all[q_frame:q_frame + 1].copy())
        qdesc = extract_descriptors(imgs_all[q_frame:q_frame + 1], dino, device, batch_size=1)[0].astype(np.float32)

        # 检索(排除 ±exclude_radius 帧,防止锚点全是查询的时间邻居 → 更接近真实苏醒)
        excl = np.where(np.abs(db_idx - q_frame) <= args.exclude_radius)[0]

        # --- VLM 语义验证(2026-08-12) ---
        # 在 DINOv2 检索后、窗口重建前,用 VLM 做"图片对"语义过滤。
        # DINOv2 靠纹理相似度检索,会把"纹理像但位置不同"的帧排进 top-K;
        # VLM 理解空间语义(房间布局/文字标识/物体),能识破这类视觉混淆。
        # 策略:对 DINOv2 top-N 候选做顺序验证,只保留 VLM 判 Yes 的帧进入下游。
        vlm_passed: list[int] | None = None  # None = 未启用
        if vlm is not None and not args.seed_expand:
            vlm_fetch = min(args.vlm_fetch * 2, len(desc_db))  # 2x over-fetch
            topk_vlm = R.retrieve(qdesc, desc_db, k=vlm_fetch, exclude_idx=excl)
            # 准备候选图片: 从 imgs_all 中取对应帧,转为 CHW→HWC uint8 供 VLM HTTP 传输
            qimg_np = (np.asarray(imgs_all[q_frame].transpose(1, 2, 0), dtype=np.float32) * 255).clip(0, 255).astype(np.uint8)
            cand_imgs_np = [
                (np.asarray(imgs_all[db_idx[i]].transpose(1, 2, 0), dtype=np.float32) * 255).clip(0, 255).astype(np.uint8)
                for i in topk_vlm
            ]
            print(f"  VLM 验证 top-{vlm_fetch} 候选...")
            try:
                vlm_passed, vlm_results = vlm.rerank_candidates(
                    qimg_np, cand_imgs_np,
                    top_k=args.k, early_stop=True, verbose=True,
                )
                if vlm_passed:
                    # 将 VLM 通过的候选映射回 db_idx
                    vlm_passed_db = topk_vlm[np.array(vlm_passed)]
                    print(f"  VLM 通过 {len(vlm_passed)}/{len(vlm_results)} 个候选: {vlm_passed_db.tolist()}")
                else:
                    print(f"  ⚠ VLM 全部拒绝 {len(vlm_results)} 个候选,退回纯 DINOv2 结果")
                    vlm_passed = None  # 退回
            except Exception as e:
                print(f"  ⚠ VLM 调用失败 ({e}),退回纯 DINOv2 结果")
                vlm_passed = None

        # --- 多假设检验(2026-08-06) ---
        # 同时准备两个候选锚点集,分别做窗口重建+Sim(3)对齐,选残差更低的
        # 假设A: patch重排(抗视觉混淆)
        # 假设B: patch重排 + 空间聚类(top-1正确时更紧凑)
        # --- 主帧+时间邻居扩展(2026-08-17) ---
        # 设计动机: DINO top-k 里可能混着"同楼同风格"的远处假场景(如 Q97 案例),
        # 直接取 top-8 当锚点必被污染。改为:
        #   1. 按 DINO 相似度从高到低逐个问 VLM "是否同一处",第一个 Yes = 主帧
        #      (主帧的身份由 VLM 语义确认,不信相似度排名)
        #   2. 其余锚点用主帧的时间邻居(±1,±2,...,按 |Δ| 从小到大)凑满 k 个。
        #      相邻帧物理上必然在主帧旁边 → 与主帧有重叠 → 窗口视图图连通,
        #      联合重建成立;假场景(时间差几百帧)结构上不可能进入窗口。
        #   3. 邻居帧的图像/位姿从完整 npz 取(imgs_all/ext_all 覆盖全部帧,
        #      不受 DB 空间子采样限制)。
        if args.seed_expand:
            fetch = min(args.vlm_fetch * 2, len(desc_db))
            cand = R.retrieve(qdesc, desc_db, k=fetch, exclude_idx=excl)
            qimg_np = (np.asarray(imgs_all[q_frame].transpose(1, 2, 0), dtype=np.float32) * 255).clip(0, 255).astype(np.uint8)
            seed_frame = None
            if vlm is not None:
                print(f"  VLM 寻主帧(按相似度顺序,最多问 {fetch} 个)...")
                for rank, i in enumerate(cand):
                    cand_np = (np.asarray(imgs_all[db_idx[i]].transpose(1, 2, 0), dtype=np.float32) * 255).clip(0, 255).astype(np.uint8)
                    try:
                        vres = vlm.verify_pair(qimg_np, cand_np)
                    except Exception as e:
                        print(f"  ⚠ VLM 调用失败 ({e}),退回 DINO top-1 作主帧")
                        break
                    mark = "✓" if vres.is_match else "✗"
                    print(f"    [{rank}] 帧{db_idx[i]} → {vres.raw_response.strip()!r} {mark} | {vres.latency_ms:.0f}ms")
                    if vres.is_match:
                        seed_frame = int(db_idx[i])
                        break
            if seed_frame is None:
                if vlm is not None:
                    print(f"  ⚠ VLM 全部拒绝 {fetch} 个候选,退回 DINO top-1 作主帧")
                seed_frame = int(db_idx[cand[0]])

            # 邻居扩展凑满 k 个锚点(--window_mode):
            # temporal: 主帧 ±1,±2,...,按 |Δ| 从小到大(要求帧号沿轨迹有序)
            # spatial:  位置门+朝向门筛选,逐步放宽(不依赖时间信息,2026-09-17)
            if args.window_mode == "spatial":
                neighbors = R.spatial_neighbors(
                    ext_all, seed_frame, args.k, baseline,
                    exclude={q_frame}, diversity=args.spatial_diversity)
            else:
                neighbors = R.temporal_neighbors(n, seed_frame, args.k,
                                                 exclude={q_frame})
            topk_frames = np.array([seed_frame] + neighbors)
            topk = None  # 锚点不按 DB 索引,统一走 topk_frames + ext_all
            h_chosen = "seed"
            print(f"  主帧={seed_frame},窗口锚点={topk_frames.tolist()}")
        elif vlm_passed is not None:
            # VLM 验证模式(2026-08-12): 用 VLM 语义过滤后的候选直接做窗口重建
            # vlm_passed_db 是 topk_vlm[passed_indices],即 indices into desc_db
            topk = np.array(vlm_passed_db)
            topk_frames = db_idx[topk]
            h_chosen = "vlm"
        elif args.multi_hypothesis:

            topk_16 = R.retrieve(qdesc, desc_db, k=min(16, len(desc_db)), exclude_idx=excl)
            q_patches = extract_patches(imgs_all[q_frame:q_frame + 1], dino, device, batch_size=1, skip=args.dino_skip)[0]
            cand_patches = extract_patches(imgs_all[db_idx[topk_16]], dino, device, batch_size=8, skip=args.dino_skip)
            topk_rerank, _ = R.rerank_dinov2_patches(q_patches, list(cand_patches), topk_16, k=args.k)
            # 假设B: 在 rerank 基础上做空间聚类(以 rerank top-1 为种子)
            topk_spatial = R.retrieve_adaptive(qdesc, desc_db, ext_db[:, :3, 3],
                                               k=args.k, exclude_idx=excl,
                                               sim_ratio=0, spatial_sort=True,
                                               baseline=baseline, over_fetch=4)
            # 但用 rerank 的 top-1 替换 spatial 的 top-1(确保种子是 rerank 选的)
            # 实际上 retrieve_adaptive 的 top-1 就是 DINOv2 top-1,rerank 可能改变它
            # 所以直接对比两个假设的残差

            hypotheses = [("rerank", topk_rerank), ("spatial", topk_spatial)]
            results_h = []
            for h_name, h_topk in hypotheses:
                h_frames = db_idx[h_topk]
                h_window = torch.cat([qimg.half(),
                                      torch.from_numpy(np.stack([imgs_all[f] for f in h_frames]))])
                h_c2w = R.run_window_c2w(model, h_window, device, dtype)
                h_s, h_R, h_t, h_res, h_inl = R.align_sim3_rotcons(h_c2w[1:], ext_db[h_topk])
                h_R_est = h_R @ h_c2w[0, :3, :3]
                h_t_est = h_s * (h_R @ h_c2w[0, :3, 3]) + h_t
                results_h.append((h_name, h_topk, h_res, h_inl, h_R_est, h_t_est))

            # 选残差更低的假设(几何一致性判据)
            # tuple: (name, topk, res, inl, R_est, t_est)
            best = min(results_h, key=lambda x: x[2])  # x[2] = residual
            topk = best[1]
            topk_frames = db_idx[topk]
            res = best[2]
            inl = best[3]
            R_est = best[4]
            t_est = best[5]
            h_chosen = best[0]
        elif args.rerank_cluster:
            # patch重排 + 空间聚类串行(2026-08-06):
            # 1. patch重排 top-16 → 更准的 top-1(抗视觉混淆)
            # 2. 以重排 top-1 为种子,从 top-16 中选空间最近的 k 帧(抗散装)
            topk_16 = R.retrieve(qdesc, desc_db, k=min(16, len(desc_db)), exclude_idx=excl)
            q_patches = extract_patches(imgs_all[q_frame:q_frame + 1], dino, device, batch_size=1, skip=args.dino_skip)[0]
            cand_patches = extract_patches(imgs_all[db_idx[topk_16]], dino, device, batch_size=8, skip=args.dino_skip)
            reranked_16, _ = R.rerank_dinov2_patches(q_patches, list(cand_patches), topk_16, k=16)
            # 以重排 top-1 为种子,从 reranked_16 中选空间最近的 k 帧
            seed_center = ext_db[reranked_16[0]][:3, 3]
            dists_to_seed = np.linalg.norm(ext_db[reranked_16][:, :3, 3] - seed_center, axis=1)
            spatial_order = np.argsort(dists_to_seed)
            topk = reranked_16[spatial_order][:args.k]
            topk_frames = db_idx[topk]
            h_chosen = "rerank_cluster"
        elif args.rerank:
            # Patch-level 重排:先取 top-16,再用 DINOv2 patch 互匹配重排
            topk_16 = R.retrieve(qdesc, desc_db, k=min(16, len(desc_db)), exclude_idx=excl)
            q_patches = extract_patches(imgs_all[q_frame:q_frame + 1], dino, device, batch_size=1, skip=args.dino_skip)[0]
            cand_patches = extract_patches(imgs_all[db_idx[topk_16]], dino, device, batch_size=8, skip=args.dino_skip)
            topk, rerank_scores = R.rerank_dinov2_patches(q_patches, list(cand_patches), topk_16, k=args.k)
            topk_frames = db_idx[topk]
            h_chosen = "rerank"
        elif args.adaptive:
            topk = R.retrieve_adaptive(qdesc, desc_db, ext_db[:, :3, 3],
                                       k=args.k, exclude_idx=excl,
                                       sim_ratio=args.sim_ratio,
                                       spatial_radius_mult=args.spatial_radius_mult,
                                       baseline=baseline,
                                       spatial_sort=args.spatial_sort,
                                       scatter_thresh=args.scatter_thresh)
            topk_frames = db_idx[topk]
            h_chosen = "adaptive"
        else:
            topk = R.retrieve(qdesc, desc_db, args.k, exclude_idx=excl)
            topk_frames = db_idx[topk]
            h_chosen = "topk"

        # 非多假设/VLM模式:统一走窗口重建+对齐
        # (多假设检验在上面已内联完成窗口重建;其余模式在此统一处理)
        if vlm_passed is not None or args.seed_expand or not args.multi_hypothesis:
            # 窗口顺序 [查询, 锚点×k]:查询帧在窗口第 0 位 = 重建坐标系参考(位置≈原点),
            # 对齐尺度噪声伤不到它 —— 这是重要的数值保护(2026-07-30 实测:放最后会放大尺度噪声)
            window_imgs = torch.cat([qimg.half(),
                                     torch.from_numpy(np.stack([imgs_all[f] for f in topk_frames]))])
            c2w_fresh = R.run_window_c2w(model, window_imgs, device, dtype)
            # ext_all[topk_frames] 对其它模式等价于 ext_db[topk](topk_frames=db_idx[topk]),
            # 对 seed_expand 模式则覆盖不在 DB 里的时间邻居帧
            s, Rm, t, res, inl = R.align_sim3_rotcons(c2w_fresh[1:], ext_all[topk_frames])
            R_est = Rm @ c2w_fresh[0, :3, :3]
            t_est = s * (Rm @ c2w_fresh[0, :3, 3]) + t

        # 检索质量:top-1 是否是该帧的时间邻居(±3 帧;排除后理论上不可能,仅作检索距离参考)
        if np.abs(topk_frames[0] - q_frame) <= 3:
            neighbor_hits += 1

        # 锚点在地图里离查询多远(区分"检索落错区域/地图漂移"与"几何误差")
        anchor_dist = np.linalg.norm(ext_all[topk_frames][:, :3, 3] - ext_all[q_frame][:3, 3], axis=1).mean()

        gt = ext_all[q_frame]
        p_err = float(np.linalg.norm(t_est - gt[:3, 3]))
        r_err = R.rot_err_deg(R_est, gt[:3, :3])
        pos_errs.append(p_err)
        rot_errs.append(r_err)
        residuals.append(res)
        anchor_dists.append(anchor_dist)
        h_info = f" [{h_chosen}]" if (args.multi_hypothesis or vlm_passed is not None or args.seed_expand) else ""
        print(f"查询帧 {q_frame:4d} | top1={topk_frames[0]} | 锚数 {len(topk_frames)} | 锚距 {anchor_dist:.3f} | "
              f"内点 {inl.sum()}/{len(topk_frames)} | 残差 {res:.4f} | "
              f"位置 {p_err:.4f} ({p_err / baseline:.2f}x基线) | "
              f"旋转 {r_err:.2f}°{h_info} | {time.time() - t0:.1f}s")

    pos_errs = np.array(pos_errs)
    rot_errs = np.array(rot_errs)
    anchor_dists = np.array(anchor_dists)
    print("\n===== 留出法测试汇总(真实视频,伪真值自洽)=====")
    print(f"检索 top-1 命中时间邻居(±3帧): {neighbor_hits}/{len(qids)}(排除 ±{args.exclude_radius} 后理应为 0)")
    # 按锚点地图距离分层:近锚点(公平测几何) vs 远锚点(检索落远处,含漂移/混淆)
    near = anchor_dists < 5 * baseline
    print(f"\n[近锚点组 {near.sum()}/{len(qids)}](锚距<5x基线,几何误差为主)")
    if near.any():
        print(f"  位置误差: 中位 {np.median(pos_errs[near]):.4f} ({np.median(pos_errs[near]) / baseline:.2f}x基线), "
              f"90分位 {np.percentile(pos_errs[near], 90):.4f}")
        print(f"  旋转误差: 中位 {np.median(rot_errs[near]):.2f}°, 90分位 {np.percentile(rot_errs[near], 90):.2f}°")
    far = ~near
    print(f"[远锚点组 {far.sum()}/{len(qids)}](锚距≥5x基线,检索混淆或地图漂移主导)")
    if far.any():
        print(f"  位置误差: 中位 {np.median(pos_errs[far]):.4f} ({np.median(pos_errs[far]) / baseline:.2f}x基线)")
        print(f"  旋转误差: 中位 {np.median(rot_errs[far]):.2f}°")
    print(f"\n全部:位置中位 {np.median(pos_errs):.4f} ({np.median(pos_errs) / baseline:.2f}x基线), "
          f"旋转中位 {np.median(rot_errs):.2f}°")
    print(f"对齐残差: 中位 {np.median(residuals):.4f}(>2x基线=可疑)")


if __name__ == "__main__":
    main()
