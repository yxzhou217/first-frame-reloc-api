"""relocalize.py — 首帧定位:共享工具库 + 检索自测

⚠ 本文件不是线上管线!当前首帧定位管线是 seed_expand(2026-08-17):
  DINO 检索(只定顺序) → VLM 逐候选验证,第一个 Yes = 主帧
  → 主帧时间邻居(±1,±2,...)凑窗口 → 窗口联合重建 → 旋转一致性 Sim(3) 对齐
  线上服务见 reloc_server.py;离线评估见 eval_heldout.py --seed_expand。
  旧的"检索 top-k 直接凑窗口"做法已废弃,相关脚本在 archive_old_topk/。

本文件内容:
  - 共享工具:位姿约定转换(w2c_to_c2w)、检索(retrieve/retrieve_adaptive)、
    DINOv2 patch 重排(rerank_dinov2_patches)、Sim(3) 对齐(umeyama /
    align_sim3_rotcons)、四元数工具、lingbot-map 模型加载(load_lingbot_model)
    和窗口重建(run_window_c2w)——被 reloc_server.py、eval_heldout.py、
    loop_closure.py、align_colmap_*.py 等依赖。
  - --selftest 检索自测(快,不加载模型):
    把数据库每帧假装成"苏醒第一帧",检查检索回的 top-k 是否在它附近。

用法(map conda 环境):
    python relocalize.py --db map_db.npz --selftest --k 5
"""

import argparse
import os

import numpy as np

# 与 my_demo.py 相同:16GB 统一内存上防碎片 OOM(必须在 import torch 前设置)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# =============================================================================
# 检索
# =============================================================================

def w2c_to_c2w(ext_w2c):
    """npz/db 里存的 extrinsic 是 w2c(demo.py 后处理多求了一次逆,2026-07-30 实证确认:
    pose_enc 直接解码就是 c2w,对 chess GT 位置残差 31.5cm vs w2c 解释 73.5cm)。
    读取时统一用这个转成 c2w 再用。ext_w2c: (N,3,4) → (N,4,4) c2w。"""
    N = ext_w2c.shape[0]
    c2w = np.tile(np.eye(4), (N, 1, 1))
    Rw = ext_w2c[:, :3, :3].astype(np.float64)
    tw = ext_w2c[:, :3, 3].astype(np.float64)
    c2w[:, :3, :3] = Rw.transpose(0, 2, 1)
    c2w[:, :3, 3] = -np.einsum('nij,nj->ni', Rw.transpose(0, 2, 1), tw)
    return c2w


def camera_centers(extrinsic):
    """extrinsic: (N,3,4) c2w → 相机在世界系的位置 (N,3),即平移列。"""
    return extrinsic[:, :3, 3].astype(np.float64)


def retrieve(query_desc, db_desc, k, exclude_idx=None):
    """余弦相似度 top-k 检索(描述子已 L2 归一化,点积=余弦)。"""
    sims = db_desc @ query_desc
    if exclude_idx is not None:
        sims[exclude_idx] = -np.inf
    return np.argsort(-sims)[:k]


def retrieve_adaptive(query_desc, db_desc, db_centers, k=8, exclude_idx=None,
                      sim_ratio=0.7, spatial_radius_mult=5.0, baseline=1.0,
                      over_fetch=2, spatial_sort=False, scatter_thresh=10.0):
    """自适应检索:相似度阈值过滤 + 空间聚类(2026-08-05 实验改进)。

    解决两个问题:
    1. 固定 k=8 在 top-1→8 衰减大时混入不相关帧 → 用 sim_ratio 阈值过滤
    2. top-8 空间散装放大 Sim(3) 误差 → 以 top-1 为中心做空间聚类

    三种空间策略:
    - spatial_sort=False & scatter_thresh=0: 半径过滤,选 spatial_radius_mult×baseline 内的候选
    - spatial_sort=True: 始终按到 top-1 的空间距离排序取最近 k 个
    - spatial_sort=False & scatter_thresh>0(默认): 混合策略——
      先算 top-k 的最大两两距离,>scatter_thresh×baseline 才用 spatial_sort,否则保持纯 top-k

    参数:
      sim_ratio:        保留相似度 >= top1 * sim_ratio 的候选(默认 0.7,设 0 禁用)
      spatial_radius_mult: 聚类半径 = spatial_radius_mult * baseline(默认 5×基线)
      over_fetch:       预取 k * over_fetch 个候选做筛选(默认 2,即 top-16)
      scatter_thresh:   混合策略阈值(top-k 最大两两距离 > 该值×baseline 才启用 spatial_sort)
    返回:选中的 db 索引数组(长度 4~k)
    """
    sims = db_desc @ query_desc
    if exclude_idx is not None:
        sims[exclude_idx] = -np.inf

    n_fetch = min(k * over_fetch, len(sims))
    topk = np.argsort(-sims)[:n_fetch]
    top_sims = sims[topk]

    # 1. 相似度阈值过滤:去掉远低于 top-1 的候选
    if sim_ratio > 0:
        sim_thresh = top_sims[0] * sim_ratio
        valid_sim = top_sims >= sim_thresh
    else:
        valid_sim = np.ones(len(topk), dtype=bool)

    # 2. 判断是否需要空间聚类(混合策略)
    topk_k = topk[:k]
    topk_centers = db_centers[topk_k]
    pairwise = np.linalg.norm(topk_centers[:, None] - topk_centers[None, :], axis=-1)
    max_pairwise = pairwise.max()
    do_spatial = spatial_sort or (scatter_thresh > 0 and max_pairwise > scatter_thresh * baseline)

    if do_spatial:
        # 以 top-1 为中心,按空间距离排序取最近的 k 个
        top1_center = db_centers[topk[0]]
        dists_to_top1 = np.linalg.norm(db_centers[topk] - top1_center, axis=1)
        spatial_order = np.argsort(dists_to_top1)
        candidates = topk[spatial_order][valid_sim[spatial_order]]
        selected = candidates[:k]
    else:
        # top-k 已经紧凑,直接用(但仍过滤相似度)
        selected = topk_k[valid_sim[:k]]
        if len(selected) < 4:
            selected = topk_k

    # 3. 不足 4 帧(Sim(3) 至少 3 点 + 1 冗余),退化为纯 top-k
    if len(selected) < 4:
        selected = topk_k

    return selected


def selftest(db, k):
    desc = db["descriptors"].astype(np.float32)
    centers = camera_centers(w2c_to_c2w(db["extrinsic"])[:, :3, :4])
    n = desc.shape[0]

    baseline = np.linalg.norm(np.diff(centers, axis=0), axis=1).mean()

    sims = desc @ desc.T
    np.fill_diagonal(sims, -np.inf)
    topk = np.argsort(-sims, axis=1)[:, :k]

    dist = np.linalg.norm(centers[:, None] - centers[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)

    top1_err = []
    hit_at_k = 0
    hit_radius = 0
    true_nearest = np.argmin(dist, axis=1)

    for i in range(n):
        retrieved = topk[i]
        top1_err.append(dist[i, retrieved[0]])
        if true_nearest[i] in retrieved:
            hit_at_k += 1
        if (dist[i, retrieved] < 5 * baseline).any():
            hit_radius += 1

    top1_err = np.array(top1_err)
    print(f"数据库规模: {n} 帧, k={k}")
    print(f"相邻帧平均间距(基线): {baseline:.3f} (地图单位,单目尺度任意)")
    print(f"top-1 距离误差: 中位 {np.median(top1_err):.3f}, "
          f"均值 {top1_err.mean():.3f}, 90 分位 {np.percentile(top1_err, 90):.3f} (地图单位)")
    print(f"recall@{k}(真正最近邻被检回的比例): {hit_at_k / n:.1%}")
    print(f"top-{k} 内命中 5x基线半径的比例: {hit_radius / n:.1%}")
    worst = np.argsort(-top1_err)[:5]
    print("检索最差的 5 帧(帧号: top-1 距离):",
          [(int(i), round(float(top1_err[i]), 2)) for i in worst])


def rerank_dinov2_patches(query_patches, candidate_patches_list, topk_indices, k=8):
    """DINOv2 patch-level 重排(EffoVPR 零样本方法,2026-08-05)。

    原理:DINOv2 [CLS] token 只捕捉全局语义相似,不验证几何一致性。
    用 patch-level 互最近邻匹配做第二阶段验证,过滤"纹理像但位置不同"的视觉混淆。

    query_patches: (P, D) L2归一化的查询帧 patch 特征
    candidate_patches_list: list of (P, D) L2归一化的候选帧 patch 特征
    topk_indices: 原始 top-k 候选的 DB 索引
    k: 返回的候选数

    返回: 重排后的 DB 索引(按 patch 互匹配数降序)
    """
    scores = []
    for cand_patches in candidate_patches_list:
        # 互最近邻匹配: cand[i] 的最佳是 query[j], 且 query[j] 的最佳是 cand[i]
        sim = cand_patches @ query_patches.T  # (P_cand, P_query)
        cand_argmax = sim.argmax(axis=1)       # (P_cand,) 每个候选patch的最佳查询patch
        query_argmax = sim.argmax(axis=0)       # (P_query,) 每个查询patch的最佳候选patch
        # 互匹配: query_argmax[cand_argmax[i]] == i
        mutual_mask = query_argmax[cand_argmax] == np.arange(len(cand_argmax))
        scores.append(int(mutual_mask.sum()))

    scores = np.array(scores)
    reranked_order = np.argsort(-scores)
    return topk_indices[reranked_order][:k], scores[reranked_order]


# =============================================================================
# Sim(3) 对齐(Umeyama)与误差度量
# =============================================================================

def umeyama(src, dst):
    """求 s,R,t 使 dst ≈ s·R·src + t。src/dst: (k,3)。k>=3 且不全共线时良态。"""
    k = src.shape[0]
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d
    C = (Y.T @ X) / k
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:  # 防反射
        D[2, 2] = -1
    R = U @ D @ Vt
    var = (X ** 2).sum() / k
    s = (S * np.diag(D)).sum() / var
    t = mu_d - s * (R @ mu_s)
    residual = np.sqrt(((dst - (s * (src @ R.T) + t)) ** 2).sum(axis=1)).mean()
    return s, R, t, residual


# --- 旋转的四元数平均(用于旋转一致性对齐) --------------------------------

def rot_to_quat(R):
    """3x3 旋转矩阵 → 四元数 (w,x,y,z)。"""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def quat_to_rot(q):
    """四元数 (w,x,y,z) → 3x3 旋转矩阵。"""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def align_sim3_rotcons(fresh_c2w, map_ext, inlier_deg=20.0):
    """旋转一致性 Sim(3) 对齐(关键帧中心近平面时,比纯 Umeyama 良态得多)。

    fresh_c2w: (k,4,4) 关键帧在新窗口坐标系的 c2w
    map_ext:   (k,3,4) 同关键帧在地图坐标系的 c2w
    思路:每个关键帧给出一个全局旋转观测 R_i = R_map_i @ R_fresh_iᵀ,
    取互相一致的内点做四元数平均;再在 R 固定下用相机中心解 s 和 t。
    返回 (s, R, t, residual, inlier_mask)。
    """
    Rs = np.stack([map_ext[i, :3, :3] @ fresh_c2w[i, :3, :3].T
                   for i in range(fresh_c2w.shape[0])])
    k = Rs.shape[0]

    # 两两角距,选"和最多人一致"的当种子
    angs = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            angs[i, j] = rot_err_deg(Rs[i], Rs[j])
    seed = int(np.argmin(np.median(angs, axis=1)))
    inliers = angs[seed] < inlier_deg
    if inliers.sum() < 2:      # 实在没有一致对,退化为全部参与
        inliers = np.ones(k, dtype=bool)

    # Markley 四元数平均(先符号对齐到种子)
    q0 = rot_to_quat(Rs[seed])
    qs = np.stack([rot_to_quat(Rs[i]) for i in range(k) if inliers[i]])
    qs *= np.sign(qs @ q0)[:, None]
    M = qs.T @ qs
    _, vecs = np.linalg.eigh(M)
    R = quat_to_rot(vecs[:, -1])

    # R 固定,用相机中心解 s,t。
    # 尺度用"点对距离比的中位数"(恒正、抗离群);最小二乘在中心近似重合时会算出无意义的负尺度
    P = fresh_c2w[:, :3, 3]            # (k,3) 新坐标系中心
    Q = map_ext[:, :3, 3].astype(np.float64)  # (k,3) 地图中心
    Pi, Qi = P[inliers], Q[inliers]
    ratios = []
    m = Pi.shape[0]
    max_spread = np.linalg.norm(Pi - Pi.mean(axis=0), axis=1).max() + 1e-12
    for a in range(m):
        for b in range(a + 1, m):
            dp = np.linalg.norm(Pi[a] - Pi[b])
            if dp > 0.1 * max_spread:  # 忽略几乎重合的点对(比值是噪声)
                ratios.append(np.linalg.norm(Qi[a] - Qi[b]) / dp)
    s = float(np.median(ratios)) if ratios else 1.0
    p_bar, q_bar = Pi.mean(axis=0), Qi.mean(axis=0)
    t = q_bar - s * (R @ p_bar)
    residual = float(np.sqrt(((Qi - (s * (Pi @ R.T) + t)) ** 2).sum(axis=1)).mean())
    return s, R, t, residual, inliers


def rot_err_deg(R_est, R_gt):
    cos = (np.trace(R_gt.T @ R_est) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# =============================================================================
# lingbot-map 模型(加载逻辑与 my_demo.py 一致:mmap + assign,省内存)
# =============================================================================

def load_lingbot_model(args, device):
    from lingbot_map.models.gct_stream_window import GCTStream
    import torch

    print("构建 GCTStream(windowed)模型...")
    model = GCTStream(
        img_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=8,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,  # Jetson 无 flashinfer,必须 sdpa
        camera_num_iterations=4,
    )
    print(f"加载 checkpoint: {args.model_path}")
    ckpt = torch.load(args.model_path, map_location="cpu", mmap=True, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    del ckpt, state_dict
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    model = model.to(device).eval()
    model.aggregator = model.aggregator.to(dtype=torch.bfloat16)
    return model


def run_window_c2w(model, window_images, device, dtype, num_scale_frames=None):
    """window_images: (N,3,H,W) [0,1] → 每帧 c2w (N,4,4) numpy(该窗口自身的坐标系)。

    num_scale_frames: 双向注意力"尺度帧"数量。默认 min(8, N-1)(留 1 帧流式);
    传 N 则全部帧双向注意力(≈原版 VGGT 的全量注意力,无流式近似)。
    """
    import torch
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    from lingbot_map.utils.geometry import closed_form_inverse_se3_general

    imgs = window_images.to(device)
    n = imgs.shape[0]
    nsf = min(8, n - 1) if num_scale_frames is None else min(num_scale_frames, n)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        preds = model.inference_windowed(
            imgs,
            window_size=n,
            num_scale_frames=nsf,
            output_device=torch.device("cpu"),
        )
    # pose_encoding_to_extri_intri 的输出直接就是 c2w(2026-07-30 对 chess GT 实证,
    # 不要再求逆!demo.py 存 npz 时多求的一次逆是上游 bug,读取方用 w2c_to_c2w 修正)
    extrinsic, _ = pose_encoding_to_extri_intri(preds["pose_enc"], imgs.shape[-2:])
    n_out = extrinsic.reshape(-1, 3, 4).shape[0]
    c2w = np.tile(np.eye(4), (n_out, 1, 1))
    c2w[:, :3, :4] = extrinsic.reshape(-1, 3, 4).float().cpu().numpy()
    return c2w


# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="首帧定位工具库 + 检索自测")
    parser.add_argument("--db", required=True, help="build_map_db.py 生成的数据库")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    db = np.load(args.db)
    if args.selftest:
        selftest(db, args.k)
    else:
        parser.error("请指定 --selftest(完整定位自测已移至 eval_heldout.py --seed_expand)")


if __name__ == "__main__":
    main()
