"""build_map_db.py — 从 my_demo.py 的建图结果(.npz)构建首帧定位数据库

原理(第一步,最简单版本):
  建图时每个关键帧 → 用 DINOv2-small 提取一个 384 维全局描述子(图像的"指纹")
  → 和这个关键帧的相机位姿(注意:npz 里是 w2c!读取用 relocalize.w2c_to_c2w 转换)、内参一起存进数据库。
  苏醒时拍一张照片,也提描述子,找数据库里最相似的关键帧,就知道自己大概在哪。

用法(在 map conda 环境下):
    HF_ENDPOINT=https://hf-mirror.com python build_map_db.py \
        --map_npz output/short.npz --out map_db.npz

输入 npz 需包含:images (1,N,3,H,W) [0,1] 浮点、extrinsic (N,3,4) **w2c**(demo.py 多求了一次逆)、intrinsic (N,3,3)
—— 正好是 my_demo.py --save_output 保存的格式。
"""

import argparse

import numpy as np
import torch
from transformers import AutoModel

# ImageNet 归一化常数(DINOv2 预训练时用的就是这个)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def extract_descriptors(images, model, device, batch_size=8):
    """images: (N,3,H,W) float16 numpy,取值 [0,1]。返回 (N,384) L2 归一化描述子。"""
    n = images.shape[0]
    descs = []
    mean = torch.from_numpy(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.from_numpy(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    for start in range(0, n, batch_size):
        batch = torch.from_numpy(images[start:start + batch_size]).float().to(device)
        batch = (batch - mean) / std  # [0,1] → ImageNet 归一化
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(pixel_values=batch)
        cls = out.last_hidden_state[:, 0]  # [CLS] token 就是全局描述子
        cls = cls.float()
        cls = cls / cls.norm(dim=-1, keepdim=True)  # L2 归一化 → 点积=余弦相似度
        descs.append(cls.cpu().numpy())
        print(f"\r提取描述子 {min(start + batch_size, n)}/{n}", end="", flush=True)
    print()
    return np.concatenate(descs, axis=0).astype(np.float16)


def main():
    parser = argparse.ArgumentParser(description="构建首帧定位数据库(描述子 + 位姿)")
    parser.add_argument("--map_npz", required=True, help="my_demo.py --save_output 的结果")
    parser.add_argument("--out", default="map_db.npz", help="输出数据库文件")
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    print(f"加载建图结果 {args.map_npz} ...")
    data = np.load(args.map_npz)
    images = data["images"][0]          # (N,3,H,W) fp16 [0,1]
    extrinsic = data["extrinsic"]       # (N,3,4) w2c(保持原样存库,读取方负责转换)
    intrinsic = data["intrinsic"]       # (N,3,3)
    n = images.shape[0]
    print(f"共 {n} 帧,图像尺寸 {images.shape[3]}x{images.shape[2]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"加载描述子模型 {args.model} ...")
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    descriptors = extract_descriptors(images, model, device, args.batch_size)

    np.savez(
        args.out,
        descriptors=descriptors,        # (N,384) fp16,L2 归一化
        extrinsic=extrinsic,            # (N,3,4) w2c!不是 c2w,读取用 relocalize.w2c_to_c2w
        intrinsic=intrinsic,            # (N,3,3)
    )
    print(f"数据库已保存到 {args.out}: {n} 个关键帧,描述子维度 {descriptors.shape[1]}")


if __name__ == "__main__":
    main()
