# 首帧定位 API

输入一张照片，返回相机在已有地图中的位置和朝向。

流程：CPU 质检 → DINOv2 检索 → patch 重排 → VLM 确认 → 空间邻近锚点 → 窗口重建 → Sim(3) 对齐。

## 1. 安装环境

需要 NVIDIA GPU。以下使用已验证的 Python 3.10、PyTorch 2.5.1 / CUDA 12.1 环境，以 VGGT-Omega 为重建后端。

```bash
conda create -n map python=3.10 -y
conda activate map

git clone https://github.com/yxzhou217/first-frame-reloc-api.git
git clone https://github.com/facebookresearch/vggt-omega.git
git -C vggt-omega checkout 282ec70363edeff59424bf43731658092fba3d37
cd first-frame-reloc-api

python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt -r ../vggt-omega/requirements.txt \
    "numpy==1.26.4" "opencv-python==4.11.0.86" "transformers==5.14.1"
python -m pip install -e ../vggt-omega --no-deps
```

最后一条是**可编辑安装**：让 `map` 环境能导入 `vggt_omega`，修改源码后无需重装；安装后保留该源码目录。`--no-deps` 表示依赖已由上一条命令安装。

从 [VGGT-Omega 官方模型页](https://huggingface.co/facebook/VGGT-Omega)申请访问并下载 `vggt_omega_1b_512.pt`，放入 `../vggt-omega/checkpoints/`。首次启动会下载 DINOv2-small；离线部署可用 `--dino_path /path/to/local/dinov2-small` 指定本地模型目录。

## 2. 准备地图

需要建图程序导出的 `map.npz`，以及按相同方向保存的地图原图目录。NPZ 中各字段须采用同一帧序：

| 字段 | 格式 |
|---|---|
| `images` | `(1,N,3,H,W)`，RGB、取值 `[0,1]`，供检索使用 |
| `extrinsic` | `(N,3,4)`，**w2c** 外参 |
| `intrinsic` | `(N,3,3)`，对应预处理后图像的内参 |
| `names` | `(N,)`，原图相对文件名；`names[i]` 必须对应第 i 帧 |

VGGT-Omega 重建会读取 `原图目录/names[i]`，独立执行官方预处理；原图必须保留，不能用 NPZ 中缩小的图代替。

`images` 必须与 `reloc_server.py` 的 `preprocess_query()` 一致：宽缩至 518，高度按 14 对齐，超过 518 时中心裁剪。查询预处理后的尺寸须与地图 `images` 一致，通常使用同相机、同画幅；Omega 的重建原图单独保留。

构建检索库：

```bash
conda activate map
CUDA_VISIBLE_DEVICES=7 python build_map_db.py --map_npz /path/to/map.npz --out /path/to/db.npz
```

## 3. 启动服务

替换下面的路径、VLM 地址和模型名。示例使用 GPU7、端口 8101，按部署机器调整；VLM 密钥只通过环境变量提供。

```bash
conda activate map
export VLM_API_KEY="你的密钥"

CUDA_VISIBLE_DEVICES=7 python reloc_server.py \
    --backend vggt_omega \
    --model_path ../vggt-omega/checkpoints/vggt_omega_1b_512.pt \
    --map_npz /path/to/map.npz \
    --db /path/to/db.npz \
    --map_image_dir /path/to/original_map_images \
    --vlm_url "https://your-vlm-endpoint" \
    --vlm_model "Qwen/Qwen3-VL-8B-Instruct" \
    --db_stride 1 --spatial_diversity --no_rotate \
    --port 8101
```

- `--map_image_dir`：Omega 必填，配合地图的 `names` 读取原图。
- `--db_stride 1`：保留全部地图帧参与检索，关键点地图使用此设置。
- `--no_rotate`：查询不额外旋转；地图帧已经旋正时使用。查询与地图方向须一致，省略此项会将查询顺时针旋转 90°。
- 质检、patch 重排和空间窗口默认开启；质检阈值默认 40，锚点数默认 8。可用 `--min_quality`、`--k` 调整；`--no_vlm` 仅供不调用 VLM 的调试。

检查就绪：

```bash
curl http://127.0.0.1:8101/health
```

返回 `"status":"ok"` 后再请求定位。前台服务按 `Ctrl+C` 停止；后台运行可将相同启动参数传给 `bash start.sh`。

## 4. 请求定位

```bash
curl -X POST http://127.0.0.1:8101/localize \
    -F "file=@/path/to/query.jpg"
```

仅在 `success=true` 时使用位姿：

- `position`：地图坐标系中的相机位置，单位与地图一致。
- `quaternion_wxyz` / `rotation_matrix`：地图坐标系中的相机朝向，四元数顺序为 **w,x,y,z**。
- `confidence`：对齐残差和内点数；`debug` 包含主帧、锚点及耗时。
- `success=false`：查看 `error` 或 `warnings`；质检拒绝时还会返回 `quality`。

## 5. 切换到 LingBot-map（可选）

在同一 `map` 环境安装：

```bash
git clone https://github.com/robbyant/lingbot-map.git ../lingbot-map
git -C ../lingbot-map checkout 1f480aeb8a47a24656090d46d053115b7fe60435
python -m pip install -e ../lingbot-map "numpy==1.26.4" "opencv-python==4.11.0.86"
```

从 [LingBot-map 官方模型页](https://huggingface.co/robbyant/lingbot-map)下载 `lingbot-map-long.pt`。沿用上面的启动命令，将 `--backend` 改为 `lingbot`、`--model_path` 改为该权重路径，并去掉 `--map_image_dir`。代码默认后端仍为 `lingbot`，使用 Omega 时必须显式指定 `--backend vggt_omega`。

许可：[Apache License 2.0](LICENSE)。
