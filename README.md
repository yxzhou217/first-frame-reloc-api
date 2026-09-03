# 首帧定位 API(First-Frame Relocalization API)

机器人"苏醒时拍一张照片 → 返回它在地图里的位姿"的 HTTP 服务。
基于 [lingbot-map](https://github.com/robbyant/lingbot-map) 做窗口联合重建,专为室内巡检机器人设计。

**管线(seed_expand)**:DINOv2 全局描述子检索 → VLM(Qwen3-VL)逐对确认主帧 → 主帧时间邻居凑窗口 → lingbot-map 窗口联合重建 → Sim(3) 对齐到地图坐标系。

## 环境安装

```bash
# 1. Python 3.10 conda 环境
conda create -n map python=3.10 -y && conda activate map

# 2. 安装上游 lingbot-map(本仓库的算法依赖:窗口重建模型)
git clone https://github.com/robbyant/lingbot-map.git
cd lingbot-map && pip install -e . && cd ..
# 按上游 README 下载模型权重,记为 /path/to/lingbot-map-long.pt

# 3. 本仓库
git clone <本仓库地址>
cd first-frame-reloc-api
pip install -r requirements.txt

# 4. DINOv2-small 描述子模型:首次运行自动从 HuggingFace 下载(~90MB)
#    国内网络: export HF_ENDPOINT=https://hf-mirror.com

# 5. VLM 验证服务(可选但强烈推荐):任意兼容 OpenAI Vision API 的端点
#    例如本地部署 Qwen3-VL-8B(vllm serve),地址传给 --vlm_url
#    或用环境变量: export VLM_URL=... VLM_API_KEY=... VLM_MODEL=...
```

## 地图准备

```bash
# ① 用 lingbot-map 对巡场视频建图,保存为 npz(images + extrinsic w2c + intrinsic)
#    (见上游 README 的 demo;若用 VidMap 等其它建图工具,转成相同 npz 格式即可)

# ② 构建检索数据库(DINOv2 描述子 + 位姿)
python build_map_db.py --map_npz output/your_map.npz --out output/your_db.npz
```

注意:
- 建图视频若已做旋正(如 VidMap 抽帧应用了旋转元数据),服务启动须加 `--no_rotate`
- 关键帧地图(只有关键帧有位姿)建议加 `--db_stride 2`(隔一取一作检索库)

## 启动服务

```bash
PYTHONPATH=/path/to/lingbot-map \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python reloc_server.py \
    --map_npz output/your_map.npz \
    --db output/your_db.npz \
    --model_path /path/to/lingbot-map-long.pt \
    --vlm_url http://127.0.0.1:8000 \
    --port 8100
```

常用参数:

| 参数 | 默认 | 说明 |
|---|---|---|
| `--map_npz` / `--db` | — | 地图 / 检索数据库 |
| `--model_path` | checkpoints/lingbot-map-long.pt | lingbot-map 权重 |
| `--k` | 8 | 窗口锚点数 |
| `--db_stride` | 无(空间子采样) | 隔 N 取一作检索库,关键帧地图建议 2 |
| `--target_db_ratio` | 0.46 | 空间子采样比例(稠密帧地图用) |
| `--no_rotate` | 关 | 查询照片不顺时针转 90°(地图帧已转正时必须加) |
| `--vlm_url` / `--vlm_model` | — | VLM 端点/模型名 |
| `--no_vlm` | 关 | 不用 VLM(仅调试,准确率会降) |
| `--port` | 8100 | 端口 |

启动约需 2~5 分钟(模型加载),`curl http://127.0.0.1:8100/health` 返回 `{"status":"ok"}` 即就绪。

### GPU 指定与启停

```bash
# 指定 GPU(代码不用改:CUDA_VISIBLE_DEVICES 会遮住其它卡,进程只看到指定的卡)
CUDA_VISIBLE_DEVICES=0 python reloc_server.py ...   # 用 0 号卡;N 换成任意卡号

# 或用仓库自带的脚本(默认 GPU 0,后台运行,日志在 reloc_server.log)
CUDA_VISIBLE_DEVICES=0 bash start.sh --map_npz output/map.npz --db output/db.npz --model_path /path/to/lingbot-map-long.pt

# 停止服务(释放显存)
bash stop.sh
# 或手动: pkill -f reloc_server.py  (前台运行的话直接 Ctrl+C)
```

## 调用 API

### 上传文件(推荐)

```bash
curl -X POST http://<服务器>:8100/localize -F "file=@photo.jpg"
```

### JSON + base64

```bash
curl -X POST http://<服务器>:8100/localize \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64 编码的图片>"}'
```

### Python

```python
import requests
with open("photo.jpg", "rb") as f:
    res = requests.post("http://<服务器>:8100/localize",
                        files={"file": ("photo.jpg", f, "image/jpeg")}, timeout=60).json()
if res["success"]:
    print(res["position"], res["quaternion_wxyz"])
```

查询图若已是地图同规格(如直接从建图 npz 取帧测试),加 `?rotate=false` 跳过旋转。

### 返回字段

| 字段 | 说明 |
|---|---|
| `success` | 是否可信。VLM 否决全部候选 / 对齐残差>2×基线 时为 false(保护机制) |
| `position` | 相机位置(地图单位;VidMap 地图=米) |
| `quaternion_wxyz` / `rotation_matrix` | 姿态 |
| `scale` | Sim(3) 对齐尺度,应 ≈1 |
| `confidence` | `residual` / `residual_x_baseline` / `inliers` |
| `debug` | `seed_frame`(主帧号)/ `vlm_calls` / `latency_ms` 等 |

## 精度评估(留出法)

```bash
# 把地图帧划分为检索库/查询池,查询池帧当"苏醒照片"打给运行中的服务
python eval_via_api.py --num_queries 100 \
    --map_npz output/your_map.npz --db output/your_db.npz
# 关键帧地图加 --db_stride 2(与服务启动参数一致);--num_queries 999 = 查询全测
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `reloc_server.py` | API 服务本体(FastAPI) |
| `relocalize.py` | 核心算法库:检索 / Sim(3) 对齐(Umeyama+旋转共识) / 窗口重建 |
| `vlm_rerank.py` | VLM 图片对验证(兼容 OpenAI Vision API) |
| `build_map_db.py` | 构建检索数据库 |
| `eval_via_api.py` | 经 HTTP 的留出法评估 |
| `eval_heldout.py` | 离线留出法评估(含 patch 重排等实验模式) |
| `start.sh` / `stop.sh` | 启停脚本(后台运行、GPU 指定、停止释放显存) |

## 许可

Apache License 2.0。本项目构建于 [lingbot-map](https://github.com/robbyant/lingbot-map)(Apache 2.0)之上,感谢其作者。
