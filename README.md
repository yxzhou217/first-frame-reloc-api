# 首帧定位 API(First-Frame Relocalization API)

机器人"苏醒时拍一张照片 → 返回它在地图里的位姿"的 HTTP 服务。
管线:DINOv2 检索 → VLM 逐对确认主帧 → 主帧时间邻居凑窗口 → [lingbot-map](https://github.com/robbyant/lingbot-map) 窗口联合重建 → Sim(3) 对齐到地图坐标系。

## 环境安装

```bash
# 1. conda 环境(Python 3.10)
conda create -n map python=3.10 -y && conda activate map

# 2. 上游 lingbot-map(窗口重建模型依赖)
git clone https://github.com/robbyant/lingbot-map.git
cd lingbot-map && pip install -e . && cd ..
# 按上游 README 下载模型权重,记为 /path/to/lingbot-map-long.pt

# 3. 本仓库
git clone https://github.com/yxzhou217/first-frame-reloc-api && cd first-frame-reloc-api
pip install -r requirements.txt
```

首次运行会自动下载 DINOv2-small(~90MB);国内网络先 `export HF_ENDPOINT=https://hf-mirror.com`。

## 准备地图

```bash
# 建图 npz(images + 位姿 w2c + 内参)由 lingbot-map / VidMap 等工具产出
# 然后构建检索数据库:
python build_map_db.py --map_npz /path/to/map.npz --out /path/to/db.npz
```

注意:建图帧若已旋正(如 VidMap 抽帧已应用旋转元数据),启动服务时须加 `--no_rotate`。

## 启动与停止

```bash
# 前台启动(测试用)
export VLM_API_KEY="<你的 VLM key>"          # VLM 鉴权 key,从环境变量读,不写进任何文件
python reloc_server.py \
    --map_npz output/map.npz \
    --db output/db.npz \
    --model_path /path/to/lingbot-map-long.pt \
    --vlm_url <VLM 服务地址> \
    --no_rotate            # 地图帧已转正时加

# 后台启动(生产用,默认 GPU 0,日志在 reloc_server.log)
CUDA_VISIBLE_DEVICES=0 bash start.sh --map_npz ... --db ... --model_path ... [其它参数同上]

# 停止
bash stop.sh               # 或 pkill -f reloc_server.py;前台运行则 Ctrl+C
```

启动需 2~5 分钟加载模型,`curl http://127.0.0.1:8100/health` 返回 `{"status":"ok"}` 即就绪。

常用参数:

| 参数 | 默认 | 说明 |
|---|---|---|
| `--map_npz` / `--db` / `--model_path` | — | 地图 / 检索库 / lingbot-map 权重(必填) |
| `--vlm_url` / `--vlm_model` | — | VLM 端点(兼容 OpenAI Vision API)/ 模型名 |
| `--no_vlm` | 关 | 不用 VLM(仅调试,精度会降) |
| `--k` | 8 | 窗口锚点数 |
| `--db_stride` | 无 | 隔 N 取一作检索库;关键帧地图建议 2(稠密帧地图用默认的空间子采样即可) |
| `--no_rotate` | 关 | 查询照片不顺时针转 90°(地图帧已转正时必须加) |
| `--port` | 8100 | 监听端口;8100 被占或多实例时才需要改 |

## 调用

```bash
# 上传文件(比如查询图的路径为photo.jpg)
curl -X POST http://127.0.0.1:8100/localize -F "file=@photo.jpg"

```

```python
import requests
with open("photo.jpg", "rb") as f:
    res = requests.post("http://127.0.0.1:8100/localize",
                        files={"file": ("photo.jpg", f, "image/jpeg")}, timeout=60).json()
if res["success"]:
    print(res["position"], res["quaternion_wxyz"])   # 位置 + 姿态(四元数 wxyz)
else:
    print(res.get("error") or res["warnings"])
```

返回字段:

| 字段 | 说明 |
|---|---|
| `success` | 是否可信(VLM 否决全部候选 / 对齐残差>2×基线 → false) |
| `position` / `quaternion_wxyz` / `rotation_matrix` | 位姿(地图单位;米制地图则为米) |
| `scale` | Sim(3) 对齐尺度;米制地图上偏离 1 是正常的,看残差判断 |
| `confidence` | `residual_x_baseline`(>2 可疑)、`inliers` |
| `debug` | `seed_frame`、`vlm_calls`、`latency_ms` 等 |

## 精度评估(可选)

```bash
# 服务运行中,把留出帧当查询打给 API(真值取自地图自身,属自洽性评估)
python eval_via_api.py --num_queries 100 --map_npz /path/to/map.npz --db /path/to/db.npz
# 关键帧地图加 --db_stride 2(与服务启动一致);--num_queries 999 = 查询全测
```

## 文件说明

| 文件 | 作用 |
|---|---|
| `reloc_server.py` | API 服务本体 |
| `relocalize.py` | 核心算法库(检索 / Sim(3) 对齐 / 窗口重建) |
| `vlm_rerank.py` | VLM 图片对验证 |
| `build_map_db.py` | 构建检索数据库 |
| `eval_via_api.py` / `eval_heldout.py` | 评估脚本(经 HTTP / 离线) |
| `start.sh` / `stop.sh` | 后台启停(含依赖检查、GPU 指定) |

## 许可

Apache License 2.0。本项目构建于 [lingbot-map](https://github.com/robbyant/lingbot-map)(Apache 2.0)之上。
