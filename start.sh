#!/bin/bash
# 启动首帧定位 API(后台运行,日志在当前目录 reloc_server.log)
# 用法: bash start.sh --map_npz output/map.npz --db output/db.npz --model_path /path/to/lingbot-map-long.pt [其它参数]
# 指定 GPU: CUDA_VISIBLE_DEVICES=0 bash start.sh ...(默认 0 号卡;N 换成想要的卡号)
# 停止: bash stop.sh

if pgrep -f "reloc_server.py" > /dev/null; then
    echo "服务已在运行 (PID: $(pgrep -f reloc_server.py | tr '\n' ' ')),先 bash stop.sh 再重启"
    exit 0
fi

# 启动前检查当前 python 环境依赖是否齐全(防止用错 conda 环境时静默失败)
if ! python -c "import torch, transformers, fastapi, uvicorn, PIL, requests" 2>/dev/null; then
    echo "错误: 当前 python($(which python))缺依赖,你是不是没激活装了依赖的 conda 环境?"
    echo "先执行: conda activate <环境名>  (环境安装见 README)"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python reloc_server.py "$@" > reloc_server.log 2>&1 &

echo "已启动 (PID: $!, GPU: $CUDA_VISIBLE_DEVICES),模型加载约需 2~5 分钟"
echo "确认就绪: curl http://127.0.0.1:8100/health  (返回 {\"status\":\"ok\"} 才算好)"
