#!/bin/bash
# 停止首帧定位 API,释放显存
if pkill -f "reloc_server.py"; then
    echo "已停止"
else
    echo "服务未在运行"
fi
