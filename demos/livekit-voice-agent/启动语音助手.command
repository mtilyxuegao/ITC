#!/bin/zsh
# 双击此文件即可启动 LiveKit 语音助手 Kelly
cd "$(dirname "$0")"
echo "正在启动语音助手 Kelly..."
echo "启动后直接对着电脑说话即可(中英文都行)。按 Ctrl+C 退出。"
echo "------------------------------------------------------------"
~/.local/bin/uv run python agent.py console
