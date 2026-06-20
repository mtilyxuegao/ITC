#!/usr/bin/env bash
#
# 在你本机(Mac/Linux)运行,把远程服务器的 8006 端口转发到本地,
# 然后浏览器打开 https://localhost:8006 即可访问 demo(自签证书,点"继续前往")。
#
# 用法:
#   SSH_KEY=~/.ssh/itc SERVER=ubuntu@192.222.53.81 bash tunnel.sh
set -euo pipefail

SSH_KEY="${SSH_KEY:-$HOME/.ssh/itc}"
SERVER="${SERVER:-ubuntu@192.222.53.81}"
PORT="${PORT:-8006}"

echo "建立隧道: 本地 $PORT -> $SERVER:$PORT (Ctrl+C 断开)"
ssh -i "$SSH_KEY" -p 22 -N -L "${PORT}:localhost:${PORT}" "$SERVER"
