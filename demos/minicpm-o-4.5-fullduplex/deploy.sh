#!/usr/bin/env bash
#
# 在 NVIDIA GPU + Linux 服务器上一键部署 MiniCPM-o 4.5 官方全功能 Web Demo。
# 实测环境:Ubuntu 22.04 + 4× H100 80GB(只用 GPU 0/1)+ Docker 28 + Compose v2 + CUDA 12.8。
#
# 复刻自我们实际跑通的步骤。详见同目录 README.md。
#
# 用法:
#   bash deploy.sh
# 可调环境变量:
#   WORKDIR     代码克隆位置(默认 ~/MiniCPM-o-Demo)
#   MODEL_DIR   模型下载位置(默认 ~/models/MiniCPM-o-4_5)
set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/MiniCPM-o-Demo}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/MiniCPM-o-4_5}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==> [0/6] 前置检查"
command -v docker >/dev/null || { echo "缺 docker"; exit 1; }
docker compose version >/dev/null || { echo "缺 docker compose v2"; exit 1; }
nvidia-smi -L || { echo "缺 NVIDIA 驱动/GPU"; exit 1; }
echo "    容器内 GPU 自检:"
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi -L

echo "==> [1/6] 克隆官方 demo 仓库"
if [ -d "$WORKDIR/.git" ]; then
  echo "    已存在,git pull"; git -C "$WORKDIR" pull --ff-only
else
  git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git "$WORKDIR"
fi
# 实测可跑通的上游 commit:a166da4 (Merge PR #44 feat/demo-tts-enable)

echo "==> [2/6] 下载模型 openbmb/MiniCPM-o-4_5 (~19GB)"
if [ -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  echo "    模型已存在,跳过"
else
  pip3 install -q --user "huggingface_hub>=0.34.0" || true
  export PATH="$HOME/.local/bin:$PATH"
  hf download openbmb/MiniCPM-o-4_5 --local-dir "$MODEL_DIR"
fi

echo "==> [3/6] 应用 requirements 依赖冲突修复"
# 上游 requirements.txt 自相矛盾:librosa>=0.10.2 vs minicpmo-utils 硬依赖 librosa==0.9.0
# 这里用 sed 幂等修复(已修则跳过),等价于 requirements.fix.patch
cd "$WORKDIR"
if grep -q "librosa>=0.10.2" requirements.txt; then
  cp requirements.txt requirements.txt.bak
  sed -i 's/^librosa>=0.10.2/librosa==0.9.0  # 随 minicpmo-utils 硬依赖/' requirements.txt
  grep -q "^setuptools" requirements.txt || \
    sed -i '/^transformers==4.51.0/i setuptools<81  # keep pkg_resources for librosa 0.9.0' requirements.txt
  echo "    已修复(备份在 requirements.txt.bak)"
else
  echo "    已是修复后版本,跳过"
fi

echo "==> [4/6] 生成自签 SSL 证书"
mkdir -p certs data
[ -f certs/cert.pem ] || openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout certs/key.pem -out certs/cert.pem -subj "/CN=minicpm-o"

echo "==> [5/6] 构建镜像并启动(默认 2 worker:GPU 0 / GPU 1)"
export MODEL_HOST_PATH="$MODEL_DIR"
docker compose up -d --build

echo "==> [6/6] 状态"
docker compose ps
echo
echo "✅ 完成。Gateway 在服务器 https://localhost:8006"
echo "   从你本机访问:见 README 的 SSH 端口转发,或运行 tunnel.sh"
