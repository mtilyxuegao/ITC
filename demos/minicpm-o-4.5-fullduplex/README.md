# MiniCPM-o 4.5 —— 全双工(full-duplex)多模态 Web Demo

[OpenBMB 官方 MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo) 的部署记录。
这是目前**公开可跑通、最接近 ITC 所复刻的 Interaction Model** 的真·全双工多模态模型:
能**边看、边听、边说**(speech+text 输出流与 video+audio 输入流互不阻塞),支持中途打断和主动开口。

> 与之对照的 **turn-based(非双工)** 实现见 [`../livekit-voice-agent`](../livekit-voice-agent)。

## 实测环境

| 项 | 值 |
| --- | --- |
| OS | Ubuntu 22.04.5 |
| GPU | 4× NVIDIA H100 80GB(本部署**只用 GPU 0 / GPU 1**) |
| Docker / Compose | 28.3.1 / v2.38.1 |
| CUDA / 驱动 | 12.8 / 570.148.08 |
| 模型 | `openbmb/MiniCPM-o-4_5`(完整 PyTorch 版,~19GB,bf16) |
| 上游 commit | `a166da4`(Merge PR #44 feat/demo-tts-enable) |

> ⚠️ 此方案**必须 NVIDIA GPU(≥28GB 显存)+ Linux + NVIDIA Container Toolkit**。
> macOS / Apple Silicon 跑不了 CUDA 版;Mac 上请用官方预编译桌面 app(llama.cpp-omni / GGUF,走 Metal)。

## 一键部署

```bash
bash deploy.sh          # 克隆 → 下模型 → 修依赖 → 出证书 → build → up
```

部署完成后从你本机访问:

```bash
SSH_KEY=~/.ssh/itc SERVER=ubuntu@<服务器IP> bash tunnel.sh
# 然后浏览器打开 https://localhost:8006(自签证书,点"高级 → 继续前往")
```

## 手动步骤(deploy.sh 做的事)

```bash
# 1. 克隆
git clone https://github.com/OpenBMB/MiniCPM-o-Demo.git && cd MiniCPM-o-Demo

# 2. 下模型(~19GB)
pip3 install --user huggingface_hub
hf download openbmb/MiniCPM-o-4_5 --local-dir ~/models/MiniCPM-o-4_5

# 3. 修依赖冲突(见下「踩坑」)
git apply ../requirements.fix.patch   # 或手动改 requirements.txt

# 4. 自签证书
mkdir -p certs data
openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
  -keyout certs/key.pem -out certs/cert.pem -subj "/CN=minicpm-o"

# 5. 构建 + 启动(默认 2 worker → GPU 0/1)
MODEL_HOST_PATH=~/models/MiniCPM-o-4_5 docker compose up -d --build
```

## 踩坑记录:requirements 依赖冲突(必修,否则 build 失败)

上游 `requirements.txt` 自相矛盾,导致 `pip install -r requirements.txt` 在镜像构建时
`ResolutionImpossible`:

- 它要求 `librosa>=0.10.2`(注释说 0.9.x 在新 setuptools 下缺 `pkg_resources` 导入即崩);
- 但它的依赖 `minicpmo-utils`(核心硬依赖,非 extra)锁死 `librosa==0.9.0`。

**修法**(见 `requirements.fix.patch`):
- 把 `librosa>=0.10.2` 改回 `librosa==0.9.0`(顺应 minicpmo-utils);
- 加 `setuptools<81`,保留 `pkg_resources`,正好解决上面那个导入崩溃。

## 架构(三层)

```
浏览器 ──HTTPS:8006──> Gateway ──> Worker ──> Backend(真正跑模型)
                     (路由/UI,    (转发隔离)  (:22500,独占 1 张 GPU)
                      torch-free)
```

- **Gateway**(`gateway.py`,:8006):对外 HTTPS、网页 UI、会话路由、录制持久化。**不吃 GPU**。
- **Worker**(`worker.py`,:22400):纯转发隔离层。
- **Backend**(`py_backend.server`,:22500):加载模型、跑推理。**每个独占一张 GPU**。
- 默认两个 `worker-backend`,`device_ids` 显式绑定 GPU 0 与 GPU 1,各占约 21.8GB 显存。
- 增减 GPU = 增删 `docker-compose.yml` 里的 `worker-backend-N` 并同步 gateway 的 `--workers` 列表。

## 推理实现:原生 PyTorch,不是 vLLM/SGLang

Backend 用 **HuggingFace transformers 4.51 + torch 2.8(eager)**,手写自回归循环
(自管 `past_key_values`、flash-attention-2 / sdpa)。两种模式:

- `turn_based`(`_stream_turn_based`):一问一答。
- `duplex`(`streaming_prefill` + `streaming_generate`):滑动窗口流式,边听边说,可打断。

**为什么不用 vLLM/SGLang**:它们是"收齐 prompt → 批量推理"的轮次范式,天生非双工,
套不进全双工的可中断/交替流式控制。各推理框架对 MiniCPM-o 4.5 的支持对比见
[`../README.md`](../README.md)。

## 常用运维

```bash
cd ~/MiniCPM-o-Demo
export MODEL_HOST_PATH=~/models/MiniCPM-o-4_5
docker compose logs -f gateway            # gateway 日志
docker compose logs -f worker-backend-0   # 模型/推理日志
docker compose ps                         # 状态
docker compose down                       # 停(释放显存)
```
