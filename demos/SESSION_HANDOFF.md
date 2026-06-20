# MiniCPM-o 4.5 — Session Handoff / 全量状态转储

> 目的：把这次会话做的所有事完整 offload，供**并行 agent** 直接接手。日期：2026-06-20。
> 关键词：MiniCPM-o 4.5 全双工多模态、llama.cpp-omni、vLLM-Omni、PyTorch demo、H100 vs B300 benchmark。

---

## 0. TL;DR — 当前 live 状态

- **B300 服务器上正在跑** MiniCPM-o 4.5 的 **Comni Web UI(llama.cpp-omni 后端,F16)**,可用。
- 访问：本机 SSH 隧道 `localhost:8010 → B300:8006`,页面 `https://localhost:8010/omni`(自签证书)。
- 刚给前端加了 **`Listen Prob Scale` 滑块**(控制"爱说 vs 易被打断"),已 docker cp 进 gateway 容器,硬刷新即可见。
- H100 服务器上有 vLLM-Omni 装好的环境(但 HTTP API 有 FastAPI 版本 bug,见 §5)。
- 结论先行:**单 request 场景 llama.cpp 最快;B300 比 H100 解码快 ~20%、TTS/延迟路径快 ~1.5x**。

---

## 1. 服务器 & 访问方式

| 机器 | 连接 | GPU | 驱动/CUDA | 备注 |
|---|---|---|---|---|
| **H100** | `ssh -i ~/.ssh/itc -p 22 ubuntu@192.222.53.81` | 4× H100 80GB | 570 / CUDA 12.8 | 共享机；**只用 GPU 0/1**,GPU 2/3 有别人(hackathon4 qwen_server)别碰 |
| **B300** | `ssh -i ~/.ssh/itc -p 22 root@86.38.238.57` | 1× B300 SXM6 275GB | 580 / CUDA 13.0 | root，独占，250GB RAM，30 cpu，~420GB 空闲盘 |
| Mac(本地) | — | Apple M4 Max 36GB | Metal | 跑过 LiveKit + Comni 桌面 app(见 §11) |

> ⚠️ 两台服务器都用同一把 key：`~/.ssh/itc`。SSH 里**别在复合命令开头用 `pkill -9 -f`**——会连带杀掉 ssh 会话导致 exit 255;按 PID kill 或拆成单独命令。

---

## 2. 部署总览（什么跑在哪）

| 部署 | 机器 | 后端 | 模型 | 端口/访问 | 状态 |
|---|---|---|---|---|---|
| PyTorch demo | H100 | py_backend(transformers+torch) | 全量 bf16 19GB | gw 8006 | 可能已 down（容器反复被重启策略拉起，注意）|
| llama.cpp Comni | H100 | llama.cpp-omni | GGUF | gw 8007 | 已 down |
| vLLM-Omni | H100 | vllm-omni 0.22 | 全量 bf16 | 8099(API坏) | 引擎能起，HTTP 坏 |
| **llama.cpp Comni** | **B300** | **llama.cpp-omni F16** | **GGUF F16** | **gw 8006 → 本地 8010** | **✅ 在跑** |

---

## 3. MiniCPM-o-Demo 部署细节

官方仓库：`https://github.com/OpenBMB/MiniCPM-o-Demo`（实测 commit `a166da4`）。两台都 clone 在 `~/MiniCPM-o-Demo`。

### 模型路径
- 全量(PyTorch)：`~/models/MiniCPM-o-4_5`（HF `openbmb/MiniCPM-o-4_5`，~19GB，bf16，4 个 safetensors 分片）
- GGUF(llama.cpp)：`~/models/MiniCPM-o-4_5-gguf`（HF `openbmb/MiniCPM-o-4_5-gguf`）
  - LLM：`MiniCPM-o-4_5-Q4_K_M.gguf`(4.7G) / `MiniCPM-o-4_5-F16.gguf`(16G)
  - 子模块：`audio/`, `tts/`(tts+projector), `vision/*.gguf`, `token2wav-gguf/*`（共 ~8G+16G）
  - 下载用 `~/.local/bin/hf download <repo> <file...> --local-dir ...`（H100 用 hf_transfer；B300 用系统 pip 装的 hf 1.20.1）

### 必修 bug：requirements 依赖冲突（PyTorch 路径）
上游 `requirements.txt` 自相矛盾：`librosa>=0.10.2` vs `minicpmo-utils` 硬依赖 `librosa==0.9.0` → 镜像构建 `ResolutionImpossible`。
**修法**（见 `requirements.fix.patch`）：`librosa==0.9.0` + 加 `setuptools<81`（保留 pkg_resources）。`deploy.sh` 已内置幂等 sed 修复。

### 启动命令
```bash
# PyTorch 路径（GPU 0/1，gw 8006）
cd ~/MiniCPM-o-Demo
MODEL_HOST_PATH=$HOME/models/MiniCPM-o-4_5 docker compose up -d --build

# C++/Comni 路径（GGUF，gw 端口可配）
GGUF_MODEL_HOST_PATH=$HOME/models/MiniCPM-o-4_5-gguf GATEWAY_HOST_PORT=8006 CPP_GPU_ID=0 \
  docker compose -f docker-compose.cpp.yml up -d --build
# 切量化：再加 GGUF_MODEL_FILE=MiniCPM-o-4_5-F16.gguf 重新 up -d（无需 rebuild）
```

### 架构（三层）
浏览器 ──HTTPS:8006──> **Gateway**(gateway.py，路由/UI/录制，torch-free) ──> **Worker**(worker.py，纯转发) ──> **Backend**(py_backend.server 或 llama-omni-server，独占 1 GPU)。
- gateway/worker 对 config 是**透明代理**（worker 是纯转发层）。
- 前端静态文件打包进 gateway 镜像 `/app/static/`（改了要 `docker cp` 进容器或重建）。

---

## 4. llama.cpp-omni 构建（B300 / Blackwell 关键）

仓库：`https://github.com/tc-mb/llama.cpp-omni`，clone 在 `~/llama.cpp-omni`。

- **B300 是 sm_103（Blackwell Ultra），必须 CUDA 13 工具链**。host 没装 nvcc → 在 CUDA13 devel 容器里编译，产物持久化到 host。
- 构建脚本 `~/build_llama.sh`（B300 上），核心：
  ```bash
  cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DCMAKE_CUDA_ARCHITECTURES="100;103"
  cmake --build build --target llama-omni-cli --target llama-omni-server -j 30
  ```
  ⚠️ `CMAKE_CUDA_ARCHITECTURES` 里的 `;` 必须**加引号**，否则 shell 把 `103` 当命令。
- 跑（需 CUDA 容器提供 runtime）：
  ```bash
  docker run --rm --gpus all -v $HOME/llama.cpp-omni:/work -v $HOME/models:/models -w /work \
    nvidia/cuda:13.0.0-devel-ubuntu22.04 ./build/bin/llama-omni-cli -m /models/.../<gguf> -ngl 99 -c 4096
  ```
- **docker-compose.cpp.yml 的 Dockerfile 已为 B300 改过**：`docker/Dockerfile.cpp-worker-backend` 里 `ARG CUDA_VERSION=13.0.0`、`ARG CMAKE_CUDA_ARCHITECTURES=100;103`（H100 上是 12.8.1 / 90）。
- CLI 测试用例：`--test <prefix> <n>`；纯音频 `audio_test_case/audio_test_case_ 2`；视觉+音频 `--omni --test omni_test_case/omni_test_case_ 9`。`--no-tts` 关语音、`--bench-vision <img>` 单测视觉编码。

---

## 5. vLLM-Omni（H100，环境已装好但 HTTP 坏）

venv：`~/vllm-omni-env`（**Python 3.12**，apt 装的）。
- **驱动 570 → CUDA 12.8 上限**，默认 vllm 拉 cu13 跑不了。解法：装 **cu129 轮子**（minor-version 兼容）：
  ```bash
  pip install "https://github.com/vllm-project/vllm/releases/download/v0.22.1/vllm-0.22.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --extra-index-url https://download.pytorch.org/whl/cu129
  pip install vllm-omni==0.22.0
  pip install librosa stepaudio2-minicpmo   # token2wav 声码器
  # 系统装 ninja-build
  ```
  装完：vllm 0.22.1+cu129，torch 2.11.0+cu129，`torch.cuda.is_available()=True`。
- 启动：`CUDA_VISIBLE_DEVICES=0,1 vllm serve ~/models/MiniCPM-o-4_5 --omni --trust-remote-code --port 8099`
  - **omni 是多 stage 引擎，单卡放不下两个 stage，需 2 GPU**。
  - 引擎能完整加载（LLM stage + token2wav stage 都 ready）。
- **❌ 已知 bug**：vllm 0.22.1 + vllm-omni 0.22.0 + FastAPI 0.138 不兼容，请求报 500 `'_IncludedRouter' object has no attribute 'path'`。降 fastapi 又和 starlette 1.x 冲突。**端到端 HTTP/TTS 在此版本组合下跑不通**。解码吞吐是用离线 `vllm bench throughput` 测的（绕过 HTTP）。

---

## 6. Benchmark 全量数据

### 6.1 单 request 解码（llama.cpp，--no-tts）
| 配置 | H100 | B300 | 提升 |
|---|---|---|---|
| Q4_K_M | 224 tok/s | **268 tok/s** | +20% |
| F16 | 157 tok/s | **188 tok/s** | +20% |

### 6.2 vLLM（H100，bf16，离线 bench）
- 单流：**129 output tok/s**
- 批量(256 并发)：**~15,500 output tok/s**（continuous batching，单卡）
- 结论：**单 request llama.cpp 更快；高并发吞吐 vLLM 碾压**。用户**只关心单 request** → llama.cpp 胜。

### 6.3 交互延迟（llama.cpp + TTS）
| 指标 | H100 Q4 | B300 Q4 | H100 F16 | B300 F16 |
|---|---|---|---|---|
| 首音(解码→第一段语音) | 127ms | **79ms** | 145ms | **92ms** |
| TTS 流式节奏 | ~400ms | **~280ms** | ~360ms | **~234ms** |
| 加载(含token2wav) | ~2-4s | ~5s | ~3-5s | ~5.6s |
| 冷启动首次 prefill(一次性) | — | ~1.1s | — | ~1.1s |
- **B300 的 TTS/延迟路径比 H100 快 ~1.5x**（比纯解码 +20% 更明显，因 token2wav 更吃算力/带宽）。
- 单 request 解码是 **launch/开销受限**，所以 B300 带宽优势在 batch=1 发挥不出（只 +20%）。

### 6.4 视觉+音频 vs 纯音频（B300，每 1s 块 prefill）
| | 纯音频 | 视觉+音频 | 差 |
|---|---|---|---|
| prefill(warm) | ~36ms | ~109ms | **+73ms** |
| 视觉编码/帧 | — | ~56ms（64 tok/帧，grid1×1）| — |
- 加视觉每块多 ~73ms，但都 ≪1000ms，**不破坏实时性**。高分辨率帧 token 更多、开销更大。

### PyTorch demo benchmark
- 工具：`benchmark.py`（全双工 per-turn 计时，LISTEN/SPEAK）。容器内跑：
  `docker exec -d minicpm-wb-0 bash -lc "cd /app && python benchmark.py --model-path /models/MiniCPM-o-4_5 --video assets/samples/compile.mp4 --gpu-id 0 ..."`
- H100：加载~14s、显存~21.8GB、SPEAK 单元 ~650-840ms（llm150-210+tts300+token2wav130）。
- **B300 上 PyTorch demo 没跑成**：镜像钉 torch 2.8+cu128 不含 sm_103，需升 cu130（torch 2.9+）。未做。

---

## 7. 推理框架对 MiniCPM-o 4.5 的支持（2026-06 调研）

| 方案 | 图/视频/音频输入 | 语音(TTS)输出 | 全双工 | 吞吐 | 备注 |
|---|---|---|---|---|---|
| 官方 PyTorch demo | ✅ | ✅ | ✅ | 低 | 参考实现，原生 eager |
| **llama.cpp-omni** | ✅ | ✅ | ✅ | 单流快 | **唯一为全双工优化**（论文作者自研，arxiv 2604.27393 §7）|
| vLLM-Omni 0.22 | ✅ | ✅(WAV) | ❌(轮次) | 高 | 有官方 MiniCPM-o 示例，本次 HTTP 坏 |
| 上游 vLLM | ✅ | ❌ | ❌ | 高 | 仅理解 |
| SGLang | ✅(仅理解) | ❌ | ❌ | 高 | `minicpmo.py` 收录 o4_5，无 TTS/duplex |
| sglang-omni | MiniCPM-o 未列入 | (有 TTS 管线) | (有 streaming-session 原语 PR #19171) | 高 | 离 MiniCPM-o 全功能最远 |
- 全双工套不进 vLLM/SGLang 的"收齐 prompt→批量"范式；只有 llama.cpp-omni / PyTorch 原生能做。
- SGLang PR #19171（streaming session + SessionAwareCache）= ITC「200ms chunk=请求、KV append 持久序列」的底层原语，对 ITC 自身有参考价值，但**不能直接跑 MiniCPM-o**。

---

## 8. 前端改动：Listen Prob Scale 控件（本次新增）

**目的**：暴露"模型爱说 vs 易被打断"的可调旋钮。
- 后端参数 **`listen_prob_scale`**（默认 1.0，≥0）：
  - **<1.0 更爱说**（话多、占话头、难打断）
  - **>1.0 更克制**（话少、倾向 listen、易被打断/让位）
  - cpp 实现：`omni.cpp:1339` `listen_bias=(scale-1)*2`，加到 `<|listen|>` token logit。
  - cpp 后端 `tools/server/ws_handler.cpp:240` **已解析** `init.config.listen_prob_scale`（无需改后端）。
- 改了 3 处（`static/omni/omni.html` + `static/omni/omni-app.js`）：见仓库 `demos/minicpm-o-4.5-fullduplex/frontend-patch/patch_omni.py`（幂等脚本，host 有 `.bak` 备份）。
  1. omni.html：「Response Length Controls」后加折叠组「Speak / Listen Balance」+ range 滑块 `omniListenProbScale`。
  2. omni-app.js：设置项注册 `{ id:'omniListenProbScale', type:'range' }`。
  3. omni-app.js：`preparePayload.config` 里加 `listen_prob_scale`。
- ⚠️ **生效时机**：和 length_penalty 一样在 **session 开始(omni_init)时**应用 → **先调滑块再点 Start**，会话中途无效。
- 应用到 live：`docker cp ~/MiniCPM-o-Demo/static/omni/{omni.html,omni-app.js} minicpm-o-demo-gateway-1:/app/static/omni/`，浏览器硬刷新。

---

## 9. 关键可调参数速查

| 参数 | 默认 | 含义 | 位置 |
|---|---|---|---|
| `listen_prob_scale` | 1.0 | speak vs listen 偏置（见 §8）| config，ws_handler.cpp:240 |
| `length_penalty` | 1.1 | 回复长度（>1 更长）| config，前端 Response Length |
| `playbackDelay`（前端「延迟」）| 200ms | 音频播放抖动缓冲；直接加在 TTFS 上；Ahead 富余够时可降到 50-100ms | omni.html |
| `CHUNK_MS` | 1000 | 全双工 **1Hz 决策周期**——TTFS ~1s 的主因，架构定的 | omni-app.js:38 |
| `force_listen_count` | 3 | 启动保护期强制前 N 次 listen | core/schemas/duplex.py |
| `max_new_speak_tokens_per_chunk` | 26 | 每 chunk 最大 speak token（便于及时被打断）| omni.h:298 |

> 前端实测指标解读：TTFS（首音端到端，~1s 主要是 1Hz 周期+200ms 缓冲，正常）；Infer（每步推理 ~76ms，和 CLI 一致）；Ahead（生成比播放提前的缓冲，越大越稳）；Drift（时间线漂移，越小越好）。

---

## 10. Gotchas / 坑

1. **SSH `pkill -9 -f` 会杀掉会话**（exit 255）→ 按 PID kill 或单独命令。
2. **H100 PyTorch 容器有"复活"现象**：`docker compose down`/`stop` 后会被 restart 策略+反复拉起；要彻底清用 `docker rm -f $(docker ps -aq --filter name=minicpm)` + `docker update --restart=no`。
3. **H100 GPU 2/3 是别人的**（hackathon4 qwen_server，pid 26850 占 4 卡），别动。
4. **B300 是 sm_103**，torch/llama.cpp 都要 CUDA 13 + arch 100;103；torch cu128 不支持。
5. vLLM-Omni HTTP 在 0.22.1+0.22.0+fastapi0.138 下坏（§5）。
6. `hf download` 用 `--include` 时若同时给了文件名会**忽略 --include**；直接把文件名作位置参数最稳。
7. cpp 后端的 `length_penalty/listen_prob_scale` 只在 ws_handler 读 init.config；**只有 cpp/Comni 路径生效，PyTorch 路径走 py 自己的**。

---

## 11. 早期工作（Mac 本地，已在 demos/ 里）

- **LiveKit voice agent**（`demos/livekit-voice-agent/`）：OpenAI Realtime 单 key 语音 agent，Mac 本地 `console` 模式跑通。turn-based 对照。Python 3.12 + uv。
- **Comni 桌面 app**（Mac）：`tc-mb/llama.cpp-omni` releases 的 `Comni-macOS-arm64.dmg`，装在 `/Applications/Comni.app`，菜单栏 app，Metal，模型懒下载（model_registry：openbmb/MiniCPM-o-4_5-gguf）。

---

## 12. 如何接手 / 常用命令

```bash
# 连 B300（当前主战场）
ssh -i ~/.ssh/itc -p 22 root@86.38.238.57

# 看 B300 Comni 栈
cd ~/MiniCPM-o-Demo && docker ps --filter name=minicpm
docker logs -f minicpm-o-demo-cpp-worker-backend-1   # 看后端/生成文本

# 本机开隧道访问 B300 前端
ssh -i ~/.ssh/itc -p 22 -fN -L 8010:localhost:8006 root@86.38.238.57
# 浏览器 https://localhost:8010/omni

# B300 跑单 request benchmark（llama.cpp）
docker run --rm --gpus all -v $HOME/llama.cpp-omni:/work -v $HOME/models:/models -w /work \
  nvidia/cuda:13.0.0-devel-ubuntu22.04 ./build/bin/llama-omni-cli \
  -m /models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-F16.gguf -ngl 99 -c 4096   # 内置音频测试用例
```

**模型计算已验证正确**（CLI 输出 "The person in the picture is skiing down the mountain"），B300 无 accuracy 问题。
