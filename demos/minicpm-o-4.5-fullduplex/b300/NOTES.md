# MiniCPM-o 4.5 on NVIDIA B300 (Blackwell Ultra) — 部署、torch.compile、可调 chunk_ms、benchmark

> B300 = Blackwell Ultra,**sm_103a**,驱动 580 / **CUDA 13.0**,275GB 显存。
> 这是在 B300 上把 llama.cpp-omni 和 PyTorch demo(含 torch.compile)跑通 + 调优单 request 延迟的全过程。
> 配套脚本都在本目录;接 [`../README.md`](../README.md) 和 [`../../SESSION_HANDOFF.md`](../../SESSION_HANDOFF.md)。

---

## 0. 环境与坑总览
- B300 是 **sm_103a**,需 **CUDA 13** 工具链;CUDA 12.8(cu128)不含 sm_103 内核 → 任何 cu128 的 torch/库都跑不动。
- 单 GPU(GPU 0),所以 compose 要改成 **单 worker**;端口用 8011/8012 避开别的栈。
- 连接:`ssh -i ~/.ssh/itc -p 22 root@<B300>`。

---

## 1. llama.cpp-omni 构建(Blackwell)
host 无 nvcc,在 **CUDA 13 devel 容器**里编,产物落 host:
```bash
docker run --rm --gpus all -v $HOME/llama.cpp-omni:/work -w /work nvidia/cuda:13.0.0-devel-ubuntu22.04 bash -c '
  apt-get update && apt-get install -y cmake git build-essential
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="100;103"     # 引号必须有,否则 ; 被 shell 当命令分隔
  cmake --build build --target llama-omni-cli --target llama-omni-server -j'
```
- docker-compose.cpp.yml 的 `docker/Dockerfile.cpp-worker-backend` 也要改:`ARG CUDA_VERSION=13.0.0`、`ARG CMAKE_CUDA_ARCHITECTURES=100;103`。
- CLI 验证(模型计算正确,无 accuracy 问题):输出 "The person in the picture is skiing down the mountain"。
- **CUDA graphs 实测无效**:`-DGGML_CUDA_GRAPHS=ON` 重编后单 request 解码 273 vs 268 tok/s(噪音内)——单 request 瓶颈不是 kernel 启动开销。

## 2. PyTorch demo 跑通(`setup_pt_demo.sh`)
3 个必改点:
1. **torch 升 cu130**:`docker/Dockerfile.worker-backend` 把 `cu128 / torch==2.8.0` 改成 **`cu130 / torch==2.9.0 torchaudio==2.9.0`**(B300 sm_103 需要)。
2. **requirements 修复**(同 H100):`librosa==0.9.0` + `setuptools<81`(原 `librosa>=0.10.2` 与 minicpmo-utils 冲突)。
3. **soundfile monkeypatch**(`sitecustomize.py`):torch 2.9 的 torchaudio 强制走 `torchcodec`(缺 ffmpeg 兼容库会崩),用 soundfile 读 WAV 绕过。挂到 `/app/sitecustomize.py`(PYTHONPATH=/app 自动加载)。
- 单 GPU compose:`docker-compose.b300pt.yml`(`setup_pt_demo.sh` 生成),1 worker(GPU0)+ gateway,端口 8011,project `minicpm-pt`。
- 模型计算在 B300 正确(输出 "There's a beach with lots of seagulls...")。

## 3. torch.compile 在 B300 跑通(`run_compile2.sh`)——单 request 最大杀器
`config.json: "compile": true`。但 slim 镜像 + 新 GPU 有 **两个工具链坑**:
1. `InductorError: Failed to find C compiler` → 容器内 **`apt install gcc g++`**。
2. `ptxas fatal: Value 'sm_103a' is not defined` → Triton 自带 ptxas 太旧不认 B300。**从 CUDA13 devel 镜像拷一个 ptxas**(`docker run ... nvidia/cuda:13.0.0-devel cp /usr/local/cuda/bin/ptxas /out/ptxas13`,认 sm_103a),设 **`TRITON_PTXAS_PATH=/ptxas13`**。
- 派生镜像 `minicpm-wb-b300-compile:dev` 把 gcc + ptxas13 + sitecustomize + `TRITON_PTXAS_PATH`/`CC` 都 bake 进去。
- worker 启动时**自动 warmup**(~150-193s 编译,一次性),首轮对话不卡。

## 4. 可调 chunk_ms 前端控件(`patch_chunk_control.py`)
**坑链(为什么改一处不生效)**:chunk 时长有 **3 层**,必须全改:
1. **AudioWorklet `capture-processor.js`**:`chunkSize` 写死 `SAMPLE_RATE_IN`(=1s)——**麦克风块大小的真正开关**,与 omni-app.js 的 `CHUNK_MS` 无关。改成 `SAMPLE_RATE_IN * CHUNK_MS / 1000`。
2. **前端 `CHUNK_MS`**(omni-app.js):feed/padding 用;改成 `let` + 全局 `window.setChunkMs()`,并把 `chunk_ms` 放进 `preparePayload.config`。
3. **模型 `CHUNK_MS`**(modeling_minicpmo*.py):模型会把音频 **padding 回 CHUNK_MS**;且 `unified.py` 传给 model 的 dict **不含 chunk_ms**。补在 **`pytorch_backend.set_duplex_config`**:`DuplexConfig(**config)` 后把 `duplex_view._model.CHUNK_MS / FIRST_CHUNK_MS` 按会话设上(见 patch)。
- 前端:omni.html 加「Chunk Size」下拉(0.1–1.0,默认 1.0),会话开始时生效。
- 验证:worker 日志出现 `[duplex] model CHUNK_MS set to <X>ms`;再量连续 unit 时间戳间隔 ≈ X。

> ⚠️ 验证 chunk 是否真生效的硬指标:`docker logs -t worker | grep "streaming_prefill: mode=OMNI"` 取连续时间戳算间隔——之前几轮"没生效"就是间隔一直 ~1.0s 暴露的。

---

## 5. Benchmark 数据(B300)

### 单 request 解码(llama.cpp,--no-tts)
| | H100 | B300 | 提升 |
|---|---|---|---|
| Q4_K_M | 224 t/s | **268 t/s** | +20% |
| F16 | 157 t/s | **188 t/s** | +20% |
> 单 request 是 launch/开销受限,B300 带宽优势发挥不出(只 +20%);CUDA graphs 无帮助。

### PyTorch 全双工单 unit(SPEAK,warm)
| | H100 eager | B300 eager | B300 + torch.compile |
|---|---|---|---|
| SPEAK unit | ~650–840ms | ~395ms | **~250ms** |
| 其中 tts | ~300ms | ~170ms | **~80ms**(compile 砍 2.3x)|
> B300 PyTorch 比 H100 ~2x(比 llama.cpp 的 +20% 大,因 PyTorch 路径更吃算力);torch.compile 再 ~1.6x。

### 交互延迟(llama.cpp + TTS)
| | H100 Q4 | B300 Q4 | H100 F16 | B300 F16 |
|---|---|---|---|---|
| 首音(解码→第一段语音) | 127ms | 79ms | 145ms | 92ms |
| TTS 流式节奏 | ~400ms | ~280ms | ~360ms | ~234ms |

### TTFT vs chunk_ms(profile 公式,实测 979ms@1.0s 校准)
```
TTFT ≈ chunk_ms/2 (决策等待) + 单元计算(~常数 200-250ms) + playbackDelay(~200ms)
```
| chunk | TTFT≈ | 质量 |
|---|---|---|
| 1.0s | ~950ms（实测~980✓）| ✅ 训练点/最优 |
| 0.5s | ~680ms | 🟡 实用甜点 |
| 0.2s | ~495ms | 🔴 off-distribution |
| 0.1s | ~435ms | 🔴🔴 |
- **有地板 ~400ms**(单元计算+缓冲),chunk 再小也下不去。
- **突破地板只能降 playbackDelay**(200→50,再省 ~150ms,与 chunk 正交)。
- **<0.5s 模型 off-distribution**(论文 §消融:训练用 1.0s,0.2/0.1s 更差;"chunk 太短→每块建模预算不足→决策不稳"),且 RTF 逼近 1(每 200ms 单元处理 ~185ms,几乎踩实时线)。

---

## 6. 本目录脚本
| 文件 | 作用 |
|---|---|
| `setup_pt_demo.sh` | 构建派生镜像(gcc+ptxas+sitecustomize)+ 单 worker compose,起 PyTorch+compile demo(端口 8011)|
| `sitecustomize.py` | torchaudio→soundfile patch(绕过 torchcodec)|
| `run_compile2.sh` | 在容器内带 sm_103a ptxas 跑 torch.compile benchmark |
| `patch_chunk_control.py` | 前端「Chunk Size」下拉 + 三层 chunk_ms 接线(omni.html / omni-app.js / pytorch_backend.py)|
| `run_chunk_sweep.sh` | 不同 chunk_ms 下量每单元计算耗时(需先让 `_extract_mp4_chunks` 跟随 CHUNK_MS)|

> 注:脚本里的路径/容器名按本次部署写死(`minicpm-pt-*`、`/root/MiniCPM-o-Demo`),复用时按需改。
