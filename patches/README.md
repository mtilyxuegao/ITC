# patches · 把 Talker–Thinker 集成进 MiniCPM-o demo

> ✅ **已在 2×H100 实测打通**:
> - (a) `control.force_speak` → 模型出声(~3s 音频),见 `tests/smoke_force_speak.py`
> - (b) 旁路监听 + 注入:observer 发现活跃会话 → 注入 force_speak → 浏览器收到音频,见 `tests/e2e_observer.py`
>
> 两个集成器(都幂等、可 `--revert`、自动备份 `*.tt.bak`):
> - `integrate_force_speak.py` —— 改 `py_backend/server.py`(分发+处理函数)、`worker.py`(放行
>   control.force_speak)、`runtime/backend_client.py`(`send_raw`),并复制 `minicpm_ext/`。**改 worker 镜像后需重建 worker。**
> - `integrate_observer.py` —— 改 `gateway.py`(register/publish/unregister 钩子 + 顶层 `/observer` 路由)。**改后需重建 gateway。**
>
> ⚠️ 关键经验:① server.py 处理函数必须插在 `def main()` **之前**(EOF 在 uvicorn 阻塞后永不执行);
> ② `/observer` 路由必须是**模块级 `@app.websocket` 装饰器**(放进函数里 install 会被 FastAPI 拒成 403);
> ③ force_speak 真身挂 `DuplexCapability`,`MiniCPMO` 上挂委托。

---

# force_speak 集成细节

为实现 AI 主动打断(CUT 的"路线 2"),需要给 demo 的小模型加一条 `duplex_force_speak` 路径。
真正的实现代码在 **本仓库** `thinker_talker/model_ext/force_speak.py`(版本受控);
`integrate_force_speak.py` 负责把它**幂等地**接进 demo。

## 它改了什么

1. 复制 `thinker_talker/model_ext/` → `<demo>/minicpm_ext/`(进 build context,容器内可 import)。
2. 改 `<demo>/py_backend/server.py`:
   - WS 分发循环加 `control.force_speak` 与 `control.stop` 分支;
   - 文件末尾追加 `_tt_handle_force_speak()` / `_tt_handle_stop()` + 懒加载 `install()`。
   `install()` 用 monkeypatch 把方法挂到模型类 / `DuplexView` / `PyTorchBackend`,**不改这三个文件**。

### stop/flush(让 CUT 真能掐断重复输出)

- `control.force_speak` 的 payload 新增 `interrupt`:
  - `true`(CUT)→ 先 `duplex_stop`(收尾当前 turn + flush 半句 TTS)+ 清空上行队列,再说 redirect;
  - `false`(INJECT)→ 仍 `_wait_finalize()` 等当前说完再接话(礼貌,不抢话)。
- `control.stop`:只停不说,用于纯掐断(截断 turn + flush TTS + 回 `listen`)。
- ✅ **已对真机源码核对**(MiniCPM-o-Demo `py_backend/server.py`,PR #45 `f133fc2`):
  - server 端**无上行输入队列**——每个 `input.append` 同步跑一次很短的 `duplex_generate`,
    "重复输出"是**客户端持续上行驱动**的。所以早期设想的"清服务端队列"无的放矢,已去掉;
    "不再被重新触发"由编排层 ASR 自门控 + epoch 围栏负责(见 `thinker_talker/orchestrator.py`)。
  - dispatch / `_op_lock` / `_wait_finalize` / `send_output_delta` 锚点在该版本仍命中。
- ⚠️ **残余播放**:已下发给浏览器、排在 `StreamingPcmPlayer` 队列里的音频会播完(约播放延迟时长)。
  真正的前端 flush 需改 vendored 的 `/static/duplex/lib/realtime-session.js`(**minified**,无源码补丁点),
  暂不做;模型侧已立即停产新音频,体感是"尾音很短一截后切到 redirect"。

## 用法

```bash
# 检查能否应用(锚点是否命中)
python patches/integrate_force_speak.py --check

# 应用(默认 demo=~/MiniCPM-o-Demo;会自动备份 server.py.tt.bak)
python patches/integrate_force_speak.py

# 撤销
python patches/integrate_force_speak.py --revert
```

## 让改动在运行的服务里生效(二选一)

代码改的是 worker 容器里的文件,需让容器看到新代码:

- **方式 A:重建镜像(干净)**
  ```bash
  cd ~/MiniCPM-o-Demo
  MODEL_HOST_PATH=~/models/MiniCPM-o-4_5 docker compose up -d --build
  ```
- **方式 B:挂载覆盖(快,调试用)** —— 在 `docker-compose.yml` 的 worker-backend 服务挂上:
  ```yaml
  volumes:
    - ./py_backend/server.py:/app/py_backend/server.py:ro
    - ./minicpm_ext:/app/minicpm_ext:ro
  ```
  (容器内代码路径以镜像里的为准,按实际调整。)

## 协议

集成后,客户端/编排层可发新消息让 Talker 立即开口:
```json
{ "type": "control.force_speak", "payload": { "text": "等一下,这里要先确认需求" } }
```
服务端返回与正常说话一致的 `response.output.delta`(kind = text / audio / listen)。

## ⚠️ 需在真机做一次联调验证(无法离线单测)

`duplex_force_speak` 复用了 streaming_generate 已验证的内部件,但有两点要在 GPU 上确认:

1. **整句音频**:当前用单次 `generate_chunk(max_new_token=2048)` 渲染整句。若 TTS 有每次调用 token
   上限导致长句被截断,改成对 hidden 分块循环 `generate_chunk` 即可(`force_speak.py` 结构已留好)。
2. **duplex 状态连贯**:force_speak 手动 feed 了 `turn_eos` 但未走 `finalize_unit` 的
   `</unit>`/`register_unit_end`/滑窗。若发现 force_speak 之后聆听异常,在收尾后补一次轻量
   unit 收尾即可。

冒烟测试建议:跑通后用一个最小 WS 客户端发 `session.init` → `control.force_speak`,确认收到
非空 `audio` delta 且时长与文本相称。
