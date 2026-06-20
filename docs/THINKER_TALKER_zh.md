# 双脑架构设计:Talker(小)+ Thinker(大)

**简体中文** · [English](./THINKER_TALKER.md)

> 本文描述 ITC 在已有全双工小模型(MiniCPM-o 4.5)之上,异步挂接一个大"思考"模型(Qwen3-235B-A22B / DeepSeek-R1 等)的后端设计。
> 重点:**会话状态管理、跨进程共享状态(上下文桥接)、Thinker 回答后如何回填触发 Talker,以及用户打断(barge-in)时如何让大模型立刻停止、其结果如何作废。**
>
> 相关:整体后端范式见 [`BACKEND_DESIGN`](待补);模型选型理由见仓库根 README 与 [`../demos/README.md`](../demos/README.md)。

---

## 0. 角色分工(一句话)

| | **Talker(小)** | **Thinker(大)** |
| --- | --- | --- |
| 模型 | MiniCPM-o 4.5(全双工) | Qwen3-235B-A22B / DeepSeek-R1 等 |
| 跑在 | 原生 PyTorch 双工 worker | **vLLM / SGLang**(轮次制、高吞吐) |
| 节奏 | 实时,200ms 微轮次 | 异步,一次思考几秒 |
| 职责 | 听/看/说、即时反应、维持对话、**决定何时升档**、**口语化复述结论**、**随时可被打断** | 深推理、长 CoT、工具/规划,**纯文本进出** |
| 类比 | System-1 / 嘴 | System-2 / 脑 |

核心原则:**Thinker 永不直接对用户发声**。它只产出文本结论,交回 Talker,由 Talker 用统一音色说出来——这样结论同样**可被打断**,音色一致。

---

## 1. 会话状态机(Talker 侧,200ms tick 驱动)

每个会话在 Talker 的实时循环里维护一个状态机。每个 200ms tick 都会根据感知(VAD/视觉)与 Thinker 事件做一次状态转移。

```
                ┌───────────────────────────────────────────────┐
                │                                               │
                ▼                                               │
          ┌───────────┐  检测到难题 / 需推理        ┌───────────┐ │
   ┌─────▶│ LISTENING │ ───────────────────────────▶│  THINKING │ │
   │      │ (感知+待命)│                            │(已升档,   │ │
   │      └───────────┘ ◀──────────┐               │ Talker 填充)│ │
   │            │   思考被取消/无需   │               └─────┬─────┘ │
   │            │ 说点即时回应         │                     │ 收到有效结论
   │            ▼                    │                     ▼       │
   │      ┌───────────┐              │               ┌───────────┐ │
   │      │  SPEAKING │──────────────┴───────────────│  SPEAKING │ │
   │      │(Talker 自答)│ 说完                         │(复述结论) │ │
   │      └─────┬─────┘                              └─────┬─────┘ │
   │            │                                          │       │
   └────────────┴──────────────────────────────────────────┘       │
                │  ◀── 任意状态下:用户开口(barge-in) ──┐          │
                ▼                                       │          │
          ┌───────────┐  flush 输出 + epoch++ + 取消 Thinker        │
          │INTERRUPTED│ ───────────────────────────────────────────┘
          └───────────┘  → 回到 LISTENING,以新输入重启
```

状态说明:

| 状态 | 含义 | 该状态下 Talker 在做什么 |
| --- | --- | --- |
| `LISTENING` | 待命/感知 | 持续吃音视频,backchannel(嗯/点头),评估是否升档 |
| `SPEAKING` | 正在说 | 输出 TTS;**仍监听 VAD,可被打断** |
| `THINKING` | 已升档,等大模型 | **延迟隐藏**:填充语("让我想想…")、追问澄清、必要时先给临时回应 |
| `INTERRUPTED` | 检测到打断 | flush 当前输出、推进 epoch、取消在跑的 Thinker 任务,瞬间回 `LISTENING` |

> `THINKING` 和 `SPEAKING` 不互斥的优化版:Thinker 流式吐 token 时,Talker 可"边收边说"第一句话,把延迟进一步压低(见 §5)。

---

## 2. 共享状态 / 上下文桥接("共享内存")

Talker 与 Thinker **不在同一进程**(Talker 在 GPU 双工 worker;Thinker 在 vLLM/SGLang 集群,甚至不同机器),所以"共享内存"实际是一个**每会话的会话状态服务**(进程内对象 → Redis/内存 KV + pub-sub),而非真正的 RAM 共享。两边**靠文本/结构化消息桥接,KV cache 不互通**。

### 2.1 共享状态对象(Blackboard)

```
SessionState {
  session_id            : str
  epoch                 : int        # ⭐ 单调递增,打断/重升档时 +1,所有取消的依据(fencing token)
  talker_state          : enum       # LISTENING / SPEAKING / THINKING / INTERRUPTED
  user_speaking         : bool       # 来自 Talker 的 VAD,逐 tick 更新

  transcript            : [Turn]     # 滚动对话日志(带时间戳),Talker 写
  context_summary       : str        # 压缩后的长程摘要,定期由 Talker/小工具刷新

  thinker_task : {                    # 当前升档任务(同一时刻至多一个)
    task_id   : str
    epoch     : int                  # ⭐ 提交时的 epoch 快照,用于回填时校验
    status    : enum                 # PENDING / RUNNING / DONE / CANCELLED
    query     : str                  # 喂给大模型的压缩上下文
    result    : str | null           # 大模型结论
  } | null
}
```

### 2.2 字段流向

| 字段 | 写者 | 读者 | 作用 |
| --- | --- | --- | --- |
| `epoch` | Talker(打断/重升档时 +1) | 双方 | **一切取消与作废的总开关** |
| `transcript` / `context_summary` | Talker | Talker(组 query 时) | 桥接给 Thinker 的素材 |
| `thinker_task.query` | Talker(升档时) | Thinker | 大模型的输入 |
| `thinker_task.result` | Thinker(完成时) | Talker(回填时) | 大模型的输出 |
| `thinker_task.status` | 双方 | 双方 | 生命周期与取消 |

### 2.3 实现建议

- **单机起步**:进程内共享对象 + `asyncio.Queue`/`Event` 做事件。
- **跨机生产**:Redis(状态 + `PUB/SUB` 事件),或轻量 gRPC 双向流。会话状态加 `epoch`/`version` 做乐观并发 + fencing。
- **只传压缩上下文,别传原始流**:升档时把"近期 transcript + 摘要"压成 `query`,而不是把音视频或整条 KV 丢给大模型。

---

## 3. 升档:Talker → Thinker(何时触发大脑)

不是每句都升档(否则又慢又贵)。Talker 每个 tick 评估:

1. **判定**:意图/难度分类——需要事实查证、多步推理、规划、工具调用 → 升档;寒暄/闲聊/即时反应 → Talker 自己用 `SPEAKING` 处理。
2. **快照**:`epoch_snapshot = state.epoch`;把近期 transcript + 摘要压成 `query`。
3. **提交**:写 `thinker_task = {task_id, epoch: epoch_snapshot, status: PENDING, query}`,经队列发给 Thinker(vLLM/SGLang 的一次 chat completion 请求)。
4. **转 `THINKING`**:立刻启动**延迟隐藏**——填充语 / 追问澄清 / backchannel,把几秒的思考盖住。
5. **去重**:同一 turn 只升档一次;若用户在思考中补充了新信息 → 走 §4 的"重升档"(取消旧的 + epoch++ + 重新提交)。

---

## 4. ⭐ 打断(barge-in):让大模型停下来 & 结果作废

**你的诉求:用户在前台打断时,大模型不该再输出。** 这里存在一个竞态——用户**恰好**在 Thinker 即将返回时开口。因此用**双重保险**:**源头取消**(让大模型别再算)+ **汇入作废**(就算算完了也不说)。两者都靠同一个 `epoch`(fencing token)。

### 4.1 打断时序

```
   用户开口 (Talker VAD 在某个 200ms tick 命中,且非 backchannel)
        │
        ▼
 ① Talker 立刻 flush 自己正在播的 TTS 音频      ← 嘴马上闭上,体验上"被打断"
 ② state.epoch += 1                            ← ⭐ 推进 fence,旧 epoch 全部失效
 ③ 向 Thinker 发 cancel(task_id)               ← 源头取消(见 4.2)
 ④ talker_state = INTERRUPTED → LISTENING       ← 以用户的新输入重启
        │
        ▼
 (稍后) Thinker 若仍返回了 result:
 ⑤ Talker 回填前校验:result.epoch == state.epoch ?
        ├─ 否(result.epoch < 当前) → 丢弃,绝不复述   ← ⭐ 汇入作废
        └─ 是 → 才允许进入 SPEAKING 复述
```

### 4.2 源头取消(让大模型真的停止计算)

Thinker 跑在 vLLM/SGLang,**支持中止在跑的请求**,两个层级:

- **强制中止**:调用后端的 abort——SGLang 的 `/abort_request`、vLLM 引擎的 `abort(request_id)`,直接停掉这次 generation,**立即释放算力**给别的会话。首选。
- **协作式中止**:若用自管循环,在每生成 N 个 token 时检查 `task.status==CANCELLED || task.epoch != state.epoch`,命中即 `break`。作为兜底。

> 取消是**省钱省卡**的关键:打断后大模型若还在跑长 CoT,既浪费 GPU 又可能堵住后续请求。

### 4.3 汇入作废(防竞态的最后一道闸)

即使取消信号晚了一步、Thinker 已经产出 `result`:**回填闸门只认 epoch**。`result.epoch != state.epoch` 一律丢弃。这样无论取消是否赶上,**过期结论永远不会被说出口**。

> 一句话记忆:**取消管"少算",epoch 管"不说"。** 两道闸,任意一道生效都安全。

### 4.4 几种打断的区分(避免误打断)

| 用户行为 | Talker 判定 | 动作 |
| --- | --- | --- |
| "嗯"、"对"、点头(backchannel) | 非打断 | **不**触发,继续当前状态 |
| 实质性插话 / 提新问题 | 打断 | 走 §4.1 全流程 |
| 思考中补充信息("我是说北京的") | 重升档 | epoch++ + 取消旧任务 + 用新上下文重新提交 |

---

## 5. 回填:Thinker → Talker(大模型答完如何触发小模型说)

1. Thinker 完成 → 写 `thinker_task.result`、`status=DONE`,**先自检 `task.epoch == state.epoch`**(过期则直接丢,连通知都不发)。
2. 通过 result 事件(queue/pub-sub)通知 Talker。
3. Talker 在下一个 200ms tick 调度里:若处于 `THINKING` 且收到**有效**(epoch 校验通过)result → 转 `SPEAKING`,把结论文本灌给**自己**的语言/语音头,用统一音色口语化说出来。
4. 复述期间是普通 `SPEAKING` 态:**仍可被再次打断**(回到 §4)。

**流式优化(降延迟)**:Thinker 以 streaming 吐 token,Talker 收到**第一句完整句子**就开始说,无需等整段 CoT 完成。注意:此时若发生打断,既要 flush 已说的音频,也要 abort 仍在流的 Thinker——§4 的双闸同样适用,只是粒度到句子。

---

## 6. 端到端时序(一次"难题 + 中途打断"的完整旅程)

```
用户:  「帮我算一下……(复杂问题)」
Talker: [LISTENING] 识别为难题 → epoch=7 快照, 提交 thinker_task(epoch=7)
Talker: [THINKING]  「嗯,我算一下哈~」        ← 延迟隐藏
Thinker:           (vLLM 上跑长 CoT, 3.2s...)
用户:  「啊不对,我说的是另一个」              ← 打断!
Talker: ① flush 填充语  ② epoch=8  ③ abort(task@epoch7)  ④ → LISTENING
Thinker:           (收到 abort, 提前停, 释放 GPU)
        └ 即便它已算完返回 result.epoch=7 → Talker 校验 7≠8 → 丢弃,不说
Talker: [LISTENING] 以新问题重新评估 → 可能重新升档(epoch=8 快照)…
```

---

## 7. 并发与边界情况速查

| 情况 | 处理 |
| --- | --- |
| 同一会话同时多个升档 | 不允许;至多一个 `thinker_task`。新升档前先取消旧的 + epoch++ |
| Thinker 超时 | 设上限(如 8s);超时即取消,Talker 用兜底话术("这个我暂时拿不准…") |
| Thinker 队列拥塞 | 准入控制 / 降级到更小思考模型;Talker 侧不能阻塞实时循环 |
| 打断信号与 result 同 tick 到达 | epoch 校验天然定序:epoch 已自增 → result 作废 |
| 多会话共享 Thinker | Thinker 在 vLLM/SGLang 上本就多会话批处理;按 `session_id`/`task_id` 路由 result |
| Talker 复述时又被打断 | 普通 `SPEAKING` 打断流程,无特殊处理 |

---

## 8. 落地清单(后续可逐项实现)

- [ ] `SessionState` 共享对象 + epoch fencing(先进程内,后 Redis)
- [ ] 升档判定器(意图/难度分类)
- [ ] Thinker 客户端:vLLM/SGLang chat + **abort 支持**(必须)
- [ ] 回填事件通道 + epoch 校验闸门
- [ ] Talker 填充/复述策略(延迟隐藏)
- [ ] backchannel vs 实质打断的 VAD 判别
- [ ] 可观测:升档率、思考延迟、打断率、被作废的 result 数(应可被监控)
```
