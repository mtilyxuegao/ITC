# Two-Brain Design: Talker (small) + Thinker (large)

[简体中文](./THINKER_TALKER_zh.md) · **English**

> This doc describes how ITC attaches a large "thinking" model (Qwen3-235B-A22B / DeepSeek-R1, etc.) asynchronously behind the existing full-duplex small model (MiniCPM-o 4.5).
> Focus: **session state management, cross-process shared state (context bridging), how the Thinker's answer triggers the Talker to speak, and how a user barge-in makes the large model stop immediately and its result get discarded.**
>
> Related: overall backend paradigm see [`BACKEND_DESIGN`](TBD); model-selection rationale see the repo-root README and [`../demos/README.md`](../demos/README.md).

---

## 0. Role Split (one line)

| | **Talker (small)** | **Thinker (large)** |
| --- | --- | --- |
| Model | MiniCPM-o 4.5 (full-duplex) | Qwen3-235B-A22B / DeepSeek-R1, etc. |
| Runs on | native PyTorch duplex worker | **vLLM / SGLang** (turn-based, high throughput) |
| Cadence | real-time, 200ms micro-turns | async, seconds per thought |
| Job | hear/see/speak, instant reactions, keep the conversation alive, **decide when to escalate**, **verbalize conclusions**, **interruptible anytime** | deep reasoning, long CoT, tools/planning, **text in / text out** |
| Analogy | System-1 / the mouth | System-2 / the brain |

Core principle: **the Thinker never speaks to the user directly.** It only produces a text conclusion, hands it back to the Talker, and the Talker says it in one consistent voice — so the conclusion is **also interruptible** and the timbre stays consistent.

---

## 1. Session State Machine (Talker side, 200ms-tick driven)

Each session keeps a state machine inside the Talker's real-time loop. Every 200ms tick performs one transition based on perception (VAD/vision) and Thinker events.

```
                ┌───────────────────────────────────────────────┐
                │                                               │
                ▼                                               │
          ┌───────────┐  hard question / needs reasoning  ┌───────────┐ │
   ┌─────▶│ LISTENING │ ───────────────────────────▶│  THINKING │ │
   │      │(perceive+ │                            │(escalated, │ │
   │      │  idle)    │ ◀──────────┐               │ Talker fills)│ │
   │      └───────────┘  cancelled │               └─────┬─────┘ │
   │            │   / not needed    │                     │ valid conclusion
   │            │ say a quick reply  │                     ▼       │
   │            ▼                    │               ┌───────────┐ │
   │      ┌───────────┐              │               │  SPEAKING │ │
   │      │  SPEAKING │──────────────┴───────────────│ (verbalize│ │
   │      │(Talker self-│ done                       │  result)  │ │
   │      │  answers) │                              └─────┬─────┘ │
   │      └─────┬─────┘                                    │       │
   │            │                                          │       │
   └────────────┴──────────────────────────────────────────┘       │
                │  ◀── any state: user speaks (barge-in) ──┐        │
                ▼                                          │        │
          ┌───────────┐  flush output + epoch++ + cancel Thinker    │
          │INTERRUPTED│ ────────────────────────────────────────────┘
          └───────────┘  → back to LISTENING, restart on new input
```

State reference:

| State | Meaning | What the Talker does here |
| --- | --- | --- |
| `LISTENING` | idle/perceiving | keep ingesting A/V, backchannel ("mm-hm"/nod), evaluate whether to escalate |
| `SPEAKING` | currently talking | emit TTS; **still monitors VAD, interruptible** |
| `THINKING` | escalated, awaiting the large model | **latency hiding**: filler ("let me think…"), clarifying question, optional interim reply |
| `INTERRUPTED` | barge-in detected | flush current output, bump epoch, cancel the running Thinker task, snap back to `LISTENING` |

> Non-exclusive optimization: while the Thinker streams tokens, the Talker can "speak-as-it-receives" the first sentence, cutting latency further (see §5).

---

## 2. Shared State / Context Bridging ("shared memory")

The Talker and Thinker **are not in the same process** (Talker on a GPU duplex worker; Thinker on a vLLM/SGLang cluster, possibly a different machine), so "shared memory" is really a **per-session session-state service** (in-process object → Redis/in-memory KV + pub-sub), not literal shared RAM. The two sides **bridge via text/structured messages; KV caches are NOT shared.**

### 2.1 Shared state object (Blackboard)

```
SessionState {
  session_id            : str
  epoch                 : int        # ⭐ monotonic; +1 on barge-in / re-escalation; basis of ALL cancellation (fencing token)
  talker_state          : enum       # LISTENING / SPEAKING / THINKING / INTERRUPTED
  user_speaking         : bool       # from Talker's VAD, updated each tick

  transcript            : [Turn]     # rolling dialogue log (timestamped), written by Talker
  context_summary       : str        # compressed long-range summary, periodically refreshed

  thinker_task : {                    # current escalation (at most one at a time)
    task_id   : str
    epoch     : int                  # ⭐ epoch snapshot at submit time, checked on write-back
    status    : enum                 # PENDING / RUNNING / DONE / CANCELLED
    query     : str                  # compressed context fed to the large model
    result    : str | null           # the large model's conclusion
  } | null
}
```

### 2.2 Field flow

| Field | Writer | Reader | Purpose |
| --- | --- | --- | --- |
| `epoch` | Talker (+1 on barge-in/re-escalation) | both | **master switch for all cancellation & invalidation** |
| `transcript` / `context_summary` | Talker | Talker (when building query) | material bridged to the Thinker |
| `thinker_task.query` | Talker (on escalation) | Thinker | the large model's input |
| `thinker_task.result` | Thinker (on completion) | Talker (on write-back) | the large model's output |
| `thinker_task.status` | both | both | lifecycle & cancellation |

### 2.3 Implementation notes

- **Start single-box**: in-process shared object + `asyncio.Queue`/`Event` for events.
- **Cross-box for prod**: Redis (state + `PUB/SUB` events), or lightweight gRPC bidi stream. Add `epoch`/`version` to session state for optimistic concurrency + fencing.
- **Pass only compressed context, never raw streams**: on escalation, compress "recent transcript + summary" into `query` rather than shipping A/V or the whole KV to the large model.

---

## 3. Escalation: Talker → Thinker (when to wake the brain)

Don't escalate every utterance (slow and expensive). Each tick the Talker evaluates:

1. **Decide**: intent/difficulty classification — needs fact-checking, multi-step reasoning, planning, tool calls → escalate; small talk / instant reactions → Talker handles it itself via `SPEAKING`.
2. **Snapshot**: `epoch_snapshot = state.epoch`; compress recent transcript + summary into `query`.
3. **Submit**: write `thinker_task = {task_id, epoch: epoch_snapshot, status: PENDING, query}`, send to the Thinker via queue (one chat completion request on vLLM/SGLang).
4. **Enter `THINKING`**: immediately start **latency hiding** — filler / clarifying question / backchannel — to cover the several seconds of thinking.
5. **Dedup**: escalate at most once per turn; if the user adds info mid-thought → take the "re-escalation" path in §4 (cancel old + epoch++ + resubmit).

---

## 4. ⭐ Barge-in: stop the large model & invalidate its result

**Your requirement: when the user interrupts in the foreground, the large model should stop outputting.** There's a race — the user interrupts *exactly* as the Thinker is about to return. So use **double protection**: **cancel at source** (stop computing) + **invalidate at sink** (don't speak even if computed). Both keyed on the same `epoch` (fencing token).

### 4.1 Barge-in timeline

```
   user speaks (Talker VAD hits on some 200ms tick, and it's not a backchannel)
        │
        ▼
 ① Talker immediately flushes the TTS audio it's playing   ← mouth shuts at once; feels "interrupted"
 ② state.epoch += 1                                        ← ⭐ advance the fence; all old epochs invalid
 ③ send cancel(task_id) to the Thinker                     ← cancel at source (see 4.2)
 ④ talker_state = INTERRUPTED → LISTENING                   ← restart on the user's new input
        │
        ▼
 (later) if the Thinker still returns a result:
 ⑤ Talker checks before write-back: result.epoch == state.epoch ?
        ├─ no (result.epoch < current) → discard, never verbalize   ← ⭐ invalidate at sink
        └─ yes → only then enter SPEAKING to verbalize
```

### 4.2 Cancel at source (actually stop the large model computing)

The Thinker runs on vLLM/SGLang, which **support aborting in-flight requests**, at two levels:

- **Hard abort**: call the backend abort — SGLang's `/abort_request`, vLLM engine's `abort(request_id)` — kills this generation and **frees compute immediately** for other sessions. Preferred.
- **Cooperative abort**: if using a self-managed loop, check `task.status==CANCELLED || task.epoch != state.epoch` every N tokens and `break` on hit. Fallback.

> Cancellation is the key to **saving money & GPUs**: after a barge-in, a Thinker still grinding a long CoT wastes GPU and can clog later requests.

### 4.3 Invalidate at sink (last gate against the race)

Even if the cancel signal is a step too late and the Thinker already produced `result`: **the write-back gate only trusts epoch.** Any `result.epoch != state.epoch` is discarded. So regardless of whether the cancel landed in time, **a stale conclusion is never spoken.**

> Mnemonic: **cancel governs "compute less," epoch governs "don't say it."** Two gates; either one firing keeps you safe.

### 4.4 Distinguishing kinds of "speaking" (avoid false interrupts)

| User behavior | Talker verdict | Action |
| --- | --- | --- |
| "mm", "yeah", nod (backchannel) | not an interrupt | do **not** trigger; stay in current state |
| substantive cut-in / new question | interrupt | run the full §4.1 flow |
| adds info mid-thought ("I mean the Beijing one") | re-escalation | epoch++ + cancel old task + resubmit with new context |

---

## 5. Write-back: Thinker → Talker (how the answer triggers the small model to speak)

1. Thinker finishes → writes `thinker_task.result`, `status=DONE`, and **self-checks `task.epoch == state.epoch` first** (if stale, drop it outright, don't even notify).
2. Notify the Talker via a result event (queue/pub-sub).
3. On the next 200ms tick, if the Talker is in `THINKING` and receives a **valid** (epoch-checked) result → it transitions to `SPEAKING`, feeds the conclusion text to **its own** language/speech head, and verbalizes it in the consistent voice.
4. During verbalization it's a normal `SPEAKING` state: **still interruptible** (back to §4).

**Streaming optimization (lower latency)**: the Thinker streams tokens; the Talker starts speaking as soon as it has the **first complete sentence**, without waiting for the whole CoT. Note: if a barge-in occurs here, you must both flush the already-spoken audio and abort the still-streaming Thinker — §4's two gates apply, just at sentence granularity.

---

## 6. End-to-end timeline (a "hard question + mid-interrupt" journey)

```
User:   "Help me compute… (complex question)"
Talker: [LISTENING] classifies as hard → snapshot epoch=7, submit thinker_task(epoch=7)
Talker: [THINKING]  "Sure, let me work that out~"        ← latency hiding
Thinker:           (long CoT on vLLM, 3.2s...)
User:   "Oh wait, I meant a different one"               ← interrupt!
Talker: ① flush filler  ② epoch=8  ③ abort(task@epoch7)  ④ → LISTENING
Thinker:           (got abort, stops early, frees GPU)
        └ even if it had returned result.epoch=7 → Talker checks 7≠8 → discard, don't speak
Talker: [LISTENING] re-evaluates on the new question → maybe re-escalate (snapshot epoch=8)…
```

---

## 7. Concurrency & edge cases cheat sheet

| Situation | Handling |
| --- | --- |
| Multiple escalations on one session | not allowed; at most one `thinker_task`. Cancel old + epoch++ before a new one |
| Thinker timeout | cap it (e.g. 8s); on timeout cancel and have the Talker fall back ("I'm not sure on that one…") |
| Thinker queue congestion | admission control / degrade to a smaller thinking model; the Talker's real-time loop must never block |
| Barge-in and result arrive same tick | epoch check orders it naturally: epoch already bumped → result invalidated |
| Multiple sessions sharing the Thinker | the Thinker batches multiple sessions on vLLM/SGLang anyway; route results by `session_id`/`task_id` |
| Talker interrupted while verbalizing | ordinary `SPEAKING` interrupt flow, no special handling |

---

## 8. Implementation checklist (to build incrementally)

- [ ] `SessionState` shared object + epoch fencing (in-process first, Redis later)
- [ ] escalation classifier (intent/difficulty)
- [ ] Thinker client: vLLM/SGLang chat + **abort support** (mandatory)
- [ ] write-back event channel + epoch-check gate
- [ ] Talker filler/verbalize strategy (latency hiding)
- [ ] backchannel vs substantive-interrupt VAD discrimination
- [ ] observability: escalation rate, think latency, interrupt rate, count of invalidated results (should be monitorable)
```
