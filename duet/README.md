# DUET-Lite — decoupled full-duplex interaction + async thinking

A 24h-hackathon system that reproduces the **Decoupled Interaction & Thinking**
architecture (DuplexOmni, arXiv:2606.09186 / Thinking Machines "Interaction Models")
on **frozen, off-the-shelf models**. The deliverable is the *control layer*, not the
model: the `Conductor` realizes the `[THINK]/[WAIT]/^/[CUT]/[PENDNS]` protocol around
models we never train.

```
 user mic ─tee─► Conductor (this package, pure asyncio, CPU)
                   ├─ STT + VAD ............ its own "ears"  → fire [THINK]/[WAIT]
                   ├─ IntentClassifier ..... knowledge intent / interrupt+new-info
                   ├─ EpochManager ......... correctness backbone of [WAIT]
                   ├─ ThinkingClient ─/v1─► Qwen3.5-35B-A3B (vLLM/SGLang) + agents
                   └─ Speaker (WRITE seam) ► MiniCPM-o 4.5 duplex (frozen, voice+vision)
```

## Module map

| file | role |
|------|------|
| `events.py`      | the control-token event log (UI hero + test oracle) |
| `state.py`       | `Phase`, `TaskState`, `EpochManager` (the epoch guard) |
| `intent.py`      | READ seam: heuristic intent classifier over the STT transcript |
| `thinking.py`    | cancellable streaming client to the MoE + local agent tools |
| `write_seam.py`  | WRITE seam: make the frozen model speak via synthetic user-turn injection |
| `conductor.py`   | the orchestrator / state machine that is the control protocol |
| `fakes/`         | in-process FakeVLLM + FakeGateway (real protocols, ephemeral ports) |
| `scripts/`       | the H0 spike + a no-GPU scenario demo |
| `tests/`         | end-to-end pytest over real loopback sockets |

## The two seams (how a frozen model gets a control protocol)

- **READ** — never tap the model's internals. The Conductor runs its **own** STT on a
  mic tee; that transcript fires `[THINK]`/`[WAIT]`.
- **WRITE** — never write into the model's KV. The Conductor injects a synthetic
  **user/system text line** through the supported input path; the model answers it in
  its own voice. `[CUT]` (true barge-in) is handled by MiniCPM's **native** duplex.

`[WAIT]` correctness does **not** depend on the abort landing instantly: every
thinking-layer emission carries an `epoch_id`, and the **epoch guard** drops anything
stale. Killing the in-flight generation (cancel coroutine + close the HTTP stream) is
an *efficiency* bonus, not the correctness mechanism.

## Run it (no GPU)

```bash
pip install -r duet/requirements.txt
python -m pytest duet/tests -q            # 7 end-to-end tests over real sockets
python -m duet.scripts.demo_scenario      # prints the control-token timeline
python -m duet.scripts.h0_spike_write_seam --demo-fake   # WRITE-seam plumbing self-test
```

## Go live

1. **Interaction layer**: bring up MiniCPM-o 4.5 duplex via
   `demos/minicpm-o-4.5-fullduplex/deploy.sh`. Then **run the H0 spike for real**:
   `MINICPM_GATEWAY_URL=ws://localhost:8006/... python -m duet.scripts.h0_spike_write_seam`.
   Edit `GatewayAdapter` in `write_seam.py` to match the real user-turn message.
   If it FAILS, swap `MiniCPMWriteSeam` for `RecordingSpeaker` as a caption sink.
2. **Thinking layer**: on a GPU node, `vllm serve Qwen/Qwen3.5-35B-A3B --enable-auto-tool-choice
   --tool-call-parser qwen3_coder`. Point `ThinkingClient(base_url=...)` at it.
3. **Ears**: replace `on_user_utterance(text)` callers with a real Silero VAD + streaming
   STT loop feeding transcripts in.

## Cluster / efficiency notes

- Hardware: Slurm `defq`, 8×**H100 80GB** per node. One node fits the whole stack
  (MiniCPM duplex on 1 GPU, Qwen3.5-35B-A3B FP8 on 1, agents share it, headroom spare).
- **SGLang PR #19171** (`SessionAwareCache`, merged): per-session KV reuse across turns,
  ~2.1–2.5× tail-latency on later turns. **Use it for the thinking-layer agents'**
  multi-turn loops (`open_session(..., streaming=True)`). It is *not* for the duplex
  audio path and is *not* a new abort primitive — for fast `[WAIT]`, still use the
  `/abort_request` path + the epoch guard. Note: streaming sessions are append-only
  (no replace/rewind), so an agent that rewrites context can't use it.
