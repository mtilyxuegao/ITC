"""DUET-Lite — decoupled full-duplex interaction + async thinking.

The 24h-hackathon deliverable is NOT the model; it is the *control layer* that
realizes the [THINK]/[WAIT]/^/[CUT] protocol around frozen, off-the-shelf models:

  - Interaction layer (frozen):  MiniCPM-o 4.5 duplex  (voice+vision, native barge-in)
  - Thinking layer    (frozen):  Qwen3.5-35B-A3B on vLLM/SGLang (OpenAI-compatible)
  - Conductor         (this pkg): pure-asyncio orchestrator that ties them together

See duet/README.md and the design doc for the full architecture.
"""

__all__ = ["events", "state", "intent", "thinking", "write_seam", "conductor"]
