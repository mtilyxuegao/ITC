#!/usr/bin/env python3
"""Speed patches for the FROZEN MiniCPM-o 4.5 native checkout (H100), adapted from
teammate's b300/ acceleration work to our non-docker setup at /home/justin/MiniCPM-o-Demo.

Run ONCE against the MiniCPM-o-Demo checkout (idempotent-guarded, makes *.speedbak backups):
    python duet/deploy/patch_minicpm_speed.py            # default BASE
    BASE=/path/to/MiniCPM-o-Demo python duet/deploy/patch_minicpm_speed.py

What it changes (all reversible via the .speedbak files):
  1. chunk_ms: dynamic, default 0.5s (latency sweet spot; 1.0s = training-optimal).
     Wired through 3 layers — omni-app.js (CHUNK_MS + payload), omni.html (dropdown),
     pytorch_backend.set_duplex_config (model.CHUNK_MS/FIRST_CHUNK_MS per session, since
     the dict handed to the model otherwise omits chunk_ms so it'd stay 1000ms).
  2. playbackDelay default 200 -> 100ms (orthogonal client-side buffer; UI-adjustable).
  3. listen_prob_scale slider (Speak/Listen balance; a 抢答 knob: >1 = more yielding).
  4. py_backend/server.py: add --compile flag so the sbatch can enable torch.compile
     (config.compile path already exists; H100 sm_90 needs no ptxas swap).
  5. AudioWorklet capture chunkSize -> follows CHUNK_MS (the dominant TTFT lever; the first
     mic packet was hardcoded to 1.0s regardless of CHUNK_MS).
  6. config.py playback_delay_ms default 200 -> 80 (server frontend_defaults overrides the HTML).
NOTE: force_speak (verbatim text write-back) is applied SEPARATELY via the teammate's
patches/integrate_force_speak.py — see the footer of this file.

Latency model (teammate's measured profile, H100-portable):
  TTFT ≈ chunk_ms/2 (decision wait) + unit_compute(~200-250ms) + playbackDelay
  1.0s -> ~980ms ; 0.5s -> ~680ms ; floor ~400ms. <0.5s is off-distribution (model trained @1.0s).
"""
import os, re, shutil, sys

BASE = os.environ.get("BASE", "/home/justin/MiniCPM-o-Demo")
JS = f"{BASE}/static/omni/omni-app.js"
HTML = f"{BASE}/static/omni/omni.html"
PB = f"{BASE}/core/processors/pytorch_backend.py"
SRV = f"{BASE}/py_backend/server.py"
DEFAULT_CHUNK_MS = 500

if "omniListenProbScale" in open(JS, encoding="utf-8").read() or "chunkSizeSel" in open(HTML, encoding="utf-8").read():
    print("ALREADY PATCHED (frontend) — aborting to avoid double-apply."); sys.exit(2)
for f in (JS, HTML, PB, SRV):
    shutil.copy(f, f + ".speedbak")
print("backups: *.speedbak")

# 1) listen_prob_scale (must precede chunk_ms: the chunk payload anchor is this line) -------
html = open(HTML, encoding="utf-8").read()
anchor = '<details class="config-group" open>\n            <summary>Vision Settings</summary>'
assert anchor in html, "HTML Vision-Settings anchor not found"
lps_block = (
    '<details class="config-group" open>\n'
    '            <summary>Speak / Listen Balance</summary>\n'
    '            <div class="cg-body">\n'
    '                <div class="cg-row" data-tip="控制模型说话欲望 vs 易被打断：&lt;1 更爱说(难打断)，&gt;1 更克制(易让位)">\n'
    '                    <div class="cg-row inline">\n'
    '                        <span class="cg-label" style="flex:none;">Listen Prob Scale</span>\n'
    "                        <input type=\"range\" id=\"omniListenProbScale\" min=\"0.2\" max=\"3.0\" step=\"0.1\" value=\"1.0\" oninput=\"document.getElementById('lpsVal').textContent=this.value\">\n"
    '                        <span id="lpsVal" style="min-width:32px;text-align:center;font-size:12px;">1.0</span>\n'
    '                    </div>\n'
    '                    <div style="font-size:10px;color:#999;margin-top:2px;">&lt;1 更爱说·难打断　|　1.0 默认　|　&gt;1 更克制·易打断</div>\n'
    '                </div>\n'
    '            </div>\n'
    '        </details>\n'
    '        '
)
open(HTML, "w", encoding="utf-8").write(html.replace(anchor, lps_block + anchor, 1))
js = open(JS, encoding="utf-8").read()
a2 = "{ id: 'omniLengthPenalty', type: 'number' },"
assert a2 in js, "JS settings-list anchor not found"
js = js.replace(a2, a2 + "\n    { id: 'omniListenProbScale', type: 'range' },", 1)
a3 = "config: { length_penalty: parseFloat(document.getElementById('omniLengthPenalty').value) || 1.0 },"
assert a3 in js, "JS payload-config anchor not found"
a3new = "config: { length_penalty: parseFloat(document.getElementById('omniLengthPenalty').value) || 1.0, listen_prob_scale: parseFloat(document.getElementById('omniListenProbScale').value) || 1.0 },"
open(JS, "w", encoding="utf-8").write(js.replace(a3, a3new, 1))
print("1) listen_prob_scale added")

# 2) tunable chunk_ms (default 0.5s) ---------------------------------------------------------
js = open(JS, encoding="utf-8").read()
js, n = re.subn(r"^(const|let) CHUNK_MS = \d+;",
                f"let CHUNK_MS = {DEFAULT_CHUNK_MS};\nif (typeof window !== 'undefined') window.setChunkMs = (v) => {{ CHUNK_MS = Math.round(parseFloat(v) * 1000); }};",
                js, count=1, flags=re.M)
assert n == 1, "CHUNK_MS declaration not found"
cfg = "listen_prob_scale: parseFloat(document.getElementById('omniListenProbScale').value) || 1.0 }"
assert cfg in js, "listen_prob_scale config anchor missing"
open(JS, "w", encoding="utf-8").write(js.replace(cfg, cfg[:-2] + ", chunk_ms: CHUNK_MS }", 1))

html = open(HTML, encoding="utf-8").read()
_lbl = lambda v: {"1": "1.0 (训练点/最稳)", "0.5": "0.5 (推荐/快)"}.get(v, v)
opts = "".join(f'<option value="{v}"{" selected" if v=="0.5" else ""}>{_lbl(v)}</option>'
               for v in ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1"])
chunk_block = (
    '<details class="config-group" open>\n'
    '            <summary>Chunk Size · 单元时长/延迟</summary>\n'
    '            <div class="cg-body">\n'
    '                <div class="cg-row" data-tip="每个决策单元时长(秒)。越小延迟越低，但模型按1.0s训练，过小会变笨/卡顿。会话开始时生效，需重开会话。">\n'
    '                    <span class="cg-label" style="flex:none;">Chunk (s)</span>\n'
    f'                    <select class="cg-select" id="chunkSizeSel" onchange="setChunkMs(this.value)">{opts}</select>\n'
    '                </div>\n'
    '            </div>\n'
    '        </details>\n'
    '        '
)
anchor_html = '<details class="config-group" open>\n            <summary>Vision Settings</summary>'
open(HTML, "w", encoding="utf-8").write(html.replace(anchor_html, chunk_block + anchor_html, 1))

pb = open(PB, encoding="utf-8").read()
anchor_pb = "        duplex_view.config = DuplexConfig(**config)"
assert anchor_pb in pb, "set_duplex_config anchor not found"
inject = anchor_pb + "\n" + (
    "        # OMNI: apply chunk_ms to the model so it actually runs at that granularity (per session)\n"
    "        try:\n"
    "            _cm = int(duplex_view.config.chunk_ms)\n"
    "            _m = duplex_view._model\n"
    "            for _obj in (_m, getattr(_m, 'model', None)):\n"
    "                if _obj is not None:\n"
    "                    setattr(_obj, 'CHUNK_MS', _cm)\n"
    "                    setattr(_obj, 'FIRST_CHUNK_MS', _cm + 35)\n"
    "            logger.info(f'[duplex] model CHUNK_MS set to {_cm}ms')\n"
    "        except Exception as _e:\n"
    "            logger.warning(f'set chunk_ms on model failed: {_e}')"
)
open(PB, "w", encoding="utf-8").write(pb.replace(anchor_pb, inject, 1))
print("2) tunable chunk_ms (default 0.5s) wired through 3 layers")

# 3) playbackDelay default 200 -> 100ms ------------------------------------------------------
html = open(HTML, encoding="utf-8").read()
pd = '<input type="number" class="cg-input-sm" id="playbackDelay" value="200" min="0" max="2000" step="50">'
assert pd in html, "playbackDelay anchor not found"
open(HTML, "w", encoding="utf-8").write(html.replace(pd, pd.replace('value="200"', 'value="100"'), 1))
print("3) playbackDelay 200 -> 100ms")

# 4) py_backend/server.py: --compile flag ----------------------------------------------------
s = open(SRV, encoding="utf-8").read()
if 'parser.add_argument("--compile"' not in s:
    a1 = '    parser.add_argument("--duplex-pause-timeout", type=float, default=None)\n    args = parser.parse_args()'
    assert a1 in s, "server.py argparse anchor not found"
    s = s.replace(a1, '    parser.add_argument("--duplex-pause-timeout", type=float, default=None)\n    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on core submodules (one-time warmup at startup)")\n    args = parser.parse_args()', 1)
    a2 = '        "compile": cfg.compile,'
    assert a2 in s, "server.py compile config anchor not found"
    s = s.replace(a2, '        "compile": args.compile or cfg.compile,', 1)
    open(SRV, "w", encoding="utf-8").write(s)
    print("4) server.py --compile flag added")
else:
    print("4) server.py already has --compile")

# 5) AudioWorklet capture chunkSize -> follow CHUNK_MS (the dominant TTFT lever) ------------
# The live-mic worklet emits a chunk every `chunkSize` samples; it was hardcoded to
# SAMPLE_RATE_IN (=16000 = 1.0s), so the FIRST mic packet the model saw was always 1s
# regardless of CHUNK_MS. Make it follow CHUNK_MS so the first packet is ~0.5s (-~500ms TTFT).
js = open(JS, encoding="utf-8").read()
wk_old = "processorOptions: { chunkSize: SAMPLE_RATE_IN },"
wk_new = "processorOptions: { chunkSize: Math.round(SAMPLE_RATE_IN * CHUNK_MS / 1000) },"
n = js.count(wk_old)
if n:
    open(JS, "w", encoding="utf-8").write(js.replace(wk_old, wk_new))
    print(f"5) worklet chunkSize -> follows CHUNK_MS ({n} sites)")
else:
    print("5) worklet chunkSize already patched")

# 6) config.py playback_delay_ms default 200 -> 80 (server frontend_defaults overrides HTML) -
# /api/frontend_defaults serves playback_delay_ms to the UI and OVERRIDES the HTML value
# (duplex-ui restore priority: HTML -> server -> localStorage), so the real lever is here.
CFG = f"{BASE}/config.py"
cfg = open(CFG, encoding="utf-8").read()
pd_field = "    playback_delay_ms: int = Field(\n        default=200,"
if pd_field in cfg:
    open(CFG, "w", encoding="utf-8").write(cfg.replace(pd_field, "    playback_delay_ms: int = Field(\n        default=80,", 1))
    print("6) config.py playback_delay_ms 200 -> 80")
else:
    print("6) config.py playback_delay_ms already 80 (or anchor moved)")

print("\nALL SPEED PATCHES APPLIED OK")
# ---------------------------------------------------------------------------------------------
# force_speak (verbatim TEXT write-back, kills the TTS->ASR garble) is a SEPARATE integration
# from the teammate's branch — apply it with their idempotent patcher (NOT this script):
#   git -C /home/justin/ITC worktree add /tmp/tt-wt origin/jisen/thinker-talker
#   cd /tmp/tt-wt && python patches/integrate_force_speak.py --demo /home/justin/MiniCPM-o-Demo --check
#   cd /tmp/tt-wt && python patches/integrate_force_speak.py --demo /home/justin/MiniCPM-o-Demo
#   # (copies minicpm_ext/, patches server.py/worker.py/runtime/backend_client.py; .tt.bak backups;
#   #  --revert to undo). The proxy then sends {"type":"control.force_speak","payload":{"text":...}}.
