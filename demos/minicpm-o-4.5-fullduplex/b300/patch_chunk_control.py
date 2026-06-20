import re, shutil
BASE = "/root/MiniCPM-o-Demo"
JS = f"{BASE}/static/omni/omni-app.js"
HTML = f"{BASE}/static/omni/omni.html"
PB = f"{BASE}/core/processors/pytorch_backend.py"
for f in (JS, HTML, PB):
    shutil.copy(f, f + ".prechunk.bak")

# ---------- 1) omni-app.js ----------
js = open(JS, encoding="utf-8").read()
# CHUNK_MS -> dynamic let + global setter
js = re.sub(r"^(const|let) CHUNK_MS = \d+;",
            "let CHUNK_MS = 1000;\nif (typeof window !== 'undefined') window.setChunkMs = (v) => { CHUNK_MS = Math.round(parseFloat(v) * 1000); };",
            js, count=1, flags=re.M)
# send chunk_ms in preparePayload.config
anchor_cfg = "listen_prob_scale: parseFloat(document.getElementById('omniListenProbScale').value) || 1.0 }"
assert anchor_cfg in js, "JS config anchor not found"
js = js.replace(anchor_cfg, anchor_cfg[:-2] + ", chunk_ms: CHUNK_MS }", 1)
open(JS, "w", encoding="utf-8").write(js)
print("omni-app.js patched")

# ---------- 2) omni.html : add a Chunk Size select before Vision Settings ----------
html = open(HTML, encoding="utf-8").read()
anchor_html = '<details class="config-group" open>\n            <summary>Vision Settings</summary>'
assert anchor_html in html, "HTML anchor not found"
opts = "".join(
    f'<option value="{v}"{" selected" if v=="1" else ""}>{("1.0 (默认/推荐)" if v=="1" else v)}</option>'
    for v in ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1"]
)
block = (
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
html = html.replace(anchor_html, block + anchor_html, 1)
open(HTML, "w", encoding="utf-8").write(html)
print("omni.html patched")

# ---------- 3) pytorch_backend.py : apply chunk_ms to the model in set_duplex_config ----------
pb = open(PB, encoding="utf-8").read()
anchor_pb = "        duplex_view.config = DuplexConfig(**config)"
assert anchor_pb in pb, "pytorch_backend anchor not found"
inject = anchor_pb + "\n" + (
    "        # OMNI: apply chunk_ms to the model so it真正按该粒度跑 (per session)\n"
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
pb = pb.replace(anchor_pb, inject, 1)
open(PB, "w", encoding="utf-8").write(pb)
print("pytorch_backend.py patched")
