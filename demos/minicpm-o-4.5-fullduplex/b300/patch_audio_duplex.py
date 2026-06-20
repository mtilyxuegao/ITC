import re, shutil
BASE = "/root/MiniCPM-o-Demo/static"
JS = f"{BASE}/audio-duplex/audio-duplex-app.js"
HTML = f"{BASE}/audio-duplex/audio_duplex.html"
OMNI_JS = f"{BASE}/omni/omni-app.js"
for f in (JS, HTML, OMNI_JS):
    shutil.copy(f, f + ".prechunk.bak")

# ---------- audio-duplex-app.js ----------
js = open(JS, encoding="utf-8").read()
js = re.sub(r"^(const|let) CHUNK_MS = \d+;",
            "let CHUNK_MS = 1000;\nif (typeof window !== 'undefined') window.setChunkMs = (v) => { CHUNK_MS = Math.round(parseFloat(v) * 1000); };",
            js, count=1, flags=re.M)
assert "chunkSize: SAMPLE_RATE_IN }" in js, "worklet anchor not found"
js = js.replace("chunkSize: SAMPLE_RATE_IN }", "chunkSize: SAMPLE_RATE_IN * CHUNK_MS / 1000 }")  # both occurrences
cfg = "config: { length_penalty: parseFloat(document.getElementById('duplexLengthPenalty').value) || 1.05 }"
assert cfg in js, "config anchor not found"
cfg_new = ("config: { length_penalty: parseFloat(document.getElementById('duplexLengthPenalty').value) || 1.05"
           ", listen_prob_scale: parseFloat(document.getElementById('duplexListenProbScale') ? document.getElementById('duplexListenProbScale').value : 1) || 1.0"
           ", chunk_ms: CHUNK_MS }")
js = js.replace(cfg, cfg_new, 1)
open(JS, "w", encoding="utf-8").write(js)
print("audio-duplex-app.js patched")

# ---------- audio_duplex.html : add controls after Length Penalty row ----------
html = open(HTML, encoding="utf-8").read()
anchor = '                <div class="cg-row inline" data-tip="Enable text-to-speech audio response for the session">'
assert anchor in html, "html anchor (TTS Response row) not found"
opts = "".join(f'<option value="{v}"{" selected" if v=="1" else ""}>{("1.0" if v=="1" else v)}</option>'
               for v in ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1"])
ctrls = (
    '                <div class="cg-row inline" data-tip="说话欲望 vs 易被打断:<1 更爱说(难打断),>1 更克制(易让位)。会话开始时生效">\n'
    '                    <span class="cg-label" style="flex:none;">Listen Prob</span>\n'
    "                    <input type=\"range\" id=\"duplexListenProbScale\" min=\"0.2\" max=\"3.0\" step=\"0.1\" value=\"1.0\" oninput=\"document.getElementById('dxLpsVal').textContent=this.value\">\n"
    '                    <span id="dxLpsVal" style="font-size:11px;color:#888;min-width:26px;">1.0</span>\n'
    '                </div>\n'
    '                <div class="cg-row inline" data-tip="单元时长(秒)。越小延迟越低,但模型按1.0s训练,过小会变笨/卡顿。会话开始时生效">\n'
    '                    <span class="cg-label" style="flex:none;">Chunk (s)</span>\n'
    f'                    <select class="cg-input-sm" id="chunkSizeSel" onchange="setChunkMs(this.value)">{opts}</select>\n'
    '                </div>\n'
)
html = html.replace(anchor, ctrls + anchor, 1)
open(HTML, "w", encoding="utf-8").write(html)
print("audio_duplex.html patched")

# ---------- omni-app.js : Metrics panel default open ----------
o = open(OMNI_JS, encoding="utf-8").read()
mk = '<details class="config-group"><summary>Metrics</summary>'
assert mk in o, "omni Metrics anchor not found"
o = o.replace(mk, '<details class="config-group" open><summary>Metrics</summary>', 1)
open(OMNI_JS, "w", encoding="utf-8").write(o)
print("omni-app.js Metrics default-open patched")
