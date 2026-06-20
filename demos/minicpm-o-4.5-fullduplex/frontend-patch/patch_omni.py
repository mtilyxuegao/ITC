import shutil
H = "/root/MiniCPM-o-Demo/static/omni/omni.html"
J = "/root/MiniCPM-o-Demo/static/omni/omni-app.js"
shutil.copy(H, H + ".bak")
shutil.copy(J, J + ".bak")

# ---- Edit 1: HTML — insert a new control group right before "Vision Settings" ----
html = open(H, encoding="utf-8").read()
anchor = '<details class="config-group" open>\n            <summary>Vision Settings</summary>'
block = (
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
assert anchor in html, "HTML anchor not found"
html = html.replace(anchor, block + anchor, 1)
open(H, "w", encoding="utf-8").write(html)
print("Edit1 HTML: OK")

# ---- Edit 2 + 3: JS ----
js = open(J, encoding="utf-8").read()
a2 = "{ id: 'omniLengthPenalty', type: 'number' },"
assert a2 in js, "JS settings anchor not found"
js = js.replace(a2, a2 + "\n    { id: 'omniListenProbScale', type: 'range' },", 1)

a3 = "config: { length_penalty: parseFloat(document.getElementById('omniLengthPenalty').value) || 1.0 },"
assert a3 in js, "JS payload anchor not found"
a3new = "config: { length_penalty: parseFloat(document.getElementById('omniLengthPenalty').value) || 1.0, listen_prob_scale: parseFloat(document.getElementById('omniListenProbScale').value) || 1.0 },"
js = js.replace(a3, a3new, 1)
open(J, "w", encoding="utf-8").write(js)
print("Edit2+3 JS: OK")
