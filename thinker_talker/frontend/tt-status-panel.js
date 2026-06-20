/* Talker–Thinker 实时状态面板(旁路,不影响原页面逻辑)。
 * 订阅 gateway 的 /observer/{session_id},显示:
 *   🗣️ 小模型说了什么 + 延迟   🧠 大模型每步决策 + 输入(证明已传入)+ 延迟
 *   🔎 联网搜索   ⚡ 注入/打断
 * 数据来源:worker 下行的 text 增量(经 hub 镜像)+ 编排层发的 tt.status。
 */
(function () {
  const panel = document.getElementById('statusPanel');
  if (!panel) return;

  panel.innerHTML = `
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;font:12px/1.5 system-ui,'PingFang SC',sans-serif">
      <b style="color:#5b9dff">🧠 双脑状态</b>
      <span id="ttConn" style="color:#999">未连接</span>
      <button id="ttToggle" style="margin-left:auto;border:1px solid #ccc;background:#f5f5f5;border-radius:6px;cursor:pointer;padding:1px 8px">收起</button>
      <div style="flex-basis:100%;height:0"></div>
      <span>🗣️小模型:<b id="ttSmall">—</b></span>
      <span>🧠大模型:<b id="ttBig">—</b></span>
      <span>动作:<b id="ttAct">—</b></span>
      <span style="color:#999">会话:<code id="ttSid">—</code></span>
    </div>
    <div id="ttLog" style="margin-top:8px;max-height:170px;overflow:auto;font:12px/1.55 ui-monospace,Menlo,monospace;background:#0f1115;color:#d6def0;border-radius:8px;padding:8px"></div>`;

  const $ = (id) => document.getElementById(id);
  $('ttToggle').onclick = () => {
    const log = $('ttLog');
    const hidden = log.style.display === 'none';
    log.style.display = hidden ? 'block' : 'none';
    $('ttToggle').textContent = hidden ? '收起' : '展开';
    panel.style.width = hidden ? '340px' : '210px';
  };
  const elLog = $('ttLog');
  const esc = (s) => (s || '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const COLOR = { small: '#5fe0d0', big: '#9cc3ff', search: '#ffb454', input: '#9aa4b2', reason: '#9aa4b2', fired: '#ff6b6b' };

  function addRow(cls, html) {
    const div = document.createElement('div');
    div.style.color = COLOR[cls] || '#d6def0';
    div.style.padding = '1px 0';
    const t = new Date().toLocaleTimeString();
    div.innerHTML = `<span style="color:#5b6675">${t}</span> ${html}`;
    elLog.appendChild(div);
    while (elLog.childElementCount > 60) elLog.removeChild(elLog.firstChild);
    elLog.scrollTop = elLog.scrollHeight;
  }

  function handle(msg) {
    if (msg.type === 'response.output.delta' && msg.kind === 'text') {
      const m = msg.metrics || {};
      const lat = m.cost_all_ms || m.cost_llm_ms;
      if (lat) $('ttSmall').textContent = Math.round(lat) + 'ms';
      if (msg.text) addRow('small', `🗣️ 小模型: ${esc(msg.text)}`);
    } else if (msg.type === 'tt.status') {
      if (msg.stage === 'thinker') {
        $('ttBig').textContent = (msg.latency_ms ?? '?') + 'ms';
        $('ttAct').textContent = msg.action || '—';
        let s = `🧠 大模型 [<b>${esc(msg.action)}</b>] ${msg.latency_ms}ms`;
        if (msg.text) s += ` → "${esc(msg.text)}"`;
        addRow('big', s);
        if (msg.searches && msg.searches.length) addRow('search', `🔎 联网搜索: ${msg.searches.map(esc).join('  /  ')}`);
        addRow('input', `↳ 实际传给大模型的输入: ${esc((msg.input || '').replace(/\n/g, ' | ').slice(-180))}`);
        if (msg.reason) addRow('reason', `   理由: ${esc(msg.reason)}`);
      } else if (msg.stage === 'asr') {
        addRow('search', `🎤 你说(ASR): ${esc(msg.text)}`);
      } else if (msg.stage === 'fired') {
        addRow('fired', `⚡ 已注入并让小模型说出(${esc(msg.action)}): "${esc(msg.text)}"`);
      }
    }
  }

  async function latestSession() {
    try {
      const r = await fetch('/observer/sessions', { cache: 'no-store' });
      const d = await r.json();
      const s = d.sessions || [];
      return s.length ? s[s.length - 1] : null;
    } catch (e) { return null; }
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function connect(sid) {
    return new Promise((resolve) => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      const ws = new WebSocket(`${proto}://${location.host}/observer/${sid}`);
      ws.onopen = () => { $('ttConn').textContent = '● 已连接'; $('ttConn').style.color = '#56d364'; };
      ws.onmessage = (ev) => { try { handle(JSON.parse(ev.data)); } catch (e) {} };
      ws.onclose = () => { $('ttConn').textContent = '○ 断开'; $('ttConn').style.color = '#999'; resolve(); };
      ws.onerror = () => {};
    });
  }

  (async function loop() {
    while (true) {
      const sid = await latestSession();
      if (!sid) { await sleep(2000); continue; }
      $('ttSid').textContent = sid;
      await connect(sid);
      await sleep(1000);
    }
  })();
})();
