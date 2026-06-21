/* Latent Lab — Talker–Thinker live status panel (English).
 * Subscribes to gateway /observer/{session_id} and shows:
 *   🗣️ Talker drafts + latency   🧠 Thinker decision + context + latency
 *   🔎 web search   ⚡ inject / cut   🎤 your speech (ASR)
 */
(function () {
  const panel = document.getElementById('statusPanel');
  if (!panel) return;

  panel.innerHTML = `
    <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;font:12px/1.5 system-ui,-apple-system,sans-serif">
      <b style="background:linear-gradient(90deg,#7c8cff,#54e0c7);-webkit-background-clip:text;background-clip:text;color:transparent;font-weight:800">🧠 Thinker</b>
      <span id="ttConn" style="color:#8b94a7">Disconnected</span>
      <button id="ttToggle" style="margin-left:auto;border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.06);color:#cdd6e6;border-radius:6px;cursor:pointer;padding:1px 8px">Collapse</button>
      <div style="flex-basis:100%;height:0"></div>
      <span style="color:#9aa4b2">🗣️ Talker: <b id="ttSmall" style="color:#5fe0d0">—</b></span>
      <span style="color:#9aa4b2">🧠 Thinker: <b id="ttBig" style="color:#9cc3ff">—</b></span>
      <span style="color:#9aa4b2">Action: <b id="ttAct" style="color:#fff">—</b></span>
      <span style="color:#6b7488">Session: <code id="ttSid">—</code></span>
    </div>
    <div id="ttLog" style="margin-top:8px;max-height:180px;overflow:auto;font:12px/1.55 ui-monospace,Menlo,monospace;background:rgba(8,10,16,.6);color:#d6def0;border-radius:9px;padding:8px"></div>`;

  const $ = (id) => document.getElementById(id);
  $('ttToggle').onclick = () => {
    const log = $('ttLog');
    const hidden = log.style.display === 'none';
    log.style.display = hidden ? 'block' : 'none';
    $('ttToggle').textContent = hidden ? 'Collapse' : 'Expand';
    panel.style.width = hidden ? '360px' : '220px';
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
      if (msg.text) addRow('small', `🗣️ Talker: ${esc(msg.text)}`);
    } else if (msg.type === 'tt.status') {
      if (msg.stage === 'thinker') {
        $('ttBig').textContent = (msg.latency_ms ?? '?') + 'ms';
        $('ttAct').textContent = msg.action || '—';
        let s = `🧠 Thinker [<b>${esc(msg.action)}</b>] ${msg.latency_ms}ms`;
        if (msg.text) s += ` → "${esc(msg.text)}"`;
        addRow('big', s);
        if (msg.searches && msg.searches.length) addRow('search', `🔎 Web search: ${msg.searches.map(esc).join('  /  ')}`);
        addRow('input', `↳ Context → Thinker: ${esc((msg.input || '').replace(/\n/g, ' | ').slice(-180))}`);
        if (msg.reason) addRow('reason', `   Reason: ${esc(msg.reason)}`);
      } else if (msg.stage === 'asr') {
        addRow('search', `🎤 You (ASR): ${esc(msg.text)}`);
      } else if (msg.stage === 'fired') {
        addRow('fired', `⚡ Injected & spoken (${esc(msg.action)}): "${esc(msg.text)}"`);
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
      ws.onopen = () => { $('ttConn').textContent = '● Connected'; $('ttConn').style.color = '#56d364'; };
      ws.onmessage = (ev) => { try { handle(JSON.parse(ev.data)); } catch (e) {} };
      ws.onclose = () => { $('ttConn').textContent = '○ Disconnected'; $('ttConn').style.color = '#8b94a7'; resolve(); };
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
