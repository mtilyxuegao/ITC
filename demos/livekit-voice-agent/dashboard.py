"""Thinker Activity dashboard — live visualization of the "second brain".

Standalone web app (decoupled from LiveKit). The agent POSTs events to /emit; this
broadcasts them over a WebSocket to any open browser, which renders a live timeline:
🧠 engaged → 🔍 searching → 📄 found → ✅ answer (and ✋ interrupted on barge-in).

Run:    python dashboard.py            # serves on 0.0.0.0:8800
Open:   http://localhost:8800          (tunnel the port if running on the server)
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

app = FastAPI(title="Thinker Activity")

_clients: set[WebSocket] = set()
_recent: deque[dict] = deque(maxlen=200)


@app.post("/emit")
async def emit(event: dict):
    _recent.append(event)
    dead = []
    for ws in list(_clients):
        try:
            await ws.send_text(json.dumps(event))
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)
    return {"ok": True, "clients": len(_clients)}


@app.get("/events.json")
async def events_json():
    return JSONResponse(list(_recent))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.add(ws)
    # replay recent so a late-joining browser has context
    for ev in list(_recent)[-40:]:
        await ws.send_text(json.dumps(ev))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)


@app.get("/")
async def index():
    return HTMLResponse(_PAGE)


_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Thinker Activity</title>
<style>
  :root { --bg:#0a0a0f; --card:#15151f; --line:#23232f; --muted:#6b6b80;
          --brain:#a78bfa; --search:#38bdf8; --ok:#4ade80; --warn:#f59e0b; --stop:#ef4444; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--bg); color:#e6e6ee; font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; padding:24px; }
  header { display:flex; align-items:center; gap:12px; margin-bottom:6px; }
  h1 { font-size:1.4rem; letter-spacing:-.3px; }
  .sub { color:var(--muted); font-size:.85rem; margin-bottom:18px; }
  .dot { width:9px; height:9px; border-radius:50%; background:var(--stop); }
  .dot.live { background:var(--ok); box-shadow:0 0 8px var(--ok); animation:pulse 1.4s infinite; }
  @keyframes pulse { 50% { opacity:.35; } }
  #feed { display:flex; flex-direction:column; gap:14px; max-width:760px; }
  .turn { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:14px 16px;
          border-left:3px solid var(--brain); animation:slide .25s ease; }
  .turn.stale { opacity:.45; border-left-color:var(--stop); }
  @keyframes slide { from { transform:translateY(8px); opacity:0; } }
  .turn .head { display:flex; align-items:center; gap:8px; font-weight:600; }
  .turn .obj { color:#cfcfe6; font-weight:500; margin:4px 0 2px; }
  .meta { color:var(--muted); font-size:.72rem; }
  .step { display:flex; gap:9px; align-items:flex-start; margin-top:10px; padding-left:2px; }
  .step .ic { flex:0 0 20px; text-align:center; }
  .step.search .lbl { color:var(--search); }
  .step.answer .lbl { color:var(--ok); }
  .step.timeout .lbl { color:var(--warn); }
  .step.spoken .lbl { color:var(--brain); }
  .lbl { font-weight:600; font-size:.86rem; }
  .results { margin-top:6px; display:flex; flex-direction:column; gap:6px; }
  .res { background:#0f0f17; border:1px solid var(--line); border-radius:9px; padding:7px 10px; }
  .res a { color:var(--search); text-decoration:none; font-size:.84rem; font-weight:600; }
  .res .sn { color:var(--muted); font-size:.78rem; margin-top:2px;
             display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
  .answer-txt { color:#e6e6ee; margin-top:3px; }
  .empty { color:var(--muted); font-style:italic; }
</style></head>
<body>
  <header><span id="live" class="dot"></span><h1>Thinker Activity</h1></header>
  <div class="sub">Live view of the background reasoning model — escalation, web search, conclusions, barge-ins.</div>
  <div id="feed"><div class="empty" id="placeholder">Waiting for the Thinker to engage… ask the assistant a hard or current-events question.</div></div>
<script>
  const feed = document.getElementById('feed');
  const liveDot = document.getElementById('live');
  const turns = {};   // task_id -> element
  const fmtTime = ts => new Date((ts||Date.now()/1000)*1000).toLocaleTimeString();

  function placeholderGone(){ const p=document.getElementById('placeholder'); if(p) p.remove(); }
  function ensureTurn(ev){
    placeholderGone();
    let el = turns[ev.task_id];
    if(!el){
      el = document.createElement('div'); el.className='turn'; el.dataset.epoch=ev.epoch;
      el.innerHTML = `<div class="head">🧠 <span>Thinker engaged</span></div>
        <div class="obj"></div><div class="meta"></div><div class="steps"></div>`;
      turns[ev.task_id]=el; feed.prepend(el);
    }
    return el;
  }
  function addStep(el, cls, ic, label, extraHtml=''){
    const s=document.createElement('div'); s.className='step '+cls;
    s.innerHTML=`<div class="ic">${ic}</div><div style="flex:1"><span class="lbl">${label}</span>${extraHtml}</div>`;
    el.querySelector('.steps').appendChild(s); return s;
  }

  function handle(ev){
    if(ev.type==='escalated'){
      const el=ensureTurn(ev);
      el.querySelector('.obj').textContent = ev.objective || '';
      el.querySelector('.meta').textContent = `reason: ${ev.reason||'—'} · epoch ${ev.epoch} · ${fmtTime(ev.ts)}`;
    } else if(ev.type==='thinking'){
      const el=ensureTurn(ev); addStep(el,'thinking','💭','reasoning…');
    } else if(ev.type==='search_start'){
      const el=ensureTurn(ev); addStep(el,'search','🔍',`searching: “${ev.query}”`);
    } else if(ev.type==='search_results'){
      const el=ensureTurn(ev);
      const rs=(ev.results||[]);
      let html='<div class="results">';
      if(!rs.length) html+='<div class="empty">no results</div>';
      for(const r of rs){ html+=`<div class="res"><a href="${r.url}" target="_blank">${r.title||r.url}</a><div class="sn">${(r.snippet||'')}</div></div>`; }
      html+='</div>';
      addStep(el,'search','📄',`found ${rs.length} result(s)`, html);
    } else if(ev.type==='answer'){
      const el=ensureTurn(ev);
      addStep(el,'answer','✅','conclusion', `<div class="answer-txt">${ev.text||''}</div>`);
    } else if(ev.type==='spoken'){
      const el=ensureTurn(ev); addStep(el,'spoken','🔊','Talker spoke it');
    } else if(ev.type==='timeout'){
      const el=ensureTurn(ev); addStep(el,'timeout','⏱','timed out — fell back');
    } else if(ev.type==='interrupted'){
      // mark any turns from older epochs as discarded
      for(const id in turns){ if(parseInt(turns[id].dataset.epoch) < ev.epoch){ turns[id].classList.add('stale'); addStep(turns[id],'timeout','✋','interrupted — result discarded'); } }
    }
  }

  function connect(){
    const proto = location.protocol==='https:'?'wss:':'ws:';
    const ws = new WebSocket(`${proto}//${location.host}/ws`);
    ws.onopen = ()=> liveDot.classList.add('live');
    ws.onclose = ()=> { liveDot.classList.remove('live'); setTimeout(connect, 1500); };
    ws.onmessage = e => { try { handle(JSON.parse(e.data)); } catch(_){} };
  }
  connect();
</script>
</body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8800)
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
