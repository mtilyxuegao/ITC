/**
 * audio-duplex-app.js — Audio Full-Duplex page entry (Layer 2 ES Module)
 *
 * Pure audio duplex — microphone or file input with waveform visualization.
 * Supports Live mode (mic only) and File mode (file-only / file+mic).
 */

// Layer 0: Pure logic
import { AudioDeviceSelector } from '../lib/audio-device-selector.js';
import { resampleAudio, arrayBufferToBase64, escapeHtml } from '../duplex/lib/duplex-utils.js';
import { RealtimeSession } from '../duplex/lib/realtime-session.js';
import { SessionRecorder } from '../duplex/lib/session-recorder.js';
import { measureLUFS } from '../duplex/lib/lufs.js';
import { MixerController } from '../duplex/lib/mixer-controller.js';

// Layer 1: UI binding
import {
    MetricsPanel,
    getStatusPanelHTML,
    initHealthCheck,
    loadFrontendDefaults,
    setDefaultPauseBtnState,
    setDefaultForceListenBtnState,
    setDuplexButtonStates,
    setQueueButtonStates,
    wireDuplexControls,
    initDataTipTooltips,
    SettingsPersistence,
    getMixerPanelHTML,
} from '../duplex/ui/duplex-ui.js';
import { startDingDongLoop, playAlarmBell, playSessionChime } from '../duplex/lib/queue-chimes.js';
import { createTtsRefController } from '../duplex/ui/tts-ref-controller.js';
import { initRefAudio } from '../duplex/ui/ref-audio-init.js';

// ============================================================================
// Constants & State
// ============================================================================
const SAMPLE_RATE_IN = 16000;
const SAMPLE_RATE_OUT = 24000;
const CHUNK_MS = 1000;
const FILE_MAX_DURATION = 300; // 5 minutes

let currentMode = 'live';
let session = null;
let media = null;          // current MediaProvider

// Save & Share
const _saveShareUI = typeof SaveShareUI !== 'undefined'
    ? new SaveShareUI({ containerId: 'save-share-container', appType: 'audio_duplex', collectComment: true })
    : null;
let selectedFile = null;

// Microphone state (live mode only)
let audioCtxIn = null;
let audioStream = null;
let captureNodeLive = null;
let audioSource = null;
let analyserNode = null;
let waveformRunning = false;

// Session recording
let sessionRecorder = null;

// 排队倒计时（使用共享 CountdownTimer 模块）
import { CountdownTimer } from '../lib/countdown-timer.js';
let _queueCountdownLabel = null;
const _duplexCountdown = new CountdownTimer(({ remaining, position, queueLength }) => {
    if (_queueCountdownLabel) {
        _queueCountdownLabel.textContent = remaining > 0
            ? `Queue ${position}/${queueLength}, ~${remaining}s`
            : `Queue ${position}/${queueLength}, overtime +${Math.abs(remaining)}s`;
    }
});

function setStatusLamp(state) {
    const lamp = document.getElementById('statusLamp');
    if (!lamp) return;
    lamp.className = 'status-lamp';
    if (state === 'hidden') { lamp.classList.remove('visible'); return; }
    lamp.classList.add('visible', state);
    const labels = { live: 'LIVE', preparing: 'Preparing', stopped: 'Stopped' };
    lamp.querySelector('.label').textContent = labels[state] || state;
}

let _queuePhase = null; // null | 'queuing' | 'almost' | 'assigned'
let _stopDingDong = null;
let lastRecordingBlob = null;
let thinkerSessionId = null;
let thinkerActiveTaskId = null;
let thinkerTurnSeq = 0;

// Mixer state
// MixerController instance (created after DOM setup, see bottom of file)
let mixerCtrl = null;

const metricsPanel = new MetricsPanel();

// ============================================================================
// Init: Status panel + health check + defaults + settings persistence
// ============================================================================
document.getElementById('panelStatus').innerHTML = getStatusPanelHTML();
document.getElementById('mixerPanel').innerHTML = getMixerPanelHTML();
let _stopHealthCheck = initHealthCheck('serviceStatus');
initDataTipTooltips();
drawIdleWaveform();

const settingsPersistence = new SettingsPersistence('audio_duplex_settings', [
    // Mode selector
    { type: 'mode', selector: '.mode-btn' },
    // File options
    { type: 'radio', name: 'fileAudioMode' },
    { id: 'padBeforeSec', type: 'number' },
    { id: 'padAfterSec', type: 'number' },
    // Session
    { id: 'playbackDelay', type: 'number' },
    { id: 'maxKvTokens', type: 'number' },
    { id: 'duplexLengthPenalty', type: 'number' },
    { id: 'thinkerModeEnabled', type: 'checkbox' },
    { id: 'outboundGreetingEnabled', type: 'checkbox' },
    { id: 'outboundGreetingText', type: 'input' },
    // System prompt
    { id: 'systemPrompt', type: 'textarea' },
    // TTS ref mode
    { type: 'radio', name: 'duplexTtsRefMode' },
    // Recording
    { id: 'recCheckbox', type: 'checkbox' },
    // Mixer
    { id: 'mxFileTarget', type: 'number' },
    { id: 'mxFileTrim', type: 'range' },
    { id: 'mxMicTarget', type: 'number' },
    { id: 'mxMicTrim', type: 'range' },
    { id: 'mxMonitor', type: 'range' },
]);

// Priority: HTML defaults → server defaults → localStorage → preset (highest)
loadFrontendDefaults().then(() => {
    settingsPersistence.restore();
    syncThinkerModeFromControl(false);
    _duplexPreset.init();
});
window._settingsPersistence = settingsPersistence;
const adxDeviceSelector = new AudioDeviceSelector({
    micSelectEl: document.getElementById('adxMicDevice'),
    speakerSelectEl: document.getElementById('adxSpeakerDevice'),
    refreshBtnEl: document.getElementById('adxBtnRefreshDevices'),
    storagePrefix: 'audio_duplex',
    onSpeakerChange: () => {
        if (session && session.audioPlayer && session.audioPlayer._ctx) {
            adxDeviceSelector.applySinkId(session.audioPlayer._ctx);
        }
    },
});
adxDeviceSelector.init();

document.getElementById('btnResetSettings')?.addEventListener('click', () => {
    if (confirm('Reset all settings to defaults?')) {
        localStorage.removeItem('audio_duplex_preset');
        adxDeviceSelector.clearSaved();
        settingsPersistence.clear();
    }
});

// ============================================================================
// Preset Selector
// ============================================================================
// ============================================================================
// Ref Audio (init before preset so preset can update it)
// ============================================================================
const duplexTtsRef = createTtsRefController('duplex', () => refAudio.getBase64());
const refAudio = initRefAudio('refAudioPlayerDuplex', {
    onTtsHintUpdate: () => duplexTtsRef.updateHint(),
});
duplexTtsRef.init();

// ============================================================================
// Preset Selector
// ============================================================================
const _duplexPreset = new PresetSelector({
    container: document.getElementById('presetSelectorDuplex'),
    page: 'audio_duplex',
    detailsEl: document.getElementById('duplexSysPromptDetails'),
    onSelect: (preset, { audioLoaded } = {}) => {
        if (preset && preset.system_prompt) {
            document.getElementById('systemPrompt').value = preset.system_prompt;
            settingsPersistence.save();
        }
        if (audioLoaded && preset && preset.ref_audio && preset.ref_audio.data) {
            refAudio.setAudio(preset.ref_audio.data, preset.ref_audio.name, preset.ref_audio.duration);
        }
    },
    storageKey: 'audio_duplex_preset',
});

// ============================================================================
// Waveform Drawing
// ============================================================================
function drawIdleWaveform() {
    const canvas = document.getElementById('waveformCanvas');
    const container = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(0,255,136,0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();
}

function startWaveformDrawing() {
    waveformRunning = true;
    document.getElementById('waveformPlaceholder').style.display = 'none';
    document.getElementById('waveformOverlay').classList.add('visible');
    requestAnimationFrame(drawWaveform);
}

function stopWaveformDrawing() {
    waveformRunning = false;
    document.getElementById('waveformOverlay').classList.remove('visible');
    drawIdleWaveform();
}

function drawWaveform() {
    if (!waveformRunning || !analyserNode) return;
    requestAnimationFrame(drawWaveform);
    const canvas = document.getElementById('waveformCanvas');
    const container = canvas.parentElement;
    const dpr = window.devicePixelRatio || 1;
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (w === 0 || h === 0) return;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    const bufLen = analyserNode.frequencyBinCount;
    const data = new Uint8Array(bufLen);
    analyserNode.getByteTimeDomainData(data);
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const sliceWidth = w / bufLen;
    let x = 0;
    for (let i = 0; i < bufLen; i++) {
        const v = data[i] / 128.0;
        const y = v * h / 2;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        x += sliceWidth;
    }
    ctx.stroke();
    ctx.strokeStyle = 'rgba(0,255,136,0.06)';
    ctx.lineWidth = 0.5;
    for (let gy = 0; gy < h; gy += h / 4) {
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    }
}

window.addEventListener('resize', () => { if (!waveformRunning) drawIdleWaveform(); });

// ============================================================================
// Chat Log UI
// ============================================================================
const chatLog = document.getElementById('chatLog');

function addSystemLog(text) {
    document.getElementById('chatEmpty').style.display = 'none';
    const el = document.createElement('div');
    el.className = 'conv-entry system';
    el.innerHTML = `<div class="conv-icon">&#x2699;</div><div class="conv-text">${escapeHtml(text)}</div>`;
    chatLog.appendChild(el);
    scrollChatLog();
}

function addUserLog(text) {
    document.getElementById('chatEmpty').style.display = 'none';
    const el = document.createElement('div');
    el.className = 'conv-entry user';
    el.innerHTML = `<div class="conv-icon">&#x1F464;</div><div class="conv-text"><span class="speaker user-tag">You:</span> ${escapeHtml(text)}</div>`;
    chatLog.appendChild(el);
    scrollChatLog();
}

function addAiLog(text) {
    document.getElementById('chatEmpty').style.display = 'none';
    const el = document.createElement('div');
    el.className = 'conv-entry ai';
    el.innerHTML = `<div class="conv-icon">&#x1F916;</div><div class="conv-text"><span class="speaker ai">AI:</span> <span class="ai-text">${escapeHtml(text)}</span></div>`;
    chatLog.appendChild(el);
    scrollChatLog();
    return el;
}

function scrollChatLog() { chatLog.scrollTop = chatLog.scrollHeight; }

// ============================================================================
// Thinker Mode
// ============================================================================
function isThinkerModeEnabled() {
    return !!document.getElementById('thinkerModeEnabled')?.checked;
}

function setThinkerStatus(text, state = 'off') {
    const el = document.getElementById('thinkerModeStatus');
    if (!el) return;
    el.textContent = text;
    el.className = `thinker-mode-status ${state}`;
}

function setThinkerLogTask(text) {
    const el = document.getElementById('thinkerLogTask');
    if (el) el.textContent = text;
}

function ensureThinkerLogPanel() {
    const panel = document.getElementById('thinkerLogPanel');
    if (panel) panel.hidden = false;
    return document.getElementById('thinkerLogLines');
}

function appendThinkerLog(text, state = 'info') {
    const lines = ensureThinkerLogPanel();
    if (!lines) return;

    const el = document.createElement('div');
    el.className = `thinker-log-line ${state}`;
    const time = new Date().toLocaleTimeString([], {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });
    el.innerHTML = `<span class="thinker-log-time">${escapeHtml(time)}</span><span class="thinker-log-msg">${escapeHtml(text)}</span>`;
    lines.appendChild(el);

    while (lines.children.length > 80) {
        lines.removeChild(lines.firstElementChild);
    }
    lines.scrollTop = lines.scrollHeight;
    // mirror to the right-side debug panel, colored by actor
    const _sa = { input: 'THINKER', muted: 'COORD', ok: 'VOICE', error: 'WARN', stream: 'THINKER', info: 'NOTE' };
    dbg(_sa[state] || 'NOTE', text);
}

// ── Right-side debug panel: a live "who is doing what" trace ──────────────────
function ensureDebugPanel() {
    if (document.getElementById('ttDebugPanel')) return;
    const style = document.createElement('style');
    style.textContent =
        '#ttDebugPanel{position:fixed;top:64px;right:10px;width:440px;max-width:46vw;height:calc(100vh - 82px);' +
        'background:#0c0e13;border:1px solid #222a36;border-radius:10px;z-index:99999;display:flex;flex-direction:column;' +
        'font:12px/1.45 ui-monospace,Menlo,monospace;box-shadow:0 10px 30px rgba(0,0,0,.45)}' +
        '#ttDbgHead{display:flex;justify-content:space-between;align-items:center;padding:7px 10px;border-bottom:1px solid #222a36;color:#cdd6e3}' +
        '#ttDbgHead button{background:#1a2230;color:#9aa7b8;border:1px solid #2a3342;border-radius:6px;padding:2px 8px;margin-left:6px;cursor:pointer;font:11px ui-monospace}' +
        '#ttDbgLines{overflow-y:auto;padding:6px 8px;flex:1}' +
        '.ttd{display:flex;gap:8px;padding:2px 0;border-bottom:1px solid #14181f}' +
        '.ttd .tm{color:#5b6677;flex:0 0 60px}.ttd .ac{flex:0 0 60px;font-weight:700;text-align:right}' +
        '.ttd .mg{color:#d7dde6;flex:1;word-break:break-word;white-space:pre-wrap}' +
        '.ac-USER{color:#4ade80}.ac-TALKER{color:#60a5fa}.ac-THINKER{color:#a78bfa}.ac-VOICE{color:#2dd4bf}' +
        '.ac-COORD{color:#9ca3af}.ac-SEARCH{color:#38bdf8}.ac-WARN{color:#f59e0b}.ac-NOTE{color:#6b7686}';
    document.head.appendChild(style);
    const p = document.createElement('div');
    p.id = 'ttDebugPanel';
    p.innerHTML = '<div id="ttDbgHead"><b>🪲 Thinker–Talker debug · who is doing what</b>' +
        '<span><button id="ttDbgClear">clear</button><button id="ttDbgHide">hide</button></span></div>' +
        '<div id="ttDbgLines"></div>';
    document.body.appendChild(p);
    p.querySelector('#ttDbgClear').onclick = () => { p.querySelector('#ttDbgLines').innerHTML = ''; };
    p.querySelector('#ttDbgHide').onclick = () => { p.style.display = 'none'; };
}

function dbg(actor, msg) {
    try {
        ensureDebugPanel();
        const box = document.getElementById('ttDbgLines');
        if (!box) return;
        const d = new Date();
        const p2 = (n) => String(n).padStart(2, '0');
        const tm = `${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}.${String(d.getMilliseconds()).padStart(3, '0').slice(0, 2)}`;
        const el = document.createElement('div');
        el.className = 'ttd';
        el.innerHTML = `<span class="tm">${tm}</span><span class="ac ac-${actor}">${actor}</span><span class="mg"></span>`;
        el.querySelector('.mg').textContent = msg;
        box.appendChild(el);
        while (box.children.length > 500) box.removeChild(box.firstChild);
        box.scrollTop = box.scrollHeight;
    } catch (_) {}
}

function syncThinkerModeFromControl(announce = true) {
    const enabled = isThinkerModeEnabled();
    if (enabled) {
        setThinkerStatus('On', 'on');
        if (announce) appendThinkerLog('mode enabled; recognized user turns will stream here', 'ok');
    } else {
        thinkerTurnSeq += 1;
        thinkerActiveTaskId = null;
        setThinkerStatus('Off', 'off');
        setThinkerLogTask('idle');
        if (announce) appendThinkerLog('mode disabled', 'muted');
    }
}

function getThinkerSessionId() {
    if (thinkerSessionId) return thinkerSessionId;
    thinkerSessionId = session?.recordingSessionId || session?.sessionId || `audio_duplex_${Date.now().toString(36)}`;
    return thinkerSessionId;
}

async function thinkerRequest(sessionId, path, options = {}) {
    const headers = { ...(options.headers || {}) };
    if (options.body) headers['Content-Type'] = 'application/json';
    const resp = await fetch(`/api/thinker/sessions/${encodeURIComponent(sessionId)}${path}`, {
        ...options,
        headers,
    });
    if (resp.ok) return resp.json();

    let detail = resp.statusText;
    try {
        const data = await resp.json();
        detail = data.detail || detail;
    } catch (_) {
        try { detail = await resp.text(); } catch (_) {}
    }
    throw new Error(detail || `Thinker request failed (${resp.status})`);
}

function buildThinkerVoiceQuery(text) {
    const systemPrompt = document.getElementById('systemPrompt')?.value?.trim();
    const promptBlock = systemPrompt
        ? `Assistant speaking style and session context:\n${systemPrompt}\n\n`
        : '';
    return `${promptBlock}The user just spoke in the MiniCPM-o real-time voice demo:\n${text.trim()}\n\nReturn a concise answer that the voice Talker can say aloud. Do not include hidden reasoning or markdown.`;
}

// ── Speech coordinator: hand off to the Thinker, never double-talk ───────────
// MiniCPM-o (the Talker) isn't trained for two-brain coordination — it can't decide
// to defer, stay quiet, or weave in the Thinker's answer. So we orchestrate it:
//   - on a "thinker task" we force the Talker to LISTEN (silent) and say a short
//     "let me check" filler, then run the Thinker async (Talker stays live for other turns);
//   - when the answer is ready we HOLD it and speak it only in a real silence gap
//     (user not talking; Talker not mid-sentence), forcing the Talker quiet during
//     delivery so the two voices never overlap.
// Voice here is the browser's TTS; MiniCPM-o-voice (chat-TTS) is the later upgrade.
let coordTalkerSpeaking = false;
let coordUserSpeaking = false;
let coordTtsBusy = false;
let coordTtsInjecting = false;   // a MiniCPM-o-voice clip is currently playing through audioPlayer
let coordChatWs = null;          // in-flight chat-TTS WS (so a barge-in can cancel it)
let coordQueue = [];   // [{text, immediate}]
let coordVadRunning = false;
let coordTick = null;
let coordLastAiMs = 0;          // last time ANY AI voice (MiniCPM-o or injected) played — for echo-guarding the ASR
let coordLastAssistantText = ''; // last assistant phrase seen by ASR echo guard
let coordPendingClassify = false; // held the Talker on speech onset; awaiting classify
let coordSafetyTimer = null;
let coordGateTimer = null;       // backstop so the duplex kill-switch can never permanently mute the model
let coordOverlapAtSpeechStart = false;
let coordSharedSilenceTimer = null;
let coordOutboundGreetingSent = false;
// Thinker-period re-hold: while a Thinker turn owns the floor the duplex SPEAK head is GATED
// (not just force_listen, which is advisory and can't stop an autonomous speak). The chat-TTS
// mouth still speaks via direct playChunk; only the model's own reflex blurt is dropped.
let coordThinkerHold = false;
// Provisional gate: raised the instant the user STOPS speaking (onUserSpeechEnd), BEFORE we
// know whether the turn needs the Thinker. It gates the duplex SPEAK head so the model can't
// blurt its own reflex answer during the browser-ASR-final lag. It is short-lived: promoted to
// a full Thinker hold if the turn escalates, or lifted (liftProvisionalGate) the moment the
// turn is classified as simple OR after PROV_GATE_MS as a backstop. Unlike coordThinkerHold it
// does NOT set thinkerHoldActive, so a normal return-to-listen can still lift it.
let coordProvisionalGate = false;
let coordProvGateTimer = null;
const PROV_GATE_MS = 1500;       // backstop: never hold a simple turn silent longer than this
// Spurious-barge-in recovery: an ANSWER cut by a *suspected* barge-in that no real user turn
// confirms (ambient / headphone leak) is re-voiced instead of being silently dropped to text.
let coordRevoiceText = null;
let coordRevoiceSeq = 0;
let coordRevoiceTimer = null;

// SIL (Synthetic Interaction Layer): when true, the MiniCPM-o duplex session is a
// PERMANENT-LISTEN perception organ ("ear") that never gets the floor — 100% of speech
// is authored by the coordinator and spoken through the chat-TTS "mouth". This makes the
// model's autonomous SPEAK head (the source of the blurt/double-talk) structurally
// impossible: there is no path that un-mutes the duplex model.
// DISABLED: SIL mutes the real MiniCPM-o talker and routes ALL speech through chat-TTS
// injection, which is unreliable during a live duplex session -> the assistant went silent.
// With SIL off the duplex model speaks again; the stop-interception + hold-by-default +
// barge-in fixes below are independent of SIL and still apply.
let SIL_MODE = false;

function holdTalker() {
    if (!session) return;
    session.forceListenActive = true;             // every input chunk now forces listen
    try { session.audioPlayer.stopAll(); } catch (_) {}  // cut any in-progress speech
}
// Enter the Thinker-period hold: gate the duplex SPEAK head for the WHOLE turn so it cannot
// blurt its own reflex answer over the Thinker's (chat-TTS) answer. force_listen alone is
// advisory and cannot abort an autonomous speak; the gate DROPS any speak frames the model
// emits (realtime-session _handleSpeak), while the chat-TTS mouth (direct playChunk) is
// unaffected. `thinkerHoldActive` tells _handleListen NOT to lift the gate mid-turn. Held
// across filler -> thinking -> answer; released by releaseTalker when the turn is fully done.
function holdTalkerForThinker() {
    coordThinkerHold = true;
    if (coordGateTimer) { clearTimeout(coordGateTimer); coordGateTimer = null; }
    // Promote any provisional gate into the full Thinker hold (cancel its auto-lift backstop so
    // the gate can't drop mid-turn). The gate set below stays up for the whole Thinker turn.
    if (coordProvGateTimer) { clearTimeout(coordProvGateTimer); coordProvGateTimer = null; }
    coordProvisionalGate = false;
    if (session) {
        session.forceListenActive = true;
        session.duplexOutputGated = true;
        session.thinkerHoldActive = true;
        try { session.audioPlayer && session.audioPlayer.stopAll(); } catch (_) {}
    }
}

// User stopped speaking: gate the duplex SPEAK head NOW so it can't blurt a reflex answer in
// the window before we classify the turn (browser-ASR-final lag). This is the main fix for the
// Talker talking garbage while the Thinker thinks. If the turn escalates, holdTalkerForThinker
// promotes this to a full hold; if it's simple, liftProvisionalGate hands the floor back. A
// genuine Thinker hold already owns the floor, so this is a no-op then.
function onUserSpeechEnd() {
    if (coordThinkerHold) return;          // already fully held for a Thinker turn
    if (!session) return;
    coordProvisionalGate = true;
    session.duplexOutputGated = true;      // drop the model's reflex speak frames for now
    session.forceListenActive = true;
    try { session.audioPlayer && session.audioPlayer.stopAll(); } catch (_) {}
    if (coordProvGateTimer) clearTimeout(coordProvGateTimer);
    // Backstop: if no classification arrives, never hold a (likely simple) turn silent forever.
    coordProvGateTimer = setTimeout(liftProvisionalGate, PROV_GATE_MS);
}

// Lift the provisional gate (NOT a Thinker hold): hand the floor back to the duplex SPEAK head.
function liftProvisionalGate() {
    if (coordProvGateTimer) { clearTimeout(coordProvGateTimer); coordProvGateTimer = null; }
    if (!coordProvisionalGate || coordThinkerHold) return;  // never lift a real Thinker hold
    coordProvisionalGate = false;
    if (session) { session.duplexOutputGated = false; session.forceListenActive = false; }
}
function releaseTalker() {
    // Releasing ends any Thinker hold and hands the floor back to the duplex head.
    coordThinkerHold = false;
    if (session) session.thinkerHoldActive = false;
    // A full release supersedes any pending provisional gate / its auto-lift backstop.
    if (coordProvGateTimer) { clearTimeout(coordProvGateTimer); coordProvGateTimer = null; }
    coordProvisionalGate = false;
    // SIL: the ear never speaks. Releasing would hand the floor to the duplex SPEAK head,
    // which free-runs and blurts. Keep it held; the mouth (chat-TTS) is the only emitter.
    if (SIL_MODE) return;
    // Releasing means we WANT the duplex head to speak this turn — lift any stale kill-switch
    // from a previous stop so the fresh answer is never silently dropped.
    if (coordGateTimer) { clearTimeout(coordGateTimer); coordGateTimer = null; }
    if (session) { session.forceListenActive = false; session.duplexOutputGated = false; }
}

function enqueueSpeech(text, immediate) {
    const clean = (text || '').trim();
    if (!clean) return;
    coordQueue.push({ text: clean, immediate: !!immediate });
    tryDeliver();
}

// Cut the currently-playing MiniCPM-o-voice clip (barge-in). Hard-stops audio and
// closes the chat-TTS WS so the server stops synthesizing.
function cancelInjection(reason) {
    if (coordChatWs) { try { coordChatWs._cancel && coordChatWs._cancel(); } catch (_) {} }
    try { session && session.audioPlayer && session.audioPlayer.stopAll(); } catch (_) {}
    // Kill-switch: gate the released duplex SPEAK head too. stopAll() only cuts buffered
    // client audio; the server keeps streaming speak frames (~18s tail) which would resume
    // playing. Gating drops those frames until the model returns to listen, so stop/barge-in
    // actually halts the ear. (No-op on the chat-TTS mouth, which plays via direct playChunk.)
    try { if (session) session.duplexOutputGated = true; } catch (_) {}
    // A stop/barge-in owns the gate now — cancel any pending provisional auto-lift so it can't
    // drop this kill-switch gate out from under us.
    if (coordProvGateTimer) { clearTimeout(coordProvGateTimer); coordProvGateTimer = null; }
    coordProvisionalGate = false;
    // Backstop: the gate normally lifts when the model returns to listen (_handleListen).
    // If the duplex head is slow to emit a clean listen (or never does), the gate must NOT
    // mute the model forever — auto-lift after a bounded window so the next user turn is
    // answerable. Re-cut once on lift so a still-streaming stale tail doesn't audibly resume.
    if (coordGateTimer) clearTimeout(coordGateTimer);
    coordGateTimer = setTimeout(() => {
        coordGateTimer = null;
        // Never auto-lift while a Thinker turn still owns the floor — that would re-open the
        // duplex blurt over the answer. The hold is released explicitly by releaseTalker.
        if (session && session.duplexOutputGated && !coordThinkerHold) {
            session.duplexOutputGated = false;
            try { session.audioPlayer && session.audioPlayer.stopAll(); } catch (_) {}
            appendThinkerLog('kill-switch auto-released (no listen boundary in time) — model responsive again', 'muted');
        }
    }, 2000);
    if (coordTtsInjecting) appendThinkerLog(`interrupted (${reason || 'barge-in'}) — stopped`, 'muted');
    coordTtsInjecting = false;
}

async function tryDeliver() {
    if (coordTtsBusy || !coordQueue.length) return;
    const next = coordQueue[0];
    if (coordUserSpeaking) return;                  // never talk over the user
    if (!next.immediate && coordTalkerSpeaking) return; // wait for a Talker gap (not for the filler)
    coordQueue.shift();
    coordTtsBusy = true;
    coordTtsInjecting = true;
    coordLastAssistantText = next.text;
    holdTalker();                                    // keep the Talker silent while we speak
    ensureBargeVad();                                // so the user can interrupt our clip
    const seq = thinkerTurnSeq;
    appendThinkerLog(`${next.immediate ? 'filler' : 'answer'} (MiniCPM-o voice): "${next.text.slice(0, 64)}"`, 'ok');
    let ok = false;
    try { ok = await synthesizeAndPlayThinkerVoice(next.text, seq, next.immediate); } catch (_) { ok = false; }
    coordTtsInjecting = false;
    coordTtsBusy = false;
    // ONE voice only: if MiniCPM-o-voice synthesis didn't run, show text but do NOT
    // speak with a second (browser) voice. Keep the Talker muted until the thinker
    // turn is fully done so it can't blurt its own reflex answer.
    if (!ok) {
        // An ANSWER (not a disposable filler) was cut by a *suspected* barge-in. If no real
        // user turn confirms it (markClassifyDone clears coordPendingClassify), the cut was
        // spurious — ambient / headphone leak tripping the RMS VAD — so re-voice it instead
        // of dropping it to text. A genuine stop/new question cancels the pending re-voice.
        if (!next.immediate && coordPendingClassify && seq === thinkerTurnSeq) {
            armAnswerRevoice(next.text, seq);
        } else {
            appendThinkerLog('(answer shown as text — could not voice it this time)', 'muted');
        }
    }
    if (!coordQueue.length && !thinkerActiveTaskId && !coordRevoiceTimer) releaseTalker();
    setTimeout(tryDeliver, 60);
}

// An answer clip was cut by the barge VAD. Wait a beat: if a real user turn gets classified
// the cut was legitimate (stop) or superseded (new question) and markClassifyDone cancels
// this. If nothing arrives, the cut was spurious (ambient / headphone-leak echo) — re-voice
// the answer so the Thinker's reply is never silently lost.
function armAnswerRevoice(text, seq) {
    if (coordRevoiceTimer) clearTimeout(coordRevoiceTimer);
    coordRevoiceText = text;
    coordRevoiceSeq = seq;
    appendThinkerLog('answer cut — confirming whether it was a real barge-in…', 'muted');
    coordRevoiceTimer = setTimeout(() => {
        coordRevoiceTimer = null;
        const t = coordRevoiceText;
        coordRevoiceText = null;
        if (!t) return;
        if (seq !== thinkerTurnSeq) return;            // superseded by a newer turn
        if (!coordPendingClassify) return;             // a real stop/question was handled — honor it
        if (coordTtsBusy || coordUserSpeaking || coordQueue.length) return;
        coordPendingClassify = false;
        appendThinkerLog('no real barge-in arrived — re-voicing the answer', 'ok');
        enqueueSpeech(t, false);
    }, 1300);
}
function clearAnswerRevoice() {
    if (coordRevoiceTimer) { clearTimeout(coordRevoiceTimer); coordRevoiceTimer = null; }
    coordRevoiceText = null;
}

// Synthesize `text` in the live duplex voice via streaming chat-TTS and play it
// through the SAME audioPlayer MiniCPM-o uses (so stopAll() can interrupt it).
// Audio arrives as response.output.delta kind=audio = raw float32 PCM @24kHz base64.
function synthesizeAndPlayThinkerVoice(text, seq, isFiller) {
    if (!session || !session.audioPlayer) return Promise.resolve(false);
    const player = session.audioPlayer;
    const ref = (duplexTtsRef && duplexTtsRef.getBase64 && duplexTtsRef.getBase64())
        || (refAudio && refAudio.getBase64 && refAudio.getBase64());
    const body = {
        messages: [{ role: 'user', content: text }],
        streaming: true,
        generation: {
            // Filler/hold-phrases are short, FIXED strings. Greedy decode (do_sample:false) on a
            // fixed string reliably hits EOS (~1.5s) instead of sampling its way past EOS and
            // rambling to the token cap — that was the 14s/54s "let me check that for you." bug.
            // The lowered cap is a hard backstop; the answer path is unchanged (220 / sampling).
            max_new_tokens: isFiller ? 48 : 220, do_sample: isFiller ? false : true,
            length_penalty: parseFloat(document.getElementById('lengthPenalty')?.value) || 1.05,
        },
        image: { max_slice_nums: 1 }, omni_mode: false,
        tts: { enabled: true, mode: 'audio_assistant' }, use_tts_template: true,
    };
    if (ref) body.tts.ref_audio_data = ref;
    let url = (location.protocol === 'https:' ? 'wss' : 'ws') + '://' + location.host + '/v1/realtime?mode=chat';
    if (window.ClientIdentity) url = window.ClientIdentity.appendToUrl(url);

    return new Promise((resolve) => {
        let ws; try { ws = new WebSocket(url); } catch (_) { resolve(false); return; }
        coordChatWs = ws;
        let firstPlayMs = 0, totalSamples = 0, started = false, settled = false, endTimer = null;
        const settle = (okv) => {
            if (settled) return; settled = true;
            if (endTimer) clearTimeout(endTimer);
            if (coordChatWs === ws) coordChatWs = null;
            try { if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'session.close', reason: 'turn_done' })); } catch (_) {}
            try { ws.close(); } catch (_) {}
            dbg('VOICE', okv ? 'finished speaking' : 'voice stopped (interrupted/superseded)');
            resolve(okv);
        };
        ws._cancel = () => settle(false);
        ws.onmessage = (ev) => {
            if (seq !== thinkerTurnSeq) { settle(false); return; }   // a newer turn superseded this
            let msg; try { msg = JSON.parse(ev.data); } catch (_) { return; }
            const t = msg.type;
            if (t === 'session.queue_done') { try { ws.send(JSON.stringify({ type: 'session.init', payload: {} })); } catch (_) {} }
            else if (t === 'session.created') { try { ws.send(JSON.stringify({ type: 'input.append', input: body })); } catch (_) {} }
            else if (t === 'response.output.delta' && msg.kind === 'audio' && msg.audio) {
                if (!started) { started = true; firstPlayMs = performance.now(); dbg('VOICE', 'speaking in MiniCPM-o voice'); try { if (!player.turnActive) player.beginTurn(); } catch (_) {} }
                try {
                    player.playChunk(msg.audio);
                    if (!player.playing && typeof player._startPlayback === 'function') {
                        setTimeout(() => {
                            try {
                                if (player.turnActive && !player.playing) player._startPlayback();
                            } catch (_) {}
                        }, 0);
                    }
                } catch (e) {
                    appendThinkerLog(`voice chunk error: ${e.message || e}`, 'error');
                }
                coordLastAiMs = performance.now();                  // our own voice is playing -> echo-guard the ASR
                totalSamples += Math.floor(msg.audio.length * 3 / 16); // base64 -> float32 samples
            }
            else if (t === 'response.done') {
                const playMs = (totalSamples / 24000) * 1000;
                const remaining = started ? Math.max(0, firstPlayMs + playMs - performance.now()) : 0;
                endTimer = setTimeout(() => { try { if (player.turnActive) player.endTurn(); } catch (_) {} settle(true); }, remaining + 120);
            }
            else if (t === 'error') { settle(false); }
        };
        ws.onerror = () => settle(false);
        setTimeout(() => settle(false), 35000); // safety
    });
}

// (browser-TTS fallback removed — one unified MiniCPM-o voice only)

// Overlap VAD on the AEC'd mic stream: mark ^ while our clip plays. The ASR
// transcript decides later whether this was a harmless backchannel or a real cut.
// Pure decision for the barge-in VAD (extracted so it can be unit-tested without a mic).
// Rule: only count a cut when the user makes a NEW speech onset AFTER the clip has played
// with at least one quiet frame. This stops the user's own trailing voice — the very turn
// that triggered this reply — and ambient noise from cutting the reply the instant it starts.
function bargeVadStep(state, injecting, rms, threshold = 0.045, frames = 4) {
    if (!injecting) return { cut: false, state: { wasInjecting: false, sawQuiet: false, above: 0 } };
    let { wasInjecting, sawQuiet, above } = state;
    if (!wasInjecting) { wasInjecting = true; sawQuiet = false; above = 0; }   // a fresh clip just started
    let cut = false;
    if (rms <= threshold) { sawQuiet = true; above = 0; }                       // user is quiet while it plays
    else if (sawQuiet) { above += 1; if (above >= frames) cut = true; }         // new onset after a quiet gap = barge-in
    return { cut, state: { wasInjecting, sawQuiet, above } };
}

function ensureBargeVad() {
    if (coordVadRunning || !audioStream) return;
    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const an = ctx.createAnalyser(); an.fftSize = 512;
        ctx.createMediaStreamSource(audioStream).connect(an);
        const buf = new Float32Array(an.fftSize);
        coordVadRunning = true;
        let vadState = { wasInjecting: false, sawQuiet: false, above: 0 };
        let overlapLogged = false;
        const tick = () => {
            if (!coordVadRunning) { try { ctx.close(); } catch (_) {} return; }
            an.getFloatTimeDomainData(buf);
            let s = 0; for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i];
            const rms = Math.sqrt(s / buf.length);
            const r = bargeVadStep(vadState, coordTtsInjecting, rms);
            vadState = r.state;
            if (r.cut) {
                // CUT THE AUDIO NOW — don't wait for the ASR transcript. The recognizer then
                // still classifies the words (stop -> stay halted; question -> answer).
                cancelInjection('barge-in: you spoke over the reply');
                coordUserSpeaking = true;
                coordOverlapAtSpeechStart = true;
                coordPendingClassify = true;
                if (!coordSafetyTimer) {
                    coordSafetyTimer = setTimeout(() => {
                        if (coordPendingClassify) {
                            coordPendingClassify = false;
                            coordOverlapAtSpeechStart = false;
                            coordUserSpeaking = false;
                            if (!coordQueue.length && !coordTtsInjecting && !thinkerActiveTaskId) releaseTalker();
                        }
                        coordSafetyTimer = null;
                    }, 3000);
                }
                if (!overlapLogged) {
                    overlapLogged = true;
                    dbg('USER', '^ barge-in over reply — cut audio, classifying');
                    appendThinkerLog('^ barge-in: cut the reply, waiting for transcript', 'stream');
                }
            }
            if (!coordTtsInjecting) overlapLogged = false;
            requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    } catch (_) {}
}
function stopBargeVad() { coordVadRunning = false; }

function normalizeTurnText(text) {
    return (text || '')
        .toLowerCase()
        .replace(/[.,!?;:"'()[\]{}，。！？；：“”‘’、]/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

function wordTokens(text) {
    const normalized = normalizeTurnText(text);
    return normalized ? normalized.split(' ') : [];
}

function isBackchannel(text) {
    const normalized = normalizeTurnText(text);
    if (!normalized) return false;
    const tokens = wordTokens(text);
    if (tokens.length > 4) return false;
    return /^(yeah|yep|yes|ok|okay|sure|right|got it|i see|uh huh|uh-huh|mhm|mm hm|mm-hm|sounds good|fine|好|好的|嗯|嗯嗯|对|是的|可以|行|明白|继续)$/.test(normalized);
}

// A bare floor command ("stop", "cancel", "never mind") is NOT a query — it must
// halt the floor, never spin up a Thinker turn. Matches the WHOLE utterance only,
// so "stop the music" (a real request) is not caught. See handleRecognizedUserTurn.
function isControlStop(text) {
    const s = normalizeTurnText(text);
    if (!s) return false;
    return /^(please )?(stop|stop it|stop stop|stop please|stop talking|cancel|cancel that|hold on|wait stop|quiet|be quiet|shush|shut up|never mind|nevermind|forget it|enough|that s enough|abort)$/.test(s)
        || /^(停|停下|停止|别说了|闭嘴|算了|取消)$/.test(s);
}

// Rough end-of-turn check: a turn ending in a dangling article/preposition/conjunction
// (e.g. "what if the") is probably an incomplete fragment, so we wait. Anything else is
// treated as a complete utterance. Used under SIL to decide answer-now vs keep-listening.
function isLikelyFragment(text) {
    const t = wordTokens(text);
    if (!t.length) return true;
    return /^(the|a|an|of|if|and|or|but|to|for|with|my|your|in|on|at|that|this|than|as)$/.test(t[t.length - 1]);
}

function hasConditionChange(text) {
    const s = normalizeTurnText(text);
    return /\b(wait|actually|instead|change|switch|skip|stop|no need|don't|do not|not that|wrong|mistake|i meant|rather|but|however|restaurant|street|scene|tone|topic|character|setting|more|less|hurry|nothing vague|confirm|already)\b/.test(s)
        || /(等等|等一下|不对|错了|改成|换成|不要|别问|不用|其实|应该|刚才|餐厅|饭店|街|语气|主题|确认)/.test(s);
}

function looksLikeAssistantEcho(text) {
    const heard = wordTokens(text);
    const last = wordTokens(coordLastAssistantText);
    if (heard.length < 3 || last.length < 4) return false;
    if (hasConditionChange(text)) return false;
    const lastSet = new Set(last);
    const overlap = heard.filter((token) => lastSet.has(token)).length;
    return overlap / Math.max(heard.length, 1) >= 0.68;
}

function clearSharedSilenceTimer() {
    if (coordSharedSilenceTimer) {
        clearTimeout(coordSharedSilenceTimer);
        coordSharedSilenceTimer = null;
    }
}

function scheduleSharedSilenceLog() {
    clearSharedSilenceTimer();
    coordSharedSilenceTimer = setTimeout(() => {
        if (!coordUserSpeaking && !coordTalkerSpeaking && !coordTtsInjecting && coordPendingClassify) {
            appendThinkerLog('[PEND1S] shared silence while waiting for the user to finish', 'stream');
            dbg('COORD', '[PEND1S] shared silence');
        }
    }, 1000);
}

// Decide whether a turn is the Thinker's job (current facts / deep reasoning).
function shouldDefer(text) {
    const s = (text || '').toLowerCase();
    return /\b(latest|current|today|tonight|right now|recent|news|price|stock|shares?|quote|worth|cost|weather|forecast|score|standings|who is|who's|when (did|does|will|is)|where is|how much|how many|why|how (do|does|can|would|should)|explain|compare|difference|calculate|analyze|evaluate|plan|research|look up|find out|figure out|summari[sz]e|write|draft|compose|story|book|website|server|availability|schedule|20(2[4-9]|3[0-9]))\b/.test(s)
        || /https?:\/\/|www\./i.test(text || '');
}

function shouldUseThinkerForTurn(text, context = {}) {
    if (shouldDefer(text)) return true;
    if (context.overlappedAssistant && hasConditionChange(text)) return true;
    if ((thinkerActiveTaskId || coordQueue.length) && hasConditionChange(text)) return true;
    return false;
}

function thinkerHoldPhrase(text, context = {}) {
    if (context.overlappedAssistant || hasConditionChange(text)) {
        return 'Got it. I’ll adjust and check that.';
    }
    if (/\b(today|current|latest|price|stock|weather|news|availability|schedule|website|server)\b|www\.|https?:\/\//i.test(text || '')) {
        return 'Let me check that for you.';
    }
    return 'Let me think for a moment.';
}

function maybeStartOutboundGreeting() {
    if (coordOutboundGreetingSent) return;
    if (!document.getElementById('outboundGreetingEnabled')?.checked) return;
    const text = (document.getElementById('outboundGreetingText')?.value || '').trim();
    if (!text) return;
    coordOutboundGreetingSent = true;
    appendThinkerLog(`[OUTBOUND] assistant opens: "${text.slice(0, 96)}"`, 'ok');
    dbg('TALKER', 'assistant-initiated opener queued');
    holdTalker();
    enqueueSpeech(text, true);
}

function markClassifyDone() {
    coordUserSpeaking = false;
    coordPendingClassify = false;
    coordOverlapAtSpeechStart = false;
    clearSharedSilenceTimer();
    if (coordSafetyTimer) {
        clearTimeout(coordSafetyTimer);
        coordSafetyTimer = null;
    }
    // A real turn was classified -> a prior answer-cut was legitimate (stop) or superseded
    // (new question). Don't re-voice the stale answer.
    clearAnswerRevoice();
}

function resetOngoingDelivery(reason) {
    coordQueue = [];
    cancelInjection(reason);
    coordTtsBusy = false;
    coordTalkerSpeaking = false;
}

function handleRecognizedUserTurn(text, context = {}) {
    const clean = (text || '').trim();
    if (!clean) return;

    // Floor command short-circuit: "stop"/"cancel"/… halts everything and goes back
    // to LISTEN. It is NOT forwarded to the classifier (hasConditionChange matches
    // "stop", which previously routed it to the Thinker and wasted a full round-trip).
    if (isControlStop(clean)) {
        markClassifyDone();
        addUserLog(clean);
        dbg('USER', `heard → "${clean}"`);
        appendThinkerLog(`[STOP] floor command — halting, not escalating: "${clean}"`, 'warn');
        dbg('COORD', '[CUT] stop command: cut audio + cancel thinker, back to LISTEN');
        resetOngoingDelivery('user stop');
        holdTalker();
        if (thinkerActiveTaskId) void interruptThinker('user stop', { announce: false });
        return;
    }

    const overlappedAssistant = !!context.overlappedAssistant;
    if (overlappedAssistant && isBackchannel(clean)) {
        markClassifyDone();
        appendThinkerLog(`[NO CUT] backchannel while assistant speaks: "${clean}"`, 'muted');
        dbg('USER', `backchannel, no cutoff → "${clean}"`);
        tryDeliver();
        return;
    }

    if (overlappedAssistant && looksLikeAssistantEcho(clean)) {
        markClassifyDone();
        appendThinkerLog(`ignored AI echo during overlap: "${clean.slice(0, 48)}"`, 'muted');
        dbg('NOTE', `ignored likely echo → "${clean.slice(0, 48)}"`);
        return;
    }

    markClassifyDone();
    addUserLog(clean);
    dbg('USER', `heard → "${clean}"`);

    const needsThinker = shouldUseThinkerForTurn(clean, { overlappedAssistant });
    if (overlappedAssistant || ((thinkerActiveTaskId || coordQueue.length) && hasConditionChange(clean))) {
        appendThinkerLog(`[CUT][WAIT] user changed conditions: "${clean.slice(0, 96)}"`, 'warn');
        dbg('COORD', '[CUT] stopped current assistant audio; [WAIT] invalidated stale thought');
        resetOngoingDelivery('condition change');
        holdTalker();
    }

    if (needsThinker) {
        dbg('COORD', 'decision: this needs the THINKER');
        appendThinkerLog(`[THINK] deferring to Thinker: "${clean}"`, 'input');
        holdTalker();
        void submitThinkerTurn(clean);
        enqueueSpeech(thinkerHoldPhrase(clean, { overlappedAssistant }), true);
        return;
    }

    if (thinkerActiveTaskId || coordQueue.length || coordThinkerHold) {
        // A Thinker turn still owns the floor. MiniCPM-o answers from its OWN un-gated
        // audio buffer, which still holds the deferred question -> releasing now makes it
        // blurt a reflex answer to THAT question over the Thinker's real answer (double-talk).
        // coordThinkerHold covers the window where the hold is set but the task isn't accepted
        // yet and the queue has drained — without it an echoed/filler turn could fall through
        // to releaseTalker() below and un-gate the head mid-think.
        dbg('COORD', 'simple turn during an active Thinker turn -> staying held (no double-talk)');
        appendThinkerLog(`held (Thinker busy), not answering: "${clean}"`, 'muted');
        return;
    }
    if (SIL_MODE) {
        // SIL: the ear never free-runs, so we can't let MiniCPM-o answer this itself.
        if (isLikelyFragment(clean)) {
            // Incomplete utterance — keep listening for the rest instead of answering noise.
            dbg('COORD', 'SIL: incomplete fragment — held, waiting for the user to finish');
            appendThinkerLog(`SIL: held (fragment): "${clean}"`, 'muted');
            return;
        }
        // A complete turn the fast classifier didn't escalate: route it to the Thinker so
        // it is answered through the mouth (with a filler for instant feedback), rather than
        // left silent. Keeps the no-blurt guarantee — the ear is never un-muted.
        dbg('COORD', 'SIL: routing complete turn to the Thinker (ear never speaks)');
        appendThinkerLog(`[THINK] SIL route: "${clean}"`, 'input');
        holdTalker();
        void submitThinkerTurn(clean);
        enqueueSpeech(thinkerHoldPhrase(clean, { overlappedAssistant }), true);
        return;
    }
    dbg('COORD', 'decision: TALKER can answer this directly');
    appendThinkerLog(`Talker handles: "${clean}"`, 'muted');
    releaseTalker();
}

// Called from pollThinkerResult when the Thinker's answer is ready: queue it for
// delivery in the next natural gap (not spoken immediately).
function speakThinkerConclusion(text) {
    enqueueSpeech(text, false);
}

// The duplex model never surfaces the USER's transcribed turn, so the thinker has
// no question text to act on. Bridge it with the browser's built-in speech
// recognition (Chrome): transcribe the user's speech and submit each final
// utterance to the thinker. Server-side ASR (MiniCPM-o chat-with-audio) is the
// robust upgrade path; this is the quick, infra-free version.
let thinkerAsr = null;

function startThinkerAsr() {
    try {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) {
            appendThinkerLog('this browser has no SpeechRecognition — use Chrome for thinker mode', 'error');
            return;
        }
        if (thinkerAsr) return;
        const rec = new SR();
        rec.continuous = true;
        rec.interimResults = false;
        const sp = document.getElementById('systemPrompt')?.value || '';
        rec.lang = /[一-鿿]/.test(sp) ? 'zh-CN' : 'en-US';
        const aiSpeakingNow = () => coordTalkerSpeaking || coordTtsInjecting
            || (performance.now() - coordLastAiMs < 450);
        rec.onspeechstart = () => {
            clearSharedSilenceTimer();
            coordUserSpeaking = true;
            const overlappedAssistant = aiSpeakingNow();
            coordOverlapAtSpeechStart = overlappedAssistant;
            if (overlappedAssistant) {
                dbg('USER', '^ overlap while assistant is speaking — waiting to classify');
                appendThinkerLog('^ overlap detected; classify as backchannel or barge-in', 'stream');
                coordPendingClassify = true;
                if (coordSafetyTimer) clearTimeout(coordSafetyTimer);
                coordSafetyTimer = setTimeout(() => {
                    if (coordPendingClassify) {
                        coordPendingClassify = false;
                        coordOverlapAtSpeechStart = false;
                        if (!coordQueue.length && !coordTtsInjecting && !thinkerActiveTaskId && !coordThinkerHold) releaseTalker();
                    }
                }, 4500);
                return;
            }
            dbg('USER', 'started speaking');
            // Real user onset: hold MiniCPM-o so it can't blurt a reflex answer before we classify.
            holdTalker();
            dbg('COORD', 'muted MiniCPM-o while you speak');
            coordPendingClassify = true;
            if (coordSafetyTimer) clearTimeout(coordSafetyTimer);
            coordSafetyTimer = setTimeout(() => {
                if (coordPendingClassify) { coordPendingClassify = false; if (!coordQueue.length && !coordTtsInjecting && !thinkerActiveTaskId && !coordThinkerHold) releaseTalker(); }
            }, 4000);
        };
        rec.onspeechend = () => {
            coordUserSpeaking = false;
            dbg('USER', 'stopped speaking');
            // Gate the duplex SPEAK head NOW (before classification) so it can't blurt a reflex
            // answer during the ASR-final lag. Promoted to a full hold if the turn escalates,
            // lifted (releaseTalker) if it's simple, or auto-lifted after PROV_GATE_MS.
            onUserSpeechEnd();
            scheduleSharedSilenceLog();
            tryDeliver();
        };
        rec.onresult = (ev) => {
            for (let i = ev.resultIndex; i < ev.results.length; i += 1) {
                if (!ev.results[i].isFinal) continue;
                const t = (ev.results[i][0].transcript || '').trim();
                if (!t) continue;
                handleRecognizedUserTurn(t, {
                    overlappedAssistant: coordOverlapAtSpeechStart || aiSpeakingNow(),
                });
            }
        };
        rec.onerror = (e) => appendThinkerLog(`ASR error: ${e.error || e.message}`, 'error');
        rec.onend = () => {
            // auto-restart while the session is live (recognition stops periodically)
            if (thinkerAsr === rec && session && session.running) {
                try { rec.start(); } catch (_) {}
            }
        };
        thinkerAsr = rec;
        rec.start();
        appendThinkerLog(`ASR listening for your questions (${rec.lang}, browser)`, 'ok');
    } catch (e) {
        appendThinkerLog(`ASR start failed: ${e.message}`, 'error');
    }
}

function stopThinkerAsr() {
    if (!thinkerAsr) return;
    const r = thinkerAsr;
    thinkerAsr = null;
    try { r.stop(); } catch (_) {}
}

async function pollThinkerResult(sessionId, taskId, seq) {
    const startedAt = performance.now();
    appendThinkerLog(`stream opened for ${taskId.slice(0, 12)}`, 'info');
    for (let attempt = 0; attempt < 30; attempt += 1) {
        if (seq !== thinkerTurnSeq || taskId !== thinkerActiveTaskId) return;
        const elapsed = ((performance.now() - startedAt) / 1000).toFixed(1);
        appendThinkerLog(`tick ${attempt + 1}: waiting for thinker output (${elapsed}s)`, 'stream');
        const data = await thinkerRequest(
            sessionId,
            `/result?timeout_ms=1000&task_id=${encodeURIComponent(taskId)}`
        );
        const result = data.result;
        if (!result) continue;
        if (result.task_id !== taskId) continue;

        if (seq !== thinkerTurnSeq) return;
        thinkerActiveTaskId = null;
        setThinkerStatus('Done', 'on');
        setThinkerLogTask('done');
        appendThinkerLog(`final output received (${result.text.length} chars)`, 'ok');
        dbg('THINKER', `answer ready → "${result.text}"`);
        addAiLog(`[Thinker] ${result.text}`);
        speakThinkerConclusion(result.text);
        return;
    }

    if (seq === thinkerTurnSeq && taskId === thinkerActiveTaskId) {
        setThinkerStatus('Waiting', 'thinking');
        appendThinkerLog('still waiting; thinker task remains active', 'stream');
    }
}

async function submitThinkerTurn(text) {
    const cleanText = (text || '').trim();
    if (!isThinkerModeEnabled() || !cleanText) return;

    const sessionId = getThinkerSessionId();
    const seq = ++thinkerTurnSeq;
    holdTalkerForThinker();   // gate the duplex SPEAK head for the whole turn (re-hold)
    setThinkerStatus('Thinking', 'thinking');
    appendThinkerLog(`user turn captured: ${cleanText.slice(0, 120)}`, 'input');

    try {
        appendThinkerLog('submitting turn to async thinker', 'info');
        const data = await thinkerRequest(sessionId, '/reescalate', {
            method: 'POST',
            body: JSON.stringify({ query: buildThinkerVoiceQuery(cleanText) }),
        });
        const taskId = data.task?.task_id;
        if (seq !== thinkerTurnSeq) return;             // superseded — newer turn owns the hold
        if (!taskId) { releaseTalker(); return; }       // no task accepted — don't stay muted

        thinkerActiveTaskId = taskId;
        setThinkerLogTask(taskId.slice(0, 12));
        appendThinkerLog(`task accepted: ${taskId.slice(0, 12)}`, 'ok');
        dbg('THINKER', 'thinking… (searching the web + reasoning)');
        await pollThinkerResult(sessionId, taskId, seq);
    } catch (e) {
        if (seq !== thinkerTurnSeq) return;
        thinkerActiveTaskId = null;
        releaseTalker();                                // don't leave the model muted on error
        setThinkerStatus('Error', 'error');
        setThinkerLogTask('error');
        appendThinkerLog(`error: ${e.message}`, 'error');
    }
}

async function interruptThinker(reason, { announce = true } = {}) {
    if (!isThinkerModeEnabled() && !thinkerActiveTaskId) return;
    const sessionId = thinkerSessionId || session?.recordingSessionId || session?.sessionId;
    thinkerTurnSeq += 1;
    thinkerActiveTaskId = null;
    // End the Thinker hold: clear the flag so the kill-switch gate (set by cancelInjection on
    // the stop/barge-in) self-lifts on the next listen boundary, the same as a normal stop.
    coordThinkerHold = false;
    if (session) session.thinkerHoldActive = false;
    clearAnswerRevoice();
    if (!sessionId) {
        setThinkerStatus(isThinkerModeEnabled() ? 'On' : 'Off', isThinkerModeEnabled() ? 'on' : 'off');
        appendThinkerLog(`interrupt requested by ${reason}; no active thinker session id`, 'muted');
        return;
    }

    try {
        appendThinkerLog(`interrupt requested by ${reason}`, 'warn');
        await thinkerRequest(sessionId, '/barge-in', { method: 'POST' });
        setThinkerStatus(isThinkerModeEnabled() ? 'Interrupted' : 'Off', isThinkerModeEnabled() ? 'on' : 'off');
        setThinkerLogTask(isThinkerModeEnabled() ? 'interrupted' : 'idle');
        if (announce) appendThinkerLog('barge-in acknowledged; stale thinker output invalidated', 'warn');
    } catch (e) {
        setThinkerStatus('Error', 'error');
        setThinkerLogTask('error');
        if (announce) appendThinkerLog(`interrupt failed: ${e.message}`, 'error');
    }
}

function resetThinkerSessionState() {
    thinkerTurnSeq += 1;
    thinkerActiveTaskId = null;
    thinkerSessionId = null;
    coordOutboundGreetingSent = false;
    coordThinkerHold = false;
    if (session) session.thinkerHoldActive = false;
    clearAnswerRevoice();
    setThinkerStatus(isThinkerModeEnabled() ? 'On' : 'Off', isThinkerModeEnabled() ? 'on' : 'off');
    setThinkerLogTask('idle');
}

// ============================================================================
// FileAudioProvider — Audio-only file mode
// ============================================================================

class FileAudioProvider {
    /**
     * @param {File} file
     * @param {{audioMode: 'file'|'mixed', padBeforeSec: number, padAfterSec: number}} opts
     */
    constructor(file, opts = {}) {
        this._file = file;
        this._audioMode = opts.audioMode || 'file';
        this._padBefore = Math.max(0, Math.floor(opts.padBeforeSec ?? 0));
        this._padAfter = Math.max(0, Math.floor(opts.padAfterSec ?? 2));

        this._fileAudioChunks = [];   // decoded + normalized file audio
        this._mainChunks = 0;

        // Padded array (file-only mode)
        this._allAudio = [];

        // Phase tracking
        this._chunkIdx = 0;
        this._mainStart = 0;
        this._mainEnd = 0;
        this._grandTotal = 0;
        this._timer = null;

        // Mic pipeline (mixed mode — AudioWorklet graph)
        this._micStream = null;
        this._micCtx = null;
        this._micSource = null;
        this._captureNode = null;
        this._micGainNode = null;
        this._fileGainNode = null;
        this._monitorGainNode = null;
        this._micAnalyserNode = null;
        this._mixAnalyserNode = null;
        this._fileSrcNode = null;
        this._graphConnected = false;

        // Playback element
        this._audioEl = document.getElementById('fileAudioEl');
        this._objectUrl = null;

        this.onChunk = null;
        this.onEnd = null;
        this.running = false;
        this.paused = false;
        this._padBeforeTimer = null;
    }

    async start() {
        addSystemLog('Processing audio file...');

        // 1. Get duration
        const rawDuration = await this._getAudioDuration();
        const cappedDuration = Math.min(rawDuration, FILE_MAX_DURATION);
        this._mainChunks = Math.floor(cappedDuration);
        if (this._mainChunks === 0) throw new Error('Audio too short');
        if (rawDuration > FILE_MAX_DURATION) {
            addSystemLog(`Audio truncated: ${rawDuration.toFixed(1)}s → ${cappedDuration}s`);
        }

        // 2. Decode file audio and normalize
        await this._decodeFileAudio(cappedDuration);

        // 3. Setup mic (mixed mode)
        if (this._audioMode === 'mixed') {
            await this._setupMic();
        }

        // 4. Phase boundaries
        this._mainStart = this._padBefore;
        this._mainEnd = this._padBefore + this._mainChunks;
        this._grandTotal = this._padBefore + this._mainChunks + this._padAfter;

        const parts = [];
        if (this._padBefore > 0) parts.push(`${this._padBefore}s pad`);
        parts.push(`${this._mainChunks}s audio`);
        if (this._padAfter > 0) parts.push(`${this._padAfter}s pad`);
        addSystemLog(`Ready: [${parts.join(' + ')}] = ${this._grandTotal} chunks, mode=${this._audioMode}`);

        // 5. Prepare playback element (user hears the original file audio)
        this._objectUrl = URL.createObjectURL(this._file);
        this._audioEl.src = this._objectUrl;

        // 6. Start feeding
        this._chunkIdx = 0;
        this.running = true;

        if (this._audioMode === 'file') {
            this._buildPaddedArrays();
            this._feedNext();
        } else {
            // mixed: mic runs for entire padded duration; file buffer includes silence padding
            // During padding, file audio = zero → mix = mic only; during main, mix = mic + file
            this._startMainMicPhase();
        }
    }

    // ==================== File-only mode: unified timer ====================

    _buildPaddedArrays() {
        const silence = () => new Float32Array(SAMPLE_RATE_IN);
        this._allAudio = [
            ...Array.from({ length: this._padBefore }, silence),
            ...this._fileAudioChunks,
            ...Array.from({ length: this._padAfter }, silence),
        ];
    }

    _feedNext() {
        if (!this.running || this.paused) return;
        if (this._chunkIdx >= this._grandTotal) {
            this.running = false;
            this._audioEl.pause();
            if (this.onEnd) this.onEnd();
            return;
        }
        // Start playback when entering main phase
        if (this._chunkIdx === this._mainStart) {
            this._audioEl.play();
        }
        const t0 = performance.now();
        const audio = this._allAudio[this._chunkIdx];
        this._chunkIdx++;
        if (this.onChunk) this.onChunk({ audio });
        const elapsed = performance.now() - t0;
        this._timer = setTimeout(() => this._feedNext(), Math.max(0, CHUNK_MS - elapsed));
    }

    // ==================== Mixed mode: phased feeding ====================

    async _setupMic() {
        const _micId = adxDeviceSelector.getSelectedMicId();
        this._micStream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, ...(_micId ? { deviceId: { exact: _micId } } : {}) },
            video: false,
        });

        this._micCtx = new AudioContext({ sampleRate: SAMPLE_RATE_IN });
        if (this._micCtx.state === 'suspended') await this._micCtx.resume();

        await this._micCtx.audioWorklet.addModule('/static/duplex/lib/capture-processor.js');

        this._micSource = this._micCtx.createMediaStreamSource(this._micStream);

        // LUFS-based gain model: effective = auto + trim
        const micTarget = parseFloat(document.getElementById('mxMicTarget')?.value) || -23;
        const micAutoGainDb = micTarget - (mixerCtrl?.micMeasuredLUFS ?? -23);
        const micTrimDb = parseInt(document.getElementById('mxMicTrim')?.value) || 0;
        const fileTrimDb = parseInt(document.getElementById('mxFileTrim')?.value) || 0;
        const monPct = parseInt(document.getElementById('mxMonitor')?.value) || 50;

        this._micGainNode = this._micCtx.createGain();
        this._micGainNode.gain.value = Math.pow(10, (micAutoGainDb + micTrimDb) / 20);

        this._fileGainNode = this._micCtx.createGain();
        this._fileGainNode.gain.value = Math.pow(10, fileTrimDb / 20); // LUFS norm already applied to PCM

        this._captureNode = new AudioWorkletNode(this._micCtx, 'capture-processor', {
            processorOptions: { chunkSize: SAMPLE_RATE_IN },
        });

        this._micAnalyserNode = this._micCtx.createAnalyser();
        this._micAnalyserNode.fftSize = 2048;

        this._mixAnalyserNode = this._micCtx.createAnalyser();
        this._mixAnalyserNode.fftSize = 2048;

        this._fileAnalyserNode = this._micCtx.createAnalyser();
        this._fileAnalyserNode.fftSize = 2048;

        this._monitorGainNode = this._micCtx.createGain();
        this._monitorGainNode.gain.value = monPct / 100;

        this._micChunkCount = 0;
        addSystemLog(`Mic ready: AudioWorklet @${SAMPLE_RATE_IN}Hz, micAutoGain=${micAutoGainDb.toFixed(1)}dB, micTrim=${micTrimDb}dB, fileTrim=${fileTrimDb}dB, monitor=${monPct}%`);
    }

    _connectMic() {
        // Build padded file AudioBuffer: [silence × padBefore] + file + [silence × padAfter]
        // Padding is transparent to the graph — mic captures the entire duration
        const silence = () => new Float32Array(SAMPLE_RATE_IN);
        const paddedChunks = [
            ...Array.from({ length: this._padBefore }, silence),
            ...this._fileAudioChunks,
            ...Array.from({ length: this._padAfter }, silence),
        ];
        const totalSamples = paddedChunks.reduce((a, c) => a + c.length, 0);
        const audioBuf = this._micCtx.createBuffer(1, totalSamples, SAMPLE_RATE_IN);
        const ch = audioBuf.getChannelData(0);
        let pos = 0;
        for (const chunk of paddedChunks) { ch.set(chunk, pos); pos += chunk.length; }
        this._fileSrcNode = this._micCtx.createBufferSource();
        this._fileSrcNode.buffer = audioBuf;

        // Connect graph:
        //   mic → micGain ──→ captureNode → mixAnalyser (waveform)
        //   file → fileGain ─┘
        //   fileGain → monitorGain → speaker (file only, no echo)
        //   micGain → micAnalyser (mic-only meter)
        this._micSource.connect(this._micGainNode);
        this._micGainNode.connect(this._captureNode);
        this._micGainNode.connect(this._micAnalyserNode);

        this._fileSrcNode.connect(this._fileGainNode);
        this._fileGainNode.connect(this._captureNode);
        this._fileGainNode.connect(this._fileAnalyserNode);
        // Speaker output uses HTML audio element (original quality); Web Audio monitor disconnected

        this._captureNode.connect(this._mixAnalyserNode);

        // Wire chunk handler
        this._captureNode.port.onmessage = (e) => {
            if (e.data.type === 'chunk') {
                this._handleMicChunk(e.data.audio);
            }
        };

        this._captureNode.port.postMessage({ command: 'start' });
        this._fileSrcNode.start();
        this._fileSrcNode.onended = () => addSystemLog('File audio in graph completed');

        this._graphConnected = true;

        // Waveform visualization uses the mix analyser
        analyserNode = this._mixAnalyserNode;
        startWaveformDrawing();
        addSystemLog('Mic connected — AudioWorklet graph mixing');
    }

    _disconnectMic() {
        stopWaveformDrawing();
        analyserNode = null;
        this._graphConnected = false;
        if (this._captureNode) {
            this._captureNode.port.postMessage({ command: 'stop' });
            try { this._captureNode.disconnect(); } catch (_) {}
        }
        if (this._fileSrcNode) {
            try { this._fileSrcNode.stop(); } catch (_) {}
            try { this._fileSrcNode.disconnect(); } catch (_) {}
            this._fileSrcNode = null;
        }
        if (this._fileGainNode) try { this._fileGainNode.disconnect(); } catch (_) {}
        if (this._micGainNode) try { this._micGainNode.disconnect(); } catch (_) {}
        if (this._micSource) try { this._micSource.disconnect(); } catch (_) {}
        if (this._monitorGainNode) try { this._monitorGainNode.disconnect(); } catch (_) {}
        if (this._micAnalyserNode) try { this._micAnalyserNode.disconnect(); } catch (_) {}
        if (this._mixAnalyserNode) try { this._mixAnalyserNode.disconnect(); } catch (_) {}
    }

    _startMainMicPhase() {
        // Use native HTML element for speaker output (original quality, not LUFS-normalized 16kHz)
        const monPct = parseInt(document.getElementById('mxMonitor')?.value) || 50;
        this._audioEl.muted = false;
        this._audioEl.volume = monPct / 100;
        // Delay file playback by padBefore — graph plays silence during leading padding
        if (this._padBefore > 0) {
            this._padBeforeTimer = setTimeout(() => {
                if (this.running) this._audioEl.play();
            }, this._padBefore * CHUNK_MS);
        } else {
            this._audioEl.play();
        }
        this._connectMic();
    }

    _handleMicChunk(mixedAudio) {
        // During padding: file audio = zero → mix = mic only
        // During main:    file audio = real → mix = mic + file
        if (this._chunkIdx >= this._grandTotal) {
            this._disconnectMic();
            this._audioEl.pause();
            addSystemLog(`Mixed mode done: ${this._micChunkCount} chunks via AudioWorklet`);
            this.running = false;
            if (this.onEnd) this.onEnd();
            return;
        }
        this._micChunkCount++;
        this._chunkIdx++;
        if (this.onChunk) this.onChunk({ audio: mixedAudio });
    }

    // ==================== Audio graph accessors ====================

    /** Expose AudioWorklet graph nodes for mixer panel and recording */
    get mixerNodes() {
        return {
            micGain: this._micGainNode,
            fileGain: this._fileGainNode,
            monitorGain: this._monitorGainNode,
            monitorEl: this._audioEl,
            micAnalyser: this._micAnalyserNode,
            fileAnalyser: this._fileAnalyserNode,
            mixAnalyser: this._mixAnalyserNode,
            captureNode: this._captureNode,
            connected: this._graphConnected,
        };
    }

    // ==================== Audio processing helpers ====================

    async _decodeFileAudio(cappedDuration) {
        try {
            const arrayBuffer = await this._file.arrayBuffer();
            const audioCtx = new AudioContext();
            const audioBuf = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
            const targetSamples = Math.ceil(cappedDuration * SAMPLE_RATE_IN);
            const offCtx = new OfflineAudioContext(1, targetSamples, SAMPLE_RATE_IN);
            const src = offCtx.createBufferSource();
            src.buffer = audioBuf;
            src.connect(offCtx.destination);
            src.start();
            const resampled = await offCtx.startRendering();
            const pcm = resampled.getChannelData(0);
            await audioCtx.close();

            // LUFS normalization: measure → compute gain → apply
            const fileTargetEl = document.getElementById('mxFileTarget');
            const targetLUFS = fileTargetEl ? parseFloat(fileTargetEl.value) || -33 : -33;
            const srcLUFS = measureLUFS(pcm, SAMPLE_RATE_IN);
            this.measuredLUFS = srcLUFS;

            let fileNormGain;
            if (this._audioMode === 'mixed' && isFinite(srcLUFS)) {
                const autoGainDb = targetLUFS - srcLUFS;
                fileNormGain = Math.pow(10, autoGainDb / 20);
                addSystemLog(`File audio: ${srcLUFS.toFixed(1)} LUFS → target ${targetLUFS} LUFS (auto ${autoGainDb.toFixed(1)} dB)`);
            } else {
                // file-only: normalize to -28 LUFS
                const foTarget = -28;
                const autoDb = isFinite(srcLUFS) ? foTarget - srcLUFS : 0;
                fileNormGain = Math.pow(10, autoDb / 20);
                addSystemLog(`File audio: ${isFinite(srcLUFS) ? srcLUFS.toFixed(1) : '—'} LUFS → ${foTarget} LUFS (gain ${autoDb.toFixed(1)} dB)`);
            }

            this._fileAudioChunks = [];
            for (let i = 0; i < pcm.length; i += SAMPLE_RATE_IN) {
                const chunk = pcm.slice(i, Math.min(i + SAMPLE_RATE_IN, pcm.length));
                for (let j = 0; j < chunk.length; j++) chunk[j] *= fileNormGain;
                this._fileAudioChunks.push(chunk);
            }

            // Update Mixer display
            const measEl = document.getElementById('mxFileMeasured');
            if (measEl) measEl.textContent = isFinite(srcLUFS) ? srcLUFS.toFixed(1) : '—';
            const autoEl = document.getElementById('mxFileAuto');
            if (autoEl && this._audioMode === 'mixed' && isFinite(srcLUFS)) {
                autoEl.textContent = (targetLUFS - srcLUFS).toFixed(1);
            }
        } catch (err) {
            addSystemLog(`Failed to decode audio: ${err.message}`);
            throw err;
        }
    }

    async _getAudioDuration() {
        return new Promise((resolve, reject) => {
            const audio = document.createElement('audio');
            audio.preload = 'metadata';
            const url = URL.createObjectURL(this._file);
            audio.src = url;
            audio.onloadedmetadata = () => {
                const d = audio.duration;
                URL.revokeObjectURL(url);
                resolve(d);
            };
            audio.onerror = () => {
                URL.revokeObjectURL(url);
                reject(new Error('Failed to load audio metadata'));
            };
        });
    }

    pause() {
        if (!this.running || this.paused) return;
        this.paused = true;
        // File-only: stop the timer
        if (this._timer) { clearTimeout(this._timer); this._timer = null; }
        // Cancel pending padBefore timer (delayed HTML play)
        if (this._padBeforeTimer) { clearTimeout(this._padBeforeTimer); this._padBeforeTimer = null; }
        // Mixed: suspend AudioContext (freezes entire graph — mic, file buffer, capture)
        if (this._micCtx && this._micCtx.state === 'running') {
            this._micCtx.suspend();
        }
        // Pause speaker output
        if (!this._audioEl.paused) this._audioEl.pause();
        addSystemLog('File provider paused');
    }

    resume() {
        if (!this.running || !this.paused) return;
        this.paused = false;
        if (this._audioMode === 'file') {
            // File-only: restart the timer and HTML element
            if (this._chunkIdx >= this._mainStart && this._chunkIdx < this._grandTotal) {
                this._audioEl.play();
            }
            this._feedNext();
        } else {
            // Mixed: resume AudioContext (unfreezes entire graph)
            if (this._micCtx && this._micCtx.state === 'suspended') {
                this._micCtx.resume();
            }
            // Resume or schedule HTML audio playback
            if (this._chunkIdx < this._mainStart) {
                // Still in leading padding — schedule delayed play for remaining pad
                const remainingPad = this._mainStart - this._chunkIdx;
                this._padBeforeTimer = setTimeout(() => {
                    if (this.running && !this.paused) this._audioEl.play();
                }, remainingPad * CHUNK_MS);
            } else if (this._chunkIdx < this._mainEnd) {
                // In main phase — resume playback immediately
                this._audioEl.play();
            }
        }
        addSystemLog('File provider resumed');
    }

    stop() {
        this.running = false;
        this.paused = false;
        if (this._timer) { clearTimeout(this._timer); this._timer = null; }
        if (this._padBeforeTimer) { clearTimeout(this._padBeforeTimer); this._padBeforeTimer = null; }
        this._disconnectMic();
        if (this._micCtx) { this._micCtx.close().catch(() => {}); this._micCtx = null; }
        if (this._micStream) { this._micStream.getTracks().forEach(t => t.stop()); this._micStream = null; }
        this._captureNode = null;
        this._micGainNode = null;
        this._fileGainNode = null;
        this._monitorGainNode = null;
        this._micAnalyserNode = null;
        this._mixAnalyserNode = null;
        this._fileSrcNode = null;
        this._micSource = null;
        this._graphConnected = false;
        this._audioEl.muted = false;
        this._audioEl.volume = 1.0;
        this._audioEl.pause();
        this._audioEl.src = '';
        if (this._objectUrl) { URL.revokeObjectURL(this._objectUrl); this._objectUrl = null; }
    }
}

// ============================================================================
// Mode Switching
// ============================================================================
function setMode(mode) {
    if (session) return;
    currentMode = mode;
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.toggle('active', btn.dataset.mode === mode));
    const isFile = mode === 'file';
    document.getElementById('fileChooser').classList.toggle('visible', isFile);
    document.getElementById('fileOptions').classList.toggle('visible', isFile);
}

function onFileAudioModeChange() {
    const audioMode = document.querySelector('input[name="fileAudioMode"]:checked')?.value;
    const showMix = audioMode === 'mixed';
    const btn = document.getElementById('btnMixerToggle');
    if (btn) btn.style.display = showMix ? '' : 'none';
    if (!showMix) mixerCtrl?.closeMixer();
}

function onFileSelected(input) {
    if (input.files.length > 0) {
        selectedFile = input.files[0];
        document.getElementById('fileName').textContent = selectedFile.name;
        const audio = document.createElement('audio');
        audio.preload = 'metadata';
        const url = URL.createObjectURL(selectedFile);
        audio.src = url;
        audio.onloadedmetadata = () => {
            const dur = audio.duration;
            const label = dur > FILE_MAX_DURATION
                ? `${dur.toFixed(1)}s (will truncate to ${FILE_MAX_DURATION}s)`
                : `${dur.toFixed(1)}s`;
            document.getElementById('fileDuration').textContent = label;
            URL.revokeObjectURL(url);
        };
        audio.onerror = () => { URL.revokeObjectURL(url); };
        // Measure file LUFS in background
        measureFileLUFS(selectedFile);
    }
}

async function measureFileLUFS(file) {
    try {
        const measEl = document.getElementById('mxFileMeasured');
        const autoEl = document.getElementById('mxFileAuto');
        if (measEl) measEl.textContent = '...';
        if (autoEl) autoEl.textContent = '...';

        const arrayBuffer = await file.arrayBuffer();
        const tmpCtx = new AudioContext();
        const decoded = await tmpCtx.decodeAudioData(arrayBuffer.slice(0));
        const cappedDuration = Math.min(decoded.duration, FILE_MAX_DURATION);
        const targetFrames = Math.ceil(cappedDuration * SAMPLE_RATE_IN);
        const offCtx = new OfflineAudioContext(1, targetFrames, SAMPLE_RATE_IN);
        const src = offCtx.createBufferSource();
        src.buffer = decoded;
        src.connect(offCtx.destination);
        src.start();
        const resampled = await offCtx.startRendering();
        const pcm = resampled.getChannelData(0);
        await tmpCtx.close();

        const srcLUFS = measureLUFS(pcm, SAMPLE_RATE_IN);
        const targetLUFS = parseFloat(document.getElementById('mxFileTarget')?.value) || -33;
        const autoGainDb = isFinite(srcLUFS) ? targetLUFS - srcLUFS : 0;

        if (measEl) measEl.textContent = isFinite(srcLUFS) ? srcLUFS.toFixed(1) : '—';
        if (autoEl) autoEl.textContent = isFinite(srcLUFS) ? autoGainDb.toFixed(1) : '—';

        addSystemLog(`File LUFS: ${isFinite(srcLUFS) ? srcLUFS.toFixed(1) : '—'} → auto ${autoGainDb.toFixed(1)} dB`);
    } catch (err) {
        console.warn('measureFileLUFS failed:', err);
    }
}

// ============================================================================
// Session Control
// ============================================================================
async function startSession() {
    if (session) return;

    if (currentMode === 'file') {
        if (!selectedFile) { alert('Please select an audio file first.'); return; }
        const audioMode = document.querySelector('input[name="fileAudioMode"]:checked')?.value || 'file';
        const _i = (id, def) => { const v = parseInt(document.getElementById(id).value, 10); return Number.isFinite(v) ? v : def; };
        media = new FileAudioProvider(selectedFile, {
            audioMode,
            padBeforeSec: _i('padBeforeSec', 0),
            padAfterSec: _i('padAfterSec', 2),
        });
    }

    session = new RealtimeSession('adx', {
        getMaxKvTokens: () => parseInt(document.getElementById('maxKvTokens').value, 10) || 8192,
        getPlaybackDelayMs: () => parseInt(document.getElementById('playbackDelay').value, 10) || 200,
        outputSampleRate: SAMPLE_RATE_OUT,
        getWsUrl: () => {
            const proto = location.protocol === 'https:' ? 'wss' : 'ws';
            const url = `${proto}://${location.host}/v1/realtime?mode=audio`;
            return window.ClientIdentity ? window.ClientIdentity.appendToUrl(url) : url;
        },
    });

    setStatusLamp('preparing');
    document.getElementById('lampTimer').textContent = '';

    // Wire hooks
    session.onMetrics = (data) => metricsPanel.update(data);
    session.onSystemLog = addSystemLog;
    session.onSpeakStart = (text) => {
        coordTalkerSpeaking = true;
        coordLastAiMs = performance.now();
        coordLastAssistantText = text || '';
        dbg('TALKER', 'MiniCPM-o speaking');
        addAiLog(text);
    };
    session.onSpeakUpdate = (el, text) => {
        coordLastAiMs = performance.now();
        coordLastAssistantText = text || coordLastAssistantText;
        const span = el.querySelector('.ai-text');
        if (span) span.textContent = text;
        scrollChatLog();
    };
    session.onSpeakEnd = () => { coordTalkerSpeaking = false; coordLastAiMs = performance.now(); dbg('TALKER', 'MiniCPM-o finished'); scrollChatLog(); tryDeliver(); };
    session.onListenResult = () => {
        // Talker went to LISTEN -> a natural gap; try to deliver a held Thinker answer.
        // (The duplex stream carries no user ASR; that comes from browser SpeechRecognition.)
        coordTalkerSpeaking = false;
        tryDeliver();
    };
    session.onRunningChange = (running) => {
        setDuplexButtonStates(running);
        // Thinker mode needs the user's question as text; the duplex stream has none,
        // so run browser ASR while the session is live. A low-freq tick makes delivery robust.
        if (running && isThinkerModeEnabled()) {
            ensureDebugPanel();
            dbg('NOTE', 'session started — thinker mode on');
            holdTalker();                                  // come up muted; the coordinator grants the floor, the Talker never free-runs
            dbg('COORD', 'hold-by-default: Talker muted until a turn is classified');
            startThinkerAsr();
            if (!coordTick) coordTick = setInterval(tryDeliver, 1200);
            setTimeout(ensureBargeVad, 500);   // audioStream is set as the session starts
        } else {
            stopThinkerAsr();
            stopBargeVad();
            if (coordTick) { clearInterval(coordTick); coordTick = null; }
            cancelInjection('session end');
            coordQueue = []; coordTtsBusy = false; coordTtsInjecting = false;
            coordTalkerSpeaking = false; coordUserSpeaking = false;
            coordPendingClassify = false; coordOverlapAtSpeechStart = false;
            coordOutboundGreetingSent = false;
            clearSharedSilenceTimer();
            clearAnswerRevoice();
            if (coordSafetyTimer) { clearTimeout(coordSafetyTimer); coordSafetyTimer = null; }
            releaseTalker();
            try { window.speechSynthesis.cancel(); } catch (_) {}
        }
    };
    session.onPauseStateChange = (state) => {
        setDefaultPauseBtnState(state);
        if (session && session.running) {
            if (state === 'active') setStatusLamp('live');
            else if (state === 'paused') setStatusLamp('preparing');
        }
        // Pause/resume media provider and recording to keep timeline aligned
        if (state === 'paused' || state === 'pausing') {
            if (media && media.pause) media.pause();
            if (sessionRecorder && sessionRecorder.pause) sessionRecorder.pause();
        } else if (state === 'active') {
            if (media && media.resume) media.resume();
            if (sessionRecorder && sessionRecorder.resume) sessionRecorder.resume();
        }
    };
    session.onQueueUpdate = (data) => {
        const lamp = document.getElementById('statusLamp');
        if (data) {
            if (_stopHealthCheck) { _stopHealthCheck(); _stopHealthCheck = null; }
            setStatusLamp('preparing');
            _queueCountdownLabel = lamp?.querySelector('.label');
            _duplexCountdown.update(data.estimated_wait_s, data.position, data.queue_length || '?');
            if (data.position === 1 && _queuePhase !== 'almost') {
                _queuePhase = 'almost';
                setQueueButtonStates('almost');
                if (!_stopDingDong) _stopDingDong = startDingDongLoop();
            } else if (data.position !== 1 && _queuePhase !== 'almost') {
                _queuePhase = 'queuing';
                setQueueButtonStates('queuing');
            }
        } else {
            _duplexCountdown.stop();
            if (_stopDingDong) { _stopDingDong(); _stopDingDong = null; }
            if (!_stopHealthCheck) { _stopHealthCheck = initHealthCheck('serviceStatus'); }
        }
    };
    session.onQueueDone = () => {
        _queuePhase = 'assigned';
        if (_stopDingDong) { _stopDingDong(); _stopDingDong = null; }
        setQueueButtonStates('assigned');
        playAlarmBell();
    };
    session.onPrepared = async () => {
        if (session.audioPlayer && session.audioPlayer._ctx) {
            adxDeviceSelector.applySinkId(session.audioPlayer._ctx);
        }
        await playSessionChime();
    };
    session.onForceListenChange = (active) => setDefaultForceListenBtnState(active);
    session.onCleanup = () => {
        _duplexCountdown.stop();
        if (_stopDingDong) { _stopDingDong(); _stopDingDong = null; }
        _queuePhase = null;
        setQueueButtonStates(null);
        setStatusLamp('stopped');
        // Finalize recording
        if (sessionRecorder && sessionRecorder.recording) {
            const result = sessionRecorder.stop();
            if (result.blob.size > 0) {
                lastRecordingBlob = result.blob;
                addSystemLog(`Recording: ${result.durationSec.toFixed(1)}s stereo WAV (${(result.blob.size / 1024).toFixed(0)} KB)`);
                const btn = document.getElementById('btnDownloadRec');
                if (btn) { btn.style.display = ''; btn.disabled = false; }
                if (_saveShareUI) _saveShareUI.setRecordingBlob(result.blob, 'wav');
            }
            sessionRecorder = null;
        }

        if (media) { media.stop(); media = null; }
        stopWaveformDrawing();
        mixerCtrl?.stopMixerMeters();
        if (captureNodeLive) {
            captureNodeLive.port.postMessage({ command: 'stop' });
            try { captureNodeLive.disconnect(); } catch (_) {}
            captureNodeLive = null;
        }
        if (audioSource) { audioSource.disconnect(); audioSource = null; }
        if (analyserNode) { analyserNode.disconnect(); analyserNode = null; }
        if (audioStream) { audioStream.getTracks().forEach(t => t.stop()); audioStream = null; }
        if (audioCtxIn) { audioCtxIn.close().catch(() => {}); audioCtxIn = null; }
        session = null;
        resetThinkerSessionState();
        // Restart standalone mixer mic if mixer is still open
        const mixerPanel = document.getElementById('mixerPanel');
        if (mixerPanel && mixerPanel.style.display === 'block') {
            mixerCtrl?.startMixerMic();
            mixerCtrl?.startMixerMeters();
        }
    };

    // Recording setup
    const recEnabled = document.getElementById('recCheckbox')?.checked;
    if (recEnabled) {
        sessionRecorder = new SessionRecorder(SAMPLE_RATE_IN, SAMPLE_RATE_OUT);
        lastRecordingBlob = null;
        const btn = document.getElementById('btnDownloadRec');
        if (btn) { btn.style.display = 'none'; btn.disabled = true; }
    }

    // Pre-start UI
    metricsPanel.reset();
    document.getElementById('chatEmpty').style.display = 'none';
    addSystemLog('Connecting...' + (recEnabled ? ' (recording enabled)' : ''));

    // Build prepare payload
    const preparePayload = {
        config: { length_penalty: parseFloat(document.getElementById('duplexLengthPenalty').value) || 1.05 },
        use_tts: document.getElementById('ttsEnabled').checked,
    };
    const refBase64 = refAudio.getBase64();
    if (refBase64) preparePayload.ref_audio_base64 = refBase64;
    const ttsRef = duplexTtsRef.getBase64();
    if (ttsRef && ttsRef !== refBase64) preparePayload.tts_ref_audio_base64 = ttsRef;

    try {
        // Wire AI audio recording hook
        if (sessionRecorder) {
            session.audioPlayer.onRawAudio = (samples, sr, ts) => {
                if (sessionRecorder) sessionRecorder.pushRight(samples, sr, ts);
            };
        }

        await session.start(
            document.getElementById('systemPrompt').value,
            preparePayload,
            currentMode === 'live' ? startMicrophone : async () => {
                media.onChunk = (chunk) => {
                    session.sendChunk({
                        type: 'audio_chunk',
                        audio_base64: arrayBufferToBase64(chunk.audio.buffer),
                    });
                    if (sessionRecorder) sessionRecorder.pushLeft(chunk.audio);
                };
                media.onEnd = () => {
                    addSystemLog('File playback completed (including padding). Auto-stopping session.');
                    stopSession();
                };
                await media.start();
                mixerCtrl?.stopMixerMic();
                mixerCtrl?.startMixerMeters();
            }
        );

        // Start recording after media is ready
        if (sessionRecorder) sessionRecorder.start();

        thinkerSessionId = session.recordingSessionId || session.sessionId || thinkerSessionId;
        document.getElementById('chatSessionInfo').textContent = session.sessionId;
        metricsPanel.update({ type: 'state', sessionState: 'Active' });
        setStatusLamp('live');
        addSystemLog('Session active — speak now');
        maybeStartOutboundGreeting();

        if (_saveShareUI && session.recordingSessionId) _saveShareUI.setSessionId(session.recordingSessionId);
    } catch (e) {
        const isCancelled = e.message?.includes('cancelled');
        if (!isCancelled) addSystemLog(`Error: ${e.message}`);
        if (session) { try { session.cleanup(); } catch (_) {} }
        session = null;
        media = null;
        setStatusLamp(isCancelled ? 'stopped' : 'hidden');
    }
}

function pauseSession() { if (session) session.pauseToggle(); }
function stopSession() {
    if (!session) return;
    void interruptThinker('session stop', { announce: false });
    if (_queuePhase) { session.cancelQueue(); } else { session.stop(); }
    session = null;
    resetThinkerSessionState();
}
function toggleForceListen() {
    if (!session) return;
    void interruptThinker('Force Listen');
    session.toggleForceListen();
}

// ============================================================================
// Microphone (Live mode — with Waveform AnalyserNode)
// ============================================================================
async function startMicrophone() {
    audioCtxIn = new AudioContext({ sampleRate: SAMPLE_RATE_IN });
    if (audioCtxIn.state === 'suspended') await audioCtxIn.resume();

    await audioCtxIn.audioWorklet.addModule('/static/duplex/lib/capture-processor.js');

    const _micId = adxDeviceSelector.getSelectedMicId();
    audioStream = await navigator.mediaDevices.getUserMedia({
        audio: _micId ? { deviceId: { exact: _micId } } : true,
    });
    audioSource = audioCtxIn.createMediaStreamSource(audioStream);

    analyserNode = audioCtxIn.createAnalyser();
    analyserNode.fftSize = 2048;

    captureNodeLive = new AudioWorkletNode(audioCtxIn, 'capture-processor', {
        processorOptions: { chunkSize: SAMPLE_RATE_IN },
    });

    // Connect: mic → analyser (waveform) → captureNode (chunk accumulation)
    audioSource.connect(analyserNode);
    analyserNode.connect(captureNodeLive);

    captureNodeLive.port.postMessage({ command: 'start' });
    captureNodeLive.port.onmessage = (e) => {
        if (e.data.type === 'chunk') {
            if (!session || !session.running || session.paused) return;
            const chunk = e.data.audio;
            session.sendChunk({
                type: 'audio_chunk',
                audio_base64: arrayBufferToBase64(chunk.buffer),
            });
            if (sessionRecorder) sessionRecorder.pushLeft(chunk);
        }
    };

    startWaveformDrawing();
}

// ============================================================================
// Wire up event listeners
// ============================================================================
wireDuplexControls({
    onStart: startSession,
    onStop: stopSession,
    onPause: pauseSession,
    onForceListen: toggleForceListen,
});

// Mode buttons
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
});

// File input
document.getElementById('audioFileInput').addEventListener('change', function() {
    onFileSelected(this);
});

// File audio mode radios
document.querySelectorAll('input[name="fileAudioMode"]').forEach(radio => {
    radio.addEventListener('change', onFileAudioModeChange);
});

// Mic calibration is handled by MixerController

// TTS ref mode radios
document.querySelectorAll('input[name="duplexTtsRefMode"]').forEach(radio => {
    radio.addEventListener('change', () => duplexTtsRef.onModeChange());
});

document.getElementById('thinkerModeEnabled')?.addEventListener('change', () => {
    if (!isThinkerModeEnabled()) void interruptThinker('mode off', { announce: false });
    syncThinkerModeFromControl(true);
});

// ============================================================================
// Mixer Controller (shared module)
// ============================================================================
mixerCtrl = new MixerController({
    sampleRate: SAMPLE_RATE_IN,
    addLog: addSystemLog,
    getMedia: () => media,
    monitorElId: 'fileAudioEl',
    isFileMode: () => currentMode === 'file',
    getFileChunks: () => media?._fileAudioChunks,
    getSelectedFile: () => selectedFile,
    getLastRecordingBlob: () => lastRecordingBlob,
    getFallbackNodes: () => ({
        micAnalyser: analyserNode,
        connected: !!captureNodeLive,
    }),
    getDownloadExt: () => 'wav',
    isMicActive: () => !!audioStream,
});

if (document.readyState !== 'loading') {
    mixerCtrl.init();
} else {
    document.addEventListener('DOMContentLoaded', () => mixerCtrl.init());
}

// Cleanup on page unload (release mic, WS, AudioContext)
window.addEventListener('beforeunload', () => {
    if (session?.running) session.stop();
});

/* ---------- end of file ---------- */
