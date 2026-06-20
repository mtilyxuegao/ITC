let pc = null;
let dc = null;
let micStream = null;
let audioCtx = null;
let analyser = null;
let rafId = 0;
let currentAiEl = null;
let muted = false;

const els = {
    statusBadge: document.getElementById('serviceStatus'),
    statusText: document.getElementById('statusText'),
    statusLamp: document.getElementById('statusLamp'),
    btnStart: document.getElementById('btnStart'),
    btnMute: document.getElementById('btnMute'),
    btnStop: document.getElementById('btnStop'),
    voiceSelect: document.getElementById('voiceSelect'),
    instructions: document.getElementById('instructions'),
    remoteAudio: document.getElementById('remoteAudio'),
    waveCanvas: document.getElementById('waveCanvas'),
    chatLog: document.getElementById('chatLog'),
    chatEmpty: document.getElementById('chatEmpty'),
    sessionInfo: document.getElementById('sessionInfo'),
};

function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (ch) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[ch]));
}

function setStatus(text, state = 'idle') {
    els.statusText.textContent = text;
    els.statusBadge.textContent = text;
    els.statusLamp.className = `status-lamp ${state}`;
}

function setRunning(running) {
    els.btnStart.disabled = running;
    els.btnMute.disabled = !running;
    els.btnStop.disabled = !running;
}

function clearEmpty() {
    els.chatEmpty.style.display = 'none';
}

function addSystem(text) {
    clearEmpty();
    const el = document.createElement('div');
    el.className = 'conv-entry system';
    el.innerHTML = `<div class="conv-icon">⚙</div><div class="conv-text">${escapeHtml(text)}</div>`;
    els.chatLog.appendChild(el);
    scrollChat();
}

function addUser(text) {
    clearEmpty();
    const el = document.createElement('div');
    el.className = 'conv-entry user';
    el.innerHTML = `<div class="conv-icon">你</div><div class="conv-text"><span class="speaker user-tag">You:</span> ${escapeHtml(text)}</div>`;
    els.chatLog.appendChild(el);
    scrollChat();
}

function startAi(text = '') {
    clearEmpty();
    const el = document.createElement('div');
    el.className = 'conv-entry ai';
    el.innerHTML = `<div class="conv-icon">AI</div><div class="conv-text"><span class="speaker ai">OpenAI:</span> <span class="ai-text">${escapeHtml(text)}</span></div>`;
    els.chatLog.appendChild(el);
    currentAiEl = el.querySelector('.ai-text');
    scrollChat();
}

function appendAi(delta) {
    if (!currentAiEl) startAi('');
    currentAiEl.textContent += delta;
    scrollChat();
}

function finishAi() {
    currentAiEl = null;
}

function scrollChat() {
    els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

async function refreshStatus() {
    try {
        const response = await fetch('/api/openai-realtime/status');
        const data = await response.json();
        const status = data.status || {};
        els.sessionInfo.textContent = `${status.model || 'OpenAI Realtime'} / ${status.voice || 'voice'}`;
        els.statusBadge.textContent = status.configured ? 'Ready' : 'Missing OPENAI_API_KEY';
    } catch (error) {
        els.statusBadge.textContent = 'Offline';
    }
}

async function startSession() {
    if (pc) return;
    setRunning(true);
    setStatus('Connecting', 'connecting');
    addSystem('Starting OpenAI Realtime session...');

    try {
        const statusResp = await fetch('/api/openai-realtime/status');
        const statusData = await statusResp.json();
        if (!statusData.status?.configured) {
            throw new Error('OPENAI_API_KEY or OPENAI_REALTIME_API_KEY is not set on the gateway.');
        }

        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        startWaveform(micStream);

        pc = new RTCPeerConnection();
        pc.ontrack = (event) => {
            els.remoteAudio.srcObject = event.streams[0];
        };
        pc.onconnectionstatechange = () => {
            if (!pc) return;
            setStatus(pc.connectionState, pc.connectionState === 'connected' ? 'live' : 'connecting');
        };

        for (const track of micStream.getAudioTracks()) {
            pc.addTrack(track, micStream);
        }

        dc = pc.createDataChannel('oai-events');
        dc.addEventListener('open', () => {
            setStatus('Live', 'live');
            addSystem('OpenAI Realtime connected.');
        });
        dc.addEventListener('message', (event) => handleRealtimeEvent(event.data));
        dc.addEventListener('close', () => addSystem('OpenAI Realtime data channel closed.'));

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        const sdpResp = await fetch('/api/openai-realtime/call', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sdp: offer.sdp,
                voice: els.voiceSelect.value,
                instructions: els.instructions.value,
            }),
        });
        if (!sdpResp.ok) {
            let detail = await sdpResp.text();
            try {
                const data = JSON.parse(detail);
                detail = data.detail || detail;
            } catch (_) {}
            throw new Error(detail);
        }

        await pc.setRemoteDescription({
            type: 'answer',
            sdp: await sdpResp.text(),
        });

        setRunning(true);
    } catch (error) {
        addSystem(`Error: ${error.message}`);
        await stopSession();
        setStatus('Error', 'error');
    }
}

async function stopSession() {
    if (dc) {
        try { dc.close(); } catch (_) {}
        dc = null;
    }
    if (pc) {
        try { pc.close(); } catch (_) {}
        pc = null;
    }
    if (micStream) {
        micStream.getTracks().forEach((track) => track.stop());
        micStream = null;
    }
    stopWaveform();
    muted = false;
    els.btnMute.textContent = 'Mute';
    setRunning(false);
    setStatus('Idle', 'idle');
}

function toggleMute() {
    if (!micStream) return;
    muted = !muted;
    for (const track of micStream.getAudioTracks()) {
        track.enabled = !muted;
    }
    els.btnMute.textContent = muted ? 'Unmute' : 'Mute';
    addSystem(muted ? 'Microphone muted.' : 'Microphone unmuted.');
}

function handleRealtimeEvent(raw) {
    let event;
    try {
        event = JSON.parse(raw);
    } catch {
        return;
    }

    switch (event.type) {
        case 'session.created':
        case 'session.updated':
            if (event.session?.id) els.sessionInfo.textContent = event.session.id;
            break;
        case 'input_audio_buffer.speech_started':
            setStatus('Listening', 'live');
            break;
        case 'input_audio_buffer.speech_stopped':
            setStatus('Thinking', 'connecting');
            break;
        case 'conversation.item.input_audio_transcription.completed':
            if (event.transcript) addUser(event.transcript);
            break;
        case 'response.audio_transcript.delta':
        case 'response.output_audio_transcript.delta':
            appendAi(event.delta || '');
            break;
        case 'response.audio_transcript.done':
        case 'response.output_audio_transcript.done':
            if (event.transcript && !currentAiEl) startAi(event.transcript);
            finishAi();
            break;
        case 'response.done':
            finishAi();
            setStatus('Live', 'live');
            break;
        case 'error':
            addSystem(`OpenAI error: ${event.error?.message || JSON.stringify(event.error || event)}`);
            setStatus('Error', 'error');
            break;
        default:
            break;
    }
}

function startWaveform(stream) {
    stopWaveform();
    audioCtx = new AudioContext();
    const source = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 2048;
    source.connect(analyser);
    drawWaveform();
}

function stopWaveform() {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
    if (audioCtx) {
        audioCtx.close().catch(() => {});
        audioCtx = null;
    }
    analyser = null;
    drawIdleWaveform();
}

function drawIdleWaveform() {
    const canvas = els.waveCanvas;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(canvas.clientWidth * dpr);
    canvas.height = Math.floor(canvas.clientHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    ctx.strokeStyle = '#d9d6cc';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const y = canvas.clientHeight / 2;
    ctx.moveTo(20, y);
    ctx.lineTo(canvas.clientWidth - 20, y);
    ctx.stroke();
}

function drawWaveform() {
    if (!analyser) return;
    const canvas = els.waveCanvas;
    const ctx = canvas.getContext('2d');
    const data = new Uint8Array(analyser.fftSize);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(canvas.clientWidth * dpr);
    canvas.height = Math.floor(canvas.clientHeight * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    analyser.getByteTimeDomainData(data);
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    ctx.strokeStyle = '#16a34a';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < data.length; i += 1) {
        const x = (i / (data.length - 1)) * canvas.clientWidth;
        const y = (data[i] / 255) * canvas.clientHeight;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();
    rafId = requestAnimationFrame(drawWaveform);
}

els.btnStart.addEventListener('click', () => { void startSession(); });
els.btnStop.addEventListener('click', () => { void stopSession(); });
els.btnMute.addEventListener('click', toggleMute);
window.addEventListener('resize', () => { if (!analyser) drawIdleWaveform(); });

drawIdleWaveform();
void refreshStatus();
