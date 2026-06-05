/* ═══════════════════════════════════════════════════════════════
   eyes.js — EYES voice assistant controller
   ═══════════════════════════════════════════════════════════════ */
'use strict';

// ── Config ────────────────────────────────────────────────────────────────────
const CFG = {
  wsUrl: `ws://${location.host}/ws`,
  blinkIdleMin: 3800,
  blinkIdleMax: 6200,
  blinkActiveMin: 2000,
  blinkActiveMax: 3600,
  blinkDuration: 130,
  sleepTimeout: 65_000,
  reconnectDelay: 2200,
  ttsBreak: /[。！？…\n.!?;；]/,
  defaultCategory: 'daily_conversation',
  defaultMaxTokens: 512,
  listenTimeoutMs: 15_000,   // hold-Space silent timeout
  winkProbability: 0.18,     // chance a blink becomes a wink
  mouseAimFadeMs: 4_000,    // idle → mouse aim window after move
};

// ── Persisted settings (localStorage) ─────────────────────────────────────────
const SETTINGS_KEY = 'mamba.eyes.settings.v1';
const DEFAULTS = {
  voiceURI: '',           // empty = auto pick best en-US local voice
  rate: 1.10,
  pitch: 1.20,
  maxTokens: 512,
  mouseTrack: true,
};
let SETTINGS = loadSettings();

function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    return raw ? { ...DEFAULTS, ...JSON.parse(raw) } : { ...DEFAULTS };
  } catch { return { ...DEFAULTS }; }
}
function saveSettings() {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(SETTINGS)); } catch { }
}

// ── State machine ─────────────────────────────────────────────────────────────
const STATES = ['idle', 'listening', 'processing', 'thinking', 'speaking', 'done', 'error', 'sleep',
  'greeting', 'startled', 'confused', 'celebrate'];
let _state = 'idle';

// ── Active character ──────────────────────────────────────────────────────────
let _char = 'eyes';   // 'eyes' | 'tars'

// ── TARS THINK / FINAL channel ────────────────────────────────────────────────
const TARS_MSGS = [
  { t: 'SYS', m: 'parsing input semantics...' },
  { t: 'MEM', m: 'loading episodic context...' },
  { t: 'PROC', m: 'tokenizing prompt...' },
  { t: 'NET', m: 'cross-referencing knowledge base...' },
  { t: 'SYS', m: 'running forward pass...' },
  { t: 'CALC', m: 'sampling probability distributions...' },
  { t: 'MEM', m: 'accessing associative store...' },
  { t: 'SYS', m: 'humor.dll engaged ↑80%' },
  { t: 'PROC', m: 'generating reasoning chain...' },
  { t: 'SYS', m: 'coherence check: passing...' },
  { t: 'NET', m: 'aligning knowledge vectors...' },
  { t: 'CALC', m: 'estimating output confidence...' },
];
let _tarsThinkT = null;
let _tarsFinalTypT = null;
let _wfAnimT = null;
let _tarsCotBuf = '';

function startTarsThink() {
  const $lines = document.getElementById('tars-term-lines');
  if (!$lines) return;
  $lines.innerHTML = '';
  _addTermLine('INIT', 'TARS cognitive engine v2.4');
  _addTermLine('SYS', 'ready for inference...');
  let msgIdx = 0;
  clearInterval(_tarsThinkT);
  _tarsThinkT = setInterval(() => {
    if (_state !== 'thinking' || _char !== 'tars') { clearInterval(_tarsThinkT); return; }
    const m = TARS_MSGS[msgIdx % TARS_MSGS.length];
    _addTermLine(m.t, m.m);
    msgIdx++;
  }, 1100);
  _startWaveform();
}

function stopTarsThink() {
  clearInterval(_tarsThinkT);
  tarsFlushCot();
  _stopWaveform();
}

// Route live CoT tokens into the TARS terminal, flushing at sentence/length boundaries
function tarsAppendCot(text) {
  if (_char !== 'tars') return;
  _tarsCotBuf += text;
  let flushed = true;
  while (flushed) {
    flushed = false;
    // flush at sentence boundary (. ! ? newline) up to 72 chars
    const m = _tarsCotBuf.match(/^([\s\S]{1,72}[.!?\n])\s*([\s\S]*)$/);
    if (m) {
      _addTermLine('COT', m[1].trim());
      _tarsCotBuf = m[2];
      flushed = true;
    } else if (_tarsCotBuf.length > 72) {
      // hard-wrap at last space
      const cut = _tarsCotBuf.lastIndexOf(' ', 72);
      const pos = cut > 12 ? cut : 72;
      _addTermLine('COT', _tarsCotBuf.slice(0, pos).trim());
      _tarsCotBuf = _tarsCotBuf.slice(pos).trimStart();
      flushed = true;
    }
  }
}

function tarsFlushCot() {
  if (_tarsCotBuf.trim()) _addTermLine('COT', _tarsCotBuf.trim());
  _tarsCotBuf = '';
}

function _addTermLine(type, text) {
  const $lines = document.getElementById('tars-term-lines');
  if (!$lines) return;
  const el = document.createElement('div');
  el.className = 'tars-term-line';
  el.dataset.t = type;
  el.textContent = `[${type}] ${text}`;
  $lines.appendChild(el);
  // Fade out older lines, keep last 7
  const kids = Array.from($lines.children);
  if (kids.length > 7) {
    kids.slice(0, kids.length - 7).forEach(c => c.remove());
    Array.from($lines.children).forEach((c, i, a) => {
      c.style.opacity = String(0.30 + (i / a.length) * 0.70);
    });
    const last = $lines.lastChild;
    if (last) last.style.opacity = '';
  }
}

function _startWaveform() {
  const $wf = document.getElementById('tars-waveform');
  if (!$wf) return;
  $wf.innerHTML = '';
  for (let i = 0; i < 18; i++) {
    const b = document.createElement('div');
    b.className = 'tars-wf-bar';
    b.style.height = '2px';
    $wf.appendChild(b);
  }
  clearTimeout(_wfAnimT);
  _tickWaveform();
}

function _tickWaveform() {
  const $wf = document.getElementById('tars-waveform');
  if (!$wf || _state !== 'thinking' || _char !== 'tars') { _wfAnimT = null; return; }
  $wf.querySelectorAll('.tars-wf-bar').forEach(b => {
    b.style.height = (2 + Math.random() * 14).toFixed(0) + 'px';
  });
  _wfAnimT = setTimeout(_tickWaveform, 85);
}

function _stopWaveform() {
  clearTimeout(_wfAnimT); _wfAnimT = null;
  const $wf = document.getElementById('tars-waveform');
  if ($wf) $wf.querySelectorAll('.tars-wf-bar').forEach(b => { b.style.height = '2px'; });
}

function setTarsFinalSentence(text) {
  const $ft = document.getElementById('tars-final-text');
  if (!$ft) return;
  $ft.textContent = '';
  let i = 0;
  clearInterval(_tarsFinalTypT);
  _tarsFinalTypT = setInterval(() => {
    if (i >= text.length) { clearInterval(_tarsFinalTypT); return; }
    i = Math.min(i + Math.ceil(1 + Math.random() * 2), text.length);
    $ft.textContent = text.slice(0, i);
  }, 28);
}

function stopTarsFinal() {
  clearInterval(_tarsFinalTypT);
  const $ft = document.getElementById('tars-final-text');
  if ($ft) $ft.textContent = '';
}

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $scene = document.getElementById('scene');
const $mascot = document.getElementById('mascot');
const $stText = document.getElementById('status-text');
const $stSub = document.getElementById('status-sub');
const $wsDot = document.getElementById('ws-dot');
const $hint = document.getElementById('hint-label');
const $transcript = document.getElementById('transcript');
const $trText = document.getElementById('transcript-text');
const $cotStream = document.getElementById('cot-stream');
const $cotContent = document.getElementById('cot-content');
const $subtitle = document.getElementById('subtitle');
const $sysBtn = document.getElementById('sys-btn');
const $sysBtnLbl = document.getElementById('sys-btn-label');
const $sysMenu = document.getElementById('sys-menu');
const $gearBtn = document.getElementById('gear-btn');
const $setPanel = document.getElementById('settings-panel');
const $setVoice = document.getElementById('set-voice');
const $setRate = document.getElementById('set-rate');
const $setRateVal = document.getElementById('set-rate-val');
const $setPitch = document.getElementById('set-pitch');
const $setPitchVal = document.getElementById('set-pitch-val');
const $setMaxtok = document.getElementById('set-maxtok');
const $setMaxVal = document.getElementById('set-maxtok-val');
const $setMouse = document.getElementById('set-mousetrack');
const $setTest = document.getElementById('set-test');
const $setReset = document.getElementById('set-reset');
const $shortcuts = document.getElementById('shortcuts-overlay');

// ── System prompt switcher ────────────────────────────────────────────────────
let _categories = [];   // [{key, title}]
let _activeCat = CFG.defaultCategory;
let _catSysPrompts = {};  // key → system prompt text

function _activeSysPrompt() {
  return (_activeCat && _catSysPrompts[_activeCat]) ? String(_catSysPrompts[_activeCat]) : '';
}

async function loadCategories() {
  try {
    const res = await fetch('/api/demo-config');
    const data = await res.json();
    const cats = (data.categories || []).map(c => ({ key: c.key, title: c.title || c.key }));
    if (cats.length === 0) return;
    _categories = cats;
    _catSysPrompts = (data.category_system_prompts && typeof data.category_system_prompts === 'object')
      ? data.category_system_prompts : {};
    // Ensure default exists
    if (!cats.find(c => c.key === _activeCat)) _activeCat = cats[0].key;
    renderSysMenu();
    updateSysLabel();
    // Proactively show KV cost of initial system prompt.
    window.PF?.reset(null, { sys: _activeSysPrompt(), user: '', history: [], reasoning: false });
  } catch { }
}

function renderSysMenu() {
  $sysMenu.innerHTML = '';
  _categories.forEach(cat => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'sys-menu-item' + (cat.key === _activeCat ? ' active' : '');
    btn.role = 'option';
    btn.textContent = cat.title;
    btn.dataset.key = cat.key;
    btn.addEventListener('click', () => {
      _activeCat = cat.key;
      CFG.defaultCategory = cat.key;
      updateSysLabel();
      closeSysMenu();
      renderSysMenu();
      window.PF?.reset(null, { sys: _activeSysPrompt(), user: '', history: [], reasoning: false });
    });
    $sysMenu.appendChild(btn);
  });
}

function updateSysLabel() {
  const cat = _categories.find(c => c.key === _activeCat);
  $sysBtnLbl.textContent = cat ? cat.title.slice(0, 14) : _activeCat.slice(0, 14);
}

function openSysMenu() {
  $sysMenu.hidden = false;
  $sysBtn.setAttribute('aria-expanded', 'true');
}
function closeSysMenu() {
  $sysMenu.hidden = true;
  $sysBtn.setAttribute('aria-expanded', 'false');
}
function toggleSysMenu() {
  if ($sysMenu.hidden) openSysMenu(); else closeSysMenu();
}

$sysBtn.addEventListener('click', e => { e.stopPropagation(); toggleSysMenu(); });
document.addEventListener('click', () => closeSysMenu());
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeSysMenu(); });

// ── WebSocket ─────────────────────────────────────────────────────────────────
let ws = null, wsReady = false, _reconTimer = null;

function connectWS() {
  if (ws && ws.readyState <= 1) return;
  setWsDot('connecting');
  ws = new WebSocket(CFG.wsUrl);
  ws.onopen = () => { wsReady = true; setWsDot('connected'); };
  ws.onmessage = ev => { try { handleWS(JSON.parse(ev.data)); } catch { } };
  ws.onerror = () => { wsReady = false; setWsDot('disconnected'); };
  ws.onclose = () => {
    wsReady = false; setWsDot('disconnected');
    if (_state !== 'sleep') _reconTimer = setTimeout(connectWS, CFG.reconnectDelay);
  };
}
function setWsDot(s) { $wsDot.className = 'ws-dot ' + s; }
function wsSend(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) { ws.send(JSON.stringify(obj)); return true; }
  return false;
}

// ── WS event handler ──────────────────────────────────────────────────────────
function handleWS(msg) {
  switch (msg.type) {
    case 'connected':
      setState('greeting');
      setTimeout(() => { if (_state === 'greeting') setState('idle'); }, 820);
      break;

    case 'meta':
      cotClear();
      setState('thinking');
      setStatus('Thinking…');
      window.PF?.reset(msg.prompt_tokens);
      break;

    case 'reasoning_token':
      // Individual CoT token streamed live from backend
      cotAppend(msg.text || '');
      break;

    case 'reasoning':
      // Final accumulated CoT blob — used as mock fallback when no
      // reasoning_token events were received (e.g. --mock mode)
      if (!_cotStreaming) {
        setState('thinking');
        setStatus('Thinking…');
        const raw = (msg.markdown || '').trim();
        if (raw) cotTypewrite(raw);
      }
      // If we already got streaming tokens, the 'reasoning' blob is redundant
      break;

    case 'token':
      if (_state !== 'speaking') {
        cotFade();
        setState('speaking');
        setStatus('Speaking…');
        _tokenCount = 0;
      }
      _tokenCount += (msg.text || '').length;
      ttsAccum(msg.text || '');
      if (msg.tok_s != null) window.PF?.tick(msg.tok_s, msg.n);
      break;

    case 'mw_inject':
      flashInject();
      break;

    case 'done':
      ttsFlush();
      scheduleDone();
      window.PF?.done(msg.tok_s, msg.prefill_ms, msg.ttft_ms, msg.total_tokens ?? msg.n);
      break;

    case 'error':
      ttsCancel();
      cotClear();
      setState('confused');
      setStatus('Something went wrong');
      setTimeout(() => setState('idle'), 2200);
      break;
  }
}

// ── Eye state ─────────────────────────────────────────────────────────────────
let _tokenCount = 0;

function setState(s) {
  if (!STATES.includes(s)) return;
  _state = s;
  STATES.forEach(x => $scene.classList.remove('state-' + x));
  $scene.classList.add('state-' + s);
  $scene.classList.remove('blinking', 'wink-l', 'wink-r', 'gaze-left', 'gaze-right', 'mouth-open');
  stopRandomGaze(); stopThinkScan(); stopConfusedGaze();
  if (s !== 'thinking') stopTarsThink();
  if (s === 'idle') startRandomGaze();
  else if (s === 'thinking') { startThinkScan(); if (_char === 'tars') startTarsThink(); }
  else if (s === 'confused') startConfusedGaze();
  scheduleBlink();
  if (['listening', 'processing', 'thinking', 'speaking'].includes(s)) resetSleepTimer();
}

function setStatus(text, sub = '') {
  $stText.textContent = text;
  $stSub.textContent = sub;
  $stText.classList.toggle('visible', !!text);
  $stSub.classList.toggle('visible', !!sub);
}

function switchCharacter(name) {
  if (name === _char) return;
  _char = name;
  ttsCancel();
  $scene.classList.toggle('char-tars', name === 'tars');
  document.querySelectorAll('.char-btn-opt').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.char === name);
  });
  setState('greeting');
  setStatus('');
  setTimeout(() => setState('idle'), 850);
}

// ── Blink ─────────────────────────────────────────────────────────────────────
let _blinkT = null;
function scheduleBlink() {
  clearTimeout(_blinkT);
  if (['sleep', 'done', 'listening'].includes(_state)) return;
  const hi = ['processing', 'thinking', 'speaking', 'confused'].includes(_state);
  const delay = rand(hi ? CFG.blinkActiveMin : CFG.blinkIdleMin,
    hi ? CFG.blinkActiveMax : CFG.blinkIdleMax);
  _blinkT = setTimeout(() => {
    if (_state === 'idle' && Math.random() < CFG.winkProbability) doWink();
    else doBlink();
    scheduleBlink();
  }, delay);
}
function doBlink() {
  $scene.classList.add('blinking');
  setTimeout(() => $scene.classList.remove('blinking'), CFG.blinkDuration + 30);
}
function doWink() {
  const dir = Math.random() < 0.5 ? 'wink-l' : 'wink-r';
  $scene.classList.add(dir);
  setTimeout(() => $scene.classList.remove(dir), Math.round(CFG.blinkDuration * 1.4));
}

// ── Idle random gaze (replaces mouse-tracking iris drift) ─────────────────────
let _gazeT = null;
function startRandomGaze() {
  clearTimeout(_gazeT);
  scheduleNextGaze();
}
function stopRandomGaze() {
  clearTimeout(_gazeT);
  $scene.classList.remove('gaze-left', 'gaze-right');
}
function scheduleNextGaze() {
  _gazeT = setTimeout(() => {
    if (_state !== 'idle') return;
    if (!SETTINGS.mouseTrack) { scheduleNextGaze(); return; }
    const dir = Math.random() < 0.5 ? 'gaze-left' : 'gaze-right';
    $scene.classList.add(dir);
    setTimeout(() => {
      $scene.classList.remove('gaze-left', 'gaze-right');
      scheduleNextGaze();
    }, rand(1200, 2400));
  }, rand(8000, 15000));
}

// ── Thinking gaze scan ────────────────────────────────────────────────────────
let _thinkScanT = null, _thinkIdx = 0;
const _THINK_SEQ = ['gaze-left', null, 'gaze-right', null];
function startThinkScan() {
  clearInterval(_thinkScanT); _thinkIdx = 0;
  _thinkScanT = setInterval(() => {
    $scene.classList.remove('gaze-left', 'gaze-right');
    const dir = _THINK_SEQ[_thinkIdx % _THINK_SEQ.length];
    if (dir) $scene.classList.add(dir);
    _thinkIdx++;
  }, 700);
}
function stopThinkScan() {
  clearInterval(_thinkScanT); _thinkScanT = null;
  $scene.classList.remove('gaze-left', 'gaze-right');
}

// ── Confused alternating gaze ─────────────────────────────────────────────────
let _confusedGazeT = null, _confusedLeft = true;
function startConfusedGaze() {
  clearInterval(_confusedGazeT);
  _confusedLeft = true;
  $scene.classList.add('gaze-left');
  _confusedGazeT = setInterval(() => {
    _confusedLeft = !_confusedLeft;
    $scene.classList.remove('gaze-left', 'gaze-right');
    $scene.classList.add(_confusedLeft ? 'gaze-left' : 'gaze-right');
  }, 900);
}
function stopConfusedGaze() {
  clearInterval(_confusedGazeT); _confusedGazeT = null;
}

// ── Inject flash ──────────────────────────────────────────────────────────────
function flashInject() {
  $scene.classList.add('inject-flash');
  setTimeout(() => $scene.classList.remove('inject-flash'), 750);
}

// ── CoT display ───────────────────────────────────────────────────────────────
// Two modes:
//   streaming  — reasoning_token events append live (real inference)
//   typewriter — single reasoning blob animated char-by-char (mock fallback)
// Both render via marked.js for markdown support.

let _cotStreaming = false;
let _cotBuf = '';
let _cotRenderT = null;
let _typeT = null, _typeTarget = '', _typePos = 0;

function _md(text) {
  if (typeof marked !== 'undefined') {
    return marked.parse(text, { breaks: true, gfm: true });
  }
  // fallback: escape HTML and preserve newlines
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>');
}

function cotAppend(text) {
  if (_char === 'tars') { tarsAppendCot(text); return; }
  if (!_cotStreaming) {
    _cotStreaming = true;
    _cotBuf = '';
    $cotContent.innerHTML = '';
    $cotStream.classList.remove('fading');
    $cotStream.classList.add('visible');
  }
  _cotBuf += text;
  clearTimeout(_cotRenderT);
  _cotRenderT = setTimeout(() => {
    const atBottom = $cotStream.scrollTop + $cotStream.clientHeight >= $cotStream.scrollHeight - 6;
    $cotContent.innerHTML = _md(_cotBuf);
    if (atBottom) $cotStream.scrollTop = $cotStream.scrollHeight;
  }, 28);
}

function cotTypewrite(text) {
  clearInterval(_typeT);
  if (_char === 'tars') {
    // feed the blob gradually into TARS terminal
    _typeTarget = text; _typePos = 0;
    _typeT = setInterval(() => {
      if (_typePos >= _typeTarget.length) { clearInterval(_typeT); tarsFlushCot(); return; }
      const chunk = _typeTarget.slice(_typePos, _typePos + 8);
      _typePos += 8;
      tarsAppendCot(chunk);
    }, 25);
    return;
  }
  _typeTarget = text; _typePos = 0; _cotBuf = '';
  $cotContent.innerHTML = '';
  $cotStream.classList.remove('fading');
  $cotStream.classList.add('visible');
  _typeT = setInterval(() => {
    if (_typePos >= _typeTarget.length) { clearInterval(_typeT); return; }
    _typePos = Math.min(_typePos + 5, _typeTarget.length);
    const slice = _typeTarget.slice(0, _typePos);
    $cotContent.innerHTML = _md(slice);
    $cotStream.scrollTop = $cotStream.scrollHeight;
  }, 20);
}

function cotFade() {
  clearInterval(_typeT);
  if (_char === 'tars') { tarsFlushCot(); return; }
  $cotStream.classList.remove('visible');
  $cotStream.classList.add('fading');
  setTimeout(() => { $cotStream.classList.remove('fading'); $cotContent.innerHTML = ''; _cotBuf = ''; }, 520);
}

function cotClear() {
  clearInterval(_typeT); clearTimeout(_cotRenderT);
  _cotStreaming = false; _cotBuf = '';
  if (_char === 'tars') { _tarsCotBuf = ''; return; }
  $cotStream.classList.remove('visible', 'fading');
  $cotContent.innerHTML = '';
}

// ── Done ──────────────────────────────────────────────────────────────────────
function scheduleDone() {
  const check = () => {
    if (!_ttsSpeaking && _ttsQueue.length === 0) doneClose();
    else setTimeout(check, 300);
  };
  check();
}
function doneClose() {
  cotClear(); hideSubtitle();
  stopTarsFinal();
  setStatus('');
  if (_tokenCount > 200) {
    _tokenCount = 0;
    setState('celebrate');
    setTimeout(() => setState('idle'), 800);
  } else {
    _tokenCount = 0;
    setState('done');
    setTimeout(() => setState('idle'), 600);
  }
}

// ── Sleep ─────────────────────────────────────────────────────────────────────
let _sleepT = null;
function resetSleepTimer() {
  clearTimeout(_sleepT);
  _sleepT = setTimeout(() => {
    if (!pttActive) { ttsCancel(); setState('sleep'); setStatus(''); }
  }, CFG.sleepTimeout);
}
function wakeUp() {
  if (_state === 'sleep') { clearTimeout(_reconTimer); connectWS(); setState('idle'); setStatus(''); }
  resetSleepTimer();
}

// ── TTS (sentence-level queue) ────────────────────────────────────────────────
let _ttsBuf = '', _ttsQueue = [], _ttsSpeaking = false;
let _synthResumeT = null;
const synth = window.speechSynthesis;

function synthResumePump() {
  clearInterval(_synthResumeT);
  _synthResumeT = setInterval(() => { if (synth.speaking) synth.resume(); }, 12_000);
}

function ttsAccum(chunk) {
  _ttsBuf += chunk;
  let m;
  while ((m = CFG.ttsBreak.exec(_ttsBuf)) !== null) {
    const sentence = _ttsBuf.slice(0, m.index + 1).trim();
    _ttsBuf = _ttsBuf.slice(m.index + 1);
    if (sentence) { _ttsQueue.push(sentence); ttsNext(); }
  }
}
function ttsFlush() {
  const rem = _ttsBuf.trim(); _ttsBuf = '';
  if (rem) { _ttsQueue.push(rem); ttsNext(); }
}
function ttsNext() {
  if (_ttsSpeaking || _ttsQueue.length === 0) return;
  const text = _ttsQueue.shift();
  _ttsSpeaking = true;
  const utt = new SpeechSynthesisUtterance(text);
  utt.lang = 'en-US';

  if (_char === 'tars') {
    utt.rate = Math.max(0.70, SETTINGS.rate * 0.88);
    utt.pitch = Math.max(0.65, SETTINGS.pitch * 0.70);
  } else {
    utt.rate = SETTINGS.rate;
    utt.pitch = SETTINGS.pitch;
  }

  const voices = synth.getVoices();
  let chosen = null;
  if (SETTINGS.voiceURI) {
    chosen = voices.find(v => v.voiceURI === SETTINGS.voiceURI);
  }
  if (!chosen) {
    if (_char === 'tars') {
      const TARS_VOICES = ['Alex', 'Daniel', 'Fred', 'Google UK English Male', 'Google US English'];
      chosen = TARS_VOICES.map(n => voices.find(v => v.name.startsWith(n)))
        .find(Boolean)
        || voices.find(v => v.lang === 'en-US' && v.localService)
        || voices.find(v => v.lang === 'en-US')
        || voices.find(v => v.lang.startsWith('en'));
    } else {
      const CUTE_VOICES = ['Zoe', 'Samantha', 'Google US English', 'Karen', 'Moira', 'Tessa'];
      chosen = CUTE_VOICES.map(n => voices.find(v => v.name.startsWith(n) && v.lang.startsWith('en')))
        .find(Boolean)
        || voices.find(v => v.lang === 'en-US' && v.localService)
        || voices.find(v => v.lang === 'en-US')
        || voices.find(v => v.lang.startsWith('en'));
    }
  }
  if (chosen) utt.voice = chosen;

  // Build subtitle DOM before speak() — CSS hides it for TARS, FINAL channel takes over
  showSubtitle(text);
  if (_char === 'tars') setTarsFinalSentence(text);
  utt.onstart = () => _startSubtitleTimer();
  utt.onboundary = ev => {
    if (ev.name === 'sentence') return;
    pulseMouth();
    if (typeof ev.charIndex === 'number') highlightSubtitleWord(ev.charIndex);
  };
  utt.onend = () => { hideSubtitle(); _ttsSpeaking = false; ttsNext(); };
  utt.onerror = () => { hideSubtitle(); _ttsSpeaking = false; ttsNext(); };

  synth.speak(utt);
  synthResumePump();
}

// ── Mouth pulse (TTS boundary-driven) ────────────────────────────────────────
let _mouthT = null;
function pulseMouth() {
  $scene.classList.add('mouth-open');
  clearTimeout(_mouthT);
  _mouthT = setTimeout(() => $scene.classList.remove('mouth-open'), 85);
}
function ttsCancel() {
  clearInterval(_synthResumeT); synth.cancel();
  _ttsBuf = ''; _ttsQueue = []; _ttsSpeaking = false;
  hideSubtitle();
  stopTarsFinal();
}

// ── Subtitle (karaoke word highlight) ─────────────────────────────────────────
let _subWords = [];   // [{el, start, end}] for current utterance
let _subTimer = null;
let _subT0 = 0;
let _subCumMs = [];   // cumulative ms to end of each word (estimated)

function showSubtitle(text) {
  clearInterval(_subTimer);
  $subtitle.innerHTML = '';
  _subWords = [];
  _subCumMs = [];

  const tokens = text.split(/(\s+)/);
  let offset = 0;
  for (const tok of tokens) {
    if (!tok) continue;
    if (/^\s+$/.test(tok)) {
      $subtitle.appendChild(document.createTextNode(' '));
    } else {
      const span = document.createElement('span');
      span.className = 'sub-word';
      span.textContent = tok;
      $subtitle.appendChild(span);
      _subWords.push({ el: span, start: offset, end: offset + tok.length });
    }
    offset += tok.length;
  }

  // Pre-compute cumulative timing for time-based fallback
  // (~68ms per alphanumeric char, min 120ms per word, adjusted by speech rate)
  let cumMs = 0;
  for (const w of _subWords) {
    const alnLen = (w.el.textContent.match(/\w/g) || []).length;
    cumMs += Math.max(120, alnLen * 68) / SETTINGS.rate;
    _subCumMs.push(cumMs);
  }

  $subtitle.classList.add('visible');
}

function _startSubtitleTimer() {
  clearInterval(_subTimer);
  _subT0 = performance.now();
  let curIdx = -1;
  _subTimer = setInterval(() => {
    if (!_subWords.length) return;
    const elapsed = performance.now() - _subT0;
    let newIdx = _subCumMs.findIndex(t => elapsed < t);
    if (newIdx < 0) newIdx = _subWords.length;
    const highlight = Math.min(Math.max(newIdx, 0), _subWords.length - 1);
    if (highlight !== curIdx) {
      curIdx = highlight;
      _applySubHighlight(curIdx);
    }
  }, 40);
}

function _applySubHighlight(idx) {
  _subWords.forEach((w, i) => {
    w.el.classList.toggle('spoken', i < idx);
    w.el.classList.toggle('active', i === idx);
  });
}

function highlightSubtitleWord(charIndex) {
  if (!_subWords.length || typeof charIndex !== 'number') return;
  // Handle whitespace gaps: if charIndex falls between words, snap to next word
  let idx = _subWords.findIndex(w => w.start <= charIndex && charIndex < w.end);
  if (idx < 0) idx = _subWords.findIndex(w => w.start >= charIndex);
  if (idx < 0) return;
  _applySubHighlight(idx);
  // Recalibrate timer to match actual speech position reported by TTS engine
  const prevMs = idx > 0 ? _subCumMs[idx - 1] : 0;
  _subT0 = performance.now() - prevMs;
}

function hideSubtitle() {
  clearInterval(_subTimer);
  $subtitle.classList.remove('visible');
  setTimeout(() => { $subtitle.innerHTML = ''; _subWords = []; _subCumMs = []; }, 350);
}
if (typeof synth !== 'undefined') {
  synth.getVoices();
  if (synth.onvoiceschanged !== undefined) synth.onvoiceschanged = () => synth.getVoices();
}

// ── Speech Recognition ────────────────────────────────────────────────────────
// Press space  → start recognition, show streaming transcript top-left
// Release space → send accumulated text, hide transcript
//
// Uses continuous:true so recognition keeps going while key is held.
// interimResults:true for live character-by-character display.

const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null, pttActive = false;
let _finalSoFar = '';   // confirmed text (from isFinal results)
let _interimNow = '';   // current unconfirmed segment

function initRec() {
  if (!SR) {
    $hint.textContent = 'Speech input requires Chrome';
    return;
  }
  rec = new SR();
  rec.lang = 'en-US';
  rec.continuous = true;   // keep running while space held
  rec.interimResults = true;   // fire partial results for live display
  rec.maxAlternatives = 1;

  rec.onresult = ev => {
    let newFinal = '', interim = '';
    for (let i = ev.resultIndex; i < ev.results.length; i++) {
      const r = ev.results[i];
      if (r.isFinal) newFinal += r[0].transcript;
      else interim += r[0].transcript;
    }
    if (newFinal) _finalSoFar += newFinal;
    _interimNow = interim;
    renderTranscript(_finalSoFar, _interimNow);
    // user is actively speaking → extend silent timeout
    if (newFinal || interim) resetListenTimeout();
  };

  rec.onerror = ev => {
    if (ev.error === 'no-speech' || ev.error === 'aborted') return;
    setState('error'); setStatus('Speech recognition failed');
    setTimeout(() => { if (_state === 'error') setState('idle'); }, 1300);
  };

  // onend fires when stop() is called or recognition naturally ends
  rec.onend = () => {
    // If PTT still active (continuous mode may auto-stop) → restart
    if (pttActive) {
      try { rec.start(); } catch { }
    }
  };
}

// ── Transcript display ────────────────────────────────────────────────────────
function renderTranscript(final, interim) {
  // Final text: full opacity; interim: dimmer via <span class="interim">
  $trText.innerHTML =
    escHtml(final) +
    (interim ? `<span class="interim">${escHtml(interim)}</span>` : '');
  $transcript.classList.add('visible');
}

function hideTranscript(flash) {
  if (flash) {
    $transcript.classList.add('sending');
    setTimeout(() => {
      $transcript.classList.remove('visible', 'sending');
      $trText.innerHTML = '';
    }, 380);
  } else {
    $transcript.classList.remove('visible', 'sending');
    $trText.innerHTML = '';
  }
}

function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── PTT start / end ───────────────────────────────────────────────────────────
let _listenTimeoutT = null;

function pttStart() {
  if (pttActive) return;
  wakeUp();

  const wasActive = ['thinking', 'speaking', 'processing'].includes(_state);
  ttsCancel();
  cotClear();
  wsSend({ action: 'abort' });

  pttActive = true;
  _finalSoFar = ''; _interimNow = '';
  $scene.classList.add('ptt-active');

  function doListen() {
    setState('listening');
    setStatus('Listening…');
    if (rec) { try { rec.start(); } catch { } }
    startMicAnalyser();
    resetListenTimeout();
  }

  if (wasActive) {
    setState('startled');
    $scene.classList.add('interrupting');
    setTimeout(() => $scene.classList.remove('interrupting'), 320);
    setTimeout(doListen, 210);
  } else {
    doListen();
  }
}

function resetListenTimeout() {
  clearTimeout(_listenTimeoutT);
  _listenTimeoutT = setTimeout(() => {
    if (!pttActive) return;
    setStatus('No speech detected', '');
    pttEnd();
  }, CFG.listenTimeoutMs);
}

function pttEnd() {
  if (!pttActive) return;
  pttActive = false;
  clearTimeout(_listenTimeoutT);
  $scene.classList.remove('ptt-active');
  if (rec) { try { rec.stop(); } catch { } }
  stopMicAnalyser();

  const text = (_finalSoFar + _interimNow).trim();
  _finalSoFar = ''; _interimNow = '';

  if (text) {
    hideTranscript(true);
    sendPrompt(text);
  } else {
    hideTranscript(false);
    setState('idle'); setStatus('');
  }
}

function sendPrompt(text) {
  if (!text.trim()) return;
  if (!wsReady) {
    // Retry once: try reconnecting, then surface error
    connectWS();
    setTimeout(() => {
      if (!wsReady) {
        setState('error');
        setStatus('Not connected', 'tap again to retry');
        setTimeout(() => setState('idle'), 1800);
        return;
      }
      _doSend(text);
    }, 500);
    return;
  }
  _doSend(text);
}

function _doSend(text) {
  setState('processing');
  setStatus('Thinking…', text.length > 34 ? text.slice(0, 34) + '…' : text);
  wsSend({
    action: 'chat',
    prompt: text,
    max_tokens: SETTINGS.maxTokens,
    reasoning: true,
    category_key: CFG.defaultCategory,
  });
}

// ── Mic analyser (drives --mic-level during listening) ────────────────────────
let _audioCtx = null, _analyser = null, _micStream = null;
let _micRaf = null, _micData = null, _micStarting = false;

async function startMicAnalyser() {
  if (_audioCtx || _micStarting) return;
  _micStarting = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    // If user released the key while we were awaiting permission, abort.
    if (!pttActive) { stream.getTracks().forEach(t => t.stop()); _micStarting = false; return; }
    _micStream = stream;
    const AC = window.AudioContext || window.webkitAudioContext;
    _audioCtx = new AC();
    const src = _audioCtx.createMediaStreamSource(_micStream);
    _analyser = _audioCtx.createAnalyser();
    _analyser.fftSize = 256;
    _analyser.smoothingTimeConstant = 0.78;
    src.connect(_analyser);
    _micData = new Uint8Array(_analyser.frequencyBinCount);
    micTick();
  } catch (e) {
    // mic blocked or unavailable — silent, recognition still works
  } finally {
    _micStarting = false;
  }
}

function micTick() {
  if (!_analyser) return;
  _analyser.getByteFrequencyData(_micData);
  let sum = 0;
  for (let i = 0; i < _micData.length; i++) sum += _micData[i];
  const avg = sum / _micData.length / 255;       // 0..1
  const level = Math.min(1, Math.pow(avg * 2.2, 1.3));  // perceptual boost
  document.documentElement.style.setProperty('--mic-level', level.toFixed(3));
  _micRaf = requestAnimationFrame(micTick);
}

function stopMicAnalyser() {
  if (_micRaf) cancelAnimationFrame(_micRaf); _micRaf = null;
  if (_micStream) _micStream.getTracks().forEach(t => t.stop());
  _micStream = null;
  if (_audioCtx) { _audioCtx.close().catch(() => { }); }
  _audioCtx = null; _analyser = null; _micData = null;
  document.documentElement.style.setProperty('--mic-level', 0);
}

// ── Keyboard: Space ───────────────────────────────────────────────────────────
let _spaceDown = false;
document.addEventListener('keydown', ev => {
  if (ev.code !== 'Space' || ev.repeat || _spaceDown) return;
  if (['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
  ev.preventDefault(); _spaceDown = true; pttStart();
});
document.addEventListener('keyup', ev => {
  if (ev.code !== 'Space' || !_spaceDown) return;
  ev.preventDefault(); _spaceDown = false; pttEnd();
});

// ── Touch ─────────────────────────────────────────────────────────────────────
let _touchId = null;
$scene.addEventListener('touchstart', ev => {
  if (_touchId !== null) return;
  _touchId = ev.changedTouches[0].identifier;
  $scene.classList.add('touch-active');
  pttStart();
}, { passive: true });
const onTouchEnd = ev => {
  for (const t of ev.changedTouches) {
    if (t.identifier === _touchId) {
      _touchId = null; $scene.classList.remove('touch-active'); pttEnd(); break;
    }
  }
};
$scene.addEventListener('touchend', onTouchEnd, { passive: true });
$scene.addEventListener('touchcancel', () => {
  _touchId = null; $scene.classList.remove('touch-active'); pttEnd();
}, { passive: true });

// ── Visibility ────────────────────────────────────────────────────────────────
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    wakeUp();
    if (!wsReady) connectWS();
    if (synth.speaking) synth.resume();
  }
});

// ── Utility ───────────────────────────────────────────────────────────────────
const rand = (a, b) => a + Math.random() * (b - a);

// ── Settings panel ────────────────────────────────────────────────────────────
function populateVoiceDropdown() {
  const voices = synth.getVoices();
  const enVoices = voices.filter(v => v.lang.startsWith('en'));
  $setVoice.innerHTML = '<option value="">Auto (best en-US)</option>';
  enVoices.forEach(v => {
    const opt = document.createElement('option');
    opt.value = v.voiceURI;
    opt.textContent = `${v.name} — ${v.lang}${v.localService ? ' ●' : ''}`;
    if (v.voiceURI === SETTINGS.voiceURI) opt.selected = true;
    $setVoice.appendChild(opt);
  });
}

function setSliderFill(el) {
  const min = +el.min, max = +el.max, v = +el.value;
  const pct = ((v - min) / (max - min) * 100).toFixed(1);
  el.style.setProperty('--p', pct + '%');
}

function applySettingsToUI() {
  $setRate.value = SETTINGS.rate;
  $setRateVal.textContent = SETTINGS.rate.toFixed(2);
  $setPitch.value = SETTINGS.pitch;
  $setPitchVal.textContent = SETTINGS.pitch.toFixed(2);
  $setMaxtok.value = SETTINGS.maxTokens;
  $setMaxVal.textContent = SETTINGS.maxTokens;
  $setMouse.checked = SETTINGS.mouseTrack;
  [$setRate, $setPitch, $setMaxtok].forEach(setSliderFill);
}

function openSettings() {
  $setPanel.hidden = false;
  $gearBtn.setAttribute('aria-expanded', 'true');
  populateVoiceDropdown();
  applySettingsToUI();
}
function closeSettings() {
  $setPanel.hidden = true;
  $gearBtn.setAttribute('aria-expanded', 'false');
}
function toggleSettings() { $setPanel.hidden ? openSettings() : closeSettings(); }

$gearBtn.addEventListener('click', e => { e.stopPropagation(); toggleSettings(); });
$setPanel.addEventListener('click', e => e.stopPropagation());

$setVoice.addEventListener('change', () => {
  SETTINGS.voiceURI = $setVoice.value; saveSettings();
});
$setRate.addEventListener('input', () => {
  SETTINGS.rate = +$setRate.value;
  $setRateVal.textContent = SETTINGS.rate.toFixed(2);
  setSliderFill($setRate); saveSettings();
});
$setPitch.addEventListener('input', () => {
  SETTINGS.pitch = +$setPitch.value;
  $setPitchVal.textContent = SETTINGS.pitch.toFixed(2);
  setSliderFill($setPitch); saveSettings();
});
$setMaxtok.addEventListener('input', () => {
  SETTINGS.maxTokens = +$setMaxtok.value;
  $setMaxVal.textContent = SETTINGS.maxTokens;
  setSliderFill($setMaxtok); saveSettings();
});
$setMouse.addEventListener('change', () => {
  SETTINGS.mouseTrack = $setMouse.checked; saveSettings();
});
$setTest.addEventListener('click', () => {
  ttsCancel();
  if (_char === 'tars') {
    _ttsQueue.push('TARS online. Humor setting at 80 percent. Ready for your command.');
  } else {
    _ttsQueue.push('Hi! How can I help you today? I am ready to listen.');
  }
  ttsNext();
});
$setReset.addEventListener('click', () => {
  SETTINGS = { ...DEFAULTS };
  saveSettings();
  applySettingsToUI();
  populateVoiceDropdown();
});

// ── Shortcuts overlay ─────────────────────────────────────────────────────────
function openShortcuts() { $shortcuts.hidden = false; }
function closeShortcuts() { $shortcuts.hidden = true; }
$shortcuts.addEventListener('click', closeShortcuts);

// ── Global key handlers (S, C, ?, Esc beyond Space) ───────────────────────────
document.addEventListener('keydown', ev => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
  if (ev.key === 'Escape') {
    closeSettings(); closeShortcuts(); closeSysMenu();
  } else if (ev.key === '?' || (ev.shiftKey && ev.code === 'Slash')) {
    ev.preventDefault();
    $shortcuts.hidden ? openShortcuts() : closeShortcuts();
  } else if (ev.key === 's' || ev.key === 'S') {
    if (!ev.metaKey && !ev.ctrlKey) { ev.preventDefault(); toggleSettings(); }
  } else if (ev.key === 'c' || ev.key === 'C') {
    if (_categories.length > 1 && !ev.metaKey && !ev.ctrlKey) {
      const idx = _categories.findIndex(c => c.key === _activeCat);
      const next = _categories[(idx + 1) % _categories.length];
      _activeCat = next.key; CFG.defaultCategory = next.key;
      updateSysLabel(); renderSysMenu();
      setStatus(next.title, '');
      setTimeout(() => { if (_state === 'idle') setStatus(''); }, 1500);
    }
  } else if (ev.key === 'x' || ev.key === 'X') {
    if (!ev.metaKey && !ev.ctrlKey) {
      ev.preventDefault();
      switchCharacter(_char === 'eyes' ? 'tars' : 'eyes');
    }
  } else if (ev.key === 'm' || ev.key === 'M') {
    if (!ev.metaKey && !ev.ctrlKey) { ev.preventDefault(); _runMockCot(); }
  } else if (ev.key === 'd' || ev.key === 'D') {
    if (!ev.metaKey && !ev.ctrlKey) { ev.preventDefault(); toggleDebugPanel(); }
  } else if (ev.key === 'p' || ev.key === 'P') {
    if (!ev.metaKey && !ev.ctrlKey) { ev.preventDefault(); window.PF?.toggle(); }
  }
});

// Click outside settings closes it
document.addEventListener('click', ev => {
  if (!$setPanel.hidden && !$setPanel.contains(ev.target) && !$gearBtn.contains(ev.target)) {
    closeSettings();
  }
});

// ── Mock CoT test (M key) ─────────────────────────────────────────────────────
const _MOCK_COT = `Let me reason through this carefully. The question touches on the nature of intelligence itself. First, I need to assess what kind of problem this is: analytical, creative, or emotional? Running classification pass... result: multidimensional. Next, pulling relevant context from associative memory. Interesting — there are three competing frameworks that apply here. Framework A suggests a linear approach, but that ignores edge cases. Framework B is more robust but computationally expensive. Framework C is novel and elegant — I'll weight it at 65%. Cross-referencing against known failure modes. Humor.dll reporting 80% engagement probability. Confidence check: 0.91. Ready to synthesize response.`;
const _MOCK_FINAL = `Based on my analysis, here's what I think: the most elegant solution combines all three approaches, weighted by context. The linear method handles the common case efficiently, while the robust framework covers edge cases. The novel approach provides the creative element that makes the response genuinely useful rather than merely correct. Sometimes the best answer isn't the most obvious one.`;

function _runMockCot() {
  ttsCancel(); cotClear();
  // ensure TARS is active for the terminal effect; if Mamba, use cot-stream
  setState('thinking');
  setStatus('Thinking…');

  let idx = 0;
  const CHUNK = 7;
  const t = setInterval(() => {
    if (idx >= _MOCK_COT.length) {
      clearInterval(t);
      setTimeout(() => {
        cotFade();
        setState('speaking');
        setStatus('Speaking…');
        _tokenCount = 0;
        let fi = 0;
        const ft = setInterval(() => {
          if (fi >= _MOCK_FINAL.length) { clearInterval(ft); scheduleDone(); return; }
          ttsAccum(_MOCK_FINAL.slice(fi, fi + 9));
          _tokenCount += 9;
          fi += 9;
        }, 45);
      }, 500);
      return;
    }
    cotAppend(_MOCK_COT.slice(idx, idx + CHUNK));
    idx += CHUNK;
  }, 32);
}
window._mockCot = _runMockCot;

// ── Debug state panel (D key) ─────────────────────────────────────────────────
const DEBUG_STATES = ['idle', 'listening', 'processing', 'thinking', 'speaking', 'done', 'error', 'greeting', 'startled', 'confused', 'celebrate', 'sleep'];

function buildDebugPanel() {
  const panel = document.createElement('div');
  panel.id = 'debug-panel';
  panel.className = 'debug-panel';
  panel.hidden = true;

  const title = document.createElement('div');
  title.className = 'debug-title';
  title.textContent = 'DEV · States';
  panel.appendChild(title);

  const grid = document.createElement('div');
  grid.className = 'debug-grid';
  DEBUG_STATES.forEach(s => {
    const b = document.createElement('button');
    b.className = 'debug-btn';
    b.textContent = s;
    b.addEventListener('click', () => {
      ttsCancel(); cotClear();
      if (s === 'thinking') {
        setState('thinking'); setStatus('Thinking…');
        // run a short mock CoT so thinking state has visible content
        let idx = 0; const chunk = 7;
        const t = setInterval(() => {
          if (idx >= _MOCK_COT.length || _state !== 'thinking') { clearInterval(t); return; }
          cotAppend(_MOCK_COT.slice(idx, idx + chunk));
          idx += chunk;
        }, 32);
      } else if (s === 'speaking') {
        setState('speaking'); setStatus('Speaking…');
        ttsAccum('This is a test sentence to verify the speaking state renders correctly.');
        _tokenCount = 40;
        setTimeout(scheduleDone, 3000);
      } else {
        setState(s);
        if (['done', 'error', 'greeting', 'startled', 'celebrate'].includes(s)) {
          setTimeout(() => setState('idle'), 1200);
        }
      }
    });
    grid.appendChild(b);
  });
  panel.appendChild(grid);

  // Mock full conversation button
  const mockBtn = document.createElement('button');
  mockBtn.className = 'debug-btn debug-btn-mock';
  mockBtn.textContent = '▶ Mock conversation';
  mockBtn.addEventListener('click', _runMockCot);
  panel.appendChild(mockBtn);

  document.body.appendChild(panel);
  return panel;
}

let _$debugPanel = null;
function toggleDebugPanel() {
  if (!_$debugPanel) _$debugPanel = buildDebugPanel();
  _$debugPanel.hidden = !_$debugPanel.hidden;
}

// ── Boot ──────────────────────────────────────────────────────────────────────
function init() {
  initRec();
  setState('idle');
  setStatus('');
  connectWS();
  scheduleBlink();
  resetSleepTimer();
  loadCategories();
  applySettingsToUI();
  document.querySelectorAll('.char-btn-opt').forEach(btn => {
    btn.addEventListener('click', e => { e.stopPropagation(); switchCharacter(btn.dataset.char); });
  });
  // populate voice dropdown when voices become available
  if (synth.onvoiceschanged !== undefined) {
    synth.onvoiceschanged = () => {
      synth.getVoices();
      if (!$setPanel.hidden) populateVoiceDropdown();
    };
  }
}

init();
