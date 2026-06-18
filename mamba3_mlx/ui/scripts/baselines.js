'use strict';
/* Baseline Showdown — Pythia-160M @step1000 (2.1B tokens) vs Mamba 3 + Tucker
 * Sequential execution: GPU is shared on Apple Silicon, running both at once
 * thrashes MPS scheduling and drops measured tok/s.
 * Payload format mirrors /ws (chat protocol): max_tokens + sampling{temperature, top_p}.
 */

const WS_PYTHIA = `ws://${location.host}/ws/baseline-pythia`;
const WS_OURS   = `ws://${location.host}/ws`;

const $ = (id) => document.getElementById(id);
const $start  = $('bl-start');
const $prompt = $('bl-prompt');

let running = false;
let wsPythia = null;
let wsOurs   = null;

const _cursors = {};

// ── Sampling state pulled from /api/demo-config (matches chat path) ──
// Without this, "ours" runs with generic 0.7/0.9 sampling and no seed,
// which produces gibberish on self_awareness. The main chat applies
// `sampling_mode_configs.self_awareness` (seed=26 verified path) — we
// mirror that here so the demo output matches the chat page exactly.
let oursSampling = { temperature: 0.7, top_p: 0.9 };
let oursMaxTokens = 512;

async function loadOursSampling() {
  try {
    const j = await (await fetch('/api/demo-config')).json();
    const cfg = (j.sampling_mode_configs && j.sampling_mode_configs.self_awareness)
              || j.sampling_defaults || null;
    if (cfg && typeof cfg === 'object') {
      // Strip non-sampling keys, keep numeric/seed values.
      oursSampling = { ...cfg };
    }
    if (j.max_new_tokens_cap) oursMaxTokens = Math.min(oursMaxTokens, j.max_new_tokens_cap);
  } catch {
    // Fall back to defaults; the chat path defaults still work for most
    // prompts, just less reliably for self_awareness.
  }
}
loadOursSampling();

// ── Status pill ───────────────────────────────────────────────────
function setStatus(side, cls, label) {
  const el = $(`status-${side}`);
  if (!el) return;
  el.className = `bl-status ${cls || ''}`.trim();
  el.textContent = label;
}
function setStats(side, ttft, toks, total) {
  if (ttft  != null) $(`ttft-${side}`).textContent  = `${Math.round(ttft)}ms`;
  if (toks  != null) $(`toks-${side}`).textContent  = Number(toks).toFixed(1);
  if (total != null) $(`total-${side}`).textContent = total;
}
function updateBar(side, tokS, max = 120) {
  const fill = $(`bar-${side}`);
  if (fill) fill.style.width = `${Math.min(100, (tokS / max) * 100).toFixed(1)}%`;
}

// ── Streaming response ─────────────────────────────────────────────
function appendResp(side, text) {
  const el = $(`resp-${side}`);
  if (!el) return;
  const ph = el.querySelector('.bl-idle');
  if (ph) ph.remove();
  const cur = _cursors[side];
  if (cur && cur.parentNode === el) el.removeChild(cur);
  const parts = text.split('\n');
  parts.forEach((part, i) => {
    if (part) el.appendChild(document.createTextNode(part));
    if (i < parts.length - 1) el.appendChild(document.createElement('br'));
  });
  const newCur = document.createElement('span');
  newCur.className = 'bl-cursor';
  el.appendChild(newCur);
  _cursors[side] = newCur;
  el.scrollTop = el.scrollHeight;
}
function doneResp(side) {
  const cur = _cursors[side];
  if (cur) cur.remove();
  delete _cursors[side];
}

// ── Think block (ours only) ────────────────────────────────────────
function showThink(side)            { const b = $(`think-${side}`); if (b) b.hidden = false; }
function appendThink(side, text)    { const e = $(`think-body-${side}`); if (e) { e.textContent += text; e.scrollTop = e.scrollHeight; } }
function doneThink(side, ms) {
  const b = $(`think-${side}`); if (b) b.classList.add('done');
  const l = $(`think-lbl-${side}`); if (l) l.textContent = `Thought for ${(ms/1000).toFixed(1)}s`;
  const t = $(`think-time-${side}`); if (t) t.textContent = `${Math.round(ms)}ms`;
}

// ── Reset ──────────────────────────────────────────────────────────
function resetAll() {
  ['pythia', 'ours'].forEach(side => {
    setStatus(side, '', 'READY');
    $(`ttft-${side}`).textContent  = '—';
    $(`toks-${side}`).textContent  = '—';
    $(`total-${side}`).textContent = '—';
    updateBar(side, 0);
    const resp = $(`resp-${side}`);
    if (resp) resp.innerHTML = '<span class="bl-idle">Waiting to start…</span>';
    const col = $(`col-${side}`);
    if (col) col.classList.remove('running', 'done');
    delete _cursors[side];
  });
  const blk = $('think-ours');
  if (blk) { blk.hidden = true; blk.classList.remove('done', 'collapsed'); }
  const body = $('think-body-ours');
  if (body) body.textContent = '';
  const lbl = $('think-lbl-ours');
  if (lbl) lbl.textContent = 'Thinking…';
  const tim = $('think-time-ours');
  if (tim) tim.textContent = '';
}

// ── Pythia baseline runner (Promise) ───────────────────────────────
function runPythia(prompt) {
  return new Promise((resolve) => {
    const side = 'pythia';
    const t0 = Date.now();
    let firstTok = null;
    let n = 0;

    setStatus(side, 'loading', 'LOADING');
    $(`col-${side}`)?.classList.add('running');

    if (wsPythia) { try { wsPythia.close(); } catch {} }
    wsPythia = new WebSocket(WS_PYTHIA);

    const finish = (r) => {
      doneResp(side);
      $(`col-${side}`)?.classList.remove('running');
      $(`col-${side}`)?.classList.add('done');
      resolve(r);
    };

    wsPythia.onopen = () => {
      // Same payload shape as /ws (chat protocol). Pythia ignores
      // `reasoning` / `category_key` — kept for interface symmetry.
      wsPythia.send(JSON.stringify({
        action: 'chat',
        prompt,
        max_tokens: 80,
        reasoning: false,
        sampling: { temperature: 0.7, top_p: 0.9 },
        category_key: 'self_awareness',
      }));
      setStatus(side, 'generating', 'GENERATING');
    };

    wsPythia.onmessage = (ev) => {
      try {
        const m = JSON.parse(ev.data);
        if (m.type === 'token') {
          if (firstTok === null) firstTok = Date.now() - t0;
          n++;
          appendResp(side, m.text || '');
          if (m.tok_s != null) {
            setStats(side, firstTok, m.tok_s, n);
            updateBar(side, m.tok_s);
          }
        } else if (m.type === 'done') {
          const ttft = firstTok ?? (Date.now() - t0);
          const tokS = m.tok_s ?? 0;
          setStats(side, ttft, tokS, m.total_tokens ?? n);
          updateBar(side, tokS);
          setStatus(side, 'done', 'DONE');
          finish({ side, ttft, tokS, n });
        } else if (m.type === 'error') {
          $(`resp-${side}`).innerHTML =
            `<span class="bl-idle">[error] ${m.message || 'unknown'}</span>`;
          setStatus(side, 'error', 'ERROR');
          finish({ side, ttft: null, tokS: 0, n });
        }
      } catch (e) { /* ignore */ }
    };

    wsPythia.onerror = () => {
      $(`resp-${side}`).innerHTML =
        '<span class="bl-idle">WebSocket error — is Pythia downloaded under baselines/?</span>';
      setStatus(side, 'error', 'ERROR');
      finish({ side, ttft: null, tokS: 0, n });
    };
  });
}

// ── Ours (417M) runner — uses main /ws chat protocol ───────────────
function runOurs(prompt) {
  return new Promise((resolve) => {
    const side = 'ours';
    const t0 = Date.now();
    let firstTok = null;
    let n = 0;
    let thinkStart = null;
    let rawMode = 'think';
    const RAW_TAGS = { '</think>': 'between', '<final>': 'final', '</final>': 'done' };

    function rawRoute(text) {
      if (!text) return;
      for (const [tag, next] of Object.entries(RAW_TAGS)) {
        const idx = text.indexOf(tag);
        if (idx < 0) continue;
        if (idx > 0) rawRoute(text.slice(0, idx));
        rawMode = next;
        if (next === 'final') {
          doneThink(side, Date.now() - (thinkStart || t0));
          setStatus(side, 'generating', 'GENERATING');
          if (firstTok === null) {
            firstTok = Date.now() - t0;
            setStats(side, firstTok, null, null);
          }
        }
        rawRoute(text.slice(idx + tag.length));
        return;
      }
      if (rawMode === 'think') appendThink(side, text);
      else if (rawMode === 'final') { appendResp(side, text); n++; }
    }

    const finish = (r) => {
      doneResp(side);
      $(`col-${side}`)?.classList.remove('running');
      $(`col-${side}`)?.classList.add('done');
      resolve(r);
    };

    setStatus(side, 'thinking', 'THINKING');
    $(`col-${side}`)?.classList.add('running');

    if (wsOurs) { try { wsOurs.close(); } catch {} }
    wsOurs = new WebSocket(WS_OURS);

    wsOurs.onopen = () => {
      wsOurs.send(JSON.stringify({
        action: 'chat',
        prompt,
        max_tokens: oursMaxTokens,
        reasoning: true,
        sampling: { ...oursSampling },
        category_key: 'self_awareness',
      }));
    };

    wsOurs.onmessage = (ev) => {
      try {
        const m = JSON.parse(ev.data);
        if (m.type === 'meta') {
          thinkStart = Date.now();
          showThink(side);
        } else if (m.type === 'reasoning_token') {
          appendThink(side, m.text || '');
        } else if (m.type === 'token') {
          rawRoute(m.text || '');
          if (m.tok_s != null && rawMode === 'final') {
            setStats(side, null, m.tok_s, null);
            updateBar(side, m.tok_s);
          }
        } else if (m.type === 'done') {
          const ttft = firstTok ?? (Date.now() - t0);
          const totalTok = m.total_tokens ?? n;
          const tokS = m.tok_s ?? 0;
          setStats(side, ttft, tokS, totalTok);
          updateBar(side, tokS);
          setStatus(side, 'done', 'DONE');
          finish({ side, ttft, tokS, totalTok });
        }
      } catch (e) { /* ignore */ }
    };

    wsOurs.onerror = () => {
      $(`resp-${side}`).innerHTML =
        '<span class="bl-idle">WebSocket error — main model not ready?</span>';
      setStatus(side, 'error', 'ERROR');
      finish({ side, ttft: null, tokS: 0, n });
    };
  });
}

// ── Collapsible think toggle ────────────────────────────────────────
(function initThinkToggle() {
  const btn = $('think-toggle-ours');
  const blk = $('think-ours');
  if (btn && blk) btn.addEventListener('click', () => blk.classList.toggle('collapsed'));
})();

// ── Orchestrator: Pythia first (garbage warm-up), then ours (hero) ──
const SEQUENCE = [
  { side: 'pythia', run: runPythia },
  { side: 'ours',   run: runOurs   },
];

$start.addEventListener('click', async () => {
  if (running) return;
  running = true;
  resetAll();
  $start.disabled = true;
  $start.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21"/></svg> Running…`;

  // Pre-mark the second slot as QUEUED so user knows it's coming
  SEQUENCE.slice(1).forEach(({ side }) => setStatus(side, 'queued', 'QUEUED'));

  const prompt = $prompt.value;
  for (const { run } of SEQUENCE) {
    try { await run(prompt); }
    catch (e) { console.error('runner error', e); }
  }

  running = false;
  $start.disabled = false;
  $start.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21"/></svg> Run Again`;
});
