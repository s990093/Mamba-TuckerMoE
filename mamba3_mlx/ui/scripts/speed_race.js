'use strict';
/* ── Speed Race v3 · PyTorch vs MLX + Performance Matrix ───────────── */

const PROMPT    = 'Who are you?';
const PT_WS_URL  = `ws://${location.host}/ws/pytorch`;
const MLX_WS_URL = `ws://${location.host}/ws`;

// ── DOM helpers ────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const $btn    = $('race-btn');
const $banner = $('speedup-banner');
const $matrix = $('perf-matrix');
const $phase  = $('phase-label');

// ── State ──────────────────────────────────────────────────────────────
let _running = false;
let _results = {};   // { pt: {tokS,ttft,prefillTps,total}, mlx: {…} }
let _mlxBenchTps = null;

// ── Fetch pre-measured bench TPS on load ──────────────────────────────
fetch('/api/bench-tps', { signal: AbortSignal.timeout(2500) })
  .then(r => r.json())
  .then(d => { _mlxBenchTps = d.mlx_bench_tps ?? null; })
  .catch(() => {});

// ── Status probe ───────────────────────────────────────────────────────
function _probePT() {
  fetch('/api/pytorch-status', { signal: AbortSignal.timeout(2500) })
    .then(r => r.json())
    .then(s => {
      const dot = $('pt-status-dot'), txt = $('pt-status-text');
      if (s.ready) {
        dot.className = 'status-dot ready';
        const info = s.info ?? {};
        txt.textContent = `PyTorch backend ready · ${info.device ?? 'mps'} · ${(info.dtype ?? '').replace('torch.','')}`;
      } else {
        dot.className = 'status-dot';
        txt.textContent = 'PyTorch loading weights in background…';
        setTimeout(_probePT, 3000);
      }
    })
    .catch(() => {
      const dot = $('pt-status-dot'), txt = $('pt-status-text');
      if (dot) dot.className = 'status-dot error';
      if (txt) txt.textContent = 'PyTorch unavailable — start server: make -C mamba3_mlx chat';
    });
}
_probePT();

// ── Column helpers ─────────────────────────────────────────────────────
function badge(id, cls, txt) {
  const el = $(`badge-${id}`);
  if (el) { el.className = `status-badge ${cls}`; el.textContent = txt; }
}
function setLive(id, key, val) {
  const el = $(`${key}-${id}`);
  if (el) el.textContent = val ?? '—';
}
function setBar(id, tps, max = 160) {
  const f = $(`bar-${id}`);
  if (f) f.style.width = `${Math.min(100, (tps / max) * 100).toFixed(2)}%`;
}

// ── Text streaming helpers ─────────────────────────────────────────────
let _cursors = {};
function appendResp(id, text) {
  if (!text) return;
  const el = $(`resp-${id}`);
  if (!el) return;
  const ph = el.querySelector('.resp-idle,.resp-note');
  if (ph) ph.remove();
  const cur = _cursors[id];
  if (cur && cur.parentNode === el) el.removeChild(cur);
  text.split('\n').forEach((p, i, a) => {
    if (p) el.appendChild(document.createTextNode(p));
    if (i < a.length - 1) el.appendChild(document.createElement('br'));
  });
  const c = document.createElement('span');
  c.className = 'cursor';
  el.appendChild(c);
  _cursors[id] = c;
  el.scrollTop = el.scrollHeight;
}
function doneResp(id) { const c = _cursors[id]; if (c) c.remove(); delete _cursors[id]; }

// ── Think helpers ──────────────────────────────────────────────────────
function showThink(id) { const b = $(`think-${id}`); if (b) b.hidden = false; }
function appendThink(id, text) {
  const e = $(`think-body-${id}`);
  if (e && text) { e.textContent += text; e.scrollTop = e.scrollHeight; }
}
function doneThink(id, ms) {
  const b = $(`think-${id}`); if (b) b.classList.add('done');
  const l = $(`think-lbl-${id}`);
  if (l) l.textContent = `Thought for ${(ms/1000).toFixed(1)}s`;
  const t = $(`think-time-${id}`);
  if (t) t.textContent = `${Math.round(ms)}ms`;
}

// Think collapse toggles
['pt','mlx'].forEach(id => {
  const btn = $(`think-toggle-${id}`), blk = $(`think-${id}`);
  if (btn && blk) btn.addEventListener('click', () => blk.classList.toggle('collapsed'));
});

// ── Phase label ────────────────────────────────────────────────────────
function setPhase(txt, cls = '') {
  if (!$phase) return;
  $phase.textContent = txt;
  $phase.className = `phase-label ${cls}`.trim();
}

// ── Animated counter ───────────────────────────────────────────────────
function animateCounter(el, targetVal, suffix = '×', duration = 800) {
  if (!el) return;
  const start = performance.now();
  const from  = 0;
  function tick(now) {
    const t = Math.min((now - start) / duration, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = (from + (targetVal - from) * ease).toFixed(1) + suffix;
    if (t < 1) requestAnimationFrame(tick);
    else el.textContent = targetVal.toFixed(1) + suffix;
  }
  requestAnimationFrame(tick);
}

// ── Matrix population ──────────────────────────────────────────────────
function populateMatrix(ptData, mlxData) {
  // Decode tok/s
  const pTps = ptData.tokS  ?? 0;
  const mTps = mlxData.tokS ?? 0;
  $('mx-decode-pt').textContent  = pTps  ? `${pTps.toFixed(1)} tok/s`  : '—';
  $('mx-decode-mlx').textContent = mTps  ? `${mTps.toFixed(1)} tok/s`  : '—';
  markWinner('mx-decode', pTps, mTps, true);

  // Prefill
  const pPre = ptData.prefillTps  ?? 0;
  const mPre = mlxData.prefillTps ?? 0;
  $('mx-prefill-pt').textContent  = pPre ? `${Math.round(pPre)} tok/s`  : '—';
  $('mx-prefill-mlx').textContent = mPre ? `${Math.round(mPre)} tok/s`  : '—';
  markWinner('mx-prefill', pPre, mPre, true);

  // TTFT (lower is better)
  const pTtft = ptData.ttft  ?? 0;
  const mTtft = mlxData.ttft ?? 0;
  $('mx-ttft-pt').textContent  = pTtft ? `${Math.round(pTtft)}ms`  : '—';
  $('mx-ttft-mlx').textContent = mTtft ? `${Math.round(mTtft)}ms`  : '—';
  markWinner('mx-ttft', mTtft > 0 ? 1/mTtft : 0, pTtft > 0 ? 1/pTtft : 0, true);
}

function markWinner(prefix, ptScore, mlxScore, higherBetter = true) {
  const ptEl  = $(`${prefix}-pt`);
  const mlxEl = $(`${prefix}-mlx`);
  if (!ptEl || !mlxEl) return;
  if (ptScore <= 0 && mlxScore <= 0) return;
  const ptWins = higherBetter ? ptScore > mlxScore : ptScore < mlxScore;
  if (ptWins) {
    ptEl.classList.add('winner-cell');
  } else if (mlxScore !== ptScore) {
    mlxEl.classList.add('winner-cell');
  }
}

// ── Show results ───────────────────────────────────────────────────────
function showResults() {
  const pt  = _results.pt;
  const mlx = _results.mlx;
  if (!pt || !mlx) return;

  const ptTps  = pt.tokS  ?? 0;
  const mlxTps = mlx.tokS ?? 0;
  const ratio  = mlxTps > 0 && ptTps > 0 ? mlxTps / ptTps : 0;

  // Banner
  $('banner-mlx-tps').textContent = mlxTps ? `${mlxTps.toFixed(1)} tok/s` : '—';
  $('banner-pt-tps').textContent  = ptTps  ? `${ptTps.toFixed(1)} tok/s`  : '—';
  animateCounter($('speedup-num'), ratio, '×');
  $banner.hidden = false;

  // Winner glow
  if (ratio > 1) {
    $('col-mlx').classList.add('winner');
    $('crown-mlx').textContent = '👑';
    $('crown-pt').textContent  = '';
  } else {
    $('col-pt').classList.add('winner');
    $('crown-pt').textContent  = '👑';
    $('crown-mlx').textContent = '';
  }

  // Matrix
  populateMatrix(pt, mlx);
  $matrix.hidden = false;
}

// ── Core WS runner (Promise-based) ────────────────────────────────────
function runEngine(id, wsUrl, payload) {
  return new Promise(resolve => {
    const t0       = Date.now();
    let thinkStart = null, firstTokMs = null;
    let tokCount   = 0;
    let prefillTps = 0;

    let rawMode = 'think';
    const TAGS = { '</think>': 'between', '<final>': 'final', '</final>': 'end' };

    function route(text) {
      if (!text) return;
      for (const [tag, next] of Object.entries(TAGS)) {
        const idx = text.indexOf(tag);
        if (idx < 0) continue;
        if (idx > 0) route(text.slice(0, idx));
        rawMode = next;
        if (next === 'final') {
          doneThink(id, Date.now() - (thinkStart || t0));
          badge(id, 'running', 'GENERATING');
          if (firstTokMs === null) {
            firstTokMs = Date.now() - t0;
            setLive(id, 'ttft', `${Math.round(firstTokMs)}ms`);
          }
        }
        route(text.slice(idx + tag.length));
        return;
      }
      if (rawMode === 'think') appendThink(id, text);
      else if (rawMode === 'final') { appendResp(id, text); tokCount++; }
    }

    badge(id, 'loading', 'THINKING');
    $(`col-${id}`).classList.remove('waiting');
    $(`col-${id}`).classList.add('running');
    showThink(id);
    thinkStart = Date.now();

    const ws = new WebSocket(wsUrl);
    ws.onopen = () => ws.send(JSON.stringify(payload));

    ws.onmessage = ev => {
      try {
        const m = JSON.parse(ev.data);
        if (m.type === 'connected') return;
        if (m.type === 'error') {
          badge(id, 'error', 'ERROR');
          const r = $(`resp-${id}`);
          if (r) r.innerHTML = `<span class="resp-note">${m.message ?? 'Connection error'}</span>`;
          resolve({ tokS: 0, ttft: null, prefillTps: 0, total: 0 });
          return;
        }
        if (m.type === 'meta') {
          thinkStart = Date.now();
          prefillTps = m.prefill_tps ?? 0;
          if (prefillTps) setLive(id, 'prefill', `${Math.round(prefillTps)}`);
        } else if (m.type === 'token') {
          route(m.text || '');
          if (m.tok_s != null && rawMode === 'final') {
            setLive(id, 'toks', Number(m.tok_s).toFixed(1));
            setBar(id, m.tok_s, 160);
          }
        } else if (m.type === 'done') {
          const ttft    = firstTokMs ?? (Date.now() - t0);
          const total   = m.total_tokens ?? tokCount;
          // Prefer bench_tps (for MLX) or per-step tok_s (for PT)
          const rawBench = (id === 'mlx' && _mlxBenchTps) ? _mlxBenchTps
                         : (m.bench_tps ?? null);
          const dispTps  = parseFloat(rawBench ?? m.tok_s ?? 0);
          const prefill  = m.prefill_tps ?? prefillTps;

          setLive(id, 'ttft',    `${Math.round(ttft)}ms`);
          setLive(id, 'toks',    dispTps.toFixed(1));
          setLive(id, 'total',   total);
          setLive(id, 'prefill', prefill ? `${Math.round(prefill)}` : '—');
          setBar(id, dispTps, 160);
          doneResp(id);
          badge(id, 'done', 'DONE ✓');
          $(`col-${id}`).classList.remove('running');
          $(`col-${id}`).classList.add('finished');
          resolve({ tokS: dispTps, ttft, prefillTps: prefill, total });
          // Keep WS open — fresh bench event may arrive a couple seconds later
        } else if (m.type === 'bench') {
          // Fresh StaticDecoder.generate() measurement (post-streaming, steady thermal state)
          const freshTps = parseFloat(m.bench_tps ?? 0);
          if (freshTps > 0) {
            setLive(id, 'toks', freshTps.toFixed(1));
            setBar(id, freshTps, 160);
            _mlxBenchTps = freshTps;   // update global for matrix
            if (_results[id]) {
              _results[id].tokS = freshTps;
              // Re-draw banner + matrix if already visible
              if ($banner && !$banner.hidden) {
                $('banner-mlx-tps').textContent = `${freshTps.toFixed(1)} tok/s`;
                const ptTps = (_results.pt && _results.pt.tokS) || 0;
                if (ptTps > 0) animateCounter($('speedup-num'), freshTps / ptTps, '×');
              }
              if ($matrix && !$matrix.hidden) {
                $('mx-decode-mlx').textContent = `${freshTps.toFixed(1)} tok/s`;
                // Re-mark winner
                const ptScore = (_results.pt && _results.pt.tokS) || 0;
                document.querySelectorAll('#mrow-decode .mt-val').forEach(el => el.classList.remove('winner-cell'));
                markWinner('mx-decode', ptScore, freshTps, true);
              }
            }
          }
        }
      } catch {}
    };

    ws.onerror = () => {
      badge(id, 'error', 'ERROR');
      const r = $(`resp-${id}`);
      if (r) r.innerHTML = `<span class="resp-note">WebSocket failed — server running?</span>`;
      resolve({ tokS: 0, ttft: null, prefillTps: 0, total: 0 });
    };
  });
}

// ── Reset ──────────────────────────────────────────────────────────────
function resetAll() {
  ['pt', 'mlx'].forEach(id => {
    badge(id, '', 'READY');
    const blk  = $(`think-${id}`);
    if (blk) { blk.hidden = true; blk.classList.remove('done','collapsed'); }
    $(`think-body-${id}`)?.removeAttribute('textContent');
    if ($(`think-body-${id}`)) $(`think-body-${id}`).textContent = '';
    if ($(`think-lbl-${id}`)) $(`think-lbl-${id}`).textContent = 'Thinking…';
    if ($(`think-time-${id}`)) $(`think-time-${id}`).textContent = '';
    const resp = $(`resp-${id}`);
    if (resp) resp.innerHTML = '<span class="resp-idle">Waiting to start…</span>';
    ['ttft','toks','total','prefill'].forEach(k => setLive(id, k, '—'));
    setBar(id, 0, 160);
    const col = $(`col-${id}`);
    if (col) col.className = 'lane';
    if ($(`crown-${id}`)) $(`crown-${id}`).textContent = '';
  });
  delete _cursors.pt; delete _cursors.mlx;
  _results = {};
  $banner.hidden = true;
  $matrix.hidden = true;
  setPhase('');
  // Clear matrix winner highlights
  document.querySelectorAll('.mt-val.winner-cell').forEach(el => el.classList.remove('winner-cell'));
}

// ── Sequential race ────────────────────────────────────────────────────
async function runRace() {
  if (_running) return;
  _running = true;
  resetAll();

  $btn.disabled  = true;
  $btn.className = 'race-btn racing';
  $btn.innerHTML = `<span class="btn-spin">●</span> Running…`;

  const payload = {
    action: 'chat', prompt: PROMPT, max_tokens: 200,
    reasoning: true, category_key: 'self_awareness',
    seed: 26, bench_mode: true,
    sampling: { temperature: 0.25, top_k: 60 },
  };

  // ── Round 1: PyTorch ──────────────────────────────────────────────
  setPhase('Round 1 / 2  ·  PyTorch MPS', 'phase-pt');
  $('col-mlx').classList.add('waiting');
  badge('mlx', 'waiting', 'UP NEXT');
  $('resp-mlx').innerHTML = '<span class="resp-idle">Waiting for PyTorch to finish…</span>';

  _results.pt = await runEngine('pt', PT_WS_URL, payload);

  // ── Round 2: MLX ──────────────────────────────────────────────────
  setPhase('Round 2 / 2  ·  MLX + Metal + Q8', 'phase-mlx');
  $('col-mlx').classList.remove('waiting');
  $('resp-mlx').innerHTML = '<span class="resp-idle">Starting…</span>';

  _results.mlx = await runEngine('mlx', MLX_WS_URL, payload);

  // ── Results ────────────────────────────────────────────────────────
  setPhase('Race complete ✓', 'phase-done');
  showResults();

  _running       = false;
  $btn.disabled  = false;
  $btn.className = 'race-btn';
  $btn.innerHTML = `<svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21"/></svg> Race Again`;
}

$btn.addEventListener('click', runRace);
