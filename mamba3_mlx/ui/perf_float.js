/* perf_float.js — Floating system + inference monitor.
 * System metrics from profiler WS on port 8765.
 * Cache sizes computed client-side — no core code touched.
 *
 * API:  window.PF.reset(prompt_tokens)                    — meta event (exact count from server)
 *       window.PF.reset(null, {sys,user,history,reasoning}) — proactive call (fetches /api/token_count)
 *       window.PF.tick(tok_s, n)              — each decoded token
 *       window.PF.done(tok_s, pre, ttft, n)   — turn complete
 *       window.PF.toggle() / show() / hide()
 *
 * Override model dims: window.PF_MODEL = { n_transformer:6, ... }
 */
(function () {
  'use strict';

  const PROFILER_PORT = 8765;
  const SPARK_N = 60;
  const CHART_N = 120;   // ~2 min of 1-Hz profiler history
  const STORE_KEY = 'mamba_pf5';

  // ── Model constants (Mamba3Config defaults) ───────────────────
  // GQA KV bytes / token = n_transformer × 2 × num_kv_heads × head_dim × bytes
  //                      = 6 × 2 × 4 × 64 × 2 = 6 144 bytes/token
  // Mamba SSM state (fixed) = n_mamba × H × P × N × bytes
  //                         = 24 × 24 × 4 × 64 × 2 = 294 912 bytes ≈ 0.281 MiB
  const MODEL = Object.assign({
    n_transformer: 6,
    n_mamba: 24,
    num_kv_heads: 4,
    head_dim: 64,
    ssm_h: 24,   // d_inner / d_head = 1536 / 64
    ssm_p: 4,    // mimo_rank
    ssm_n: 64,   // d_state
    bytes: 2,    // bf16
    max_seq: 2048,
  }, window.PF_MODEL || {});

  const KV_PER_TOK = MODEL.n_transformer * 2 * MODEL.num_kv_heads * MODEL.head_dim * MODEL.bytes;
  const SSM_BYTES = MODEL.n_mamba * MODEL.ssm_h * MODEL.ssm_p * MODEL.ssm_n * MODEL.bytes;
  const SSM_MIB = SSM_BYTES / (1048576);

  function kvMib(n) { return KV_PER_TOK * n / 1048576; }
  function fmtMib(v) {
    if (v < 1) return (v * 1024).toFixed(0) + 'K';
    if (v < 100) return v.toFixed(2) + 'M';
    return v.toFixed(1) + 'M';
  }

  // ── Ring buffers ──────────────────────────────────────────────
  const sparkBuf = new Float32Array(SPARK_N);
  let sparkPtr = 0, sparkDirty = false;

  // System metrics (1 Hz from profiler WS)
  const tsGpu = new Float32Array(CHART_N);
  const tsCpu = new Float32Array(CHART_N);
  const tsMem = new Float32Array(CHART_N);
  const tsTs = new Float32Array(CHART_N);
  let tsPtr = 0, tsLen = 0;

  // KV cache history — updated every profiler WS tick with current totalN
  const tsKv = new Float32Array(CHART_N);
  let kvPtr = 0, kvLen = 0;

  // ── State ─────────────────────────────────────────────────────
  let sysGpu = null, sysCpu = null, sysMem = null;
  let llmToks = null, llmPre = null, llmTtft = null;
  let llmN = null;      // generated tokens this turn
  let pfxN = 0;         // prompt_tokens from meta (system + user)
  let profWs = null, profRetry = null;
  let expanded = false;
  let chartDirty = false;

  function totalN() { return (pfxN || 0) + (llmN || 0); }

  // ── DOM ───────────────────────────────────────────────────────
  const root = document.createElement('div');
  root.innerHTML = `
<div id="pf-panel" class="pf-panel">
  <div class="pf-handle" id="pf-handle">
    <span class="pf-dot" id="pf-dot"></span>
    <span class="pf-title">PERF</span>
    <span class="pf-mini-stats" id="pf-mini">—</span>
    <div class="pf-hbtns">
      <button class="pf-hbtn" id="pf-expand" title="Charts">⊞</button>
      <button class="pf-hbtn" id="pf-min" title="Collapse">−</button>
      <button class="pf-hbtn" id="pf-close" title="Close">×</button>
    </div>
  </div>
  <div class="pf-body" id="pf-body">

    <!-- ══ Compact view ══ -->
    <div id="pf-compact">
      <div class="pf-section">
        <div class="pf-row-bar">
          <span class="pf-row-lbl">GPU</span>
          <div class="pf-bar-wrap"><div class="pf-bar pf-bar-gpu" id="pf-gpu-bar"></div></div>
          <span class="pf-row-val" id="pf-gpu-val">—</span>
        </div>
        <div class="pf-row-sub" id="pf-gpu-sub" style="display:none"></div>
        <div class="pf-row-bar">
          <span class="pf-row-lbl">CPU</span>
          <div class="pf-bar-wrap"><div class="pf-bar pf-bar-cpu" id="pf-cpu-bar"></div></div>
          <span class="pf-row-val" id="pf-cpu-val">—</span>
        </div>
        <div class="pf-row-bar">
          <span class="pf-row-lbl">Mem</span>
          <div class="pf-bar-wrap"><div class="pf-bar pf-bar-mem" id="pf-mem-bar"></div></div>
          <span class="pf-row-val" id="pf-mem-val">—</span>
        </div>
      </div>

      <!-- LLM stats -->
      <div class="pf-divider" id="pf-llm-divider" style="display:none"></div>
      <div class="pf-section" id="pf-llm" style="display:none">
        <div class="pf-llm-grid">
          <div class="pf-cell"><div class="pf-val" id="pf-toks">—</div><div class="pf-lbl">tok/s</div></div>
          <div class="pf-cell"><div class="pf-val" id="pf-ttft">—</div><div class="pf-lbl">TTFT</div></div>
          <div class="pf-cell"><div class="pf-val" id="pf-pre">—</div><div class="pf-lbl">Prefill</div></div>
          <div class="pf-cell"><div class="pf-val" id="pf-n">—</div><div class="pf-lbl">Tokens</div></div>
        </div>
        <canvas class="pf-spark" id="pf-spark" height="22"></canvas>
      </div>

      <!-- Cache (client-side formula, appears after meta event) -->
      <div class="pf-divider" id="pf-cache-divider" style="display:none"></div>
      <div id="pf-cache" style="display:none">
        <div class="pf-cache-hdr">
          <span class="pf-cache-title">Context Cache</span>
          <span class="pf-cache-ctx" id="pf-cache-ctx"></span>
        </div>
        <!-- GQA: two-segment canvas bar (prefill | decode) -->
        <div class="pf-row-bar" style="margin-top:5px">
          <span class="pf-row-lbl pf-lbl-gqa">GQA</span>
          <canvas class="pf-kv-canvas" id="pf-kv-canvas" height="4"></canvas>
          <span class="pf-row-val" id="pf-kv-val">—</span>
        </div>
        <div class="pf-row-sub pf-kv-legend-row">
          <span class="pf-kv-seg-dot pf-kv-seg-pre"></span><span id="pf-kv-pre-lbl">pre —</span>
          <span class="pf-kv-seg-dot pf-kv-seg-dec"></span><span id="pf-kv-dec-lbl">dec —</span>
        </div>
        <!-- SSM: fixed size -->
        <div class="pf-row-bar" style="margin-top:4px">
          <span class="pf-row-lbl pf-lbl-ssm">SSM</span>
          <div class="pf-bar-wrap pf-bar-wrap-ssm">
            <div class="pf-bar pf-bar-ssm" id="pf-ssm-bar"></div>
          </div>
          <span class="pf-row-val" id="pf-ssm-val">—</span>
        </div>
        <div class="pf-row-sub pf-ssm-note">Mamba · ${MODEL.n_mamba}L · H×P×N · <em>fixed</em></div>
      </div>
    </div>

    <!-- ══ Chart view ══ -->
    <div id="pf-charts" style="display:none">
      <div class="pf-chart-block">
        <div class="pf-chart-hdr">
          <span class="pf-chart-lbl pf-clbl-gpu">GPU</span>
          <span class="pf-chart-cur" id="pfc-gpu-cur">—</span>
        </div>
        <canvas class="pf-chart-canvas" id="pfc-gpu"></canvas>
      </div>
      <div class="pf-chart-block">
        <div class="pf-chart-hdr">
          <span class="pf-chart-lbl pf-clbl-cpu">CPU</span>
          <span class="pf-chart-cur" id="pfc-cpu-cur">—</span>
        </div>
        <canvas class="pf-chart-canvas" id="pfc-cpu"></canvas>
      </div>
      <div class="pf-chart-block">
        <div class="pf-chart-hdr">
          <span class="pf-chart-lbl pf-clbl-mem">RAM</span>
          <span class="pf-chart-cur" id="pfc-mem-cur">—</span>
        </div>
        <canvas class="pf-chart-canvas" id="pfc-mem"></canvas>
      </div>
      <!-- KV cache chart: GQA (grows) + SSM (fixed dashed) -->
      <div class="pf-chart-block" id="pfc-kv-block">
        <div class="pf-chart-hdr">
          <span class="pf-chart-lbl pf-clbl-kv">Cache</span>
          <div class="pf-kv-legend">
            <span class="pf-legend-dot pf-legend-gqa"></span>
            <span class="pf-legend-lbl" id="pfc-legend-gqa">GQA</span>
            <span class="pf-legend-dot pf-legend-ssm"></span>
            <span class="pf-legend-lbl" id="pfc-legend-ssm">SSM ${fmtMib(SSM_MIB)}</span>
          </div>
        </div>
        <canvas class="pf-chart-canvas pf-chart-canvas-kv" id="pfc-kv"></canvas>
      </div>
      <div class="pf-time-axis" id="pf-time-axis"></div>
    </div>

    <div class="pf-rzh" id="pf-rzh">⌟</div>
  </div>
</div>`;
  document.body.appendChild(root.firstElementChild);

  const panel = document.getElementById('pf-panel');
  const body = document.getElementById('pf-body');
  const dotEl = document.getElementById('pf-dot');
  const miniEl = document.getElementById('pf-mini');
  const minBtn = document.getElementById('pf-min');
  const expBtn = document.getElementById('pf-expand');
  const llmSec = document.getElementById('pf-llm');
  const llmDiv = document.getElementById('pf-llm-divider');
  const compact = document.getElementById('pf-compact');
  const charts = document.getElementById('pf-charts');
  const sCtx = document.getElementById('pf-spark').getContext('2d');

  // ── Restore state ─────────────────────────────────────────────
  try {
    const s = JSON.parse(localStorage.getItem(STORE_KEY) || '{}');
    if (s.x != null) { panel.style.left = s.x + 'px'; panel.style.top = s.y + 'px'; }
    else { panel.style.right = '16px'; panel.style.top = '16px'; }
    if (s.w) panel.style.width = s.w + 'px';
    if (s.mini) { body.classList.add('pf-hidden'); minBtn.textContent = '+'; }
    else minBtn.textContent = '−';
    if (s.hide) panel.style.display = 'none';
    if (s.expanded) setExpanded(true, true);
  } catch { panel.style.right = '16px'; panel.style.top = '16px'; }

  function save() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify({
        x: panel.offsetLeft, y: panel.offsetTop, w: panel.offsetWidth,
        mini: body.classList.contains('pf-hidden'),
        hide: panel.style.display === 'none', expanded,
      }));
    } catch { }
  }

  // ── Drag ──────────────────────────────────────────────────────
  let dragging = false, ddx = 0, ddy = 0;
  document.getElementById('pf-handle').addEventListener('mousedown', e => {
    if (e.button !== 0 || e.target.closest('.pf-hbtn')) return;
    dragging = true;
    const r = panel.getBoundingClientRect();
    panel.style.right = ''; panel.style.bottom = '';
    panel.style.left = r.left + 'px'; panel.style.top = r.top + 'px';
    ddx = e.clientX - r.left; ddy = e.clientY - r.top;
    e.preventDefault();
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    panel.style.left = Math.max(0, Math.min(e.clientX - ddx, innerWidth - panel.offsetWidth)) + 'px';
    panel.style.top = Math.max(0, Math.min(e.clientY - ddy, innerHeight - panel.offsetHeight)) + 'px';
  });
  document.addEventListener('mouseup', () => { if (dragging) { dragging = false; save(); } });

  // ── Resize ────────────────────────────────────────────────────
  let resizing = false, rw0 = 0, rx0 = 0;
  document.getElementById('pf-rzh').addEventListener('mousedown', e => {
    if (e.button !== 0) return;
    resizing = true; rw0 = panel.offsetWidth; rx0 = e.clientX;
    e.preventDefault(); e.stopPropagation();
  });
  document.addEventListener('mousemove', e => {
    if (!resizing) return;
    panel.style.width = Math.max(expanded ? 280 : 192, rw0 + e.clientX - rx0) + 'px';
    sparkDirty = true;
    chartDirty = true;
  });
  document.addEventListener('mouseup', () => { if (resizing) { resizing = false; save(); } });

  // ── Buttons ───────────────────────────────────────────────────
  minBtn.addEventListener('click', () => {
    const c = body.classList.toggle('pf-hidden');
    minBtn.textContent = c ? '+' : '−'; save();
  });
  document.getElementById('pf-close').addEventListener('click', () => { panel.style.display = 'none'; save(); });
  expBtn.addEventListener('click', () => { setExpanded(!expanded); save(); });

  function setExpanded(on, silent) {
    expanded = on;
    expBtn.style.color = on ? '#cc785c' : '';
    expBtn.style.opacity = on ? '1' : '';
    compact.style.display = on ? 'none' : '';
    charts.style.display = on ? '' : 'none';
    if (on) {
      if (!panel.style.width || parseInt(panel.style.width) < 280) panel.style.width = '320px';
      chartDirty = true;
    }
    if (!silent) save();
  }

  // ── DPR-aware canvas helper ───────────────────────────────────
  // Returns a 2d context scaled for device pixel ratio.
  // The canvas CSS size must be set via CSS (width:100%, height:Npx).
  function getCtx(id) {
    const cnv = document.getElementById(id);
    if (!cnv) return null;
    const dpr = window.devicePixelRatio || 1;
    const cssW = cnv.clientWidth || (panel.offsetWidth - 22);
    const cssH = cnv.clientHeight || 56;
    cnv.width = Math.round(cssW * dpr);
    cnv.height = Math.round(cssH * dpr);
    const ctx = cnv.getContext('2d');
    ctx.resetTransform();
    ctx.scale(dpr, dpr);
    return { ctx, W: cssW, H: cssH };
  }

  // ── Sparkline ─────────────────────────────────────────────────
  function drawSpark() {
    const cnv = document.getElementById('pf-spark');
    const dpr = window.devicePixelRatio || 1;
    const cssW = Math.max(60, panel.offsetWidth - 22);
    cnv.width = Math.round(cssW * dpr);
    cnv.height = Math.round(22 * dpr);
    sCtx.resetTransform();
    sCtx.scale(dpr, dpr);
    const W = cssW, H = 22, pad = 2;
    sCtx.clearRect(0, 0, W, H);
    let max = 0;
    for (let i = 0; i < SPARK_N; i++) if (sparkBuf[i] > max) max = sparkBuf[i];
    if (max < 1) { sparkDirty = false; return; }
    const step = W / (SPARK_N - 1);
    sCtx.beginPath();
    for (let i = 0; i < SPARK_N; i++) {
      const idx = (sparkPtr + i) % SPARK_N;
      const x = i * step;
      const y = H - pad - (sparkBuf[idx] / max) * (H - pad * 2);
      i === 0 ? sCtx.moveTo(x, y) : sCtx.lineTo(x, y);
    }
    sCtx.strokeStyle = 'rgba(204,120,92,0.75)'; sCtx.lineWidth = 1.5; sCtx.lineJoin = 'round';
    sCtx.stroke();
    sCtx.lineTo((SPARK_N - 1) * step, H); sCtx.lineTo(0, H); sCtx.closePath();
    sCtx.fillStyle = 'rgba(204,120,92,0.09)'; sCtx.fill();
    sparkDirty = false;
  }

  // ── Generic chart (GPU / CPU / RAM) ───────────────────────────
  const CC = {
    gpu: { stroke: 'rgba(204,120,92,0.90)', fill: 'rgba(204,120,92,0.10)', grid: 'rgba(255,255,255,0.06)' },
    cpu: { stroke: 'rgba(145,205,110,0.85)', fill: 'rgba(145,205,110,0.08)', grid: 'rgba(255,255,255,0.06)' },
    mem: { stroke: 'rgba(220,175,65,0.85)', fill: 'rgba(220,175,65,0.08)', grid: 'rgba(255,255,255,0.06)' },
  };

  function drawChart(id, buf, maxVal, c) {
    const g = getCtx(id); if (!g) return;
    const { ctx, W, H } = g;
    ctx.clearRect(0, 0, W, H);
    const n = Math.min(tsLen, CHART_N);
    if (n < 2) return;

    // Grid lines
    ctx.strokeStyle = c.grid; ctx.lineWidth = 0.5;
    for (const f of [0.25, 0.5, 0.75]) {
      const y = (H - f * H) | 0;
      ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(W, y + 0.5); ctx.stroke();
    }

    // Line
    const step = W / (n - 1);
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const idx = (tsPtr - n + i + CHART_N) % CHART_N;
      const y = H - 1 - Math.max(0, Math.min(1, buf[idx] / maxVal)) * (H - 3);
      i === 0 ? ctx.moveTo(i * step, y) : ctx.lineTo(i * step, y);
    }
    ctx.strokeStyle = c.stroke; ctx.lineWidth = 1.8; ctx.lineJoin = 'round'; ctx.stroke();

    // Fill
    ctx.lineTo((n - 1) * step, H); ctx.lineTo(0, H); ctx.closePath();
    ctx.fillStyle = c.fill; ctx.fill();

    // Y labels
    ctx.fillStyle = c.stroke; ctx.font = 'bold 9px ui-monospace,monospace'; ctx.textAlign = 'right';
    const topLbl = maxVal > 20 ? Math.round(maxVal) + '%' : maxVal.toFixed(1) + 'G';
    ctx.fillText(topLbl, W - 3, 10);
    ctx.fillStyle = 'rgba(138,135,125,0.45)'; ctx.font = '9px ui-monospace,monospace';
    ctx.fillText('0', W - 3, H - 3);
  }

  // ── KV cache chart (dynamic Y, two lines) ─────────────────────
  function drawKvChart() {
    const g = getCtx('pfc-kv'); if (!g) return;
    const { ctx, W, H } = g;
    ctx.clearRect(0, 0, W, H);

    const n = Math.min(kvLen, CHART_N);
    const curKv = kvMib(totalN());

    // Dynamic Y-axis: at least 2× current KV, or 1 MiB floor
    const dynMax = Math.max(curKv * 2, kvMib(256), 1);

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 0.5;
    for (const f of [0.25, 0.5, 0.75]) {
      const y = (H - f * H) | 0;
      ctx.beginPath(); ctx.moveTo(0, y + 0.5); ctx.lineTo(W, y + 0.5); ctx.stroke();
    }

    // ── Line 1: SSM (fixed horizontal dashed) ─────────────────
    const ssmFrac = Math.min(1, SSM_MIB / dynMax);
    const ssmY = H - 1 - ssmFrac * (H - 3);
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = 'rgba(140,205,100,0.70)';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(0, ssmY); ctx.lineTo(W, ssmY); ctx.stroke();
    ctx.restore();

    // SSM label on line
    ctx.fillStyle = 'rgba(140,205,100,0.55)';
    ctx.font = '9px ui-monospace,monospace'; ctx.textAlign = 'left';
    ctx.fillText(fmtMib(SSM_MIB), 4, ssmY - 3);

    // ── Line 2: GQA KV (grows with tokens) ────────────────────
    if (n >= 2) {
      const step = W / (n - 1);
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const idx = (kvPtr - n + i + CHART_N) % CHART_N;
        const v = tsKv[idx];
        const y = H - 1 - Math.max(0, Math.min(1, v / dynMax)) * (H - 3);
        i === 0 ? ctx.moveTo(i * step, y) : ctx.lineTo(i * step, y);
      }
      ctx.strokeStyle = 'rgba(155,110,220,0.90)';
      ctx.lineWidth = 1.8; ctx.lineJoin = 'round'; ctx.stroke();
      ctx.lineTo((n - 1) * step, H); ctx.lineTo(0, H); ctx.closePath();
      ctx.fillStyle = 'rgba(155,110,220,0.08)'; ctx.fill();
    }

    // Y labels
    ctx.fillStyle = 'rgba(155,110,220,0.60)';
    ctx.font = 'bold 9px ui-monospace,monospace'; ctx.textAlign = 'right';
    ctx.fillText(fmtMib(dynMax), W - 3, 10);
    ctx.fillStyle = 'rgba(138,135,125,0.45)'; ctx.font = '9px ui-monospace,monospace';
    ctx.fillText('0', W - 3, H - 3);

    // Legend values
    const lg = document.getElementById('pfc-legend-gqa');
    if (lg) lg.textContent = `GQA ${fmtMib(curKv)} (${totalN()}t)`;
  }

  function drawTimeAxis() {
    const el = document.getElementById('pf-time-axis');
    if (!el || tsLen < 2) { if (el) el.textContent = ''; return; }
    const n = Math.min(tsLen, CHART_N);
    const newest = tsTs[(tsPtr - 1 + CHART_N) % CHART_N];
    const oldest = tsTs[(tsPtr - n + CHART_N) % CHART_N];
    const span = newest - oldest;
    if (span < 1) return;
    const steps = [5, 10, 15, 30, 60];
    const tickStep = steps.find(s => span / s >= 2 && span / s <= 6) || Math.round(span / 4);
    el.innerHTML = ''; el.style.position = 'relative';
    for (let t = Math.ceil(oldest / tickStep) * tickStep; t <= newest; t += tickStep) {
      const frac = (t - oldest) / span;
      const ago = Math.round(newest - t);
      const sp = document.createElement('span');
      sp.textContent = ago === 0 ? 'now' : `-${ago}s`;
      sp.style.cssText = `position:absolute;left:${(frac * 100).toFixed(1)}%;transform:translateX(-50%)`;
      el.appendChild(sp);
    }
  }

  function drawAllCharts() {
    if (!expanded) return;
    const mt = sysMem?.total_gb || 16;
    drawChart('pfc-gpu', tsGpu, 100, CC.gpu);
    drawChart('pfc-cpu', tsCpu, 100, CC.cpu);
    drawChart('pfc-mem', tsMem, mt, CC.mem);
    drawKvChart();
    drawTimeAxis();
    const gv = sysGpu?.usage_percent, cv = sysCpu?.usage_percent, mv = sysMem?.used_gb;
    const gEl = document.getElementById('pfc-gpu-cur');
    const cEl = document.getElementById('pfc-cpu-cur');
    const mEl = document.getElementById('pfc-mem-cur');
    if (gEl) gEl.textContent = gv != null ? gv.toFixed(1) + '%' : '—';
    if (cEl) cEl.textContent = cv != null ? cv.toFixed(1) + '%' : '—';
    if (mEl) mEl.textContent = mv != null ? mv.toFixed(2) + 'G' : '—';
  }

  // ── rAF ───────────────────────────────────────────────────────
  function raf() {
    if (panel.style.display !== 'none' && !body.classList.contains('pf-hidden')) {
      if (sparkDirty && llmSec.style.display !== 'none' && !expanded) drawSpark();
      if (chartDirty && expanded) { drawAllCharts(); chartDirty = false; }
    }
    requestAnimationFrame(raf);
  }
  raf();

  // ── UI helpers ────────────────────────────────────────────────
  function setBar(id, pct) {
    const el = document.getElementById(id);
    if (el) el.style.width = Math.max(0, Math.min(100, pct || 0)).toFixed(1) + '%';
  }
  function fmt1(v) { return v != null ? Number(v).toFixed(1) : '—'; }
  function fmt0(v) { return v != null ? Number(v).toFixed(0) : '—'; }

  function updateSysUI() {
    const g = sysGpu, c = sysCpu, m = sysMem;
    const gv = g?.usage_percent, cv = c?.usage_percent;
    const mu = m?.used_gb, mp = m?.percent;

    setBar('pf-gpu-bar', gv); setBar('pf-cpu-bar', cv); setBar('pf-mem-bar', mp);

    const gpuEl = document.getElementById('pf-gpu-val');
    const cpuEl = document.getElementById('pf-cpu-val');
    const memEl = document.getElementById('pf-mem-val');
    const subEl = document.getElementById('pf-gpu-sub');
    if (gpuEl) gpuEl.textContent = gv != null ? fmt1(gv) + '%' : '—';
    if (cpuEl) cpuEl.textContent = cv != null ? fmt1(cv) + '%' : '—';
    if (memEl) memEl.textContent = mu != null ? fmt1(mu) + 'G' : '—';
    if (subEl) {
      const parts = [];
      if (g?.frequency_mhz != null) parts.push(g.frequency_mhz + 'MHz');
      if (g?.power_mw != null) parts.push((g.power_mw / 1000).toFixed(1) + 'W');
      subEl.textContent = parts.join('  ');
      subEl.style.display = parts.length ? '' : 'none';
    }
    miniEl.textContent =
      `${gv != null ? fmt0(gv) + '%' : '—'} · ${cv != null ? fmt0(cv) + '%' : '—'} · ${mu != null ? fmt1(mu) + 'G' : '—'}`;
    dotEl.className = 'pf-dot ' + (
      gv == null ? 'pf-dot-idle' : gv > 80 ? 'pf-dot-hot' : gv > 40 ? 'pf-dot-warm' : 'pf-dot-cool');
  }

  function updateLlmUI() {
    const show = llmToks != null || llmPre != null;
    llmSec.style.display = show ? '' : 'none';
    llmDiv.style.display = show ? '' : 'none';
    if (!show) return;
    const t = document.getElementById('pf-toks');
    const f = document.getElementById('pf-ttft');
    const p = document.getElementById('pf-pre');
    const n = document.getElementById('pf-n');
    if (t) t.textContent = llmToks != null ? Number(llmToks).toFixed(1) : '—';
    if (f) f.textContent = llmTtft != null ? Number(llmTtft).toFixed(0) + 'ms' : '—';
    if (p) p.textContent = llmPre != null ? Number(llmPre).toFixed(0) + 'ms' : '—';
    if (n) n.textContent = llmN != null ? String(totalN()) : '—';
  }

  function updateCacheUI() {
    const sec = document.getElementById('pf-cache');
    const div = document.getElementById('pf-cache-divider');
    // Show as soon as meta fires (pfxN > 0), even before any tokens
    const show = pfxN > 0 || llmN != null;
    if (sec) sec.style.display = show ? '' : 'none';
    if (div) div.style.display = show ? '' : 'none';
    if (!show) return;

    const decN = llmN ?? 0;
    const tn = pfxN + decN;           // total = prefill + decode
    const kvPre = kvMib(pfxN);           // KV from system+user prompt
    const kvDec = kvMib(decN);           // KV from generated tokens
    const kvTot = kvPre + kvDec;
    const maxKv = kvMib(MODEL.max_seq);

    // ── Context usage label ──────────────────────────────────────
    const ctxEl = document.getElementById('pf-cache-ctx');
    if (ctxEl) ctxEl.textContent = `${tn} / ${MODEL.max_seq}`;

    // ── Two-segment canvas bar ───────────────────────────────────
    // Left (dim): prefill (sys+user).  Right (bright): decode tokens.
    const cnv = document.getElementById('pf-kv-canvas');
    if (cnv) {
      const dpr = window.devicePixelRatio || 1;
      const cssW = cnv.clientWidth || (panel.offsetWidth - 100);
      const cssH = 4;
      cnv.width = Math.round(cssW * dpr);
      cnv.height = Math.round(cssH * dpr);
      const ctx = cnv.getContext('2d');
      ctx.resetTransform();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, cssW, cssH);
      // Track bg
      ctx.fillStyle = 'rgba(255,255,255,0.07)';
      ctx.roundRect(0, 0, cssW, cssH, 2);
      ctx.fill();
      if (tn > 0) {
        const preFrac = pfxN / MODEL.max_seq;
        const decFrac = decN / MODEL.max_seq;
        const preW = Math.min(cssW, preFrac * cssW);
        const decW = Math.min(cssW - preW, decFrac * cssW);
        // Prefill segment — muted violet
        if (preW > 0) {
          ctx.fillStyle = 'rgba(140,95,200,0.60)';
          ctx.fillRect(0, 0, preW, cssH);
        }
        // Decode segment — bright violet
        if (decW > 0) {
          ctx.fillStyle = 'rgba(175,130,240,0.90)';
          ctx.fillRect(preW, 0, decW, cssH);
        }
      }
    }

    // ── Value label ──────────────────────────────────────────────
    const kvEl = document.getElementById('pf-kv-val');
    if (kvEl) kvEl.textContent = fmtMib(kvTot);

    // ── Breakdown sub-labels ─────────────────────────────────────
    const preLbl = document.getElementById('pf-kv-pre-lbl');
    const decLbl = document.getElementById('pf-kv-dec-lbl');
    if (preLbl) preLbl.textContent = `pre ${fmtMib(kvPre)} (${pfxN}t)`;
    if (decLbl) decLbl.textContent = `dec ${fmtMib(kvDec)} (${decN}t)`;

    // ── SSM fixed bar ────────────────────────────────────────────
    const ssmPct = Math.min(100, (SSM_MIB / maxKv) * 100);
    setBar('pf-ssm-bar', ssmPct);
    const ssmEl = document.getElementById('pf-ssm-val');
    if (ssmEl) ssmEl.textContent = fmtMib(SSM_MIB);
  }

  // ── /api/token_count — exact prefill size from backend tokenizer ──
  // Called on reset() so the cache bar fills immediately showing sys+user cost.
  // Falls back gracefully if the endpoint isn't available (mock / no backend).
  let _tcAbort = null;
  function fetchTokenCount(sysPrompt, userMsg, history, reasoning) {
    if (_tcAbort) _tcAbort.abort();
    _tcAbort = new AbortController();
    const chatPort = location.port || (location.protocol === 'https:' ? '443' : '80');
    const base = `${location.protocol}//${location.hostname}:${chatPort}`;
    fetch(`${base}/api/token_count`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        system_prompt: sysPrompt || '',
        user_message: userMsg || '',
        history: history || [],
        reasoning: !!reasoning,
      }),
      signal: _tcAbort.signal,
    })
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (!d) return;
        pfxN = d.total_tokens || 0;
        updateCacheUI();
        if (expanded) chartDirty = true;
      })
      .catch(() => { });  // silently ignore network / abort errors
  }

  // ── Profiler WS ───────────────────────────────────────────────
  function ingestHistory(history) {
    if (!Array.isArray(history) || history.length === 0) return;
    const src = history.slice(-CHART_N);
    tsLen = src.length; tsPtr = 0;
    for (let i = 0; i < src.length; i++) {
      const h = src[i];
      tsGpu[i] = h.gpu?.usage_percent ?? 0;
      tsCpu[i] = h.cpu?.usage_percent ?? 0;
      tsMem[i] = h.memory?.used_gb ?? 0;
      tsTs[i] = h.timestamp ?? 0;
      tsPtr = i + 1;
    }
    // Append current KV value to its own ring (same timestamp cadence)
    tsKv[kvPtr] = kvMib(totalN());
    kvPtr = (kvPtr + 1) % CHART_N;
    kvLen = Math.min(kvLen + 1, CHART_N);
    chartDirty = true;
  }

  function connectProfiler() {
    if (profWs && profWs.readyState <= 1) return;
    const url = `ws://${location.hostname}:${PROFILER_PORT}/ws`;
    try { profWs = new WebSocket(url); } catch { scheduleRetry(); return; }
    profWs.onmessage = ev => {
      let d; try { d = JSON.parse(ev.data); } catch { return; }
      if (!d.snapshot) return;
      sysGpu = d.snapshot.gpu || null;
      sysCpu = d.snapshot.cpu || null;
      sysMem = d.snapshot.memory || null;
      updateSysUI();
      if (d.history) ingestHistory(d.history);
    };
    profWs.onclose = profWs.onerror = () => { profWs = null; scheduleRetry(); };
  }

  function scheduleRetry() {
    clearTimeout(profRetry);
    profRetry = setTimeout(connectProfiler, 5000);
  }
  setTimeout(connectProfiler, 800);

  // ── Public API ────────────────────────────────────────────────
  // reset(promptTokens)              — called from meta event (server gives exact count)
  // reset(null, {sys, user, history, reasoning}) — proactive call before turn starts
  function reset(promptTokens, ctx) {
    llmN = null;
    llmToks = null;
    const t = document.getElementById('pf-toks');
    if (t) { t.textContent = '—'; t.classList.remove('pf-val-live'); }

    if (promptTokens != null) {
      // Server already gave us the exact count — use it immediately
      pfxN = +promptTokens;
      updateLlmUI();
      updateCacheUI();
      if (expanded) chartDirty = true;
    } else if (ctx) {
      // Proactive call: optimistically show 0 then update when backend replies
      pfxN = 0;
      updateLlmUI();
      updateCacheUI();
      fetchTokenCount(ctx.sys, ctx.user, ctx.history, ctx.reasoning);
    } else {
      pfxN = 0;
      updateLlmUI();
      updateCacheUI();
      if (expanded) chartDirty = true;
    }
  }

  function tick(tok_s, n) {
    if (tok_s != null) {
      llmToks = tok_s;
      sparkBuf[sparkPtr] = +tok_s;
      sparkPtr = (sparkPtr + 1) % SPARK_N;
      sparkDirty = true;
      const el = document.getElementById('pf-toks');
      if (el) { el.textContent = Number(tok_s).toFixed(1); el.classList.add('pf-val-live'); }
    }
    if (n != null) {
      llmN = n;
      const el = document.getElementById('pf-n');
      if (el) el.textContent = String(totalN());
    }
    updateLlmUI();
    updateCacheUI();

    // Keep the chart tip live: overwrite the most-recent KV slot so the
    // rightmost point on the chart moves with each token, not just at 1 Hz.
    if (kvLen > 0) {
      const tip = (kvPtr - 1 + CHART_N) % CHART_N;
      tsKv[tip] = kvMib(totalN());
    }
    if (expanded) chartDirty = true;
  }

  function done(tok_s, pre, ttft, n) {
    llmToks = tok_s ?? llmToks;
    llmPre = pre ?? llmPre;
    llmTtft = ttft ?? llmTtft;
    llmN = n ?? llmN;
    const el = document.getElementById('pf-toks');
    if (el) el.classList.remove('pf-val-live');
    updateLlmUI();
    updateCacheUI();
    // Commit final KV value to the chart tip
    if (kvLen > 0) {
      const tip = (kvPtr - 1 + CHART_N) % CHART_N;
      tsKv[tip] = kvMib(totalN());
    }
    if (expanded) chartDirty = true;
  }

  window.PF = {
    reset, tick, done,
    show: () => { panel.style.display = ''; save(); },
    hide: () => { panel.style.display = 'none'; save(); },
    toggle: () => { panel.style.display === 'none' ? window.PF.show() : window.PF.hide(); },
    model: MODEL,
  };
})();
