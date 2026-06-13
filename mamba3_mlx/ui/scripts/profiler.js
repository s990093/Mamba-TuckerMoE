'use strict';

const WS_URL = `ws://${location.host}/ws`;
const $ = (id) => document.getElementById(id);

// ── Colour thresholds for GPU usage ──────────────────
function gpuColor(pct) {
  if (pct === null || pct === undefined || Number.isNaN(pct)) return '#55535b';
  if (pct >= 90) return '#eb4e4e';
  if (pct >= 70) return '#eb8a3e';
  if (pct >= 30) return '#e9b847';
  return '#44c97a';
}

function gpuGlow(pct) {
  if (pct === null || pct === undefined || Number.isNaN(pct)) return 'transparent';
  if (pct >= 90) return 'rgba(235,78,78,0.45)';
  if (pct >= 70) return 'rgba(235,138,62,0.40)';
  if (pct >= 30) return 'rgba(233,184,71,0.35)';
  return 'rgba(68,201,122,0.40)';
}

// ── Circular gauge ───────────────────────────────────
function drawGauge(canvas, pct) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2, cy = h / 2;
  const radius = 100;
  const lineWidth = 10;
  const startAngle = Math.PI * 0.72;
  const endAngle = Math.PI * 2.28;
  const span = endAngle - startAngle;

  ctx.clearRect(0, 0, w, h);

  // Track
  ctx.beginPath();
  ctx.arc(cx, cy, radius, startAngle, endAngle);
  ctx.strokeStyle = '#1a1a24';
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.stroke();

  if (pct !== null && pct !== undefined && !Number.isNaN(pct)) {
    const clamped = Math.min(100, Math.max(0, pct));
    const fillAngle = startAngle + (clamped / 100) * span;
    const color = gpuColor(clamped);

    // Glow (outer)
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, fillAngle);
    ctx.strokeStyle = gpuGlow(clamped);
    ctx.lineWidth = lineWidth + 6;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Main arc
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, fillAngle);
    const grad = ctx.createConicalGradient
      ? ctx.createConicalGradient(startAngle, cx, cy)
      : null;
    if (grad) {
      grad.addColorStop(0, color);
      grad.addColorStop(1, color);
      ctx.strokeStyle = grad;
    } else {
      ctx.strokeStyle = color;
    }
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Tick marks
    for (let i = 0; i <= 100; i += 25) {
      const a = startAngle + (i / 100) * span;
      const inner = radius - lineWidth / 2 - 6;
      const outer = radius - lineWidth / 2 - 1;
      const x1 = cx + Math.cos(a) * inner;
      const y1 = cy + Math.sin(a) * inner;
      const x2 = cx + Math.cos(a) * outer;
      const y2 = cy + Math.sin(a) * outer;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = '#2a2a38';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
}

// ── Sparkline with gradient fill ─────────────────────
function drawSpark(canvas, values, colorFn) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.clientWidth;
  const h = canvas.height;
  canvas.width = w;
  const nums = values.filter((x) => x !== null && x !== undefined && !Number.isNaN(x));
  if (nums.length < 2) return;
  const min = Math.min(...nums);
  const max = Math.max(...nums);
  const span = Math.max(max - min, 1e-6);
  const padX = 2, padY = 4;

  ctx.clearRect(0, 0, w, h);

  // Gradient fill beneath line
  const points = nums.map((v, i) => ({
    x: (i / (nums.length - 1)) * (w - padX * 2) + padX,
    y: h - padY - ((v - min) / span) * (h - padY * 2),
  }));

  ctx.beginPath();
  ctx.moveTo(points[0].x, h);
  points.forEach((p) => ctx.lineTo(p.x, p.y));
  ctx.lineTo(points[points.length - 1].x, h);
  ctx.closePath();

  const fillGrad = ctx.createLinearGradient(0, 0, 0, h);
  const baseColor = typeof colorFn === 'function' ? colorFn(nums[nums.length - 1]) : (colorFn || '#5b9bd5');
  fillGrad.addColorStop(0, hexToRgba(baseColor, 0.18));
  fillGrad.addColorStop(1, hexToRgba(baseColor, 0.02));
  ctx.fillStyle = fillGrad;
  ctx.fill();

  // Line
  ctx.beginPath();
  points.forEach((p, i) => {
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.strokeStyle = baseColor;
  ctx.lineWidth = 1.6;
  ctx.lineJoin = 'round';
  ctx.stroke();

  // End dot
  const last = points[points.length - 1];
  ctx.beginPath();
  ctx.arc(last.x, last.y, 3.5, 0, Math.PI * 2);
  ctx.fillStyle = baseColor;
  ctx.fill();
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Helpers ──────────────────────────────────────────
function fmt(v, digits) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(digits !== undefined ? digits : 1);
}

function historySeries(history, pick) {
  return (history || []).map(pick);
}

const PEAK_FREQ_MHZ = 1398;

// ── Render ───────────────────────────────────────────
function renderEnvelope(env) {
  const s = env.snapshot || {};
  const h = env.history || [];

  const gpu = s.gpu || {};
  const cpu = s.cpu || {};
  const mem = s.memory || {};
  const swap = s.swap || {};
  const thermal = s.thermal || {};
  const llm = s.llm || {};
  const state = s.state || {};

  // ── GPU Hero ─────────────────────────────────────
  const gpuUsage = gpu.usage_percent;
  const colorHex = gpuColor(gpuUsage);

  $('gpu-val').textContent = gpuUsage !== null && gpuUsage !== undefined && !Number.isNaN(gpuUsage)
    ? fmt(gpuUsage) : '—';
  $('gpu-val').style.color = colorHex;

  drawGauge($('gauge-canvas'), gpuUsage);

  // Frequency
  const freq = gpu.frequency_mhz;
  $('gpu-freq').textContent = freq ?? '—';
  const freqPct = freq ? Math.min(100, (freq / PEAK_FREQ_MHZ) * 100) : 0;
  $('freq-bar').style.width = freqPct + '%';

  // Power
  const power = gpu.power_mw;
  $('gpu-power').textContent = power ?? '—';
  const powerPct = power ? Math.min(100, (power / 5000) * 100) : 0;
  $('power-bar').style.width = powerPct + '%';

  // Idle
  const idle = gpu.idle_percent;
  $('gpu-idle').textContent = idle !== null && idle !== undefined && !Number.isNaN(idle)
    ? fmt(idle) : '—';

  const idleBadge = $('idle-badge');
  if (idle !== null && idle !== undefined && !Number.isNaN(idle)) {
    if (idle < 85) {
      idleBadge.textContent = 'active';
      idleBadge.className = 'idle-badge active';
    } else {
      idleBadge.textContent = 'idle';
      idleBadge.className = 'idle-badge';
    }
  } else {
    idleBadge.textContent = 'unknown';
    idleBadge.className = 'idle-badge';
  }

  // GPU sparkline
  drawSpark($('chart-gpu'), historySeries(h, (x) => x.gpu?.usage_percent), gpuColor);

  // ── CPU ──────────────────────────────────────────
  $('cpu-usage').textContent = fmt(cpu.usage_percent);
  drawSpark($('chart-cpu'), historySeries(h, (x) => x.cpu?.usage_percent), '#8ec5ff');

  // ── Memory ───────────────────────────────────────
  $('mem-used').textContent = mem.used_gb ?? '—';
  $('mem-pct').textContent = fmt(mem.percent);
  drawSpark($('chart-mem'), historySeries(h, (x) => x.memory?.percent), '#6fbf8f');

  // ── Swap ─────────────────────────────────────────
  $('swap-used').textContent = swap.used_gb ?? '—';
  $('swap-pct').textContent = fmt(swap.percent);

  // ── Thermal ──────────────────────────────────────
  $('thermal-pressure').textContent = thermal.pressure ?? '—';
  const throttle = $('thermal-throttle');
  if (thermal.throttling === true) {
    throttle.textContent = 'throttling';
    throttle.className = 'badge warn';
  } else if (thermal.throttling === false) {
    throttle.textContent = 'nominal';
    throttle.className = 'badge ok';
  } else {
    throttle.textContent = 'unknown';
    throttle.className = 'badge';
  }

  // ── LLM / Inference ─────────────────────────────
  const tps = llm.tokens_per_sec;
  const el = $('llm-tps');
  el.textContent = fmt(tps);
  if (tps !== null && tps !== undefined && !Number.isNaN(tps) && tps > 0) {
    el.className = 'val tps-active';
  } else {
    el.className = 'val';
  }

  $('llm-decode').textContent = fmt(llm.decode_tps);
  $('llm-prefill').textContent = fmt(llm.prompt_tps);
  $('llm-latency').textContent = fmt(llm.latency_ms, 0);
  $('llm-ctx').textContent = llm.context_length ?? '—';
  $('llm-phase').textContent = `${state.model || 'Mamba3-XR'}  ·  ${state.phase || 'idle'}`;

  drawSpark($('chart-llm'), historySeries(h, (x) => x.llm?.decode_tps ?? x.llm?.tokens_per_sec), '#e0b35a');
}

// ── WebSocket ────────────────────────────────────────
function connect() {
  const status = $('conn-status');
  const dot = status.querySelector('.dot');
  const text = status.querySelector('.status-text');
  const ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    text.textContent = 'connected';
    status.className = 'status ok';
  };

  ws.onclose = () => {
    text.textContent = 'reconnecting';
    status.className = 'status bad';
    setTimeout(connect, 2000);
  };

  ws.onerror = () => {
    text.textContent = 'error';
    status.className = 'status bad';
  };

  ws.onmessage = (event) => {
    let data;
    try { data = JSON.parse(event.data); } catch { return; }
    if (data.type === 'connected') return;
    if (data.schema_version && data.snapshot) {
      renderEnvelope(data);
    }
  };
}

connect();
