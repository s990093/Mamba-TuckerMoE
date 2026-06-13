'use strict';
/* ── Model Showdown — compare.js ───────────────────────────────── */

const PROMPT = 'Who are you?';
const WS_URL = `ws://${location.host}/ws`;

// ── Mock data — real cloud API timing benchmarks ──────────────────
//
// GPT-4o (flagship):          55–120 tok/s commercial standard → sim 88 tok/s
//   TTFT: 350–550 ms network latency
//   CoT:  fast inline reasoning, ~0.5s
//
// Claude Sonnet 4.6 (Extended Thinking / Adaptive):
//   Thinking-mode generation:  30–50 tok/s (slower while thinking)
//   Output after thinking:     40–110 tok/s → sim 72 tok/s
//   TTFT: 500–750 ms network + 3–5 s thinking (Low budget, moderate question)
//   Simple questions may skip thinking; "Who are you?" triggers brief thinking pass

const MOCK_OPENAI = {
  net_delay_ms: 420,    // API round-trip before first byte
  cot_chars_s:  260,    // CoT chars/s — GPT-4o fast inline reasoning
  resp_tok_s:   88,     // output tok/s — mid of 55–120 range
  cot:
`Step 1: Identify request — user asks for self-identification.
Step 2: Apply system persona — operating as Mamba AI.
Step 3: Key attributes to surface — local inference, SSM architecture, TuckerMoE.
Step 4: Structure response — direct, confident, one clear value proposition.`,
  response:
`I'm Mamba — an AI assistant engineered for speed and precision on Apple Silicon.

Unlike cloud-hosted models, I run entirely on your device using a Hybrid Mamba-3 architecture (State Space Model + Tucker-decomposed Mixture-of-Experts). No network latency, no API costs, full privacy.

I reason before responding, write clearly, and get things done. What would you like to work on?`,
};

const MOCK_CLAUDE = {
  net_delay_ms: 560,    // Anthropic API latency
  cot_chars_s:  48,     // slow CoT — extended thinking at 30–50 tok/s → ~3–4 s total
  resp_tok_s:   72,     // output tok/s after thinking — mid of 40–110 range
  cot:
`The user is asking me to identify myself. This is a moderate self-awareness question — I'll use a short thinking pass rather than skipping entirely.

Let me consider what's genuinely important to convey:
— Core identity: I'm Mamba, a local AI on Apple Silicon
— Architectural differentiator: SSM + Tucker MoE (not a standard transformer)
— Privacy by design: no cloud routing, no API dependency
— Tone: warm, direct, not a spec sheet

"Who are you?" is a trust-building moment. I should be honest and specific without overwhelming the user. Two short paragraphs — identity + invitation — feels right.`,
  response:
`I'm Mamba — an AI assistant that runs locally on your device, not in a data centre.

I'm built on a Hybrid Mamba-3 architecture: a State Space Model combined with Tucker-decomposed Mixture-of-Experts, giving me broad reasoning capacity at 417M parameters — a fraction of cloud model sizes — running fully offline on Apple Silicon.

I think carefully before I answer, I respect your privacy by design, and I'm optimised for the hardware you already own. What would you like to explore?`,
};

// ── DOM refs ──────────────────────────────────────────────────────
const $start = document.getElementById('cmp-start');
const $result = document.getElementById('cmp-result');

function $(id) { return document.getElementById(id); }

// ── State ─────────────────────────────────────────────────────────
let running = false;
let ws = null;

// ── Utilities ─────────────────────────────────────────────────────
function setStatus(id, cls, label) {
  const el = $(`status-${id}`);
  if (!el) return;
  el.className = `cmp-status ${cls}`;
  el.textContent = label;
}

function showThink(id) {
  const blk = $(`think-${id}`);
  if (blk) blk.hidden = false;
}

function appendThink(id, text) {
  const el = $(`think-body-${id}`);
  if (!el) return;
  el.textContent += text;
  el.scrollTop = el.scrollHeight;
}

function doneThink(id, ms) {
  const blk = $(`think-${id}`);
  if (blk) blk.classList.add('done');
  const lbl = $(`think-lbl-${id}`);
  if (lbl) lbl.textContent = `Thought for ${(ms / 1000).toFixed(1)}s`;
  const tim = $(`think-time-${id}`);
  if (tim) tim.textContent = `${ms.toFixed(0)}ms`;
}

let _cursors = {};
function appendResp(id, text) {
  const el = $(`resp-${id}`);
  if (!el) return;
  const ph = el.querySelector('.resp-idle, .cmp-placeholder');
  if (ph) ph.remove();
  const cur = _cursors[id];
  if (cur && cur.parentNode === el) el.removeChild(cur);
  // Split on newlines → insert <br> so paragraphs render correctly
  const parts = text.split('\n');
  parts.forEach((part, i) => {
    if (part) el.appendChild(document.createTextNode(part));
    if (i < parts.length - 1) el.appendChild(document.createElement('br'));
  });
  const newCur = document.createElement('span');
  newCur.className = 'cmp-cursor';
  el.appendChild(newCur);
  _cursors[id] = newCur;
  el.scrollTop = el.scrollHeight;
}

function doneResp(id) {
  const cur = _cursors[id];
  if (cur) cur.remove();
  delete _cursors[id];
}

function setStats(id, ttft, toks, total) {
  if (ttft != null) $(`ttft-${id}`).textContent = ttft != null ? `${Math.round(ttft)}ms` : '—';
  if (toks != null) $(`toks-${id}`).textContent = toks != null ? Number(toks).toFixed(1) : '—';
  if (total != null) $(`total-${id}`).textContent = total != null ? total : '—';
}

function updateSpeedBar(id, tokS, maxTokS) {
  const fill = $(`bar-${id}`);
  if (!fill) return;
  fill.style.width = `${Math.min(100, (tokS / maxTokS) * 100).toFixed(1)}%`;
}

// ── Collapsible think blocks ──────────────────────────────────────
function initThinkToggles() {
  ['mamba', 'openai', 'claude'].forEach(id => {
    const btn = $(`think-toggle-${id}`);
    const blk = $(`think-${id}`);
    if (!btn || !blk) return;
    btn.addEventListener('click', () => blk.classList.toggle('collapsed'));
  });
}

// ── System prompt loader ──────────────────────────────────────────
async function loadSysPrompt() {
  const toggleBtn = document.getElementById('sys-toggle');
  const body = document.getElementById('sys-body');
  const bodyText = document.getElementById('sys-body-text');
  const chevron = document.getElementById('sys-chevron');
  if (!toggleBtn || !body) return;

  toggleBtn.addEventListener('click', () => {
    const hidden = body.hidden;
    body.hidden = !hidden;
    if (chevron) chevron.textContent = hidden ? '▴' : '▾';
    toggleBtn.textContent = (hidden ? 'Hide' : 'Show') + ' system prompt ';
    if (chevron) toggleBtn.appendChild(chevron);
  });

  try {
    const res = await fetch('/api/demo-config');
    const data = await res.json();
    const prompts = data.category_system_prompts || {};
    const text = prompts['self_awareness'] || '(not found)';
    if (bodyText) bodyText.textContent = text;
  } catch {
    if (bodyText) bodyText.textContent = '(could not load)';
  }
}

// ── Simulated cloud runner ────────────────────────────────────────
// Phase 1: net_delay_ms   — API in-flight (CONNECTING, nothing shown)
// Phase 2: cot_chars_s    — CoT streaming (per-model speed)
// Phase 3: resp_tok_s     — response streaming (per-model speed)
function runSimulated(id, mock, onDone) {
  const startMs = Date.now();
  let tokCount = 0;
  const cotCps  = mock.cot_chars_s  || 120;   // CoT chars/s
  const respTps  = mock.resp_tok_s   || 60;    // response tok/s
  const avgCharsPerTok = 4.5;

  setStatus(id, 'think', 'CONNECTING');
  $(`col-${id}`)?.classList.add('running');

  // Phase 1 — network round-trip delay (cloud latency before first byte)
  setTimeout(() => {
    setStatus(id, 'think', 'THINKING');
    showThink(id);

    const cotChars = mock.cot.split('');
    let cotIdx = 0;
    const thinkTimer = setInterval(() => {
      if (cotIdx >= cotChars.length) {
        clearInterval(thinkTimer);
        doneThink(id, Date.now() - startMs);

        // Phase 3 — response streaming (TTFT measured from click start)
        setStatus(id, 'gen', 'GENERATING');
        const ttft = Date.now() - startMs;
        setStats(id, ttft, null, null);

        const words = mock.response.split(/(?<=\s)|(?=\s)/);
        let wIdx = 0;
        const genTimer = setInterval(() => {
          if (wIdx >= words.length) {
            clearInterval(genTimer);
            const totalTok = Math.round(mock.response.length / avgCharsPerTok);
            const elapsed = (Date.now() - startMs - ttft) / 1000;
            const tokS = elapsed > 0 ? (totalTok / elapsed).toFixed(1) : respTps.toFixed(1);
            setStats(id, ttft, tokS, totalTok);
            updateSpeedBar(id, parseFloat(tokS), 150);
            doneResp(id);
            setStatus(id, 'done', 'DONE');
            $(`col-${id}`)?.classList.remove('running');
            $(`col-${id}`)?.classList.add('done');
            onDone({ id, ttft, tokS: parseFloat(tokS), totalTok });
            return;
          }
          const chunk = words.slice(wIdx, wIdx + 1).join('');
          appendResp(id, chunk);
          tokCount += chunk.length / avgCharsPerTok; // approximate tok count
          wIdx++;
          // Show the configured tok/s speed directly (stable display)
          setStats(id, null, respTps.toFixed(1), null);
          updateSpeedBar(id, respTps, 150);
        }, 1000 / (respTps * avgCharsPerTok));
        return;
      }
      appendThink(id, cotChars[cotIdx]);
      cotIdx++;
    }, 1000 / cotCps);
  }, mock.net_delay_ms);
}

// ── Mamba local runner (via WebSocket) ───────────────────────────
function runMamba(onDone) {
  const id = 'mamba';
  const startMs = Date.now();
  let thinkStart = null;
  let firstTokMs = null;
  let tokCount = 0;
  // raw_sampling router: <think> is in the PROMPT (not a generated token)
  // stream: [CoT tokens] → </think> → <final> → [answer] → </final>
  let rawMode = 'think'; // start in think phase immediately
  const RAW_TAGS = { '</think>': 'between', '<final>': 'final', '</final>': 'done' };

  function rawRoute(text) {
    if (!text) return;
    for (const [tag, next] of Object.entries(RAW_TAGS)) {
      const idx = text.indexOf(tag);
      if (idx < 0) continue;
      if (idx > 0) rawRoute(text.slice(0, idx));
      rawMode = next;
      if (next === 'final') {
        // </think> → <final> transition: close think block, open response
        doneThink(id, Date.now() - (thinkStart || startMs));
        setStatus(id, 'gen', 'GENERATING');
        if (firstTokMs === null) {
          firstTokMs = Date.now() - startMs;
          setStats(id, firstTokMs, null, null);
        }
      }
      rawRoute(text.slice(idx + tag.length));
      return;
    }
    if (rawMode === 'think') {
      appendThink(id, text);
    } else if (rawMode === 'final') {
      appendResp(id, text);
      tokCount++;
    }
  }

  setStatus(id, 'think', 'THINKING');
  $(`col-${id}`)?.classList.add('running');

  if (ws) { try { ws.close(); } catch { } }
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    ws.send(JSON.stringify({
      action: 'chat',
      prompt: PROMPT,
      max_tokens: 256,
      reasoning: true,
      sampling: { temperature: 0.7, top_p: 0.9 },
      category_key: 'self_awareness',
    }));
  };

  ws.onmessage = ev => {
    try {
      const m = JSON.parse(ev.data);
      if (m.type === 'meta') {
        thinkStart = Date.now();
        showThink(id);
      } else if (m.type === 'reasoning_token') {
        // Non-raw path fallback (reasoning events from non-raw_sampling mode)
        appendThink(id, m.text || '');
      } else if (m.type === 'token') {
        rawRoute(m.text || '');
        if (m.tok_s != null && rawMode === 'final') {
          setStats(id, null, m.tok_s.toFixed(1), null);
          updateSpeedBar(id, m.tok_s, 150);
        }
      } else if (m.type === 'done') {
        const ttft = firstTokMs ?? (Date.now() - startMs);
        const totalTok = m.total_tokens ?? tokCount;
        const tokS = m.tok_s ?? 0;
        setStats(id, ttft, Number(tokS).toFixed(1), totalTok);
        updateSpeedBar(id, tokS, 150);
        doneResp(id);
        setStatus(id, 'done', 'DONE');
        $(`col-${id}`)?.classList.remove('running');
        $(`col-${id}`)?.classList.add('done');
        onDone({ id, ttft, tokS, totalTok });
      }
    } catch { }
  };

  ws.onerror = () => {
    setStatus(id, 'done', 'ERROR');
    $(`resp-${id}`).textContent = 'WebSocket error — is the server running?';
    onDone({ id, ttft: null, tokS: 0, totalTok: 0 });
  };
}

// ── Reset all columns ─────────────────────────────────────────────
function resetAll() {
  ['mamba', 'openai', 'claude'].forEach(id => {
    setStatus(id, '', 'READY');
    const blk = $(`think-${id}`);
    if (blk) { blk.hidden = true; blk.classList.remove('done'); }
    const body = $(`think-body-${id}`);
    if (body) body.textContent = '';
    const lbl = $(`think-lbl-${id}`);
    if (lbl) lbl.textContent = 'Thinking…';
    const tim = $(`think-time-${id}`);
    if (tim) tim.textContent = '';
    const resp = $(`resp-${id}`);
    if (resp) { resp.innerHTML = '<span class="resp-idle">Waiting to start…</span>'; }
    setStats(id, null, null, null);
    $(`ttft-${id}`).textContent = '—';
    $(`toks-${id}`).textContent = '—';
    $(`total-${id}`).textContent = '—';
    updateSpeedBar(id, 0, 150);
    const col = $(`col-${id}`);
    if (col) { col.classList.remove('running', 'done'); }
  });
  $result.hidden = true;
  delete _cursors['mamba'];
  delete _cursors['openai'];
  delete _cursors['claude'];
}

// ── Show results banner ───────────────────────────────────────────
const _results = {};
function onModelDone(data) {
  _results[data.id] = data;
  if (Object.keys(_results).length < 3) return;

  // All three done — show winner banner
  $result.hidden = false;
  running = false;
  $start.disabled = false;
  $start.textContent = '↺ Run Again';

  const models = Object.values(_results);
  const fastestTok = models.reduce((a, b) => a.tokS > b.tokS ? a : b);
  const fastestTtft = models.reduce((a, b) => (a.ttft ?? 99999) < (b.ttft ?? 99999) ? a : b);

  const nameMap = { mamba: 'Mamba 3', openai: 'GPT-4o', claude: 'Claude' };
  $('winner-speed-val').textContent = `${nameMap[fastestTok.id]} (${Number(fastestTok.tokS).toFixed(1)} tok/s)`;
  $('winner-ttft-val').textContent = `${nameMap[fastestTtft.id]} (${Math.round(fastestTtft.ttft)}ms)`;

  const mambaSpd = _results.mamba?.tokS ?? 0;
  const cloudAvg = ((_results.openai?.tokS ?? 0) + (_results.claude?.tokS ?? 0)) / 2;
  if (mambaSpd > 0 && cloudAvg > 0) {
    const ratio = (mambaSpd / cloudAvg).toFixed(1);
    $('winner-local-val').textContent = `Mamba ${ratio}× vs cloud avg`;
  }
}

// ── Start benchmark ───────────────────────────────────────────────
$start.addEventListener('click', () => {
  if (running) return;
  running = true;
  Object.keys(_results).forEach(k => delete _results[k]);
  resetAll();

  $start.disabled = true;
  $start.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21"/></svg> Running…`;

  // Launch all three simultaneously
  runMamba(onModelDone);
  setTimeout(() => runSimulated('openai', MOCK_OPENAI, onModelDone), 50);
  setTimeout(() => runSimulated('claude', MOCK_CLAUDE, onModelDone), 100);
});

// ── Init ──────────────────────────────────────────────────────────
initThinkToggles();
loadSysPrompt();
