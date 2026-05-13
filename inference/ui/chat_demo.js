const $ = (s) => document.querySelector(s);
const chatScroll = $('#chat-scroll');
const welcomeWrap = $('#welcome-wrap');
const input = $('#prompt-input');
const sendBtn = $('#send-btn');
const wsPill = $('#ws-pill');
const wsLabel = $('#ws-label');
const maxTokensSlider = $('#max-tokens');
const maxTokensVal = $('#max-tokens-val');
const drawer = $('#drawer');
const overlay = $('#drawer-overlay');
const mockBanner = $('#mock-banner');
const sidebar = $('#sidebar');
const sidebarCat = $('#sidebar-categories');
const systemPromptMd = $('#system-prompt-md');
const sysCard = $('#sys-card');
const sysCardBody = $('#sys-card-body');
const sysCatSelect = $('#sys-cat-select');
const btnPlaySys = $('#btn-play-sys');
const modePill = $('#mode-pill');
const modePillLabel = $('#mode-pill-label');
const modePillIcon = $('#mode-pill-icon');
const modeMenu = $('#mode-menu');
const jumpBtn = $('#jump-bottom');

/** Auto-scroll controller.
 *  - `autoScroll` follows streaming output as long as the user is parked near
 *    the bottom of `chatScroll`.
 *  - If the user scrolls up while a turn is streaming, we stop fighting them
 *    and reveal the "Jump to latest" pill instead.
 *  - All scroll writes go through `nudgeScroll()` which uses rAF to wait until
 *    after the DOM has been morphed by `streamRenderInto`. */
const NEAR_BOTTOM_PX = 96;
let autoScroll = true;
let _scrollPending = false;
let _suppressScrollEvent = 0;

function isAtBottom() {
  if (!chatScroll) return true;
  return (chatScroll.scrollHeight - chatScroll.scrollTop - chatScroll.clientHeight) <= NEAR_BOTTOM_PX;
}

function updateJumpBtn() {
  if (!jumpBtn) return;
  const show = !autoScroll;
  jumpBtn.classList.toggle('show', show);
  if (show) jumpBtn.removeAttribute('hidden');
  else jumpBtn.setAttribute('hidden', '');
}

function nudgeScroll(force = false) {
  if (!chatScroll) return;
  if (!force && !autoScroll) return;
  if (_scrollPending) return;
  _scrollPending = true;
  requestAnimationFrame(() => {
    _suppressScrollEvent++;
    chatScroll.scrollTop = chatScroll.scrollHeight;
    requestAnimationFrame(() => {
      chatScroll.scrollTop = chatScroll.scrollHeight;
      _scrollPending = false;
      requestAnimationFrame(() => { _suppressScrollEvent = Math.max(0, _suppressScrollEvent - 1); });
    });
  });
}

function forceFollow() {
  autoScroll = true;
  updateJumpBtn();
  nudgeScroll(true);
}

chatScroll?.addEventListener('scroll', () => {
  if (_suppressScrollEvent > 0) return;
  autoScroll = isAtBottom();
  updateJumpBtn();
}, { passive: true });

chatScroll?.addEventListener('wheel', () => {
  if (!isAtBottom()) {
    autoScroll = false;
    updateJumpBtn();
  }
}, { passive: true });

chatScroll?.addEventListener('touchmove', () => {
  if (!isAtBottom()) {
    autoScroll = false;
    updateJumpBtn();
  }
}, { passive: true });

jumpBtn?.addEventListener('click', forceFollow);

window.addEventListener('resize', () => nudgeScroll());
const samplingGrid = $('#sampling-grid');
const samplingJsonEl = $('#sampling-json');
const btnSamplingReset = $('#btn-sampling-reset');

/** Sampling controls — keep in sync with `chat_demo.py` CLI args. */
const SAMPLING_FIELDS = [
  { key: 'temperature', label: 'Temperature', cli: '--temp',     min: 0,    max: 1.5, step: 0.01, decimals: 2 },
  { key: 'top_k',       label: 'Top-K',       cli: '--top_k',    min: 1,    max: 200, step: 1,    decimals: 0 },
  { key: 'top_p',       label: 'Top-P',       cli: '--top_p',    min: 0,    max: 1,   step: 0.01, decimals: 2 },
  { key: 'min_p',       label: 'Min-P',       cli: '--min_p',    min: 0,    max: 0.5, step: 0.005,decimals: 3 },
  { key: 'repetition_penalty', label: 'Rep. pen.', cli: '--rep_pen', min: 1, max: 1.6, step: 0.01, decimals: 2 },
];
/** Mutable current values, sent with every chat payload as `sampling: {...}`. */
let samplingState = {};
/** CLI defaults from `/api/demo-config.sampling_defaults` for the Reset button. */
let samplingDefaults = {};

/** Response-mode dropdown.
 *  - 'think'  → reasoning ON, prompt injects <think> after the assistant marker.
 *  - 'direct' → reasoning OFF, model answers immediately, no <think> block.
 *  The chosen mode drives the `reasoning` flag in every chat WS payload. */
const MODE_LABELS = { think: 'Thinking', direct: 'Direct' };
const MODE_ICONS = {
  think:
    '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a7 7 0 00-4.95 11.95c.6.6.95 1.41.95 2.26V17h8v-.79c0-.85.35-1.66.95-2.26A7 7 0 0012 2z"/></svg>',
  direct:
    '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
};
let currentMode = 'think';

function isReasoningOn() {
  return currentMode === 'think';
}

function applyModeToPill() {
  if (modePillLabel) modePillLabel.textContent = MODE_LABELS[currentMode] || 'Thinking';
  if (modePillIcon) modePillIcon.innerHTML = MODE_ICONS[currentMode] || MODE_ICONS.think;
  if (modePill) {
    modePill.classList.toggle('thinking', currentMode === 'think');
    modePill.title = currentMode === 'think'
      ? 'Thinking mode · prompt injects <think>…</think> before the answer'
      : 'Direct mode · skip <think>, answer immediately';
  }
  modeMenu?.querySelectorAll('.mode-item').forEach((it) => {
    const active = it.dataset.mode === currentMode;
    it.classList.toggle('active', active);
    it.setAttribute('aria-selected', active ? 'true' : 'false');
  });
}

function setMode(mode) {
  if (mode !== 'think' && mode !== 'direct') return;
  currentMode = mode;
  applyModeToPill();
}

function setReasoning(on) {
  setMode(on ? 'think' : 'direct');
}

function openModeMenu() {
  if (!modeMenu || !modePill) return;
  modeMenu.hidden = false;
  modePill.setAttribute('aria-expanded', 'true');
}

function closeModeMenu() {
  if (!modeMenu || !modePill) return;
  modeMenu.hidden = true;
  modePill.setAttribute('aria-expanded', 'false');
}

function toggleModeMenu() {
  if (!modeMenu) return;
  if (modeMenu.hidden) openModeMenu(); else closeModeMenu();
}

modePill?.addEventListener('click', (e) => {
  e.stopPropagation();
  toggleModeMenu();
});

modeMenu?.querySelectorAll('.mode-item').forEach((it) => {
  it.addEventListener('click', (e) => {
    e.stopPropagation();
    setMode(it.dataset.mode || 'think');
    closeModeMenu();
  });
});

document.addEventListener('click', (e) => {
  if (!modeMenu || modeMenu.hidden) return;
  if (modeMenu.contains(e.target) || modePill?.contains(e.target)) return;
  closeModeMenu();
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && modeMenu && !modeMenu.hidden) closeModeMenu();
});

applyModeToPill();

function formatSamplingValue(field, value) {
  const v = Number(value);
  if (!Number.isFinite(v)) return String(value);
  return field.decimals > 0 ? v.toFixed(field.decimals) : String(Math.round(v));
}

function refreshSamplingJson() {
  if (!samplingJsonEl) return;
  samplingJsonEl.textContent = JSON.stringify(samplingState);
}

function renderSamplingControls() {
  if (!samplingGrid) return;
  samplingGrid.innerHTML = '';
  SAMPLING_FIELDS.forEach((f) => {
    const row = document.createElement('div');
    row.className = 'sampling-row';
    row.dataset.field = f.key;
    const label = document.createElement('div');
    label.className = 'sr-name';
    label.innerHTML = `${f.label}<span class="sr-key">${f.cli}</span>`;
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = String(f.min);
    slider.max = String(f.max);
    slider.step = String(f.step);
    const cur = samplingState[f.key] != null ? Number(samplingState[f.key]) : Number(samplingDefaults[f.key] ?? f.min);
    slider.value = String(cur);
    const val = document.createElement('div');
    val.className = 'sr-val';
    val.textContent = formatSamplingValue(f, cur);
    slider.addEventListener('input', () => {
      const v = f.decimals > 0 ? Number(slider.value) : parseInt(slider.value, 10);
      samplingState[f.key] = v;
      val.textContent = formatSamplingValue(f, v);
      const defv = Number(samplingDefaults[f.key]);
      const dirty = Number.isFinite(defv) ? Math.abs(defv - v) > (f.step / 2) : false;
      row.classList.toggle('dirty', dirty);
      refreshSamplingJson();
    });
    row.appendChild(label);
    row.appendChild(slider);
    row.appendChild(val);
    samplingGrid.appendChild(row);
  });
  refreshSamplingJson();
}

function initSamplingDefaults(defaults) {
  samplingDefaults = (defaults && typeof defaults === 'object') ? { ...defaults } : {};
  samplingState = {};
  SAMPLING_FIELDS.forEach((f) => {
    const dv = Number(samplingDefaults[f.key]);
    samplingState[f.key] = Number.isFinite(dv) ? dv : Number(f.min);
  });
  renderSamplingControls();
}

btnSamplingReset?.addEventListener('click', () => {
  initSamplingDefaults(samplingDefaults);
});

let ws = null;
let isGenerating = false;
let turnCount = 0;
let currentMsg = null;
let typingDots = null;
let streamText = '';
/** @type {string|null} */
let pendingExampleId = null;
/** True while Mamba is streaming the turn-1 system-prompt intro bubble. */
let inIntro = false;

/** Full mock persona (drawer + fallback for system card). */
let globalSystemMarkdown = '';
/** Per sidebar `key` → short SFT/export string from `/api/demo-config`. */
let categorySystemPrompts = {};

/** @see cot_dataset/GUIDE.md — structured assistant output (tables, headings, lists, email blocks). */
const MD_PURIFY = { USE_PROFILES: { html: true } };

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

/** Inline: `code`, **bold**, *italic*, [text](https://...) */
function inlineMarkdown(s) {
  const parts = String(s).split('`');
  return parts
    .map((chunk, i) => {
      if (i % 2 === 1) return '<code>' + escapeHtml(chunk) + '</code>';
      let t = escapeHtml(chunk);
      t = t.replace(
        /\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
      );
      t = t.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
      t = t.replace(/\*([^*]+)\*/g, '<em>$1</em>');
      return t;
    })
    .join('');
}

/** Lenient: a row starts with `|` and has ≥2 pipes. Allowing an unclosed
 *  trailing `|` lets the *currently streaming* row also appear inside the
 *  <table> while data is still arriving (cells are padded to header width). */
function isTableRow(ln) {
  const t = ln.trim();
  if (!t.startsWith('|')) return false;
  return (t.match(/\|/g) || []).length >= 2;
}

function isTableSeparatorRow(ln) {
  if (!isTableRow(ln)) return false;
  const cells = splitTableRow(ln);
  return cells.length > 0 && cells.every((c) => /^:?-{2,}:?$/.test(c));
}

function splitTableRow(row) {
  let t = row.trim();
  if (t.startsWith('|')) t = t.slice(1);
  if (t.endsWith('|')) t = t.slice(0, -1);
  return t.split('|').map((c) => c.trim());
}

function consumeTable(lines, start) {
  if (start + 1 >= lines.length || !isTableSeparatorRow(lines[start + 1])) {
    return { html: '<p>' + inlineMarkdown(lines[start].trim()) + '</p>', end: start + 1 };
  }
  const headerCells = splitTableRow(lines[start]);
  let i = start + 2;
  const bodyLines = [];
  while (i < lines.length && isTableRow(lines[i])) {
    bodyLines.push(lines[i]);
    i++;
  }
  const n = headerCells.length;
  let html = '<table><thead><tr>';
  headerCells.forEach((c) => {
    html += '<th>' + inlineMarkdown(c) + '</th>';
  });
  html += '</tr></thead><tbody>';
  bodyLines.forEach((row) => {
    const cells = splitTableRow(row);
    html += '<tr>';
    for (let k = 0; k < n; k++) {
      const c = cells[k] !== undefined ? cells[k] : '';
      html += '<td>' + inlineMarkdown(c) + '</td>';
    }
    html += '</tr>';
  });
  html += '</tbody></table>';
  return { html, end: i };
}

function isBlockFence(ln) {
  return ln.trim().startsWith('```');
}

function consumeFencedCode(lines, start) {
  let i = start + 1;
  const body = [];
  while (i < lines.length && !isBlockFence(lines[i])) {
    body.push(lines[i]);
    i++;
  }
  if (i < lines.length) i++;
  return { html: '<pre><code>' + escapeHtml(body.join('\n')) + '</code></pre>', end: i };
}

function isSpecialBlockStart(lines, i) {
  const ln = lines[i];
  if (isBlockFence(ln)) return true;
  if (/^#{1,6}\s/.test(ln)) return true;
  if (/^(\*{3,}|-{3,}|_{3,})$/.test(ln.trim())) return true;
  if (/^[-*]\s+/.test(ln)) return true;
  if (/^\d+\.\s+/.test(ln)) return true;
  if (isTableRow(ln) && i + 1 < lines.length && isTableSeparatorRow(lines[i + 1])) return true;
  return false;
}

/**
 * Minimal Markdown → HTML aligned with GUIDE-style model outputs (no CDN parser).
 */
function renderGuideMarkdown(text) {
  const lines = String(text || '')
    .replace(/\r\n/g, '\n')
    .split('\n');
  const out = [];
  let i = 0;
  const n = lines.length;

  while (i < n) {
    const ln = lines[i];
    if (ln.trim() === '') {
      i++;
      continue;
    }

    if (isBlockFence(ln)) {
      const { html, end } = consumeFencedCode(lines, i);
      out.push(html);
      i = end;
      continue;
    }

    if (isTableRow(ln) && i + 1 < n && isTableSeparatorRow(lines[i + 1])) {
      const { html, end } = consumeTable(lines, i);
      out.push(html);
      i = end;
      continue;
    }

    const hm = ln.match(/^(#{1,6})\s+(.*)$/);
    if (hm) {
      const level = Math.min(hm[1].length, 6);
      out.push('<h' + level + '>' + inlineMarkdown(hm[2].trim()) + '</h' + level + '>');
      i++;
      continue;
    }

    if (/^(\*{3,}|-{3,}|_{3,})$/.test(ln.trim())) {
      out.push('<hr>');
      i++;
      continue;
    }

    if (/^[-*]\s+/.test(ln)) {
      const items = [];
      while (i < n && /^[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^[-*]\s+/, ''));
        i++;
      }
      out.push('<ul>' + items.map((it) => '<li>' + inlineMarkdown(it) + '</li>').join('') + '</ul>');
      continue;
    }

    if (/^\d+\.\s+/.test(ln)) {
      const items = [];
      while (i < n && /^\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\d+\.\s+/, ''));
        i++;
      }
      out.push('<ol>' + items.map((it) => '<li>' + inlineMarkdown(it) + '</li>').join('') + '</ol>');
      continue;
    }

    const buf = [];
    buf.push(ln);
    i++;
    while (i < n && lines[i].trim() !== '' && !isSpecialBlockStart(lines, i)) {
      buf.push(lines[i]);
      i++;
    }
    if (buf.length) {
      out.push('<p>' + buf.map((l) => inlineMarkdown(l)).join('<br>') + '</p>');
    }
  }
  return out.join('\n');
}

function renderMd(text) {
  const raw = renderGuideMarkdown(text || '');
  if (typeof DOMPurify !== 'undefined') {
    return DOMPurify.sanitize(raw, MD_PURIFY);
  }
  return raw;
}

/** Container tags whose children we morph individually so new <tr> / <li>
 *  fade in without re-animating the surrounding <table>/<ul>/<ol>. */
const MORPH_CONTAINERS = new Set(['TABLE', 'TBODY', 'THEAD', 'TR', 'UL', 'OL']);

/** In-place children morph: same index + same tag → patch innerHTML;
 *  different tag → replace + tag .fresh; extras at the tail → append .fresh;
 *  surplus old kids → drop. Containers recurse so growing rows/items animate
 *  individually rather than the whole table/list re-flashing. */
function morphChildren(parent, newKids) {
  const oldKids = Array.from(parent.children);
  const minLen = Math.min(oldKids.length, newKids.length);
  for (let i = 0; i < minLen; i++) {
    const o = oldKids[i];
    const n = newKids[i];
    if (o.tagName !== n.tagName) {
      n.classList.add('fresh');
      parent.replaceChild(n, o);
      continue;
    }
    if (o.outerHTML === n.outerHTML) continue;
    if (MORPH_CONTAINERS.has(o.tagName)) {
      morphChildren(o, Array.from(n.children));
    } else {
      o.innerHTML = n.innerHTML;
    }
  }
  for (let i = oldKids.length; i < newKids.length; i++) {
    newKids[i].classList.add('fresh');
    parent.appendChild(newKids[i]);
  }
  while (parent.children.length > newKids.length) {
    parent.lastElementChild.remove();
  }
}

/** Progressive render: parse the partial stream as markdown, then morph the
 *  body in place so already-visible nodes don't re-animate. New blocks (and
 *  new <tr>/<li> inside tables/lists) pick up `.fresh` → CSS fade-slide-in. */
function streamRenderInto(bodyEl, raw) {
  const html = postProcessMarkers(renderMd(raw || ''));
  const tmp = document.createElement('div');
  tmp.innerHTML = html;
  morphChildren(bodyEl, Array.from(tmp.children));
}

function postProcessMarkers(html) {
  return html
    .replace(
      /\[CALL:\s*([^\]\s]+)\s*\]/g,
      '<code class="ctl-tok ctl-call" title="System Call"><span class="b">[</span><span class="k">CALL:</span> <span class="i">$1</span><span class="b">]</span></code>'
    )
    .replace(
      /\[SYSTEM_RESULT:\s*([^\]]+?)\s*\]/g,
      '<code class="ctl-tok ctl-result" title="Host-injected payload"><span class="b">[</span><span class="k">SYSTEM_RESULT:</span> <span class="s">$1</span><span class="b">]</span></code>'
    );
}
const postProcessCalls = postProcessMarkers;

/** Sidebar category icons (stroke SVG, no emoji). */
const CAT_ICON_SVGS = {
  _default:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="12" cy="12" r="9"/></svg>',
  emotion:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M12 21a9 9 0 009-9H3a9 9 0 009 9z"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><path d="M9 9h.01M15 9h.01"/></svg>',
  self_awareness:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="12" cy="8" r="3"/><path d="M6 20v-1a4 4 0 014-4h4a4 4 0 014 4v1"/></svg>',
  email_summary:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 4h16v16H4z"/><path d="M4 8l8 5 8-5"/></svg>',
  movie_intro:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><rect x="2" y="4" width="20" height="14" rx="2"/><path d="M8 4v14M16 4v14M2 10h20"/></svg>',
  daily_conversation:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M21 15a4 4 0 01-4 4H8l-5 3V7a4 4 0 014-4h10a4 4 0 014 4z"/></svg>',
  system_call:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 6h16M4 12h10M4 18h14"/><path d="M18 10l2 2-2 2"/></svg>',
  deep_dive:
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/><path d="M8 7h8M8 11h8M8 15h5"/></svg>',
};

const EX_ICON_SVG =
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6"/><path d="M16 13H8M16 17H8M10 9H8"/></svg>';

function catIconSvg(key) {
  return CAT_ICON_SVGS[key] || CAT_ICON_SVGS._default;
}

function renderControlBlock(raw) {
  const t = (raw || '').trim();
  if (!t) return '';
  const html = postProcessMarkers(escapeHtml(t));
  return `<pre class="ctl-block"><code>${html}</code></pre>`;
}

function finalizeAssistantBody(bodyEl, raw) {
  bodyEl.classList.remove('streaming', 'markdown-streaming');
  streamRenderInto(bodyEl, raw);
}

function renderSidebarCategories(categories) {
  if (!sidebarCat) return;
  sidebarCat.innerHTML = '';
  (categories || []).forEach((cat) => {
    const wrap = document.createElement('div');
    wrap.className = 'side-cat';
    const t = document.createElement('div');
    t.className = 'side-cat-title';
    const tic = document.createElement('span');
    tic.className = 'side-cat-icon';
    tic.setAttribute('aria-hidden', 'true');
    tic.innerHTML = catIconSvg(cat.key || '');
    const ttxt = document.createElement('span');
    ttxt.className = 'side-cat-title-text';
    ttxt.textContent = cat.title || cat.key || '';
    t.appendChild(tic);
    t.appendChild(ttxt);
    wrap.appendChild(t);
    (cat.examples || []).forEach((ex) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'side-ex';
      btn.dataset.prompt = ex.user || '';
      btn.dataset.exampleId = ex.id || '';
      btn.dataset.categoryKey = cat.key || '';
      const eic = document.createElement('span');
      eic.className = 'side-ex-icon';
      eic.setAttribute('aria-hidden', 'true');
      eic.innerHTML = EX_ICON_SVG;
      const mid = document.createElement('span');
      mid.className = 'side-ex-mid';
      const line = document.createElement('span');
      line.className = 'ex-line';
      line.textContent = ex.user || ex.id || 'Example';
      const sub = document.createElement('span');
      sub.className = 'ex-sub';
      sub.textContent = [cat.key, ex.subcategory].filter(Boolean).join(' · ');
      mid.appendChild(line);
      mid.appendChild(sub);
      btn.appendChild(eic);
      btn.appendChild(mid);
      btn.addEventListener('click', () => {
        if (isGenerating || !ws || ws.readyState !== 1) return;
        pendingExampleId = ex.id || null;
        input.value = ex.user || '';
        input.dispatchEvent(new Event('input'));
        if (sysCatSelect && cat.key) {
          sysCatSelect.value = cat.key;
          applySysCardForCategory(cat.key);
        }
        doSend();
      });
      wrap.appendChild(btn);
    });
    sidebarCat.appendChild(wrap);
  });
  refreshExampleDisabled();
}

function refreshExampleDisabled() {
  const ok = ws && ws.readyState === 1;
  document.querySelectorAll('.side-ex').forEach((b) => {
    b.disabled = !ok || isGenerating;
  });
  if (btnPlaySys) {
    btnPlaySys.disabled =
      !ok || isGenerating || !sysCatSelect || sysCatSelect.options.length === 0;
  }
}

function renderStyleAndTools(constraints, tools) {
  const stylePanel = document.getElementById('style-constraints');
  if (stylePanel) {
    const c = constraints || {};
    const pills = [];
    if (c.no_emoji) pills.push('no emoji');
    if (c.no_motivational_cliches) pills.push('no clichés');
    if (c.deep_dive_only_on_request) pills.push('deep-dive on request');
    if (c.default_voice) pills.push(c.default_voice);
    stylePanel.innerHTML = pills.length
      ? pills.map((p) => `<span class="style-pill">${escapeHtml(p)}</span>`).join('')
      : '<span class="style-pill muted">no constraints declared</span>';
  }
  const toolList = document.getElementById('tool-registry');
  if (toolList) {
    const rows = (tools || [])
      .map((t) => {
        const examples = (t.trigger_examples || []).map((q) => `<li>${escapeHtml(q)}</li>`).join('');
        return (
          `<div class="tool-entry">` +
          `<div class="tool-name"><code class="tool-name-code">${escapeHtml(t.name || '')}</code> <span class="tool-purpose">${escapeHtml(t.purpose || '')}</span></div>` +
          `<div class="tool-result">${renderControlBlock(t.system_result_example || '')}</div>` +
          (examples ? `<ul class="tool-eg">${examples}</ul>` : '') +
          `</div>`
        );
      })
      .join('');
    toolList.innerHTML = rows || '<p class="drawer-note">No registered tools.</p>';
  }
}

function fillSysCategorySelect(categories) {
  if (!sysCatSelect) return;
  sysCatSelect.innerHTML = '';
  (categories || []).forEach((cat) => {
    const key = cat.key || '';
    if (!key) return;
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = cat.title || key;
    sysCatSelect.appendChild(opt);
  });
  if (sysCatSelect.options.length && !sysCatSelect.value) sysCatSelect.selectedIndex = 0;
  refreshExampleDisabled();
}

/** System card body: show SFT short string for `categoryKey` when known; else full mock persona. */
function applySysCardForCategory(categoryKey) {
  if (!sysCardBody) return;
  const k = categoryKey != null ? String(categoryKey).trim() : '';
  const short = k && categorySystemPrompts[k] ? String(categorySystemPrompts[k]) : '';
  const src = short || globalSystemMarkdown;
  if (!src) return;
  sysCardBody.innerHTML = postProcessMarkers(renderMd(src));
}

async function loadDemoConfig() {
  try {
    const r = await fetch('/api/demo-config');
    const j = await r.json();
    if (j.mock) mockBanner.hidden = false;
    globalSystemMarkdown = j.system_prompt_markdown || '';
    categorySystemPrompts =
      j.category_system_prompts && typeof j.category_system_prompts === 'object'
        ? j.category_system_prompts
        : {};
    if (globalSystemMarkdown && systemPromptMd) {
      systemPromptMd.innerHTML = postProcessMarkers(renderMd(globalSystemMarkdown));
    }
    fillSysCategorySelect(j.categories);
    applySysCardForCategory(sysCatSelect && sysCatSelect.value ? sysCatSelect.value : null);
    renderStyleAndTools(j.style_constraints, j.tool_registry);
    renderSidebarCategories(j.categories);
    initSamplingDefaults(j.sampling_defaults || {});
  } catch {
    globalSystemMarkdown = '';
    categorySystemPrompts = {};
    fillSysCategorySelect([]);
    renderSidebarCategories([]);
    applySysCardForCategory(null);
    initSamplingDefaults({});
  }
}

function setDrawerTab(tab) {
  document.querySelectorAll('.drawer-tab').forEach((b) => {
    b.classList.toggle('active', b.dataset.tab === tab);
  });
  document.querySelectorAll('.drawer-panel').forEach((p) => {
    p.classList.toggle('hidden', p.dataset.panel !== tab);
  });
}

document.querySelectorAll('.drawer-tab').forEach((btn) => {
  btn.addEventListener('click', () => setDrawerTab(btn.dataset.tab || 'metrics'));
});

$('#btn-info').onclick = () => {
  drawer.classList.add('show');
  overlay.classList.add('show');
};
$('#drawer-close').onclick = overlay.onclick = () => {
  drawer.classList.remove('show');
  overlay.classList.remove('show');
};

$('#btn-sidebar-toggle')?.addEventListener('click', () => {
  sidebar?.classList.toggle('collapsed-mobile');
});

maxTokensSlider.oninput = () => {
  maxTokensVal.textContent = maxTokensSlider.value;
};

input.oninput = () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 200) + 'px';
};
input.onkeydown = (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    doSend();
  }
};

function clearChatUi() {
  chatScroll.innerHTML = '';
  if (sysCard) {
    sysCard.open = true;
    chatScroll.appendChild(sysCard);
  }
  chatScroll.appendChild(welcomeWrap);
  welcomeWrap.style.display = '';
  applySysCardForCategory(sysCatSelect && sysCatSelect.value ? sysCatSelect.value : null);
  turnCount = 0;
  $('#m-turns').textContent = '0';
  ['m-toks', 'm-ttft', 'm-prefill', 'm-total'].forEach((id) => {
    const el = $('#' + id);
    if (el) el.textContent = '—';
  });
}

function doClear() {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ action: 'clear' }));
  clearChatUi();
}

$('#btn-clear').onclick = doClear;
$('#btn-new-sidebar')?.addEventListener('click', doClear);

function setWs(state, text) {
  wsPill.className = 'ws-pill ' + state;
  wsLabel.textContent = text || state;
  const mws = $('#m-ws');
  if (mws) mws.textContent = state === 'open' ? 'Live' : state;
  const ok = state === 'open';
  sendBtn.disabled = !ok || isGenerating;
  refreshExampleDisabled();
}

function connectWs() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(proto + '//' + location.host + '/ws');
  setWs('connecting', 'Connecting');
  ws.onopen = () => {
    setWs('open', 'Connected');
    fetchInfo();
  };
  ws.onclose = () => {
    setWs('closed', 'Reconnecting…');
    setTimeout(connectWs, 2000);
  };
  ws.onerror = () => setWs('closed', 'Error');
  ws.onmessage = (e) => {
    try {
      handleMsg(JSON.parse(e.data));
    } catch {
      /* ignore */
    }
  };
}

async function fetchInfo() {
  try {
    const d = await (await fetch('/api/status')).json();
    if (d.config) {
      const map = [
        ['info-dmodel', d.config.d_model],
        ['info-layers', d.config.num_layers],
        ['info-experts', d.config.kmoe_num_experts],
        ['info-topk', d.config.kmoe_top_k],
        ['info-quant', d.config.quantize ? d.config.quantize + 'b' : 'off'],
        ['info-dtype', d.config.dtype],
      ];
      map.forEach(([id, v]) => {
        const el = $('#' + id);
        if (el) el.textContent = v != null ? String(v) : '—';
      });
    }
    if (d.load_timings && d.load_timings.total_ms != null) {
      const el = $('#info-load');
      if (el) el.textContent = (d.load_timings.total_ms / 1000).toFixed(1) + 's';
    }
  } catch {
    /* ignore */
  }
}

function addRow(role, text) {
  welcomeWrap.style.display = 'none';
  const row = document.createElement('div');
  row.className = 'msg-row ' + role;
  const inner = document.createElement('div');
  inner.className = 'msg-inner';
  const av = document.createElement('div');
  av.className = 'msg-avatar';
  av.textContent = role === 'user' ? 'You' : 'M';
  const content = document.createElement('div');
  content.className = 'msg-content';
  const roleName = document.createElement('div');
  roleName.className = 'msg-role';
  roleName.textContent = role === 'user' ? 'You' : 'Mamba';
  const body = document.createElement('div');
  body.className = 'msg-text md-body';
  if (role === 'user') {
    body.innerHTML = postProcessMarkers(renderMd(text || ''));
  } else {
    body.classList.add('markdown-streaming', 'streaming');
    body.textContent = text || '';
  }
  content.appendChild(roleName);
  content.appendChild(body);
  inner.appendChild(av);
  inner.appendChild(content);
  row.appendChild(inner);
  chatScroll.appendChild(row);
  nudgeScroll();
  return { row, body, content, role };
}

function addToolRow(call, systemResult, phase) {
  welcomeWrap.style.display = 'none';
  const row = document.createElement('div');
  row.className = 'msg-row tool-route';
  const card = document.createElement('div');
  card.className = 'tool-card';
  const phaseLabel = phase === 'response' ? 'Host · injected payload' : 'Host · function tool';
  const callHtml = call ? `<div class="tool-line">${renderControlBlock(call)}</div>` : '';
  const injHtml = systemResult
    ? `<div class="tool-line tool-line-result"><div class="tool-sublabel">Injected back into the conversation</div>${renderControlBlock(systemResult)}</div>`
    : '';
  card.innerHTML =
    '<div class="tool-inner">' +
    `<div class="tool-label"><span class="tool-dot"></span>${phaseLabel}</div>` +
    callHtml +
    injHtml +
    '</div>';
  row.appendChild(card);
  chatScroll.appendChild(row);
  nudgeScroll();
}

function addDots(obj) {
  const d = document.createElement('div');
  d.className = 'typing-dots';
  d.innerHTML = '<span></span><span></span><span></span>';
  obj.body.appendChild(d);
  return d;
}

function addPrefill(obj, data) {
  const p = document.createElement('div');
  p.className = 'prefill-pill';
  const cat = data.category ? ` · ${data.category}` : '';
  const eid = data.example_id ? ` · ${data.example_id}` : '';
  p.textContent =
    (data.prompt_tokens ?? '') +
    ' tok · ' +
    Number(data.prefill_ms).toFixed(1) +
    'ms' +
    cat +
    eid +
    (data.turns > 1 ? ' · turn ' + data.turns : '');
  obj.content.insertBefore(p, obj.body);
}

function addMetrics(obj, data) {
  const m = document.createElement('div');
  m.className = 'msg-metrics';
  m.innerHTML =
    '<span>' +
    Number(data.tok_s).toFixed(1) +
    ' tok/s</span><span>TTFT ' +
    Number(data.ttft_ms).toFixed(1) +
    'ms</span><span>Prefill ' +
    Number(data.prefill_ms).toFixed(1) +
    'ms</span><span>' +
    data.total_tokens +
    ' tok</span><span>' +
    Number(data.total_ms).toFixed(0) +
    'ms</span>';
  obj.content.appendChild(m);
}

function ensureReasoningBlock() {
  if (!currentMsg) return null;
  let det = currentMsg.content.querySelector('details.reasoning-block');
  if (!det) {
    det = document.createElement('details');
    det.className = 'reasoning-block';
    det.open = true;
    det.innerHTML =
      '<summary>Chain of thought <span class="cot-tag">CoT · JSON field</span></summary><div class="reasoning-md md-body"></div>';
    currentMsg.content.insertBefore(det, currentMsg.body);
  }
  return det.querySelector('.reasoning-md');
}

function handleMsg(m) {
  if (m.type === 'connected' || m.type === 'pong') return;
  if (m.type === 'cleared') {
    turnCount = 0;
    const mt = $('#m-turns');
    if (mt) mt.textContent = '0';
    return;
  }
  if (m.type === 'error') {
    if (currentMsg) {
      if (typingDots) {
        typingDots.remove();
        typingDots = null;
      }
      currentMsg.body.classList.remove('streaming', 'markdown-streaming');
      currentMsg.body.textContent = 'Error: ' + m.message;
      currentMsg.body.style.color = '#ef5350';
      currentMsg = null;
    }
    isGenerating = false;
    sendBtn.disabled = false;
    refreshExampleDisabled();
    return;
  }
  if (m.type === 'reasoning') {
    const slot = ensureReasoningBlock();
    if (slot) slot.innerHTML = postProcessMarkers(renderMd(m.markdown || ''));
    nudgeScroll();
    return;
  }
  if (m.type === 'tool_action') {
    addToolRow(m.call, m.system_result, m.phase);
    return;
  }
  if (m.type === 'intro_start') {
    inIntro = true;
    nudgeScroll();
    if (currentMsg) {
      if (typingDots) {
        typingDots.remove();
        typingDots = null;
      }
      currentMsg.row.classList.add('intro-bubble');
      const roleEl = currentMsg.row.querySelector('.msg-role');
      if (roleEl) {
        const tit = String(m.category_title || '').trim();
        roleEl.textContent = tit ? 'Mamba · ' + tit + ' · SFT system' : 'Mamba · System prompt';
      }
      const avEl = currentMsg.row.querySelector('.msg-avatar');
      if (avEl) avEl.textContent = 'S';
    }
    return;
  }
  if (m.type === 'assistant_split') {
    if (currentMsg) {
      if (typingDots) {
        typingDots.remove();
        typingDots = null;
      }
      finalizeAssistantBody(currentMsg.body, streamText);
    }
    currentMsg = addRow('assistant', '');
    currentMsg.body.classList.add('streaming', 'markdown-streaming');
    typingDots = addDots(currentMsg);
    streamText = '';
    inIntro = false;
    nudgeScroll();
    return;
  }
  if (m.type === 'meta') {
    if (typingDots) {
      typingDots.remove();
      typingDots = null;
    }
    if (currentMsg) addPrefill(currentMsg, m);
    const mp = $('#m-prefill');
    if (mp) mp.textContent = Number(m.prefill_ms).toFixed(0) + 'ms';
    if (m.turns) {
      const mt = $('#m-turns');
      if (mt) mt.textContent = m.turns;
    }
    nudgeScroll();
    return;
  }
  if (m.type === 'token') {
    streamText += m.text;
    if (currentMsg) {
      if (typingDots) {
        typingDots.remove();
        typingDots = null;
      }
      currentMsg.body.classList.remove('markdown-streaming');
      currentMsg.body.classList.add('streaming');
      streamRenderInto(currentMsg.body, streamText);
    }
    nudgeScroll();
    if (!inIntro) {
      const mtok = $('#m-toks');
      const mtot = $('#m-total');
      if (mtok) mtok.textContent = Number(m.tok_s).toFixed(1);
      if (mtot) mtot.textContent = m.n;
    }
    return;
  }
  if (m.type === 'done') {
    const playOnly = Boolean(m.play_system_only);
    if (currentMsg) {
      finalizeAssistantBody(currentMsg.body, streamText);
      if (!playOnly) addMetrics(currentMsg, m);
    }
    const mtok = $('#m-toks');
    const mtt = $('#m-ttft');
    const mpf = $('#m-prefill');
    const mtot = $('#m-total');
    const mtr = $('#m-turns');
    if (mtok) mtok.textContent = Number(m.tok_s).toFixed(1);
    if (mtt) mtt.textContent = Number(m.ttft_ms).toFixed(0) + 'ms';
    if (mpf) mpf.textContent = Number(m.prefill_ms).toFixed(0) + 'ms';
    if (mtot) mtot.textContent = m.total_tokens;
    nudgeScroll();
    if (!playOnly) turnCount++;
    if (mtr) mtr.textContent = turnCount;
    currentMsg = null;
    typingDots = null;
    streamText = '';
    inIntro = false;
    isGenerating = false;
    sendBtn.disabled = false;
    refreshExampleDisabled();
    input.focus();
  }
}

function playSystemPrompt() {
  if (!ws || ws.readyState !== 1 || isGenerating) return;
  const key = (sysCatSelect && sysCatSelect.value) || '';
  if (!key) return;
  isGenerating = true;
  sendBtn.disabled = true;
  refreshExampleDisabled();
  welcomeWrap.style.display = 'none';
  if (sysCard && sysCard.open) sysCard.open = false;
  currentMsg = addRow('assistant', '');
  typingDots = addDots(currentMsg);
  streamText = '';
  inIntro = false;
  ws.send(JSON.stringify({ action: 'play_system_prompt', category_key: key }));
}

btnPlaySys?.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  playSystemPrompt();
});

sysCatSelect?.addEventListener('change', () => {
  applySysCardForCategory(sysCatSelect.value);
});

function doSend() {
  const t = input.value.trim();
  if (!t || isGenerating || !ws || ws.readyState !== 1) return;
  isGenerating = true;
  sendBtn.disabled = true;
  refreshExampleDisabled();
  input.value = '';
  input.style.height = 'auto';
  if (sysCard && sysCard.open) sysCard.open = false;
  forceFollow();
  addRow('user', t);
  currentMsg = addRow('assistant', '');
  typingDots = addDots(currentMsg);
  streamText = '';
  inIntro = false;
  forceFollow();
  const payload = {
    action: 'chat',
    prompt: t,
    max_tokens: parseInt(maxTokensSlider.value, 10),
    reasoning: isReasoningOn(),
    sampling: { ...samplingState },
  };
  if (sysCatSelect && sysCatSelect.value) {
    payload.category_key = sysCatSelect.value;
  }
  if (pendingExampleId) {
    payload.example_id = pendingExampleId;
    pendingExampleId = null;
  }
  ws.send(JSON.stringify(payload));
}

sendBtn.onclick = doSend;

loadDemoConfig().then(() => connectWs());
