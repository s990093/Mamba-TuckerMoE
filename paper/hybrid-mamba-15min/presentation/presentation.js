/* ═══════════════════════════════════════════════════════════
   From O(N²) to Linear — Presentation Navigation
   ═══════════════════════════════════════════════════════════ */
document.addEventListener("DOMContentLoaded", () => {
  // ── KaTeX rendering ──
  if (typeof renderMathInElement === "function") {
    renderMathInElement(document.getElementById("stage"), {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  }

  const sl = Array.from(document.querySelectorAll(".slide")),
    N = sl.length;
  let c = 0;
  const pg  = document.getElementById("pg"),
    pgt = document.getElementById("pgt"),
    ct  = document.getElementById("ct"),
    ct2 = document.getElementById("ctr-top"),
    bp  = document.getElementById("bp"),
    bn  = document.getElementById("bn");
  const p2 = (n) => String(n).padStart(2, "0");

  const fsOverlay = document.getElementById("fs-overlay");
  const fsImg = document.getElementById("fs-img");
  const fsExit = document.getElementById("fs-exit");
  const isFsOpen = () => fsOverlay && fsOverlay.classList.contains("open");

  function openFigureFullscreen(src, alt) {
    if (!fsOverlay || !fsImg) return;
    fsImg.src = src;
    fsImg.alt = alt || "";
    fsOverlay.classList.add("open");
    fsOverlay.setAttribute("aria-hidden", "false");
    document.body.classList.add("fs-open");
  }

  function closeFigureFullscreen() {
    if (!fsOverlay || !fsImg) return;
    fsOverlay.classList.remove("open");
    fsOverlay.setAttribute("aria-hidden", "true");
    fsImg.src = "";
    fsImg.alt = "";
    document.body.classList.remove("fs-open");
  }

  // ── TD-MoE iframe：全螢幕（原生 Fullscreen API，不支援則偽全螢幕）──
  const simWrap = document.getElementById("sim-embed-wrap");
  const simFrame = document.getElementById("sim-embed-frame");
  const simFsBtn = document.getElementById("sim-fs-btn");
  const simPseudoExit = document.getElementById("sim-pseudo-fs-exit");

  function simIsNativeFs() {
    const el = document.fullscreenElement || document.webkitFullscreenElement;
    if (!el) return false;
    return el === simWrap || el === simFrame;
  }
  function simIsPseudoFs() {
    return !!(simWrap && simWrap.classList.contains("is-pseudo-fullscreen"));
  }
  function simIsAnyFs() {
    return simIsNativeFs() || simIsPseudoFs();
  }
  function syncSimFsBtn() {
    if (!simFsBtn) return;
    simFsBtn.textContent = simIsAnyFs() ? "退出全螢幕" : "全螢幕";
  }
  function simEnterFs() {
    if (!simWrap || simIsAnyFs()) return;

    const enterPseudo = () => {
      simWrap.classList.add("is-pseudo-fullscreen");
      document.body.classList.add("sim-fs-open");
      if (simPseudoExit) {
        simPseudoExit.hidden = false;
        simPseudoExit.setAttribute("aria-hidden", "false");
      }
      syncSimFsBtn();
    };

    const webkitOrPseudo = () => {
      try {
        if (simWrap.webkitRequestFullscreen) {
          simWrap.webkitRequestFullscreen();
          syncSimFsBtn();
          return;
        }
      } catch (_) {
        /* fall through */
      }
      try {
        if (simFrame && simFrame.webkitRequestFullscreen) {
          simFrame.webkitRequestFullscreen();
          syncSimFsBtn();
          return;
        }
      } catch (_) {
        /* fall through */
      }
      enterPseudo();
    };

    if (simWrap.requestFullscreen) {
      simWrap
        .requestFullscreen()
        .then(() => syncSimFsBtn())
        .catch(() => {
          if (simFrame && simFrame.requestFullscreen) {
            return simFrame.requestFullscreen().then(() => syncSimFsBtn());
          }
          return Promise.reject(new Error("wrap+frame fs failed"));
        })
        .catch(webkitOrPseudo);
      return;
    }
    webkitOrPseudo();
  }
  async function simExitFs() {
    if (simIsNativeFs()) {
      try {
        if (document.exitFullscreen) await document.exitFullscreen();
        else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
      } catch (_) {
        /* ignore */
      }
    }
    if (simIsPseudoFs()) {
      simWrap.classList.remove("is-pseudo-fullscreen");
      document.body.classList.remove("sim-fs-open");
      if (simPseudoExit) {
        simPseudoExit.hidden = true;
        simPseudoExit.setAttribute("aria-hidden", "true");
      }
    }
    syncSimFsBtn();
  }

  if (simWrap && simFsBtn) {
    simFsBtn.addEventListener("click", (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      if (simIsAnyFs()) void simExitFs();
      else simEnterFs();
    });
    document.addEventListener("fullscreenchange", syncSimFsBtn);
    document.addEventListener("webkitfullscreenchange", syncSimFsBtn);
  }
  if (simPseudoExit) {
    simPseudoExit.addEventListener("click", () => void simExitFs());
  }

  function go(d) {
    const nx = Math.max(0, Math.min(N - 1, c + d));
    if (nx === c && d !== 0) return;
    if (nx !== c && simIsAnyFs()) void simExitFs();
    const prev = c;

    // Exit: current slide direction
    sl[prev].classList.remove("active");
    if (d > 0) sl[prev].classList.add("exit-left");

    // Entrance: next slide from correct direction
    if (d < 0) {
      sl[nx].style.transform = "translateY(-12px) scale(0.99)";
    } else {
      sl[nx].style.transform = "translateY(12px) scale(0.99)";
    }
    sl[nx].style.opacity = "0";
    sl[nx].getBoundingClientRect(); // force reflow
    sl[nx].style.transform = "";
    sl[nx].style.opacity = "";

    c = nx;
    sl.forEach((s, i) => s.classList.toggle("active", i === c));

    setTimeout(() => {
      sl[prev].classList.remove("exit-left");
      sl[prev].style.transform = "";
      sl[prev].style.opacity = "";
    }, 300);

    const progress = (((c + 1) / N) * 100).toFixed(1) + "%";
    if (pg)  pg.style.width = progress;
    if (pgt) pgt.style.width = progress;
    const lb = p2(c + 1) + " / " + p2(N);
    if (ct)  ct.textContent = lb;
    if (ct2) ct2.textContent = lb;
    if (bp) bp.disabled = c === 0;
    if (bn) bn.disabled = c === N - 1;
    sl[c].scrollTop = 0;
  }

  // ── Button Navigation ──
  if (bp) bp.addEventListener("click", () => go(-1));
  if (bn) bn.addEventListener("click", () => go(1));

  document.querySelectorAll("[data-fullscreen-img]").forEach((btn) => {
    btn.addEventListener("click", () => {
      openFigureFullscreen(btn.dataset.fullscreenImg, btn.dataset.fullscreenAlt || btn.textContent);
    });
  });
  if (fsExit) fsExit.addEventListener("click", closeFigureFullscreen);
  if (fsOverlay) {
    fsOverlay.addEventListener("click", (e) => {
      if (e.target === fsOverlay) closeFigureFullscreen();
    });
  }

  // ── Auto-Play ──
  let autoPlayInterval = null;
  const bpa = document.getElementById("bpa");
  if (bpa) {
    bpa.addEventListener("click", () => {
      if (autoPlayInterval) {
        clearInterval(autoPlayInterval);
        autoPlayInterval = null;
        bpa.textContent = "自動播放";
        bpa.style.borderColor = "";
        bpa.style.color = "";
      } else {
        autoPlayInterval = setInterval(() => {
          if (c < N - 1) {
            go(1);
          } else {
            go(-c); // back to start
          }
        }, 4000);
        bpa.textContent = "暫停播放";
        bpa.style.borderColor = "var(--gold)";
        bpa.style.color = "var(--gold)";
      }
    });
  }

  // ── Keyboard Navigation ──
  document.addEventListener("keydown", (e) => {
    const t = e.target;
    if (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable) return;

    if (isFsOpen()) {
      e.preventDefault();
      if (e.key === "Escape") closeFigureFullscreen();
      return;
    }

    if (simWrap && simIsAnyFs()) {
      if (e.key === "Escape" && simIsPseudoFs()) {
        e.preventDefault();
        void simExitFs();
        return;
      }
      const k = e.key;
      if (
        k === "ArrowRight" ||
        k === "ArrowDown" ||
        k === " " ||
        k === "PageDown" ||
        k === "ArrowLeft" ||
        k === "ArrowUp" ||
        k === "PageUp" ||
        k === "Home" ||
        k === "End"
      ) {
        e.preventDefault();
      }
      return;
    }

    if (e.key === "ArrowRight" || e.key === "ArrowDown" || e.key === " " || e.key === "PageDown") {
      e.preventDefault(); go(1);
    } else if (e.key === "ArrowLeft" || e.key === "ArrowUp" || e.key === "PageUp") {
      e.preventDefault(); go(-1);
    } else if (e.key === "Home") {
      e.preventDefault(); go(-c);
    } else if (e.key === "End") {
      e.preventDefault(); go(N - 1 - c);
    }
  });

  // ── Touch Navigation ──
  let tx = null;
  const st = document.getElementById("stage");
  st.addEventListener("touchstart", (e) => {
    tx = e.changedTouches[0].screenX;
  }, { passive: true });
  st.addEventListener("touchend", (e) => {
    if (simWrap && simIsAnyFs()) return;
    if (tx === null) return;
    const dx = e.changedTouches[0].screenX - tx;
    tx = null;
    if (Math.abs(dx) < 45) return;
    go(dx < 0 ? 1 : -1);
  }, { passive: true });

  // ── Initialize ──
  go(0);
  if (bp) bp.disabled = true;
});
