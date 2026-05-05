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

  function go(d) {
    const nx = Math.max(0, Math.min(N - 1, c + d));
    if (nx === c && d !== 0) return;
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
