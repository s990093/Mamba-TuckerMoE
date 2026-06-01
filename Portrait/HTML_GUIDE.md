# Portrait HTML Guide

## Main Poster (entry point)

| File                   | Purpose                                                                                                                                                                                          | Lines |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| `academic_poster.html` | A0 academic poster (841x1189mm). NKUST logo top-left, title (Hybrid Mamba-TuckerMoE for On-Device LLM Inference), authors, then 7 sections. Uses JS slot-loader to `fetch()` sub-files into DOM. | 545   |

**Sections (in order):**

1. Header — NKUST logo + title + authors + email
2. Problems → Solutions — 3 pain points vs 3 solutions, connected by SVG chevron arrow
3. Phase 1 (slot ← `s_phase1.html`) — Macro Block Architecture & TuckerMoE
4. Phase 3 (slot ← `s_phase3.html`) — Multi-Strategy Speculative Jacobi Decoding
5. Phase 2 (slot ← `s_phase2.html`) — Multi-Stage Training & CoT Loss Design
6. CoT Loss (slot ← `s_cot_loss.html`) — 5-Weights Product + FCP + MoE Aux
7. Edge Application (slot ← `s_cot.html`) — Real-Time On-Device AI Assistant
8. Results (slot ← `s_results.html`) — Main Results, Compression Study & Conclusion

**CSS:** Full `<style>` block (~400 lines), CSS custom properties, 2-column grid.

---

## Slot fragments (loaded into `academic_poster.html

These are `<div>` fragments with **no** `<html>/<head>/<body>`. They rely on parent CSS classes (`sec`, `sh`, `sc`, `sb`, `sn`, `sl`).

| File              | Content                                                                                                                                    | Lines |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| `s_phase1.html`   | **(a)** Mamba-3 block data-flow, **(c)** TuckerMoE 3-panel (Tucker decomp, Expert pool + Top-k, Step-by-step comparison grid)              | 435   |
| `s_phase2.html`   | 3-stage training pipeline (FineWeb-Edu → UltraChat → Custom CoT 13M tok), 5-Weight loss cards                                              | 122   |
| `s_phase3.html`   | N-gram Jacobi decode pipeline SVG, Mamba parallel verify (associative scan), 3 cache panels (NGramCache, SuffixRetriever, CoTPhaseTracker) | 537   |
| `s_cot_loss.html` | Full loss formula L_total, 3-column: 5-Weights, FCP (lambda=0.2), MoE Aux. Stats bar: 20,078 samples / 13M tok                             | 131   |
| `s_cot.html`      | Edge assistant demo (user_story2.png), Visual FSM pipeline SVG, 3 guard badges (Format Guard, Watchdog, Token Budget)                      | 137   |
| `s_results.html`  | 5 key stats row, Decode comparison table (Mamba3 vs Transformer), SJD speedup table, Tucker vs SVD MSE table, Key takeaways                | 401   |

---

## Template panels (reusable SVG diagrams)

All contain **only** `<div>` + inline SVG, no document structure.

| File                       | Content                                                                                                               | Lines |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----- |
| `templates/s_phase1a.html` | **(A)** Mamba-3 Block internal data-flow SVG — in_proj, z bypass, Chunk Parallel Scan, SiLU gate, out_proj            | 127   |
| `templates/s_phase1b.html` | **(B)** Transformer Block internal data-flow SVG — Q/K/V split, GQA Attention, SwiGLU FFN, residual ⊕ nodes           | 147   |
| `templates/s_phase1c.html` | **(C)** TuckerMoE tensor decomposition SVG — isometric 3D cube, U_in·G_e·U_out factorization, dimension annotations   | 514   |
| `templates/s_phase1d.html` | **(D)** Chunk Parallel Scan O(log n) — integral-image analogy, sequential vs parallel contrast, binary tree reduction | 187   |

---

## Archive (older/alternative versions)

| File                                   | Content                                                                                                       | Lines |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ----- |
| `archive/html/s_edge.html`             | Simplified edge stats: 3,800 tok/s prefill, 68 tok/s decode, O(1) KV, 0 dead experts. Compact single section. | 26    |
| `archive/html/s_speculative.html`      | Speculative decoding results: 3.32x speedup, SJD K=16, n-gram cache. 2x2 metric grid + AR vs SJD flowchart.   | 120   |
| `archive/html/Per-TurnDecodeFlow.html` | Full HTML doc — per-turn decode FSM pipeline from `cot_middleware.py`. 6-step flow + FSM state chain.         | 119   |

---

## Reference

| File                                     | Content                                                                                                                                               | Lines |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| `ref/deepseek_html_20260515_1c9468.html` | Reference poster: "Loss Engineering for Chain-of-Thought Reasoning". Card-based layout, SCALe cosine SVG, responsive CSS. Used as design inspiration. | 413   |

---

## Assets

| File                     | Usage                                     |
| ------------------------ | ----------------------------------------- |
| `assets/NKUST_Logo.png`  | School logo, top-left of header (48×48mm) |
| `assets/mamba-logo.png`  | Mamba logo (unused)                       |
| `assets/mlx-logo.png`    | MLX logo (unused)                         |
| `assets/user_story2.png` | Edge assistant screenshot in `s_cot.html` |
| `assets/ce_loss.png`     | Training loss chart (unused in HTML)      |

---

## Design System (from `academic_poster.html`)

| Variable         | Value                               |
| ---------------- | ----------------------------------- |
| Poster size      | A0 portrait (841×1189mm)            |
| Layout           | 2-column CSS Grid (1fr 1fr)         |
| `--deep-navy`    | #1B2A47                             |
| `--gray-blue`    | #5C7B9E                             |
| `--soft-purple`  | #9374B1                             |
| `--mustard-gold` | #D8B868                             |
| `--rust-red`     | #8A3324                             |
| `--deep-green`   | #1B5E34                             |
| `--ice-blue`     | #E8EDF5                             |
| `--sand-beige`   | #FFF9F0                             |
| Title font       | 62.4pt, weight 800, Helvetica/Arial |
| Body font        | 19.5pt, "Times New Roman"           |
| Section header   | Helvetica/Arial, 23.4pt, weight 800 |

## Slot-loader (JS at bottom of `academic_poster.html`)

Uses `fetch()` to load 6 fragments into DOM slots by ID:

- `slot-phase1` ← `s_phase1.html`
- `slot-phase3` ← `s_phase3.html`
- `slot-phase2` ← `s_phase2.html`
- `slot-cot-loss` ← `s_cot_loss.html`
- `slot-edge` ← `s_cot.html`
- `slot-results` ← `s_results.html`
