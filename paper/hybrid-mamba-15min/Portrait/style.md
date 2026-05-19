# Poster Design System — Color Tokens & Rules
# Updated: 2026-05-16 — Aligned to Q1 Conference Design Directive

## Brand Palette

| Token         | Hex       | Usage |
|---------------|-----------|-------|
| `--primary`   | `#1B365D` | Headers, top bar, section borders, left border accents |
| `--secondary` | `#5C7B9E` | Sub-headers, UI elements, step indicators, secondary borders |
| `--accent`    | `#D35400` | Key metrics, "Ours"/TuckerMoE highlights, pain-point borders, CTA labels |
| `--bg`        | `#FFFFFF` | Poster background |
| `--fill`      | `#F4F7F9` | Content box background (off-white/light gray) |
| `--card`      | `#FFFFFF` | Inner card background |
| `--tx`        | `#0F172A` | Body text |
| `--mu`        | `#4B5563` | Muted / caption text |
| `--bo`        | `#C8CDD8` | Borders, dividers |

> **Old `--navy` (#1455A4) replaced by `--primary` (#1B365D).**
> **Old `--burg` (#C0392B) replaced by `--accent` (#D35400).**

---

## Typography Rules

| Element                | Font Family                          | Size  | Weight | Color       |
|------------------------|--------------------------------------|-------|--------|-------------|
| Poster title           | Helvetica, Arial, sans-serif         | 62pt  | 800    | `#1B365D`   |
| Section header (`.sh`) | Helvetica, Arial, sans-serif         | 30pt  | 700    | `#111827`   |
| Phase label / sub-head | Helvetica, Arial, sans-serif         | 13pt  | 700    | varies      |
| Body / bullet          | Times New Roman, Times, serif        | 19pt  | 400    | `#0F172A`   |
| Math equations         | Times New Roman, Times, serif        | 13pt  | 400    | `#111827`   |
| Diagram labels         | Times New Roman, Times, serif        | 11pt  | 400    | dark        |
| Caption                | Times New Roman, Times, serif        | 15pt  | 400    | `#4B5563`   |
| Monospace / code       | Courier New, Courier, monospace      | 11pt  | 400    | `#111827`   |
| Stat number (`.sn`)    | Helvetica, Arial, sans-serif         | 48pt  | 800    | `#D35400`   |
| Stat label (`.sl`)     | Helvetica, Arial, sans-serif         | 15pt  | 400    | `#4B5563`   |

**Rule**: All heading/section text → Helvetica/Arial (sans-serif).
**Rule**: All body, math, and diagram labels → Times New Roman (serif).

---

## Section Header Rule (`.sh`)

All section headers MUST use:
- Background: `#EEF2F7` (light primary tint)
- Left border: `6px solid #1B365D`
- Bottom border: `1.5px solid #BFDBFE`
- Text: `#111827` — **never colored**
- Font: Helvetica, Arial, sans-serif — `30pt` bold

---

## Pipeline Strip (main chevron, academic_poster.html)

Chevron shapes, left to right:

| Stage       | Fill      | Text color |
|-------------|-----------|------------|
| INPUT       | `#0F3460` | `#93C5FD`  |
| ① Design    | `#1B365D` | `#FFFFFF`  |
| ② Training  | `#2C4F7C` | `#FFFFFF`  |
| ③ Inference | `#7C4D0A` | `#FFFFFF`  |
| OUTPUT      | `#0F172A` | `#93C5FD`  |

---

## Section Accent Colors (per phase)

Each phase card uses its own accent for sub-headers and borders —
**the `.sh` section header is ALWAYS `#1B365D` navy, never phase-colored.**

| Section                    | Accent BG   | Accent Border | Accent Text  |
|----------------------------|-------------|---------------|--------------|
| Phase 1 Design             | `#EEF2F7`   | `#1B365D`     | `#1B365D`    |
| Phase 2 — Pre-train strip  | `#F4F7F9`   | `#1B365D`     | `#1B365D`    |
| Phase 2 — Indie SFT strip  | `#EEF2F9`   | `#5C7B9E`     | `#1B365D`    |
| Phase 2 — CoT SFT strip    | `#FEF3EC`   | `#D35400`     | `#D35400`    |
| Phase 2 — Pain points box  | `#FEF3EC`   | `#D35400`     | `#D35400`    |
| Phase 2 — Pre-train loss   | `#F4F7F9`   | `#1B365D`     | `#1B365D`    |
| Phase 2 — CoT total loss   | `#FEF3EC`   | `#D35400`     | `#92400E`    |
| Phase 2 — FCP box          | `#fff`      | `#D35400`     | `#D35400`    |
| Phase 2 — SFT-GO box       | `#fff`      | `#5C7B9E`     | `#1B365D`    |
| Phase 2 — SCALe box        | `#fff`      | `#3B0764`     | `#3B0764`    |
| Phase 3 Mamba              | `#F0FAF5`   | `#065F46`     | `#065F46`    |
| Phase 3 GQA                | `#FFFBF0`   | `#92400E`     | `#78350F`    |
| CoT think zone             | `#FFF0F0`   | `#D35400`     | `#111827`    |
| CoT final zone             | `#EEF2F7`   | `#1B365D`     | `#111827`    |

---

## Stat Box (`.sb`)

- Background: `#F4F7F9`
- Border: `1px solid #C8CDD8`
- Top border: `3px solid #1B365D`
- Number (`.sn`): `48pt 800` weight `#D35400`
- Label (`.sl`): `15pt` `#4B5563`

---

## Pain Points Banner (Phase 2 specific)

Use for the "WHY SPECIAL TRAINING?" banner at the top of Phase 2:

- Container background: `#FEF3EC`
- Container border: `2px solid #D35400`, border-radius: `7px`
- Header font: Helvetica/Arial, `12pt 800`, color `#D35400`
- Pain point cards: `background:#fff`, `border-left: 4px solid [accent]`
  - Pain #1 (EOS leak): accent `#D35400`
  - Pain #2 (boundary): accent `#5C7B9E`
  - Pain #3 (CoT dominates): accent `#1B365D`

---

## Metric Badge (inline, Phase 2)

Used for before/after metric labels (e.g. "EOS: 0.20→0.02"):

```css
background: #D35400;
color: #fff;
padding: 1mm 2mm;
border-radius: 3px;
font-family: Helvetica, Arial, sans-serif;
font-weight: 700;
font-size: 10pt;
```

---

## Bullet List (`.bl li`)
- `▸` prefix in `#D35400`
- `strong` in `#1B365D`
- `.acc` in `#D35400` bold

---

## Bottom Summary Strip (Phase 2)

Dark bar summarizing key metrics across the bottom of Phase 2:
- Background: `#1B365D`
- Text labels: `#93C5FD`, Helvetica/Arial, `11pt 700`
- Metric values: `#D35400`, bold
- Dividers: `#5C7B9E`

---

## Image Paths (relative to Portrait/)

| Asset                        | Path                         |
|------------------------------|------------------------------|
| NKUST Logo                   | `NKUST_Logo.png`             |
| User story photo             | `user_story2.png`            |
| MLX benchmark chart          | `mlx_inference_benchmark.png`|
