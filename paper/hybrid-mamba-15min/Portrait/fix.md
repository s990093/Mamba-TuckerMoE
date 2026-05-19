請根據 首頁去看完整 /Users/hungwei/Desktop/Proj/Mamba3-XR/paper/hybrid-mamba-15min/report/report.md
跟

/Users/hungwei/Desktop/Proj/Mamba3-XR/paper/hybrid-mamba-15min/Portrait/ref/SUMMARY.md

了解目前內容 與 assets 內容是相關圖片 跟 prototypes 是 重要圖html

跟NKUST_Logo.png 是 學校 icon

整體用 英文撰寫

了解 @Portrait 內容 在改 這個塊 Phase 2 — Multi-Stage Training & CoT Loss Design
目前也是 都是字 不夠直覺請設計成 1 分鐘就可以了解版本

目前沒有說到痛點

[System Objective]
You are an expert academic poster designer and LaTeX/TikZ specialist. Your task is to redesign and optimize a high-density computer science research poster titled "Breaking the Memory Wall: Compute-Bound TuckerMoE for Hybrid State Space Models" for a top-tier (Q1) conference. The goal is to maximize readability from 2 meters away, ensure visual hierarchy, and maintain a rigorous, professional academic aesthetic.

[Global Design System & Color Palette]
Strictly adhere to the following color palette and typography rules to reduce visual noise:

1. Primary Color (Headers, Top Bar, Borders): #1B365D (Deep Blue)
2. Secondary Color (Sub-headers, UI elements): #5C7B9E (Slate Blue)
3. Background Fill (Content boxes): #F4F7F9 (Off-white/Light Gray)
4. Accent Color (Key metrics, Ours/TuckerMoE highlights): #D35400 (Burnt Orange)
5. Typography: Use a clean Sans-Serif font (Helvetica or Arial) for all Headings and Section Titles. Use Serif (Times New Roman) for all body text, mathematical equations, and diagram labels to maintain academic rigor.
6. Spacing: Increase internal padding within all "Phase" bounding boxes by 20%. Maintain consistent, wider gutters between the three main columns.

[Section-Specific Modification Directives]

# 1. Header & Top Summary (Input -> Output)

- Background: Solid #1B365D.
- Text: White. Ensure the 5-step pipeline (Input -> Model Design -> Training -> Inference -> Output) is perfectly center-aligned with clear, distinct icons or separators.

# 2. Phase 1: Macro Block Architecture & TuckerMoE Design

- Diagram Refactoring (CRITICAL INSTRUCTION): Redesign the GQA Attention and Mamba block diagrams. You MUST replace all traditional, bulky rectangular "matrix boxes" with direct line connections between input tokens and the Q, K, V matrices. Use a clean, high-definition node-based graphical style (resembling TikZ) to reduce clutter and show direct data flow.
- TuckerMoE Tensor Decomposition: Simplify the 3D tensor visuals. Keep the top-k router clear, but remove unnecessary background grid lines.

# 3. Phase 2: Multi-Stage Training & Loss Design

- Reduce Text Density: Do not use paragraphs. Convert the "Pre-train -> Indie -> CoT SFT" logic and "CoT Inference Format" into high-level bullet points or minimal structured pseudo-code.
- Equation Visibility: Enlarge the mathematical formulas for the Loss functions. Ensure standard LaTeX math rendering.

# 4. Main Results, Compression Study & Conclusion (Right Column)

- Metric Highlight: The four key numbers at the top (82.87%, ~90%, 2.4B, 0.00135) must be extracted and placed in a highly visible metric row. Use a massive font size, bold weight, and the Accent Color (#D35400) for these numbers.
- Chart Standardization: For the Perplexity and MSE charts, ensure the "Ours" (Tucker) line uses the Accent Color (#D35400) with a distinct marker (e.g., star). All axes labels and tick marks must be increased by 30% in size for legibility.

# 5. Phase 3 & Edge Application: Autoregressive Decode & Inference

- Highlight Hardware Performance: Visually emphasize the "Throughput (M2 Pro)" statistics. Make the "3,800 tok/s" text bold and prominent.
- Flowchart Cleanup: Standardize the rounded corners of all flowchart boxes in the "Real-Time On-Device AI Assistant" section. Use the Secondary Color (#5C7B9E) for standard steps and Accent Color (#D35400) for critical output paths.

然後先更新句上面更新 style.md 在設計
