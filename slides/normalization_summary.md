---
marp: true
theme: default
paginate: false
html: true
style: |
  section {
    font-size: 13px;
    padding: 32px 40px;
  }
  h1 { font-size: 20px; border-bottom: 3px solid #3b7dd8; padding-bottom: 6px; margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; font-size: 11px; }
  th { background: #1a2a3a; color: white; padding: 6px 10px; text-align: left; }
  td { padding: 6px 10px; vertical-align: top; border-bottom: 1px solid #e0e6ef; }
  tr:nth-child(even) td { background: #f7f9fc; }
  .rb { background: #fffbe6; border: 1.5px solid #e8c840; border-radius: 6px; padding: 8px 14px; margin-bottom: 14px; font-size: 11px; color: #5a4000; }
  code { background: #fff3c0; padding: 1px 5px; border-radius: 3px; font-size: 10.5px; }
---

# Image Normalization Strategies — SubCellAE Patch Prep

<div class="rb">
<strong>Pre-step (optional, all modes) — Rolling Ball Background Subtraction</strong><br>
<code>img = img − rolling_ball(img, radius=10)</code> &nbsp;·&nbsp; Removes slow background variation before normalization &nbsp;·&nbsp; Percentile stats re-computed on RB-corrected pixels &nbsp;·&nbsp; No zero-clipping applied
</div>

| | **Dataset Percentile** | **Image Percentile** | **Cell Inside/Outside** | **Cell Min-Max** |
|---|---|---|---|---|
| **mode** | `"dataset"` | `"image"` | `"cell_insideoutside"` | `"cell_minmax"` |
| **Step 1** | `p_lo, p_hi = percentile(`<br>`  all pixels in set, 0.2/99.8)` | `p_lo, p_hi = percentile(`<br>`  this image, 1/99)` | `int1 = img − median(outside)` | `int1 = img − mean(outside)` |
| **Step 2** | `out = clip(`<br>`  (img−p_lo)/(p_hi−p_lo), 0,1)` | `out = clip(`<br>`  (img−p_lo)/(p_hi−p_lo), 0,1)` | `int2 = int1 / mean(int1[inside])` | `scale = percentile(`<br>`  int1[inside], 99)` |
| **Step 3** | — | — | `out = int2 / scale` *(default 5)* | `out = clip(int1/scale, 0, 1)` |
| **Stats scope** | Whole dataset | Per image | Per image + mask | Per image + mask |
| **BG reference** | Global percentile | Per-image percentile | Median of outside pixels | Mean of outside pixels |
| **Output range** | [0, 1] | [0, 1] | ~[0, 1] (fixed scale) | [0, 1] |
| **Mask needed** | No | No | Yes | Yes |
| **Notes** | Cross-image comparable. Requires full dataset pass. Recommended baseline. | No cross-image comparability. No pre-computation. | Robust to BG outliers via median. Scale keeps range predictable. | Matches Alana's notebook. 99th-pct avoids single-pixel compression. |
