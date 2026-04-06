---
marp: true
theme: default
paginate: true
style: |
  section {
    font-size: 22px;
    font-family: Arial, sans-serif;
  }
  h1 { font-size: 36px; color: #2c5f8a; }
  h2 { font-size: 28px; color: #2c5f8a; border-bottom: 2px solid #2c5f8a; padding-bottom: 4px; }
  h3 { font-size: 22px; color: #444; }
  code { background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }
  table { font-size: 18px; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 2em; }
  .highlight { background: #e8f4fd; border-left: 4px solid #2c5f8a; padding: 8px 12px; }
  .warning { background: #fff3cd; border-left: 4px solid #e6a817; padding: 8px 12px; }
  .success { background: #d4edda; border-left: 4px solid #28a745; padding: 8px 12px; }
---

# SubCellAE
## Subcellular Feature Learning with Autoencoders

**Goal:** Learn compact representations of focal adhesion (FA) patches from fluorescence microscopy images, enabling unsupervised and semi-supervised analysis of FA morphology and position.

---

## Pipeline Overview

```
Raw .czi microscopy images
         │
         ▼
┌─────────────────────┐
│   Patch Preparation │  Extract, segment, normalize 32×32 patches
└─────────────────────┘
         │  patch .tif files  +  distance-to-edge features CSV
         ▼
┌─────────────────────┐
│  Autoencoder (AE)   │  Learn 8-dim latent representation
│  (5 variants)       │  ae · vae · semisup · contrastive · supcon
└─────────────────────┘
         │  latents.csv  (z₀ … z₇ per patch)
         ▼
┌─────────────────────┐
│  Classification     │  LightGBM on latent (+ distance) features
│  (LightGBM)         │  FA-type · Position
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  New-Data Validation│  Apply frozen models to new experiment
│                     │  Predict · Visualize · Evaluate vs. labels
└─────────────────────┘
```

---

# Part 1
# Patch Preparation

---

## Patch Prep — Input Data

**Images:** Multi-channel `.czi` fluorescence microscopy files
- Channel of interest: paxillin (PAX) channel (`major_ch = 1`)
- Two experimental conditions: **control** and **ycomp**

**Cell segmentation masks:** `.tif` files (one per image)
- Used to restrict patch extraction to cell regions

**Two separate runs** (one config per condition):
```
config_control_czi_dataset_norm.yaml   →  patch_control/
config_ycomp_czi_dataset_norm.yaml     →  patch_ycomp/
```

---

## Patch Prep — Processing Steps

<div class="columns">
<div>

**1. Normalization**
- Mode: `dataset` — compute 0.2–99.8 percentile stats across all images in the set, apply uniformly
- Ensures consistent intensity scale across images and conditions

**2. Grid extraction**
- Pad image by 64 px on each side
- Overlay regular grid of 32×32 windows
- Extract every grid position

</div>
<div>

**3. Mask filtering**
- Keep patch only if ≥ 40% of pixels fall inside the cell mask
- Discards background / edge patches

**4. Distance-to-edge features**
- Cast 8 rays from patch centre to cell boundary
- Normalize by cell equivalent diameter
- Cyclically shift so minimum distance is first
- Output: `d00 … d07` (rotation-invariant)

</div>
</div>

---

## Patch Prep — Output

**Per condition output folder:**
```
patch_control/
  ├── f0000x0176y0336ps32.tif    (32×32 float32)
  ├── f0000x0176y0368ps32.tif
  ├── ...
  └── data_prep_record_*_to_N.csv
```

**CSV columns include:**
- `crop_img_filename` — patch filename (join key for annotations)
- `x_c, y_c` — patch centre coordinates in original image
- `mean_intensity` — mean paxillin signal
- `d00 … d07` — normalized distances to cell edge (8 directions)
- `equiv_diam` — equivalent cell diameter

**Scale:** thousands to tens of thousands of patches per condition

---

# Part 2
# Autoencoder Training

---

## AE — Architecture (All Variants)

**Shared encoder–decoder backbone:**

```
Input (1 × 32 × 32)
    │
    ├── Conv2d(1→32, stride=2)   →  32 × 16 × 16
    ├── Conv2d(32→64, stride=2)  →  64 ×  8 ×  8
    ├── Conv2d(64→128, stride=2) →  128 × 4 ×  4
    ├── Flatten → FC → FC        →  latent z  (dim = 8)
    │
    ├── FC → FC → Reshape        →  128 × 4 × 4
    ├── ConvTranspose2d(128→64)  →  64 ×  8 ×  8
    ├── ConvTranspose2d(64→32)   →  32 × 16 × 16
    └── ConvTranspose2d(32→1)    →  1  × 32 × 32  (Sigmoid)
```

**Latent dimension:** 8   ·   **Input channels:** 1   ·   **Patch size:** 32 × 32 px

---

## AE Variants Overview

| Variant | Extra components | Loss | Key parameters |
|---------|-----------------|------|----------------|
| **AE** | — | MSE reconstruction | — |
| **VAE** | μ, σ heads; reparameterization | MSE + β·KL | `beta`, `beta_anneal` |
| **SemiSup AE** | 1–2 classification heads | MSE + λ·CE | `lambda_cls`, `lambda_cls_2` |
| **Contrastive AE** | Projection head; augmented view | MSE + λ·NT-Xent | `lambda_contrast`, `noise_prob`, `use_flip` |
| **SupCon AE** | Projection head; class-aware positives | MSE + λ·SupCon | `temperature`, `use_flip` |

**All variants** share:
- Adam optimizer, LR = 0.001, group-aware train/val split (80/20)
- Batch size 128, up to 500 epochs with early stopping

---

## VAE — Variational Autoencoder

**Encoder outputs two vectors** (μ and log σ²) instead of a single point.
Latent sample drawn via reparameterization: `z = μ + ε·σ,  ε ~ N(0,I)`

**ELBO loss:**
```
L = MSE(x̂, x)  +  β · KL( q(z|x) ‖ p(z) )
```

**β-VAE:** β > 1 encourages disentangled representations.

**Beta annealing** (`beta_anneal: true`):
- Linearly ramp β from 0 → β over first half of training
- Prevents **posterior collapse** (KL → 0, encoder ignores input)

---

## Contrastive AE (NT-Xent)

**Two views of each patch:**
- **Clean view** → encode → decode (reconstruction)
- **Augmented view** (salt-and-pepper noise 5% ± random H/V flips) → encode → projection head

**NT-Xent loss** (SimCLR-style):
```
L = λ_recon · MSE(x̂, x)  +  λ_contrast · NT-Xent(proj(z_clean), proj(z_aug))
```

**Positives:** the paired augmented view only.
**Negatives:** all other patches in the 2N-sized batch.

**Goal:** Learn representations invariant to noise/orientation while preserving reconstruction quality.

---

# Part 3
# Semi-Supervised AE
## (Focus Section)

---

## SemiSup AE — Motivation

**Standard AE problem:** Latent space organizes by whatever drives reconstruction variation — not necessarily by biologically meaningful categories.

**Goal:** Nudge the encoder so that FA-type (and/or subcellular position) is reflected in the latent space — without requiring labels on every patch.

**Key insight:** Only ~20% of patches are annotated. Semi-supervised learning lets unlabelled patches contribute via reconstruction while labelled patches provide class structure.

---

## SemiSup AE — Architecture

```
                    ┌──────────────────────────────────┐
                    │           Shared Encoder          │
Input patch (32²) ──►  Conv × 3  →  FC × 2  →  z (8-dim)
                    └─────────────┬────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
      ┌──────────────┐   ┌───────────────┐   ┌───────────────┐
      │   Decoder    │   │ Classifier 1  │   │ Classifier 2  │
      │  (recon x̂)  │   │  FA-type (5)  │   │  Position (4) │
      └──────────────┘   └───────────────┘   └───────────────┘
```

**Classifier head:** `FC(8→64) → ReLU → FC(64→num_classes)`

Labels:
- **FA-type (5 classes):** Nascent Adhesion · Focal Complex · Focal Adhesion · Fibrillar Adhesion · No Adhesion
- **Position (4 classes):** Protruding Edge · Cell Periphery · Lamella · Cell Body

---

## SemiSup AE — Loss Function

**Total loss per batch:**

```
L = λ_recon · MSE(x̂, x)
  + λ_cls   · CE(logits₁, y_FAtype)     [labelled patches only]
  + λ_cls_2 · CE(logits₂, y_position)   [labelled patches only]
```

**Unlabelled patches** (label = −1): contribute only to reconstruction.
**Labelled patches**: contribute to all three terms.

**Current settings:**
```yaml
lambda_recon : 1.0   # reconstruction dominates
lambda_cls   : 0.5   # FA-type guidance
lambda_cls_2 : 0.2   # Position guidance (weaker)
```

The asymmetry ensures the model does not sacrifice reconstruction quality for classification.

---

## SemiSup AE — Training Stability

### Two-phase warmup

| Phase | Epochs | Active losses | Purpose |
|-------|--------|--------------|---------|
| **Phase 1 — Warmup** | 0 → 200 | Reconstruction only | Stabilize encoder; build a good latent foundation |
| **Phase 2 — Full** | 200 → 500 | Recon + Classification | Organize latent space by FA-type / Position |

**LR scheduler reset at transition:** `ReduceLROnPlateau` can over-reduce LR during the reconstruction-only warmup plateau; LR is reset to its initial value at epoch 200 so Phase 2 starts with full headroom.

**`min_epochs_for_best = 200`:** Prevents saving a degenerate early-epoch checkpoint as "best".

---

# Part 3b
# Supervised Contrastive AE
## (New Addition)

---

## SupCon AE — Motivation

**Contrastive AE limitation:** NT-Xent treats each patch's augmented twin as the only positive. Patches from different images of the same FA class are pushed apart even if they look similar.

**Supervised Contrastive Loss (SupCon):** For labelled patches, all same-class patches in the batch are treated as positives — pulling the latent space together by class identity, not just augmentation pairing.

<div class="highlight">

Mixed labeled/unlabeled batches are handled gracefully: labeled anchors use class-based positives; unlabeled anchors fall back to NT-Xent with their paired augmented view.

</div>

---

## SupCon AE — Loss Design

**Batch structure:** concatenate clean and augmented views → 2N samples

```
For each anchor i  (in the 2N batch):

  if anchor i is labelled:
      positives = all j ≠ i  where label_j == label_i
                              (including augmented twins + other same-class patches)
  else:
      positives = only the paired augmented twin

L_SupCon = mean over anchors of:
    − (1/|P_i|) · Σ_{j ∈ P_i} sim(z_i, z_j)/τ
                + log Σ_{k ≠ i} exp(sim(z_i, z_k)/τ)
```

**Total loss:**
```
L = λ_recon · MSE(x̂, x)  +  λ_contrast · L_SupCon
```

---

## SupCon AE — Augmentation & Ablation

**Augmentation pipeline** (applied to create the second view):

1. **Salt-and-pepper noise** — 5% of pixels set to 0 or 1 randomly
2. **Random flips** (controlled by `use_flip`) — 50% chance each of horizontal and vertical flip

**Why flips for FA?** Focal adhesions have no preferred orientation; H/V flips create genuinely different views that the encoder should map to the same class region.

**Ablation design:**

| Variant | Noise | Flips | Purpose |
|---------|-------|-------|---------|
| `contrastive` | ✓ | ✗ | NT-Xent baseline |
| `supcon` | ✓ | ✓ | Full supervised contrastive |
| `supcon_noflip` | ✓ | ✗ | Isolate whether flips help vs. noise-only SupCon |

---

# Part 4
# Classification

---

## Classification Pipeline

**Input:** `latents.csv` from AE pipeline
- Columns: `z_0 … z_7` (8 latent dimensions)
- Plus metadata: `filename`, `condition`, `split`, `mean_intensity`

**Optional distance features:**
- `d00 … d07` (distance to cell edge in 8 directions)
- Loaded from patch-prep `data_prep_record_*.csv`
- Scaled ×100 to match latent dimension range

**Classifier:** LightGBM (gradient boosted trees)
- 500 estimators, LR 0.05, `class_weight="balanced"`
- 5-fold stratified cross-validation

**Two targets:** FA-type (5 classes) · Position (4 classes)

---

## Classification — Feature Modes

| Mode | Features | Dir name |
|------|----------|----------|
| Latent only | `z_0 … z_7` (8 dims) | `fa_cls_lat8` |
| Latent + distance | `z_0 … z_7` + `d00 … d07` (16 dims) | `fa_cls_lat8dist8` |

**Distance features** capture where the FA sits within the cell (edge vs. interior), complementing the morphology captured by the latent dimensions.

**Split strategy — `from_csv`:** reuses the same group-aware train/val partition as AE training, preventing information leaks where latents shaped by classification loss appear in the LightGBM test set.

---

## Classification — Outputs

**Metrics:**
- Accuracy, balanced accuracy, macro-F1 (train + val + CV)
- Per-class precision / recall / F1

**Visualizations:**
- Confusion matrices (raw counts + row-normalized), accuracy in title
- UMAP projections colored by true / predicted label
- `umap_all_model.pkl` saved for reuse on new data

**Patch sorting:**
```
patch_sort/
  gt0pred0/   ← correct: true=0, predicted=0
  gt0pred1/   ← error:   true=0, predicted=1
  ...
```

---

# Part 5
# New-Data Validation

---

## New-Data Validation — Overview

Apply **frozen, pre-trained** models from a reference run to a new experiment — no retraining.

```
New .czi images
      │
      ▼  Stage 1: patchprep (same settings as training)
      │  patches/control/tiff_patches32/
      │  patches/ycomp/tiff_patches32/
      │
      ▼  Stage 2: AE apply  (load model_final.pt, encode)
      │  latents_newdata.csv  +  recon/visual/ PNGs
      │
      ▼  Stage 3: Analysis  (UMAP via .transform() on training UMAP)
      │  umap plots in training latent space
      │
      ▼  Stage 4: Cls apply  (load lgbm_model.pkl, predict)
      │  predictions_all.csv  (string class names)
      │
      ▼  Stage 5: Cross-vis  (UMAP + crosstabs + confusion matrices)
      │
      ▼  Stage 6: Overlay  (whole-image coloured bounding boxes)
```

---

## New-Data Validation — UMAP Consistency

**Problem:** If a new UMAP is fitted on new data, the 2-D space is not comparable to the training UMAP — conditions cannot be visually compared.

**Solution:** Load the `umap_all_model.pkl` saved during classification training and call `.transform()` on new latents.

- **Analysis stage** (`run_analysis_from_config.py`): `umap_model_pkl` config key
- **Cls apply stage** (`run_cls_apply_from_config.py`): `umap_model_pkl` config key
- **Cross-vis stage** (`run_cross_classification_vis.py`): `umap_model_pkl` config key

All three project new data into the **same 2-D coordinate space** as training data.

---

## New-Data Validation — Reconstruction

AE apply can save reconstruction visuals alongside latents:

```
recon/
  visual/   ← side-by-side raw vs. reconstruction PNG per source image
  images/   ← full-canvas raw and recon .tif files
  patches/  ← per-patch raw and recon .tif files
```

Useful for checking whether the trained AE generalizes — poor reconstruction quality on new data indicates a distribution shift.

---

## New-Data Validation — Prediction & Labels

**Cls apply:** predicted labels stored as **string class names** in `predictions_all.csv`.

When ground-truth annotations are available for a subset of new patches:

- Annotation CSV joined via canonical filename (`f\d+x\d+y\d+ps\d+\.tif`) — condition prefixes like `ctrl_ch1_` stripped automatically
- **Confusion matrices with accuracy** generated for both FA-type and position classifiers
- "Uncertain" labels excluded from accuracy calculation but visible in crosstab rows

---

## New-Data Validation — Whole-Image Overlay

Each source image gets a PNG with coloured bounding boxes drawn at each patch location, coloured by predicted label.

<div class="columns">
<div>

**FA-type colour scheme:**
- 🔴 Nascent Adhesion
- 🟠 Focal Complex
- 🟢 Focal Adhesion
- 🔵 Fibrillar Adhesion
- ⚫ No Adhesion

</div>
<div>

**Position colour scheme:**
- 🔴 Cell Protruding Edge
- 🟠 Cell Periphery/other
- 🟢 Lamella
- 🔵 Cell Body

</div>
</div>

Overlays produced for each AE × classifier × feature-set combination (12 total: 3 AE × 2 targets × 2 feature sets).

---

## Summary — AE Variants Compared

| AE | Latent guidance | Augmentation | Expected strength |
|----|----------------|--------------|-------------------|
| **Baseline AE** | None | — | Unbiased reconstruction |
| **SemiSup FA** | FA-type CE loss | — | FA-type separation |
| **SemiSup Both** | FA-type + Position CE | — | Dual label structure |
| **Contrastive** | NT-Xent (self-supervised) | Noise only | Noise robustness |
| **SupCon** | SupCon (class-aware) | Noise + flips | Class clustering |
| **SupCon (no flip)** | SupCon (class-aware) | Noise only | Ablation: flip effect |

Classification evaluated with **lat8** (latent only) and **lat8dist8** (latent + distance) features, for both FA-type and position targets.

---

## Summary & Next Steps

**Completed:**
- Five AE variants with unified training pipeline
- Supervised contrastive loss with mixed labeled/unlabeled batch support
- Flip augmentation ablation (`use_flip` flag)
- New-data validation pipeline (6-stage, fully scripted)
- UMAP consistency via pre-trained transform across all stages
- Confusion matrices with accuracy for new-data evaluation
- Whole-image prediction overlay plots

**Open questions:**
- Do flips help or hurt SupCon for FA data? (compare `supcon` vs `supcon_noflip`)
- Which AE variant gives best generalization to new experimental conditions?
- Does latent + distance feature combination consistently improve over latent-only?

---

# Thank you

**Repository:** `SubCellAE`

> *To render these slides:*
> ```bash
> # Install Marp CLI
> npm install -g @marp-team/marp-cli
> # Export to PDF
> marp slides/subcellae_overview.md --pdf
> # Export to PowerPoint
> marp slides/subcellae_overview.md --pptx
> # Live preview
> marp --watch slides/subcellae_overview.md
> ```
