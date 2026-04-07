---
marp: true
theme: default
paginate: true
style: |
  section {
    font-size: 21px;
    font-family: Arial, sans-serif;
  }
  h1 { font-size: 34px; color: #1a4a7a; }
  h2 { font-size: 26px; color: #1a4a7a; border-bottom: 2px solid #1a4a7a; padding-bottom: 4px; }
  h3 { font-size: 21px; color: #333; }
  table { font-size: 17px; border-collapse: collapse; width: 100%; }
  th { background: #1a4a7a; color: white; padding: 6px 10px; }
  td { padding: 5px 10px; border: 1px solid #ccc; }
  tr:nth-child(even) { background: #f2f6fb; }
  .good  { color: #1a7a3a; font-weight: bold; }
  .bad   { color: #9a1a1a; font-weight: bold; }
  .note  { background: #fff8e1; border-left: 4px solid #f0a500; padding: 8px 14px; margin-top: 10px; }
  .tbd   { color: #888; font-style: italic; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 2em; }
---

# SubCellAE — ML Meeting
## Where We Stand

**Core question:** Can we use autoencoders to predict **FA type** and **subcellular position** from fluorescence microscopy patches?

---

## What Data Are We Looking At?

<div class="columns">
<div>

**Experimental conditions:**
- **Control** — unperturbed cells
- **Y-comp** — Y-27632 compound treatment (Rho kinase inhibitor)

**Imaging:** Multi-channel `.czi` fluorescence microscopy
| Channel | Marker |
|---------|--------|
| Ch 1 | Paxillin (PAX) — primary |
| Ch 2 | eGFP-Zyxin (Z) |
| Ch 3 | Phalloidin / Actin (A) |
| Ch 4 | Vinculin (V) |

</div>
<div>

**Ground-truth labels (manual annotation):**

*FA type (5 classes):*
- Nascent Adhesion · Focal Complex
- Focal Adhesion · Fibrillar Adhesion · No Adhesion

*Position (4 classes):*
- Cell Protruding Edge · Cell Periphery/other
- Lamella · Cell Body

**Scale:** ~5,800 patches per condition (control + ycomp)
~60 manually labelled patches in new validation set

</div>
</div>

---

## Normalization

**Strategy: per-dataset percentile stretch**

- Compute **0.2nd–99.8th percentile** intensity across **all images** in the same condition set
- Apply a single linear mapping: `I_norm = clip((I − p_low) / (p_high − p_low), 0, 1)`
- Applied uniformly to every image in the set → all images share the same intensity scale

**Why this is similar to per-cell normalization:**
- Because cells tend to fill the field of view, the dataset percentiles are driven by cell signal rather than background
- Avoids patch-to-patch intensity variation that would confuse the AE

**Current scope:** normalization applied to the **paxillin channel only** (the analysis channel)

---

## Autoencoder — Basic Rules

<div class="columns">
<div>

**Patch extraction:**
- Size: **32 × 32 pixels**
- Grid-based, inside cell mask (≥ 40% overlap)
- Padded image (64 px) → coordinates are deterministic

**Architecture:**
- 3× Conv encoder → **8-dimensional latent z**
- 3× ConvTranspose decoder → sigmoid output
- Single channel input (paxillin)

</div>
<div>

**Training:**
- Group-aware 80/20 split — all patches from the same image stay together
- Adam optimizer, LR = 0.001, batch = 128
- Up to 500 epochs

**Additional features:**
- 8 distance-to-cell-edge features (`d00–d07`, rotation-invariant)
- Used as optional supplement to latent features in classifier

</div>
</div>

---

## Research Question

<div style="text-align:center; font-size: 26px; margin: 30px 0 20px;">

**Can we use an autoencoder to predict<br>FA type and subcellular position?**

</div>

**Approach:** Train AE on all patches → extract 8-dim latent `z` → train LightGBM classifier on `z`

**Four AE strategies tested:**

| Strategy | Idea |
|----------|------|
| **Baseline AE** | Pure reconstruction — latent organizes by image content |
| **Semi-supervised AE** | Add classification head during AE training using labelled patches |
| **Contrastive AE** | Pull augmented views of the same patch together (NT-Xent) |
| **Supervised Contrastive AE** | Like contrastive, but same-class patches are positive pairs |

**Feature sets:** latent only (`lat8`) · latent + distance (`lat8+dist8`)

---

# Results
## Using Paxillin Only

---

## Paxillin Only — Training Set Performance (lat8)

Accuracy on held-out validation split (same experiment as training)

| Model | Position | FA Classification |
|-------|----------|-------------------|
| Baseline AE | 64% | 62% |
| SemiSup AE (FA only) | 60% | **92%** |
| SemiSup AE (FA + Position) | **95%** | **93%** |
| Contrastive AE (NT-Xent) | 58% | 41% |
| SupCon AE (+ flips) | 52% | 37% |
| SupCon AE (noise only) | 54% | 41% |

<div class="note">

**Note:** SemiSup models trained on this dataset — high accuracy expected. Contrastive models receive no label supervision → lower classification accuracy but potentially better generalization.

</div>

---

## Paxillin Only — New Data Validation (Dy2, lat8)

Apply frozen trained models to **new control images** (Dy2 dataset), evaluate on 60 manually labelled patches

| Model | Position (Dy2) | FA Classification (Dy2) |
|-------|----------------|-------------------------|
| Baseline AE | 47% | 47% |
| SemiSup AE (FA only) | 35% | 37% |
| SemiSup AE (FA + Position) | 41% | 34% |
| Contrastive AE (NT-Xent) | <span class="tbd">pending</span> | <span class="tbd">pending</span> |
| SupCon AE (+ flips) | <span class="tbd">pending</span> | <span class="tbd">pending</span> |
| SupCon AE (noise only) | <span class="tbd">pending</span> | <span class="tbd">pending</span> |

<div class="note">

Semi-supervised models show a large drop from training to new data — see notes slide.

</div>

---

## Paxillin Only — Combined Summary (lat8)

| Model | Pos (train) | FA (train) | Pos (Dy2) | FA (Dy2) |
|-------|------------|-----------|----------|---------|
| Baseline AE | 64% | 62% | 47% | 47% |
| SemiSup (FA) | 60% | 92% | 35% | 37% |
| SemiSup (Both) | 95% | 93% | 41% | 34% |
| Contrastive | 58% | 41% | — | — |
| SupCon + flips | 52% | 37% | — | — |
| SupCon no flip | 54% | 41% | — | — |

**Adding distance features (lat8 + dist8), training set:**

| Model | Position | FA Classification |
|-------|----------|-------------------|
| Baseline AE | 76% | 67% |
| SemiSup (Both) | 94% | 94% |

Distance features improve baseline position by **+12 pp**; SemiSup unchanged.

---

## Notes — Semi-Supervised AE

**Training setup:**
- Two classification heads on top of shared encoder: FA type + subcellular position
- Classification loss active **only on labelled patches** (~20% of training set)
- Two-phase training: 200 epochs reconstruction-only warmup, then classification heads activated
- Loss: `λ_recon · MSE + λ_cls · CE(FA) + λ_cls_2 · CE(Position)`

**Why the large train → Dy2 gap?**

<div class="note">

The reference run (`test_run_overfit_20260322`) was intentionally set up to probe overfitting. The model has seen the training images many times with label supervision — the latent space is shaped around those specific cells. On new images from a different imaging session, the class structure learned during training does not transfer cleanly.

</div>

**Next steps:**
- Test with stricter regularization / lower `lambda_cls`
- Compare against baseline on more new-data images (currently only 60 labelled patches)
- Run contrastive variants on Dy2 to see if unsupervised methods generalize better

---

# Multi-Channel Results
## (In Progress)

---

## Paxillin + Actin (P + A)

| Model | Position | FA Classification | Pos (Dy2) | FA (Dy2) |
|-------|----------|-------------------|----------|---------|
| Baseline AE | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> |
| SemiSup AE (FA only) | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> |
| SemiSup AE (FA + Pos) | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> |
| Contrastive AE | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> |

Multi-channel pipeline implemented; runs pending.

---

## Paxillin + Zyxin (P + Z)

| Model | Position | FA Classification | Pos (Dy2) | FA (Dy2) |
|-------|----------|-------------------|----------|---------|
| Baseline AE | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> |
| SemiSup AE (FA only) | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> |
| SemiSup AE (FA + Pos) | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> |
| Contrastive AE | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> |

---

## Paxillin + Vinculin (P + V)

| Model | Position | FA Classification | Pos (Dy2) | FA (Dy2) |
|-------|----------|-------------------|----------|---------|
| Baseline AE | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> |
| SemiSup AE (FA only) | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> |
| SemiSup AE (FA + Pos) | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> |
| Contrastive AE | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> | <span class="tbd">—</span> |

---

## Where We Stand — Summary

| | Done | In progress |
|--|------|-------------|
| **Data** | Paxillin-only pipeline, patch extraction, normalization | Multi-channel (P+A, P+Z, P+V) runs |
| **Models** | Baseline, SemiSup (FA / Both), Contrastive, SupCon | Hyperparameter tuning; contrastive Dy2 eval |
| **Training eval** | All 6 model variants × lat8 + lat8dist8 | — |
| **New-data (Dy2)** | Baseline + SemiSup (3 variants) labelled | Contrastive variants; more labelled patches |

**Key finding so far:**
- SemiSup AE achieves high training-set accuracy (92–95%) but **does not generalize** to Dy2 in this overfit test run
- Baseline AE (no label supervision) is more consistent across train and Dy2 (~47–64%)
- Distance-to-edge features consistently help baseline position accuracy (+12 pp)

**Open question for the group:**
> Is the train→Dy2 drop a fundamental limitation of semi-supervision, or can we fix it with better regularization / more labelled data?

---

# Thank you

> *Slides rendered with Marp:*
> ```bash
> marp slides/ML_meeting_presentation_subcellAE.md --pdf
> marp slides/ML_meeting_presentation_subcellAE.md --pptx
> ```
