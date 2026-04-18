# SubCellAE

SubCellAE is a Python pipeline for learning latent representations of subcellular fluorescence microscopy patches using autoencoders, then analysing and classifying those representations.

The pipeline has four sequential stages, each driven by a YAML config file and a dedicated runner script:

| Stage | Script | Config |
|-------|--------|--------|
| 1. Patch preparation | `scripts/run_patchprep_from_config.py` | `config/config_*.yaml` |
| 2. Autoencoder training | `scripts/run_ae_from_config.py` | `config/config_ae.yaml` |
| 3. Latent analysis | `scripts/run_analysis_from_config.py` | `config/config_analysis.yaml` |
| 4. Classification | `scripts/run_classification_from_config.py` | `config/config_classification.yaml` |

Additional pipelines for applying trained models to new data and for cross-classification visualization are described below.

---

## Setup

Install the package in editable mode so all `subcellae.*` imports resolve correctly:

**CPU:**

```bash
conda env create -f environment.yml
conda activate subcellae
```

**GPU (CUDA 12.4):**

```bash
conda env create -f environment-cuda.yml
conda activate subcellae-cuda
```

Both environments install the package in editable mode (`pip install -e .`) so local edits are immediately reflected.

---

## Repository Layout

```text
subcellae/
  dataprep/         patch extraction, normalization, and preprocessing helpers
  modelling/
    autoencoders.py   AE / VAE32 / SemiSupAE / ContrastiveAE / SupConAE models + training loops
    dataset.py        PatchDataset (unified, annotation-aware)
  clustering/       KMeans and DBSCAN utilities
  classification/   helper functions for sklearn-based classifiers
  analysis/         feature extraction helpers (legacy)
  utils/
    label_colors.py   fixed tab10 colour map for FA-type and position labels
  pipeline/
    patchprep_pipeline.py     Stage 1 orchestration
    ae_pipeline.py            Stage 2 orchestration
    analysis_pipeline.py      Stage 3 orchestration (also used for newdata analysis)
    classification_pipeline.py Stage 4 orchestration
    ae_apply_pipeline.py      Apply trained AE to new data → latents_newdata.csv
    cls_apply_pipeline.py     Apply trained classifier to new-data latents → predictions_all.csv

config/
  config_control_czi_cell_minmax.yaml       patch prep – CZI, cell_minmax norm + rolling ball
  config_ycomp_czi_cell_minmax.yaml
  config_control_czi_cell_insideoutside.yaml patch prep – CZI, cell_insideoutside norm + rolling ball
  config_ycomp_czi_cell_insideoutside.yaml
  config_control_czi_rb_dataset.yaml        patch prep – CZI, rolling ball + dataset percentile
  config_ycomp_czi_rb_dataset.yaml
  config_ae.yaml                            autoencoder training
  config_analysis.yaml                      latent analysis
  config_classification.yaml               LightGBM classification
  test_config/                             configs for cell_minmax batch test run
  test_config_cio/                         configs for cell_insideoutside batch test run
  contrastive_config/                      configs for contrastive / supcon AE runs
  newdata_config/                          configs for new-data validation pipeline

scripts/
  run_patchprep_from_config.py
  run_ae_from_config.py
  run_analysis_from_config.py
  run_classification_from_config.py
  run_ae_apply_from_config.py
  run_cls_apply_from_config.py
  run_cross_classification_vis.py
  run_whole_image_overlay.py
  run_cellminmax_test_configs.sh           end-to-end batch run (cell_minmax norm)
  run_cellinsideoutside_test_configs.sh    end-to-end batch run (cell_insideoutside norm)
  run_other_paxillin.sh                   apply pre-trained models to other_paxillin datasets
  pack_run_to_h5.py                        pack PNG/TIFF outputs into HDF5 for download
  view_run_h5.py                           interactive local viewer for packed HDF5 files
  pack_interactive_h5.py                  pack one result dir into a structured HDF5 for viewer
  view_interactive.py                     interactive Panel+Bokeh patch viewer (hover+tap)
  label_patches.py                        browser-based manual patch labeling tool (multi-user)
  run_newdata_validation.sh               end-to-end new-data validation pipeline
  run_supcon.sh                           supervised contrastive AE pipeline
  run_supcon_noflip.sh                    supervised contrastive AE (noise-only, no flip)
```

---

## Stage 1 — Patch Preparation

Extracts centred square patches from CZI or NPY microscopy images, applies normalization, and saves individual `.tif` files.

For each accepted patch the pipeline also computes **8 rotation- and scale-invariant distance-to-boundary features** (`d00`–`d07`) by casting rays in 8 evenly-spaced directions from the patch centre to the cell boundary. Distances are normalised by the cell equivalent diameter and cyclically shifted so the direction of minimum distance comes first.

### Normalization modes

| `norm_mode` | Description | Mask required |
|---|---|---|
| `null` | No normalization | No |
| `"image"` | Per-image percentile stretch [0, 1] | No |
| `"dataset"` | Per-dataset percentile stretch [0, 1]; stats computed once over all patches | No |
| `"cell_insideoutside"` | Subtract median of outside pixels → divide by mean of inside pixels → fixed scale | Yes |
| `"cell_minmax"` | Subtract mean of outside pixels → divide by 99th-percentile of inside pixels → clip [0, 1] | Yes |

### Rolling ball preprocessing (optional)

All normalization modes support an optional rolling ball background subtraction step applied **before** normalization:

```yaml
preprocessing:
  rolling_ball_radius : 10   # set to null to disable; radius 10 recommended (Alana's validation)
```

When enabled, `img = img - rolling_ball(img, radius)` is applied per channel. For `"dataset"` normalization, percentile statistics are computed on rolling-ball-corrected pixels for consistency.

**Config keys:**

| Key | Description |
|-----|-------------|
| `file_type` | `"czi"` or `"npy"` |
| `cell_mask_folder` | path to pre-computed segmentation masks; omit or leave null for on-the-fly segmentation |
| `norm_mode` | see table above |
| `patch_size` | side length in pixels (e.g. `32`) |
| `patch_prefix` | string prepended to every patch filename (e.g. `"control"`) |
| `major_ch` | channel index used for patch detection |

**Run:**

```bash
python scripts/run_patchprep_from_config.py config/config_control_czi_cell_minmax.yaml
```

**Output** – one `.tif` per patch, named:

```
{prefix}_f{image_id}x{x_centre}y{y_centre}ps{patch_size}.tif
```

Plus a CSV record per image file with spatial metadata and `d00`–`d07` distance features.

---

## Stage 2 — Autoencoder Training

Trains one of several model types on the extracted patches, then saves the trained model, loss curves, a latent feature CSV, and reconstruction outputs.

### Model types

| `model_type` | Description |
|---|---|
| `"ae"` | Standard convolutional autoencoder |
| `"vae"` | Variational AE / β-VAE (VAE32); uses `mu` as latent |
| `"semisup"` | Semi-supervised AE with a classification head trained on annotated patches |
| `"contrastive"` | Contrastive AE with NT-Xent self-supervised loss |
| `"supcon"` | Supervised contrastive AE — same-class positives for labelled patches, NT-Xent fallback for unlabelled |

All decoders use `Hardtanh(0, 1)` as the output activation (replaces Sigmoid) to avoid saturation collapse on sparse patches.

### Key config sections

```yaml
data:
  patch_dirs:
    - path: "/path/to/patches"
      condition: 0
      condition_name: "control"

model:
  model_type   : "ae"    # ae | vae | semisup | contrastive | supcon
  latent_dim   : 8
  input_ps     : 32

annotation:              # required for semisup / supcon; used by all models for CSV labels
  annotation_file: "/path/to/labels.csv"
  label_col      : "classification"
  filename_col   : "unique_ID"

training:
  epochs         : 500
  lr             : 0.001
  group_split    : true       # keeps all patches from the same image in one split
  warmup_epochs  : 200        # recon-only phase before activating cls/contrastive loss
  lr_scheduler   : "cosine"   # "cosine" | "plateau" | "none"
  lr_min         : 1e-5

reconstruction:
  save_recon      : true
  recon_pad_size  : 64
  recon_image_size: 1024
```

**Run:**

```bash
python scripts/run_ae_from_config.py config/config_ae.yaml
```

**Outputs** written to `output.result_dir`:

```
model_final.pt
latents.csv               # filename · condition · group · split · recon_mse · z_0…z_{N-1} · …
{prefix}_train_val_loss.png
recon/
  patches_raw.tif         # (N, H, W) stacked TIFF of all raw patches
  patches_recon.tif       # (N, H, W) stacked TIFF of all reconstructed patches
  patches_index.csv       # frame · split · condition · condition_name · group · name
  images_raw.tif          # (N, H, W) stacked TIFF of whole-image canvases (one frame per group×channel)
  images_recon.tif
  images_index.csv        # frame · group · channel
  visual.tif              # (N, H, W, 3) uint8 RGB stack of raw vs reconstruction comparisons
  visual_index.csv        # frame · group
```

> All reconstruction outputs are stacked TIFFs with companion CSV index files rather than individual files. Open in FIJI and use the CSV to identify frames by patch name, condition, or group.

---

## Stage 3 — Latent Analysis

Reads `latents.csv` directly — no model reload required — and produces embeddings, clustering, and diagnostic plots.

**Run:**

```bash
python scripts/run_analysis_from_config.py config/config_analysis.yaml
```

**Outputs** written to `output.out_dir`:

```
umap/  by_condition.png · by_annotation.png · by_kmeans.png · …
latent_correlation.png / .csv
latent_by_condition.png · latent_mean_by_condition.png
mse_distribution.png · mse_by_condition_split.png
analysis_results.csv
kmeans_model.pkl · umap_model.pkl
```

---

## Stage 4 — Classification

Trains a classifier on latent features (`z_*`) and optionally combined latent + distance features (`z_*` + `d00`–`d07`).

**Run:**

```bash
python scripts/run_classification_from_config.py config/config_classification.yaml
```

**Outputs** written to `output.out_dir`:

```
lgbm_model.pkl · umap_all_model.pkl
metrics.txt · metrics.csv
confusion_matrix_{counts,norm}_{train,val}.png
feature_importance.png · f1_per_class.png · prob_by_true_class.png
predictions_all.csv · umap_predicted_label.png
```

---

## Batch Test Runs

Two end-to-end shell scripts run the full pipeline (patchprep → AE × 4 → analysis × 4 → classification × 16 → visualization × 8) for a given normalization strategy. Each run lands in a **timestamped directory** so repeated runs never overwrite each other.

### cell_minmax

```bash
bash scripts/run_cellminmax_test_configs.sh 2>&1 | tee logs/test_run_minmax.log
```

Results: `ae_results/test_run_minmax_YYMMDDHH/{baseline,semisup_fa,semisup_pos,semisup_both}/`

### cell_insideoutside

```bash
bash scripts/run_cellinsideoutside_test_configs.sh 2>&1 | tee logs/test_run_cio.log
```

Results: `ae_results/test_run_cio_YYMMDDHH/{baseline,semisup_fa,semisup_pos,semisup_both}/`

Both scripts pack outputs into HDF5 files after each AE training stage and at the end of the full run, printing the scp paths for easy cluster download.

### Output packing for cluster use

```bash
# Pack a result directory into a single HDF5 (all PNG/TIFF → one file)
python scripts/pack_run_to_h5.py ae_results/test_run_cio_26041416/baseline

# Browse locally after scp
python scripts/view_run_h5.py baseline_outputs.h5
python scripts/view_run_h5.py baseline_outputs.h5 --filter recon
```

The viewer prints a numbered list; type a number to display that image, `f <word>` to filter, `q` to quit.

---

## New-Data Validation Pipeline

Applies frozen, pre-trained AE models and classifiers to new data without retraining. Orchestrated by `scripts/run_newdata_validation.sh`.

| Stage | Script | What it does |
|-------|--------|--------------|
| 1. Patch prep | `run_patchprep_from_config.py` | Extract patches from new CZI files |
| 2. AE apply | `run_ae_apply_from_config.py` | Encode patches → `latents_newdata.csv` |
| 3. Analysis | `run_analysis_from_config.py` | UMAP on new latents (reuses pre-trained UMAP) |
| 4. Cls apply | `run_cls_apply_from_config.py` | Predict labels → `predictions_all.csv` |
| 5. Cross-vis | `run_cross_classification_vis.py` | UMAP + crosstab plots |
| 6. Overlay | `run_whole_image_overlay.py` | Whole-image coloured bounding-box visualization |

### Whole-image overlay

Reads `predictions_all.csv`, assembles a raw-patch canvas per source image, and draws coloured bounding boxes by predicted label.

```yaml
input:
  predictions_csv : "/path/to/predictions_all.csv"
labels:
  label_order: [...]
misc:
  pad_size   : 64
  image_size : 1024
  linewidth  : 0.6
  dpi        : 300
```

Output:
```
overlay.tif          # (N, H, W, 3) uint8 RGB stack, one frame per condition × image
overlay_index.csv    # frame · condition · img_id
```

---

## Other-Paxillin Dataset Pipeline

Applies frozen pre-trained models (from `test_run_overfit_20260322`) to four unlabelled datasets — vinculin (`vinc`), phospho-paxillin (`pfak`), paxillin (`ppax`), and NIH 3T3 (`nih3t3`) — each containing `control` and `ycomp` subfolders.

Key differences from the training pipeline:
- **No labels** — all `annotation_csv` fields are `null`; vis plots colour by predicted label only
- **`norm_mode: "dataset"`** — per-dataset percentile stretch, matching the training data normalization
- **On-the-fly segmentation** — `cell_mask_folder` is `null`; the pipeline segments cells at runtime
- **4 AE variants applied** — `baseline`, `semisup_fa`, `semisup_pos`, `semisup_both`

Configs: `config/other_paxillin_config/` (128 files: 8 patchprep + 120 generated)

```bash
# Regenerate non-patchprep configs from the generator script
python scripts/generate_other_paxillin_configs.py

# Run the full pipeline
bash scripts/run_other_paxillin.sh
bash scripts/run_other_paxillin.sh 2>&1 | tee logs/run_other_paxillin.log
```

Results land under:
```
ae_results/other_paxillin/
  patches/{vinc,pfak,ppax,nih3t3}/{control,ycomp}/tiff_patches32/
  {baseline,semisup_fa,semisup_pos,semisup_both}/{vinc,pfak,ppax,nih3t3}/
    latents_newdata.csv
    analysis/
    fa_cls_lat8/  pos_cls_lat8/
    vis_lat8/
```

---

## Interactive Patch Viewer

A local browser-based tool (Panel + Bokeh) for exploring UMAP embeddings and inspecting individual patches. Two directions of interaction are supported:

```
┌──────────────────────┬──────────────────────┬──────────────────────┐
│  [Colour by ▼]       │  [Image selector ▼]  │  ### Details         │
│                      │                      │                      │
│  UMAP scatter        │  Full paxillin canvas│  Patch: ctrl_f0000…  │
│  • hover dot →       │  • coloured patch    │  FA: focal adhesion  │
│    patch thumbnail   │    boxes (FA type)   │  Pos: Lamella        │
│    instantly         │  • click a patch →   │                      │
│  • tap dot →         │    nearest patch     │  [Raw patch]         │
│    detail updates;   │    highlighted;      │  [Recon patch]       │
│    canvas switches   │    red dot on UMAP;  │                      │
│    to source image   │    detail updates    │  ────────────────    │
│                      │                      │  FA-type legend      │
│                      │                      │  Position legend     │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

**Direction A — UMAP → canvas:**
- Hover any UMAP dot: raw + reconstructed patch appear in a tooltip instantly (client-side, embedded base64 PNG — no server round-trip).
- Tap a dot: the detail panel updates with patch images and FA/Position prediction text; the canvas panel automatically switches to the source image and highlights the patch with a white border.

**Direction B — Canvas → UMAP:**
- Select any source image from the dropdown (labelled `condition | group_key`).
- All patches for that image are drawn as coloured bounding boxes (colour = FA type prediction).
- Click anywhere on the canvas: the nearest patch is found, a large red dot appears on the UMAP at its embedding location, and the detail panel updates.

The colour legend in the bottom-right lists both FA type and Position palettes for reference.

### Step 1 — Pack a result directory into a viewer HDF5

Run on the cluster (or locally) for each `variant/dataset` result directory:

```bash
python scripts/pack_interactive_h5.py \
    ae_results/other_paxillin/baseline/vinc \
    --out ae_results/other_paxillin/baseline/vinc/interactive.h5

# Optional: downscale canvas images to halve file size (try 0.5 for large datasets)
python scripts/pack_interactive_h5.py <result_dir> --image-scale 0.5
```

The packer reads:
- `latents_newdata.csv` — patch metadata, latent coordinates
- `analysis/analysis_results.csv` — UMAP_1, UMAP_2
- `fa_cls_lat8/predictions_all.csv` and `pos_cls_lat8/predictions_all.csv` — FA and Position predictions
- `recon/patches_raw.tif` + `patches_recon.tif` + `patches_index.csv` — patch image stacks
- `recon/images_raw.tif` + `images_index.csv` — full canvas image stacks

`run_other_paxillin.sh` automatically packs all 16 dataset-variant combinations at the end (Stage 6).

> **Old-format fallback:** if the pipeline ran before the stacked-TIFF refactor (producing individual `recon/patches/raw_*.tif` and `recon/images/raw_*.tif` files instead of stacked TIFFs), `pack_interactive_h5.py` will still write a valid HDF5 containing only metadata. The viewer detects this and **lazy-loads patches and canvas images directly from disk** on demand — no re-running the pipeline is required.

### Step 2 — View locally

```bash
# Standalone — opens browser at http://localhost:5006 automatically
python scripts/view_interactive.py path/to/interactive.h5

# Via panel serve (equivalent)
panel serve scripts/view_interactive.py --args path/to/interactive.h5 --show
```

**Dependencies** (not in base environment; install once):

```bash
pip install panel bokeh h5py pillow tifffile
```

---

## Patch Labeling Tool

A browser-based manual annotation tool that works directly with `interactive.h5`.  
Shows the full canvas image with all patch boxes as a grid; click a label button then click a patch to assign it.  Multiple annotators can work simultaneously — each browser tab is an independent session.

```
┌─────────────────────────────────────────────────────────────────┐
│  Patch Labeling Tool · interactive.h5  in GardelNas: Liya/…    │
│  Annotator: [____________]   [Image selector ▼]                 │
│  Active label: [Nascent Adhesion][focal complex][focal adhesion][No adhesion]   Labeled: N   [Finish & Save] │
│  Select a label then click a patch.                             │
│                                                                 │
│  [Full canvas — unlabeled patches shown as faint grid;          │
│   labeled patches filled in label colour at high opacity]       │
└─────────────────────────────────────────────────────────────────┘
```

**Interaction:**
- Type your name in the **Annotator** field (used in the output filename)
- Select an image from the dropdown
- Click a label button to set the active label
- Click any patch on the canvas — it fills with the label colour instantly
- Switch images freely; labels accumulate across all images in the session
- **Finish & Save** writes `labels_{name}_{timestamp}.csv` next to the H5, with columns `filename, label, annotator`

### Local use (single user)

```bash
python scripts/label_patches.py path/to/interactive.h5
```

### Lab server mode (shared NAS, multiple users)

Run once on any machine that has the Python environment and can see the NAS:

```bash
# One dataset
python scripts/label_patches.py /mnt/p/Liya/FA_patch_group_label/pfak/interactive.h5 \
    --serve --nas-mount /mnt/p/ --nas-name "GardelNas Expansion"

# Multiple datasets — each gets its own URL path
python scripts/label_patches.py \
    /mnt/p/Liya/FA_patch_group_label/pfak/interactive.h5 \
    /mnt/p/Liya/FA_patch_group_label/nih3t3/interactive.h5 \
    --serve --nas-mount /mnt/p/ --nas-name "GardelNas Expansion"
# prints:
#   http://128.135.108.109:5007/pfak    →  …/pfak/interactive.h5
#   http://128.135.108.109:5007/nih3t3  →  …/nih3t3/interactive.h5
```

Lab members open the printed URL in any browser — no Python install needed.

If the page loads but the app is unresponsive from another machine, the Ubuntu firewall may be blocking the port:

```bash
sudo ufw allow 5007/tcp
```

**Output CSV columns:** `filename`, `label`, `annotator`

---

## Contrastive / Supervised Contrastive AE

`config/contrastive_config/` contains configs for three comparable AE variants:

| Variant | Config | Description |
|---------|--------|-------------|
| `contrastive` | `ae_contrastive.yaml` | NT-Xent self-supervised loss |
| `supcon` | `ae_supcon.yaml` | Supervised contrastive loss + H/V flips + noise |
| `supcon_noflip` | `ae_supcon_noflip.yaml` | Supervised contrastive loss + noise only |

```bash
bash scripts/run_supcon.sh
bash scripts/run_supcon_noflip.sh
```

---

## End-to-end Example

```bash
# 1. Extract patches with cell_minmax normalization + rolling ball
python scripts/run_patchprep_from_config.py config/config_control_czi_cell_minmax.yaml
python scripts/run_patchprep_from_config.py config/config_ycomp_czi_cell_minmax.yaml

# 2. Train autoencoder
python scripts/run_ae_from_config.py config/test_config/ae_baseline.yaml

# 3. Analyse latent space
python scripts/run_analysis_from_config.py config/test_config/analysis_baseline.yaml

# 4. Classify
python scripts/run_classification_from_config.py config/test_config/cls_baseline_fa_lat8.yaml

# Or run the full batch pipeline (all 4 AE variants, timestamped output)
bash scripts/run_cellminmax_test_configs.sh
```

All scripts accept `--root_folder /path` to override absolute paths for running on a different machine or cluster.

---

## License

This project is released under the MIT License. See `LICENSE`.
