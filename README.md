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
  dataprep/         patch extraction and preprocessing helpers
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
  config_control_czi_dataset_norm.yaml   patch prep – CZI, per-dataset norm
  config_ycomp_czi_dataset_norm.yaml     patch prep – CZI, per-dataset norm
  config_npy_shared_no_norm.yaml         patch prep – NPY, no norm
  config_ae.yaml                         autoencoder training
  config_analysis.yaml                   latent analysis
  config_classification.yaml             LightGBM classification
  contrastive_config/                    configs for contrastive / supcon AE runs
  newdata_config/                        configs for new-data validation pipeline

scripts/
  run_patchprep_from_config.py
  run_ae_from_config.py
  run_analysis_from_config.py
  run_classification_from_config.py
  run_ae_apply_from_config.py
  run_cls_apply_from_config.py
  run_cross_classification_vis.py
  run_whole_image_overlay.py
  run_newdata_validation.sh              end-to-end new-data validation pipeline
  run_supcon.sh                          supervised contrastive AE pipeline
  run_supcon_noflip.sh                   supervised contrastive AE (noise-only, no flip)
```

---

## Stage 1 — Patch Preparation

Extracts centred square patches from CZI or NPY microscopy images, applies optional normalisation (per-image or per-dataset percentile stretch), and saves individual `.tif` files.

For each accepted patch the pipeline also computes **8 rotation- and scale-invariant distance-to-boundary features** (`d00`–`d07`) by casting rays in 8 evenly-spaced directions from the patch centre to the cell boundary.  Distances are normalised by the cell equivalent diameter (`equiv_diam = 2√(area/π)`) and cyclically shifted so the direction of minimum distance comes first.  These features are written to the patch record CSV alongside the spatial metadata.

**Config keys:**

| Key | Description |
|-----|-------------|
| `file_type` | `"czi"` or `"npy"` |
| `cell_mask_folder` | path to pre-computed segmentation masks; omit or leave null for on-the-fly segmentation |
| `norm_mode` | `null` · `"image"` · `"dataset"` · `"cell_insideoutside"` |
| `patch_size` | side length in pixels (e.g. `32`) |
| `patch_prefix` | string prepended to every patch filename (e.g. `"control"`) |
| `major_ch` | channel index used for patch detection |
| `norm_channels` | channels to normalise (must include `major_ch`) |
| `n_dist_orientations` | number of ray directions for distance features (default `8`) |
| `use_timestamp` | `false` keeps output paths fixed for downstream configs |

**Run:**

```bash
python scripts/run_patchprep_from_config.py config/config_control_czi_dataset_norm.yaml
```

**Output** – one `.tif` per patch, named:

```
{prefix}_f{image_id}x{x_centre}y{y_centre}ps{patch_size}.tif
```

Plus a CSV record per image file:

```
data_prep_record_{condition}_ch{ch}_f_{start}_to_{end}.csv
  columns: image_folder · filename · filenameID · x_c · y_c · rand_angle ·
           rand_tx · rand_ty · x/y_corner1–4 · movie_partitioned_data_dir ·
           crop_img_filename · movie_plot_dir · plot_filename ·
           equiv_diam · d00 … d07
```

---

## Stage 2 — Autoencoder Training

Trains one of several model types on the extracted patches, then saves the trained model, loss curves, a latent feature CSV, and reconstruction images.

### Model types

| `model_type` | Description |
|---|---|
| `"ae"` | Standard convolutional autoencoder |
| `"vae"` | Variational AE / β-VAE (VAE32); uses `mu` as latent |
| `"semisup"` | Semi-supervised AE with a classification head trained on annotated patches |
| `"contrastive"` | Contrastive AE with NT-Xent self-supervised loss; supports random flips and salt-and-pepper noise augmentation |
| `"supcon"` | Supervised contrastive AE — for labelled anchors, positives are all same-class patches in the batch; for unlabelled, falls back to NT-Xent with the paired augmented view |

Switch between them by editing `model.model_type` in the YAML config.

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
  # Contrastive / supcon options:
  noise_prob        : 0.05    # salt-and-pepper noise probability
  use_flip          : true    # random H/V flips (set false for noise-only ablation)
  temperature       : 0.5
  lambda_contrast   : 0.5
  proj_dim          : 64

annotation:              # required for semisup / supcon; used by all models for CSV labels
  annotation_file: "/path/to/labels.csv"
  label_col      : "classification"
  filename_col   : "unique_ID"

training:
  epochs      : 200
  group_split : true     # keeps all patches from the same image in one split

reconstruction:
  save_recon      : true
  recon_pad_size  : 64   # padding used during patch extraction
  recon_image_size: 1024 # canvas size for whole-image outputs
```

**Run:**

```bash
python scripts/run_ae_from_config.py config/config_ae.yaml
```

**Outputs** written to `output.result_dir`:

```
model_final.pt
latents.csv               # filename · condition · group · split · recon_mse ·
                          # mean_intensity · norm_mse · z_0…z_{N-1} ·
                          # annotation_label · annotation_label_name
loss_curve.png
recon/
  patches/                # raw_{split}_{name}.tif  &  recon_{split}_{name}.tif
  images/                 # raw_{group}.tif  &  recon_{group}.tif  (fixed canvas)
  visual/                 # {group}_comparison.png  (side-by-side raw vs reconstruction)
```

> **Note on filename convention** — patch files use an underscore before the coordinate block (`control_f0001x…`) while annotation CSVs use a hyphen (`control-f0001x…`). The dataset loader and classification pipeline both normalise this automatically.

---

## Stage 3 — Latent Analysis

Reads `latents.csv` directly — no model reload required — and produces embeddings, clustering, and a comprehensive set of diagnostic plots.

**Config keys:**

```yaml
input:
  latents_csv  : "/path/to/latents.csv"
  split_filter : "all"      # "all" | "train" | "val"

embedding:
  methods: [UMAP]           # UMAP and/or PHATE
  umap_n_neighbors: 15
  umap_min_dist   : 0.1
  umap_model_pkl  : null    # provide path to reuse a pre-trained UMAP (e.g. for newdata)

clustering:
  kmeans_enabled   : true
  kmeans_n_clusters: 5
  dbscan_enabled   : false
  boxplot_kind     : "box"  # "box" | "violin"

label_orders:
  annotation_label_name: [...]
  condition_name        : [...]
```

When `umap_model_pkl` points to an existing `umap_all_model.pkl` (saved by the classification stage), the analysis stage calls `.transform()` instead of re-fitting, projecting new data into the same 2-D space as the training data.

**Run:**

```bash
python scripts/run_analysis_from_config.py config/config_analysis.yaml
```

**Outputs** written to `output.out_dir`:

```
umap/
  by_condition.png · by_split.png · by_annotation.png · by_kmeans.png
latent_correlation.png / .csv
latent_by_condition.png · latent_by_annotation.png
latent_mean_by_condition.png · latent_mean_by_annotation.png
distribution_condition.png · distribution_annotation.png
mse_distribution.png · mse_by_condition_split.png · mse_by_annotation_split.png
norm_mse_distribution.png · norm_mse_by_condition_split.png · ...
intensity_vs_latent.png     # mean_intensity vs each z_i scatter + Pearson r
analysis_results.csv        # latents.csv augmented with UMAP coords + cluster labels
kmeans_model.pkl
umap_model.pkl              # saved only when a new UMAP is fitted
```

---

## Stage 4 — Classification

Trains a LightGBM classifier on latent features (`z_*`) and optionally on combined latent + distance features (`z_*` + `d00`–`d07`), then evaluates against ground-truth labels from an external annotation CSV.

### Feature modes

| Mode | Feature set | Typical `out_dir` name |
|------|-------------|------------------------|
| Latent only | `z_0`…`z_{N-1}` | `fa_cls_lat8` |
| Latent + distance | `z_0`…`z_{N-1}` + `d00`…`d07` | `fa_cls_lat8dist8` |

Distance features come from the `data_prep_record_*_to_<N>.csv` files written in Stage 1.  The pipeline finds the file with the largest `N` in each supplied directory, concatenates them, and merges on `crop_img_filename`.  Because the normalised distances are on a 0–1 ratio scale while latent dimensions are larger in magnitude, a `feature_weight` multiplier (default `100`) is applied to all `d*` columns before both the classifier and UMAP.

**Config keys:**

```yaml
input:
  latents_csv: "/path/to/latents.csv"

labels:
  label_col   : "classification"
  label_csv   : "/path/to/annotation.csv"   # leave "" to use labels in latents.csv
  filename_col: "unique_ID"                  # join key in the annotation CSV
  label_order : [...]
  exclude_labels: null

features:
  include_mean_intensity: false

split:
  strategy: "from_csv"   # reuse group-aware split from AE training
                         # or "stratified" for a fresh random split

lightgbm:
  n_estimators : 500
  learning_rate: 0.05
  class_weight : "balanced"
  n_cv_folds   : 5        # 0 = skip CV

dist_features:
  patch_prep_dirs:         # omit or set to null for latent-only mode
    - "/path/to/control/plots"
    - "/path/to/ycomp/plots"
  feature_weight: 100.0   # multiplier applied to d* cols before classifier + UMAP

patch_sort:
  sort_labelled  : true   # copy train/val patches into gt{x}pred{y} folders
  sort_unlabelled: false  # copy unlabelled patches into test/gtnpred{y} folders
```

**Run:**

```bash
python scripts/run_classification_from_config.py config/config_classification.yaml
```

**Outputs** written to `output.out_dir`:

```
lgbm_model.pkl
umap_all_model.pkl                  # UMAP fitted on full feature set (lat or lat+dist)
metrics.txt                         # accuracy · balanced accuracy · F1 · CV summary
metrics.csv                         # per-class precision / recall / F1
confusion_matrix_counts_train.png
confusion_matrix_norm_train.png
confusion_matrix_counts_val.png
confusion_matrix_norm_val.png
feature_importance.png
f1_per_class.png
prob_by_true_class.png
classification_results.csv
umap_predicted_label.png
umap_true_label.png
patch_sort/
  train/gt{x}pred{y}/
  val/gt{x}pred{y}/
  test/gtnpred{y}/
```

---

## New-Data Validation Pipeline

Applies a set of **frozen, pre-trained** AE models and classifiers to new experimental data without any retraining.  The full pipeline is orchestrated by `scripts/run_newdata_validation.sh`.

### Stages

| Stage | Script | What it does |
|-------|--------|--------------|
| 1. Patch prep | `run_patchprep_from_config.py` | Extract patches from new CZI files |
| 2. AE apply | `run_ae_apply_from_config.py` | Encode patches → `latents_newdata.csv`; optionally save reconstruction visuals |
| 3. Analysis | `run_analysis_from_config.py` | UMAP / clustering on new latents (uses pre-trained UMAP via `umap_model_pkl`) |
| 4. Cls apply | `run_cls_apply_from_config.py` | Predict labels with trained LightGBM → `predictions_all.csv` |
| 5. Cross-vis | `run_cross_classification_vis.py` | UMAP + crosstab plots; confusion matrices with accuracy when ground-truth labels are supplied |
| 6. Overlay | `run_whole_image_overlay.py` | Whole-image PNG with coloured bounding boxes per predicted label |

### AE apply (`run_ae_apply_from_config.py`)

```yaml
reconstruction:
  save_recon       : true    # set true to produce recon/visual/ comparison PNGs
  recon_pad_size   : 64
  recon_image_size : 1024
```

Outputs `latents_newdata.csv` (same columns as `latents.csv` but with `split = "newdata"`) and, when `save_recon: true`:

```
recon/
  visual/    # {group}_comparison.png  — raw vs reconstruction side-by-side
  images/    # raw_{group}.tif  &  recon_{group}.tif
  patches/   # per-patch raw_*.tif  &  recon_*.tif
```

### Cls apply (`run_cls_apply_from_config.py`)

```yaml
model:
  model_pkl      : "/path/to/lgbm_model.pkl"
  umap_model_pkl : "/path/to/umap_all_model.pkl"   # reuse training-data UMAP space

labels:
  label_order:
    - "Nascent Adhesion"
    - "focal complex"
    - ...
```

Predicted labels are always stored as **string class names** in `predictions_all.csv` (`pred_label` column).  Outputs:

```
predictions_all.csv      # all patches + pred_label + per-class probabilities
umap_pred.png            # UMAP coloured by predicted label
umap_condition.png       # UMAP coloured by condition
pred_distribution.png    # bar chart of predicted-label counts
```

### Cross-classification vis (`run_cross_classification_vis.py`)

Merges FA-type and position classifier predictions, builds UMAP using a pre-trained model (`.transform()`), and generates crosstab heatmaps.

When an `annotation_csv` is provided (optional), true labels are joined on `crop_img_filename` and per-classifier **confusion matrices with accuracy** are added:

```yaml
input:
  umap_model_pkl   : "/path/to/umap_all_model.pkl"   # pre-trained UMAP from training run
  annotation_csv   : "/path/to/labels.csv"            # optional ground-truth for newdata
```

The annotation join normalises filenames with a regex (`f\d+x\d+y\d+ps\d+\.tif`) so condition prefixes like `ctrl_ch1_` are stripped automatically.

Additional outputs when true labels are available:

```
confusion_fa_type_{subset}.png        # true vs predicted FA type (counts), accuracy in title
confusion_fa_type_norm_{subset}.png   # row-normalised version
confusion_position_{subset}.png       # true vs predicted position
confusion_position_norm_{subset}.png
```

### Whole-image overlay (`run_whole_image_overlay.py`)

Reads `predictions_all.csv`, assembles a raw-patch canvas per source image, and draws coloured bounding boxes by predicted label.

```yaml
input:
  predictions_csv : "/path/to/predictions_all.csv"
labels:
  label_order: [...]   # determines colour assignment
misc:
  pad_size   : 64
  image_size : 1024
  linewidth  : 0.6
  dpi        : 300
```

Output: one PNG per source image → `overlay_{condition}_{img_id}.png`

### Config layout for newdata

```
config/newdata_config/
  patchprep_control.yaml          patchprep_ycomp.yaml
  ae_apply_baseline.yaml          ae_apply_semisup_fa.yaml
  ae_apply_semisup_both.yaml
  analysis_baseline.yaml          analysis_semisup_fa.yaml
  analysis_semisup_both.yaml
  cls_apply_{ae}_{target}_{feat}.yaml   # 12 combinations
  vis_{ae}_{feat}.yaml                  # 5 active combinations
  overlay_{ae}_{target}_{feat}.yaml     # 12 combinations
```

---

## Contrastive / Supervised Contrastive AE

`config/contrastive_config/` contains configs for three comparable AE variants:

| Variant | Config | Description |
|---------|--------|-------------|
| `contrastive` | `ae_contrastive.yaml` | NT-Xent self-supervised loss |
| `supcon` | `ae_supcon.yaml` | Supervised contrastive loss + H/V flips + noise |
| `supcon_noflip` | `ae_supcon_noflip.yaml` | Supervised contrastive loss + noise only (`use_flip: false`) |

Run the full 4-stage pipeline for each variant:

```bash
bash scripts/run_supcon.sh         # supcon with flips
bash scripts/run_supcon_noflip.sh  # supcon noise-only (ablation)
```

### Supervised contrastive loss

For labelled anchors, positives = all same-class patches in the 2N-sized batch (original + augmented view).  For unlabelled patches, positives = only the paired augmented view (NT-Xent fallback).  The `use_flip` flag controls whether random horizontal/vertical flips are applied in addition to salt-and-pepper noise.

---

## End-to-end Example

```bash
# 1. Extract patches (computes d00–d07 distance features automatically)
python scripts/run_patchprep_from_config.py config/config_control_czi_dataset_norm.yaml
python scripts/run_patchprep_from_config.py config/config_ycomp_czi_dataset_norm.yaml

# 2. Train autoencoder (edit config to select model_type)
python scripts/run_ae_from_config.py config/config_ae.yaml

# 3. Analyse latent space
python scripts/run_analysis_from_config.py config/config_analysis.yaml

# 4a. Classify using latent features only
python scripts/run_classification_from_config.py config/config_classification.yaml

# 4b. Classify using latent + distance features (set dist_features.patch_prep_dirs in config)
python scripts/run_classification_from_config.py config/config_classification.yaml

# Apply trained models to new data
bash scripts/run_newdata_validation.sh
```

All scripts support `--dry_run` (prints resolved config without running) and `--log_level DEBUG`.  Most also accept `--root_folder /path` to override all absolute paths for running on a different machine.

---

## License

This project is released under the MIT License. See `LICENSE`.
