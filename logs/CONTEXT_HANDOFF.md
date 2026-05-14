# SubCellAE-contrastive — Session Handoff Context
_Generated: 2026-05-14. Covers work done on `exp/contrastive_projector` branch._

---

## 1. What This Project Is

**SubCellAE-contrastive** is a fork/extension of `SubCellAE` that adds contrastive learning
autoencoders (NT-Xent / SimCLR-style) for analyzing focal adhesion patches from fluorescence
microscopy images (vinculin channel, 32×32 px patches, 1 channel).

- Repo: `/home/lding/lding/gitcode/SubCellAE-contrastive`  
- Active branch: `exp/contrastive_projector`  
- Remote: `github.com:Center-for-Living-Systems/SubCellAE.git`

**PYTHONPATH conflict**: An older editable install of the base `SubCellAE` package lives at
`/home/lding/lding/gitcode/SubCellAE` and pollutes `PYTHONPATH`. All `run_*.py` scripts
fix this with `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` at the top.
The shell pipeline script also exports `PYTHONPATH` explicitly:
```bash
export PYTHONPATH="/home/lding/lding/gitcode/SubCellAE-contrastive"
```

---

## 2. Key Conda Environment

```bash
conda activate subcellae-cuda
```

---

## 3. Data Paths (local machine)

All results live under a single root:
```
root_folder = /home/lding/lding/fa_data_analysis
```

| Purpose | Path |
|---|---|
| Vinc control patches (CIO-RB) | `root_folder/ae_results/patches/cio_rb/vinc/control/tiff_patches32/` |
| Vinc ycomp patches (CIO-RB) | `root_folder/ae_results/patches/cio_rb/vinc/ycomp/tiff_patches32/` |
| Vinc combined labels CSV | `root_folder/labelling/vinc_combined_labels.csv` |
| Current run results | `root_folder/ae_results/contrastive_run/contrastive_cio_rb_vinc_lat12proj8/` |

**vinc_combined_labels.csv** — 959 rows, key columns:
- `unique_ID`: join key = `"{condition}_{crop_img_filename}"` e.g. `control_f0000x0240y0752ps32.tif`
- `classification`: FA type label (No adhesion, focal complex, focal adhesion, fibrillar adhesion, Nascent Adhesion, Uncertain)
- `Position`: spatial label (Cell Protruding Edge, Cell Periphery/other, Lamella, Cell Body)
- `condition`: control / ycomp

---

## 4. Model Architecture — ContrastiveAE

File: `subcellae/modelling/autoencoders.py`

The model has **two output representations**:

| Name | Source | CSV columns | Purpose |
|---|---|---|---|
| `z_recon` | encoder bottleneck | `z_0 … z_11` (latent_dim=12) | reconstruction + classification |
| `z_proj` | projection head output | `p_0 … p_7` (proj_dim=8) | NT-Xent contrastive loss only |

**Projector architecture** (changed this session from hardcoded 128-hidden to scaled):
```
latent_dim=12  →  hidden=48 (latent_dim×4)  →  proj_dim=8
```
This follows SimCLR convention: representation must be larger than projection.

**Key model call patterns:**
```python
x_hat, z = model(x)          # z = z_recon (bottleneck)
z_proj = model.project(z)    # z_proj from projection head
```

**Config knobs** (`ae_contrastive_cio_rb_vinc_lat12proj8.yaml`):
```yaml
model:
  latent_dim: 12
  proj_dim: 8
  noise_prob: 0.05
  temperature: 0.5
  lambda_recon: 1.0
  lambda_contrast: 0.5
  intensity_scale_range: [0.8, 1.2]
training:
  epochs: 500
  lr: 0.001
  batch_size: 128
  group_split: true   # all patches from one image stay in same split
```

---

## 5. Pipeline Overview (4 Stages)

The full pipeline is driven by a single shell script:
```bash
bash scripts/run_contrastive_cio_rb_vinc_lat12proj8.sh 2>&1 | tee logs/vinc_run.log
```

### Stage 1 — AE Training
```bash
python scripts/run_ae_from_config.py \
    config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8.yaml
```
Outputs: `latents.csv` (with `z_*` AND `p_*` columns), `model.pt`, loss curves, reconstructions.

### Stage 2 — Analysis (dual UMAP)
```bash
python scripts/run_analysis_from_config.py \
    config/contrastive_config/analysis_contrastive_cio_rb_vinc_lat12proj8.yaml
```
Outputs:
- `analysis/umap/` — UMAP on z_recon
- `analysis/umap_proj/` — UMAP on z_proj (new this session)

### Stage 3 — Classification (4 runs: 2 targets × 2 feature sets)
```bash
# FA type with z_recon
python scripts/run_classification_from_config.py \
    config/contrastive_config/cls_contrastive_cio_rb_vinc_lat12proj8_fa_zrecon.yaml
# FA type with z_proj
python scripts/run_classification_from_config.py \
    config/contrastive_config/cls_contrastive_cio_rb_vinc_lat12proj8_fa_zproj.yaml
# Position with z_recon
python scripts/run_classification_from_config.py \
    config/contrastive_config/cls_contrastive_cio_rb_vinc_lat12proj8_pos_zrecon.yaml
# Position with z_proj
python scripts/run_classification_from_config.py \
    config/contrastive_config/cls_contrastive_cio_rb_vinc_lat12proj8_pos_zproj.yaml
```
`feature_prefix: "z_"` or `"p_"` in config selects which latent columns to use.

### Stage 4 — Cross-Classification Visualization (2 runs)
```bash
python scripts/run_cross_classification_vis.py \
    config/contrastive_config/vis_contrastive_cio_rb_vinc_lat12proj8_zrecon.yaml
python scripts/run_cross_classification_vis.py \
    config/contrastive_config/vis_contrastive_cio_rb_vinc_lat12proj8_zproj.yaml
```

---

## 6. Key Source Files Modified This Session

| File | What changed |
|---|---|
| `subcellae/modelling/autoencoders.py` | Projector hidden: hardcoded 128 → `latent_dim * 4` |
| `subcellae/pipeline/ae_pipeline.py` | `_extract_latents`: collects `z_proj`; `_save_latent_csv`: saves as `p_*` columns |
| `subcellae/pipeline/analysis_pipeline.py` | Detects `p_*` cols; Step 2b runs UMAP on z_proj → `umap_proj/` |
| `subcellae/pipeline/classification_pipeline.py` | `ClassificationConfig.feature_prefix: str = "z_"` |
| `subcellae/pipeline/cls_apply_pipeline.py` | `ClsApplyConfig.feature_prefix: str = "z_"` |
| `scripts/run_ae_from_config.py` | `sys.path.insert` fix |
| `scripts/run_analysis_from_config.py` | `sys.path.insert` fix |
| `scripts/run_classification_from_config.py` | `sys.path.insert` fix; passes `feature_prefix` from config |

---

## 7. New Files Added This Session

| File | Purpose |
|---|---|
| `scripts/run_contrastive_cio_rb_vinc_lat12proj8.sh` | **Main entry point** — runs all 4 stages end-to-end |
| `scripts/reextract_proj_latents.py` | One-shot: re-extract z_proj from a saved model and patch `p_*` columns into existing latents.csv |
| `config/contrastive_config/ae_contrastive_cio_rb_vinc_lat12proj8.yaml` | AE training config (vinc, lat12, proj8) |
| `config/contrastive_config/analysis_contrastive_cio_rb_vinc_lat12proj8.yaml` | Analysis config |
| `config/contrastive_config/cls_contrastive_cio_rb_vinc_lat12proj8_fa_zrecon.yaml` | FA cls, z_recon features |
| `config/contrastive_config/cls_contrastive_cio_rb_vinc_lat12proj8_fa_zproj.yaml` | FA cls, z_proj features |
| `config/contrastive_config/cls_contrastive_cio_rb_vinc_lat12proj8_pos_zrecon.yaml` | Position cls, z_recon |
| `config/contrastive_config/cls_contrastive_cio_rb_vinc_lat12proj8_pos_zproj.yaml` | Position cls, z_proj |
| `config/contrastive_config/vis_contrastive_cio_rb_vinc_lat12proj8_zrecon.yaml` | Visualization, z_recon |
| `config/contrastive_config/vis_contrastive_cio_rb_vinc_lat12proj8_zproj.yaml` | Visualization, z_proj |

---

## 8. Current Status

- All code changes committed to `exp/contrastive_projector` (commit `eef1512`) and **pushed** to GitHub.
- The vinc pipeline has **not been run yet** on this new code (GPU failure halted the previous run).
- The old (pre-fix) run on nih3t3 patches is irrelevant — ignore any existing results under `contrastive_cio_rb_lat12proj8/`.
- A fresh run of `run_contrastive_cio_rb_vinc_lat12proj8.sh` is needed on the DSI cluster.

---

## 9. Known Issues / Context

### Low classification accuracy (from previous run on wrong dataset)
- Train ~60%, val ~10% on nih3t3 patches (wrong dataset, also possibly wrong labels).
- The overfitted reference model (standard AE) got train 100% / val 60-90%.
- Root cause hypothesis: `lambda_contrast=0.5` makes NT-Xent loss dominate (~98% of gradient),
  and the 180° rotation augmentation used by SimCLR is rotation-invariant — which destroys the
  orientation signal that separates FA types.
- **This needs re-evaluation on the correct vinc dataset before drawing conclusions.**
- Possible next experiments: reduce `lambda_contrast`, try supervised contrastive (supcon configs
  already exist in `config/contrastive_config/ae_supcon*.yaml`).

### label_csv filename join
- Patch filenames on disk: `control_f0000x0240y0752ps32.tif`
- Original label CSVs (Margaret_Control_V2, Margaret_Ycomp_V2): bare `f0000x0240y0752ps32.tif`
- Fix: `vinc_combined_labels.csv` has `unique_ID = condition + "_" + crop_img_filename`
- Config uses `filename_col: "unique_ID"` to match against latents.csv

### Excluded labels in cls configs
- FA type: excludes `"Uncertain"` (only 20 samples, unreliable)
- Position: excludes `"No Category/uncertain"`

---

## 10. Quick Start on DSI Cluster

```bash
# 1. Clone / pull
git clone git@github.com:Center-for-Living-Systems/SubCellAE.git SubCellAE-contrastive
cd SubCellAE-contrastive
git checkout exp/contrastive_projector

# 2. Copy data (or mount / symlink)
#    - patches: fa_data_analysis/ae_results/patches/cio_rb/vinc/{control,ycomp}/tiff_patches32/
#    - labels:  fa_data_analysis/labelling/vinc_combined_labels.csv
#    Update root_folder in YAML configs if the path differs from /home/lding/lding/fa_data_analysis

# 3. Run
conda activate subcellae-cuda
bash scripts/run_contrastive_cio_rb_vinc_lat12proj8.sh 2>&1 | tee logs/vinc_run.log
```

If `root_folder` differs on DSI, do a sed replacement across all vinc configs:
```bash
find config/contrastive_config -name "*vinc*" -exec \
    sed -i 's|/home/lding/lding/fa_data_analysis|/YOUR/DSI/PATH|g' {} \;
```

---

## 11. Git History (recent)

```
eef1512  add z_proj extraction, dual UMAP, and vinc CIO-RB lat12proj8 pipeline  ← current
fa3c66b  add weight_decay to train_ae and strategy template configs
4f2b43a  add recon metrics, cross-dataset eval, latent distance analysis, and CIO-RB pipeline
239df8f  add patch labeling tool and improve interactive viewer
9fa62b6  update README with full interactive viewer documentation
```
