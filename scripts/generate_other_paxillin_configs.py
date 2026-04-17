"""
Generate ae_apply / analysis / cls_apply / vis / overlay configs
for the other_paxillin datasets.

Run once:
    python scripts/generate_other_paxillin_configs.py

Output: config/other_paxillin_config/{ae_apply,analysis,cls_apply,vis,overlay}_*.yaml
"""

from pathlib import Path
import textwrap

ROOT     = "/home/lding/lding/fa_data_analysis"
MODEL_RUN = "test_run_overfit_20260322"   # pre-trained models
OUT_DIR   = Path("config/other_paxillin_config")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["vinc", "pfak", "ppax", "nih3t3"]
VARIANTS = ["baseline", "semisup_fa", "semisup_pos", "semisup_both"]

FA_ORDER = [
    "Nascent Adhesion", "focal complex", "focal adhesion",
    "fibrillar adhesion", "No adhesion",
]
POS_ORDER = [
    "Cell Protruding Edge", "Cell Periphery/other", "Lamella", "Cell Body",
]
FA_COLORS = ["#e6194b", "#f58231", "#3cb44b", "#4363d8", "#aaaaaa"]
POS_COLORS = ["#e6194b", "#f58231", "#3cb44b", "#4363d8"]


def label_list(items, indent=4):
    pad = " " * indent
    return "\n".join(f"{pad}- \"{v}\"" for v in items)


def color_list(items, indent=4):
    pad = " " * indent
    return "\n".join(f"{pad}- \"{v}\"" for v in items)


def write(path: Path, content: str):
    path.write_text(textwrap.dedent(content).lstrip())
    print(f"  wrote {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# ae_apply  (4 variants × 4 datasets)
# ─────────────────────────────────────────────────────────────────────────────
for variant in VARIANTS:
    for ds in DATASETS:
        patches = f"{ROOT}/ae_results/other_paxillin/patches/{ds}"
        out     = f"{ROOT}/ae_results/other_paxillin/{variant}/{ds}"
        model   = f"{ROOT}/ae_results/{MODEL_RUN}/{variant}/model_final.pt"
        write(OUT_DIR / f"ae_apply_{variant}_{ds}.yaml", f"""\
# AE Apply: {variant} | dataset: {ds}
root_folder : "{ROOT}"
model:
  model_pt : "{model}"

data:
  patch_dirs:
    - path          : "{patches}/control/tiff_patches32"
      condition     : 0
      condition_name: "control"
    - path          : "{patches}/ycomp/tiff_patches32"
      condition     : 1
      condition_name: "ycomp"

output:
  out_dir : "{out}"

inference:
  batch_size : 128

reconstruction:
  save_recon       : true
  recon_pad_size   : 64
  recon_image_size : 1024

misc:
  device    : "auto"
  log_level : "INFO"
""")

# ─────────────────────────────────────────────────────────────────────────────
# analysis  (4 variants × 4 datasets)
# ─────────────────────────────────────────────────────────────────────────────
for variant in VARIANTS:
    for ds in DATASETS:
        ae_out    = f"{ROOT}/ae_results/other_paxillin/{variant}/{ds}"
        umap_pkl  = f"{ROOT}/ae_results/{MODEL_RUN}/{variant}/fa_cls_lat8/umap_all_model.pkl"
        write(OUT_DIR / f"analysis_{variant}_{ds}.yaml", f"""\
# Analysis: {variant} | dataset: {ds}
root_folder : "{ROOT}"
input:
  latents_csv  : "{ae_out}/latents_newdata.csv"
  split_filter : "all"

output:
  out_dir : "{ae_out}/analysis"

embedding:
  methods:
    - UMAP
  umap_model_pkl : "{umap_pkl}"

clustering:
  kmeans_enabled    : true
  kmeans_n_clusters : 5
  dbscan_enabled    : false
  boxplot_kind      : box

label_orders:
  condition_name:
    - "control"
    - "ycomp"

misc:
  log_level : "INFO"
""")

# ─────────────────────────────────────────────────────────────────────────────
# cls_apply  (4 variants × 4 datasets × 2 targets: fa + pos, lat8 only)
# ─────────────────────────────────────────────────────────────────────────────
for variant in VARIANTS:
    for ds in DATASETS:
        ae_out = f"{ROOT}/ae_results/other_paxillin/{variant}/{ds}"

        # FA type
        cls_dir = f"{ROOT}/ae_results/{MODEL_RUN}/{variant}/fa_cls_lat8"
        write(OUT_DIR / f"cls_apply_{variant}_fa_lat8_{ds}.yaml", f"""\
# Cls Apply: FA type | {variant} | lat8 | dataset: {ds}
root_folder : "{ROOT}"
input:
  latents_csv : "{ae_out}/latents_newdata.csv"

model:
  model_pkl      : "{cls_dir}/lgbm_model.pkl"
  umap_model_pkl : "{cls_dir}/umap_all_model.pkl"

output:
  out_dir : "{ae_out}/fa_cls_lat8"

labels:
  label_order:
{label_list(FA_ORDER)}

features:
  feature_cols : null

dist_features:
  patch_prep_dirs: null
  feature_weight : 20.0

misc:
  log_level : "INFO"
""")

        # Position
        cls_dir = f"{ROOT}/ae_results/{MODEL_RUN}/{variant}/pos_cls_lat8"
        write(OUT_DIR / f"cls_apply_{variant}_pos_lat8_{ds}.yaml", f"""\
# Cls Apply: Position | {variant} | lat8 | dataset: {ds}
root_folder : "{ROOT}"
input:
  latents_csv : "{ae_out}/latents_newdata.csv"

model:
  model_pkl      : "{cls_dir}/lgbm_model.pkl"
  umap_model_pkl : "{cls_dir}/umap_all_model.pkl"

output:
  out_dir : "{ae_out}/pos_cls_lat8"

labels:
  label_order:
{label_list(POS_ORDER)}

features:
  feature_cols : null

dist_features:
  patch_prep_dirs: null
  feature_weight : 20.0

misc:
  log_level : "INFO"
""")

# ─────────────────────────────────────────────────────────────────────────────
# vis  (4 variants × 4 datasets)
# ─────────────────────────────────────────────────────────────────────────────
for variant in VARIANTS:
    for ds in DATASETS:
        ae_out   = f"{ROOT}/ae_results/other_paxillin/{variant}/{ds}"
        umap_pkl = f"{ROOT}/ae_results/{MODEL_RUN}/{variant}/fa_cls_lat8/umap_all_model.pkl"
        write(OUT_DIR / f"vis_{variant}_lat8_{ds}.yaml", f"""\
# Cross-cls vis: {variant} | lat8 | dataset: {ds}
root_folder : "{ROOT}"
input:
  latents_csv          : "{ae_out}/latents_newdata.csv"
  fa_type_results_csv  : "{ae_out}/fa_cls_lat8/predictions_all.csv"
  position_results_csv : "{ae_out}/pos_cls_lat8/predictions_all.csv"
  umap_model_pkl       : "{umap_pkl}"
  annotation_csv       : null   # no labels for other_paxillin

output:
  out_dir : "{ae_out}/vis_lat8"

labels:
  fa_type_label_col  : "classification"
  position_label_col : "Position"
  fa_type_order:
{label_list(FA_ORDER)}
  position_order:
{label_list(POS_ORDER)}

misc:
  random_state : 42
  log_level    : "INFO"
""")

# ─────────────────────────────────────────────────────────────────────────────
# overlay  (4 variants × 4 datasets × 2 targets: fa + pos)
# ─────────────────────────────────────────────────────────────────────────────
for variant in VARIANTS:
    for ds in DATASETS:
        ae_out = f"{ROOT}/ae_results/other_paxillin/{variant}/{ds}"

        write(OUT_DIR / f"overlay_{variant}_fa_lat8_{ds}.yaml", f"""\
# Overlay: FA type | {variant} | lat8 | dataset: {ds}
root_folder : "{ROOT}"
input:
  predictions_csv : "{ae_out}/fa_cls_lat8/predictions_all.csv"
output:
  out_dir : "{ae_out}/fa_cls_lat8/overlay"
labels:
  label_order:
{label_list(FA_ORDER)}
  colors:
{color_list(FA_COLORS)}
misc:
  pad_size     : 64
  image_size   : 1024
  linewidth    : 0.6
  dpi          : 300
  title_prefix : "FA type | {variant} | lat8 | {ds}"
  log_level    : "INFO"
""")

        write(OUT_DIR / f"overlay_{variant}_pos_lat8_{ds}.yaml", f"""\
# Overlay: Position | {variant} | lat8 | dataset: {ds}
root_folder : "{ROOT}"
input:
  predictions_csv : "{ae_out}/pos_cls_lat8/predictions_all.csv"
output:
  out_dir : "{ae_out}/pos_cls_lat8/overlay"
labels:
  label_order:
{label_list(POS_ORDER)}
  colors:
{color_list(POS_COLORS)}
misc:
  pad_size     : 64
  image_size   : 1024
  linewidth    : 0.6
  dpi          : 300
  title_prefix : "Position | {variant} | lat8 | {ds}"
  log_level    : "INFO"
""")

print(f"\nDone. {len(list(OUT_DIR.glob('*.yaml')))} configs total in {OUT_DIR}")
