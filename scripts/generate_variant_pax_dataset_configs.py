#!/usr/bin/env python3
"""
Generate config/variant_pax_dataset_config/ — all 232 yaml configs for the
dataset-normalization variant-pax pipeline (8 variants × 4 datasets, full pipeline).

Run once from the repo root:
    python scripts/generate_variant_pax_dataset_configs.py
"""

from pathlib import Path

# ── constants ────────────────────────────────────────────────────────────────
VARIANTS   = ['baseline', 'semisup_fa', 'semisup_pos', 'semisup_both',
               'conae', 'semicon_fa', 'semicon_pos', 'semicon_both']
DATASETS   = ['vinc', 'ppax', 'pfak', 'nih3t3']
CONDITIONS = ['control', 'ycomp']
TARGETS    = ['fa', 'pos']   # classifier / overlay targets

ROOT = '"/home/lding/lding/fa_data_analysis"'   # yaml string literal

DS_INFO = {
    'vinc': {
        'desc':    'vinc (20250311 – Vinculin/rb647 + paxillin/m568)  [= training experiment]',
        'subdir':  '20250311_eGFPZyxin488_Phalloidin405_Vinculin(rb)647_paxillin(m)568',
        'cond_dir':{'control': 'Control', 'ycomp': 'Ycomp'},
    },
    'ppax': {
        'desc':    'ppax (20250721 – pPaxy118/rb647 + paxillin/m568)  [= previously "newdata"]',
        'subdir':  '20250721_eGFP-Zyxin 488_Phalloidin405_pPaxy118(rb) 647_Pax(m)568',
        'cond_dir':{'control': 'Control', 'ycomp': 'Y-comp'},
    },
    'pfak': {
        'desc':    'pfak (20250720 – pFAK/rb647 + paxillin/m568)  [truly new dataset]',
        'subdir':  '20250720_eGFP-Zyxin 488, Phalloidin 405, pFAK (rb) 647, paxillin(m)568/072025',
        'cond_dir':{'control': 'Control', 'ycomp': 'Ycomp'},
    },
    'nih3t3': {
        'desc':    'nih3t3 (20260227 – NIH3T3, Vinc/rb647 + pax/m555)  [truly new, different cell line]',
        'subdir':  '20260227_NIH3T3_ZyxinGFP,Phalloidin405,Vinc_rb647,Pax_m555_reduced_size_AH',
        'cond_dir':{'control': 'Control', 'ycomp': 'YCompound'},
    },
}

FA_LABEL_ORDER = [
    '"Nascent Adhesion"', '"focal complex"', '"focal adhesion"',
    '"fibrillar adhesion"', '"No adhesion"',
]
FA_COLORS = ['"#e6194b"', '"#f58231"', '"#3cb44b"', '"#4363d8"', '"#aaaaaa"']

POS_LABEL_ORDER = [
    '"Cell Protruding Edge"', '"Cell Periphery/other"',
    '"Lamella"', '"Cell Body"',
]
POS_COLORS = ['"#e6194b"', '"#f58231"', '"#3cb44b"', '"#4363d8"']

CLS_FOLDER = {'fa': 'fa_cls_lat8', 'pos': 'pos_cls_lat8'}

# ── helpers ──────────────────────────────────────────────────────────────────
def _label_items(label_order):
    return '\n'.join(f'    - {l}' for l in label_order)

def _color_items(colors):
    return '\n'.join(f'    - {c}' for c in colors)

def write(path: Path, content: str) -> None:
    path.write_text(content)
    print(f'  wrote {path.name}')

# ── patchprep (8 files) ──────────────────────────────────────────────────────
def gen_patchprep(out: Path) -> None:
    for ds, info in DS_INFO.items():
        for cond in CONDITIONS:
            cond_dir = info['cond_dir'][cond]
            content = f"""\
# Dataset: {info['desc']}  |  condition: {cond}
# Normalization: dataset  (match training data)
paths:
  root_folder      : {ROOT}
  image_folder     : root_folder + "/fa_data/other_paxillin/{info['subdir']}/{cond_dir}"
  cell_mask_folder : null
  patch_output_dir : root_folder + "/ae_results/variant_pax/patches_dataset/{ds}/{cond}/tiff_patches32"
  plot_output_dir  : root_folder + "/ae_results/variant_pax/patches_dataset/{ds}/{cond}/plot_patches32"
experiment:
  condition : "{cond}"
  major_ch  : 1
input:
  file_type : "czi"
  start_ind : 0
  end_ind   : 999
patch:
  patch_size   : 32
  mask_ratio   : 0.4
  pad_size     : 64
  patch_prefix : "{cond}"
preprocessing:
  rolling_ball_radius : null
normalization:
  norm_mode : "dataset"
  norm_lo   : 0.2
  norm_hi   : 99.8
segmentation:
  seg_ch                  : 3
  seg_threshold           : 0.1
  seg_close_size          : 5
  seg_min_size_initial    : 3
  seg_min_size_post_close : 10
  seg_min_size_final      : 30000
augmentation:
  rand_trans    : false
  max_shift_px  : 0
  rand_rota     : false
  max_angle_deg : 0.0
misc:
  dpi           : 256
  debug         : false
  log_level     : "INFO"
  use_timestamp : false
"""
            write(out / f'patchprep_{ds}_{cond}.yaml', content)

# ── ae_apply (32 files) ──────────────────────────────────────────────────────
def gen_ae_apply(out: Path) -> None:
    for var in VARIANTS:
        for ds in DATASETS:
            content = f"""\
# AE Apply (dataset): {var} | dataset: {ds}
# MODEL_RUN is substituted at runtime by run_variant_pax_dataset.sh
root_folder : {ROOT}
model:
  model_pt : root_folder + "/ae_results/MODEL_RUN/{var}/model_final.pt"

data:
  patch_dirs:
    - path          : root_folder + "/ae_results/variant_pax/patches_dataset/{ds}/control/tiff_patches32"
      condition     : 0
      condition_name: "control"
    - path          : root_folder + "/ae_results/variant_pax/patches_dataset/{ds}/ycomp/tiff_patches32"
      condition     : 1
      condition_name: "ycomp"

output:
  out_dir : root_folder + "/ae_results/variant_pax/MODEL_RUN/{var}/{ds}"

inference:
  batch_size : 128

reconstruction:
  save_recon       : true
  recon_pad_size   : 64
  recon_image_size : 1024

misc:
  device    : "auto"
  log_level : "INFO"
"""
            write(out / f'ae_apply_{var}_{ds}.yaml', content)

# ── analysis (32 files) ──────────────────────────────────────────────────────
def gen_analysis(out: Path) -> None:
    for var in VARIANTS:
        for ds in DATASETS:
            content = f"""\
# Analysis (dataset): {var} | dataset: {ds}
# MODEL_RUN is substituted at runtime by run_variant_pax_dataset.sh
root_folder : {ROOT}
input:
  latents_csv  : root_folder + "/ae_results/variant_pax/MODEL_RUN/{var}/{ds}/latents_newdata.csv"
  split_filter : "all"

output:
  out_dir : root_folder + "/ae_results/variant_pax/MODEL_RUN/{var}/{ds}/analysis"

embedding:
  methods:
    - UMAP
  umap_model_pkl    : root_folder + "/ae_results/MODEL_RUN/{var}/fa_cls_lat8/umap_all_model.pkl"
  umap_n_neighbors  : 15
  umap_min_dist     : 0.1
  umap_random_state : 42

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
"""
            write(out / f'analysis_{var}_{ds}.yaml', content)

# ── cls_apply (64 files: 8 var × 4 ds × 2 targets) ──────────────────────────
def gen_cls_apply(out: Path) -> None:
    for var in VARIANTS:
        for ds in DATASETS:
            for tgt in TARGETS:
                folder   = CLS_FOLDER[tgt]
                lo = FA_LABEL_ORDER if tgt == 'fa' else POS_LABEL_ORDER
                content = f"""\
# Cls Apply ({tgt.upper()} type, dataset): {var} | lat8 | dataset: {ds}
# MODEL_RUN is substituted at runtime by run_variant_pax_dataset.sh
root_folder : {ROOT}
input:
  latents_csv : root_folder + "/ae_results/variant_pax/MODEL_RUN/{var}/{ds}/latents_newdata.csv"

model:
  model_pkl      : root_folder + "/ae_results/MODEL_RUN/{var}/{folder}/lgbm_model.pkl"
  umap_model_pkl : root_folder + "/ae_results/MODEL_RUN/{var}/{folder}/umap_all_model.pkl"

output:
  out_dir : root_folder + "/ae_results/variant_pax/MODEL_RUN/{var}/{ds}/{folder}"

labels:
  label_order:
{_label_items(lo)}

features:
  feature_cols : null

dist_features:
  patch_prep_dirs: null
  feature_weight : 20.0

misc:
  log_level : "INFO"
"""
                write(out / f'cls_apply_{var}_{tgt}_lat8_{ds}.yaml', content)

# ── vis (32 files: 8 var × 4 ds) ─────────────────────────────────────────────
def gen_vis(out: Path) -> None:
    for var in VARIANTS:
        for ds in DATASETS:
            content = f"""\
# Cross-cls vis (dataset): {var} | lat8 | dataset: {ds}
# MODEL_RUN is substituted at runtime by run_variant_pax_dataset.sh
root_folder : {ROOT}
input:
  latents_csv          : root_folder + "/ae_results/variant_pax/MODEL_RUN/{var}/{ds}/latents_newdata.csv"
  fa_type_results_csv  : root_folder + "/ae_results/variant_pax/MODEL_RUN/{var}/{ds}/fa_cls_lat8/predictions_all.csv"
  position_results_csv : root_folder + "/ae_results/variant_pax/MODEL_RUN/{var}/{ds}/pos_cls_lat8/predictions_all.csv"
  umap_model_pkl       : root_folder + "/ae_results/MODEL_RUN/{var}/fa_cls_lat8/umap_all_model.pkl"
  annotation_csv       : null

output:
  out_dir : root_folder + "/ae_results/variant_pax/MODEL_RUN/{var}/{ds}/vis_lat8"

labels:
  fa_type_label_col  : "classification"
  position_label_col : "Position"
  fa_type_order:
{_label_items(FA_LABEL_ORDER)}
  position_order:
{_label_items(POS_LABEL_ORDER)}

misc:
  random_state : 42
  log_level    : "INFO"
"""
            write(out / f'vis_{var}_lat8_{ds}.yaml', content)

# ── overlay (64 files: 8 var × 4 ds × 2 targets) ────────────────────────────
def gen_overlay(out: Path) -> None:
    for var in VARIANTS:
        for ds in DATASETS:
            for tgt in TARGETS:
                folder = CLS_FOLDER[tgt]
                lo     = FA_LABEL_ORDER if tgt == 'fa' else POS_LABEL_ORDER
                colors = FA_COLORS      if tgt == 'fa' else POS_COLORS
                content = f"""\
# Overlay ({tgt.upper()} type, dataset): {var} | lat8 | dataset: {ds}
# MODEL_RUN is substituted at runtime by run_variant_pax_dataset.sh
root_folder : {ROOT}
input:
  predictions_csv : root_folder + "/ae_results/variant_pax/MODEL_RUN/{var}/{ds}/{folder}/predictions_all.csv"
output:
  out_dir : root_folder + "/ae_results/variant_pax/MODEL_RUN/{var}/{ds}/{folder}/overlay"
labels:
  label_order:
{_label_items(lo)}
  colors:
{_color_items(colors)}
misc:
  pad_size     : 64
  image_size   : 1024
  linewidth    : 0.6
  dpi          : 300
  title_prefix : "{tgt.upper()} type | {var} | lat8 | {ds}"
  log_level    : "INFO"
"""
                write(out / f'overlay_{var}_{tgt}_lat8_{ds}.yaml', content)

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    out = Path(__file__).parents[1] / 'config' / 'variant_pax_dataset_config'
    out.mkdir(exist_ok=True)
    print(f'Writing to {out}')
    gen_patchprep(out)
    gen_ae_apply(out)
    gen_analysis(out)
    gen_cls_apply(out)
    gen_vis(out)
    gen_overlay(out)
    total = len(list(out.glob('*.yaml')))
    print(f'\nDone — {total} configs generated')

if __name__ == '__main__':
    main()
