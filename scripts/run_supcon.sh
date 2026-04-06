#!/usr/bin/env bash
# =============================================================================
# run_supcon.sh
# Full pipeline for the supervised contrastive AE variant (SupCon):
#   1 AE training → analysis → 4 classifications → 2 visualizations
#
# Key differences vs run_contrastive.sh:
#   - model_type: "supcon" uses SupCon loss (same-class patches as positives)
#   - augmentation: random H/V flips + salt-and-pepper (no intensity scaling)
#   - requires annotation_file in config (labels needed for SupCon loss)
#
# Results tree:  ae_results/contrastive_run/supcon/
#   latents.csv
#   analysis/
#   fa_cls_lat8/          pos_cls_lat8/
#   fa_cls_lat8dist8/     pos_cls_lat8dist8/
#   vis_lat8/             vis_lat8dist8/
#
# Usage:
#   conda activate subcellae-cuda
#   cd /path/to/SubCellAE
#   bash scripts/run_supcon.sh
#   bash scripts/run_supcon.sh 2>&1 | tee logs/supcon_run.log
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Change ROOT_FOLDER when working on a different computer.
# ---------------------------------------------------------------------------
ROOT_FOLDER="/home/lding/lding/fa_data_analysis"
PYTHON="python"   # activate subcellae-cuda conda env before running this script
CFG="config/contrastive_config"

mkdir -p logs

echo "======================================================================"
echo " STAGE 1 — Supervised Contrastive AE training"
echo "======================================================================"

echo "--- [1/1] supcon AE ---"
$PYTHON scripts/run_ae_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/ae_supcon.yaml

echo ""
echo "======================================================================"
echo " STAGE 2 — Analysis"
echo "======================================================================"

echo "--- [1/1] supcon analysis ---"
$PYTHON scripts/run_analysis_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/analysis_supcon.yaml

echo ""
echo "======================================================================"
echo " STAGE 3 — Classification  (4 runs: 2 targets × 2 feature sets)"
echo "======================================================================"

echo "--- [1/4] supcon | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/cls_supcon_fa_lat8.yaml

echo "--- [2/4] supcon | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/cls_supcon_pos_lat8.yaml

echo "--- [3/4] supcon | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/cls_supcon_fa_lat8dist8.yaml

echo "--- [4/4] supcon | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/cls_supcon_pos_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " STAGE 4 — Cross-classification visualization  (2 feature sets)"
echo "======================================================================"

echo "--- [1/2] supcon | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/vis_supcon_lat8.yaml

echo "--- [2/2] supcon | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/vis_supcon_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " ALL DONE — results in: $ROOT_FOLDER/ae_results/contrastive_run/supcon/"
echo "======================================================================"
