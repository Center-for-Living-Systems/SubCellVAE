#!/usr/bin/env bash
# =============================================================================
# run_supcon_noflip.sh
# SupCon AE variant with salt-and-pepper noise ONLY (no geometric flips).
# Use to isolate whether flips help or hurt compared to run_supcon.sh.
#
# Results: ae_results/contrastive_run/supcon_noflip/
#
# Usage:
#   conda activate subcellae-cuda
#   cd /path/to/SubCellAE
#   bash scripts/run_supcon_noflip.sh 2>&1 | tee logs/supcon_noflip_run.log
# =============================================================================

set -euo pipefail

ROOT_FOLDER="/home/lding/lding/fa_data_analysis"
PYTHON="python"
CFG="config/contrastive_config"

mkdir -p logs

echo "======================================================================"
echo " STAGE 1 — SupCon AE training  (noise only, no flips)"
echo "======================================================================"
$PYTHON scripts/run_ae_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/ae_supcon_noflip.yaml

echo ""
echo "======================================================================"
echo " STAGE 2 — Analysis"
echo "======================================================================"
$PYTHON scripts/run_analysis_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/analysis_supcon_noflip.yaml

echo ""
echo "======================================================================"
echo " STAGE 3 — Classification  (4 runs)"
echo "======================================================================"
echo "--- [1/4] FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/cls_supcon_noflip_fa_lat8.yaml

echo "--- [2/4] Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/cls_supcon_noflip_pos_lat8.yaml

echo "--- [3/4] FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/cls_supcon_noflip_fa_lat8dist8.yaml

echo "--- [4/4] Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/cls_supcon_noflip_pos_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " STAGE 4 — Visualization"
echo "======================================================================"
echo "--- [1/2] lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/vis_supcon_noflip_lat8.yaml

echo "--- [2/2] lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/vis_supcon_noflip_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " ALL DONE — results in: $ROOT_FOLDER/ae_results/contrastive_run/supcon_noflip/"
echo "======================================================================"
