#!/usr/bin/env bash
# =============================================================================
# run_contrastive.sh
# Full pipeline for the contrastive AE variant:
#   1 AE training → analysis → 4 classifications → 2 visualizations
#
# Results tree:  ae_results/contrastive_run/contrastive/
#   latents.csv
#   analysis/
#   fa_cls_lat8/          pos_cls_lat8/
#   fa_cls_lat8dist8/     pos_cls_lat8dist8/
#   vis_lat8/             vis_lat8dist8/
#
# Usage:
#   conda activate subcellae-cuda
#   cd /path/to/SubCellAE
#   bash scripts/run_contrastive.sh
#   bash scripts/run_contrastive.sh 2>&1 | tee logs/contrastive_run.log
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
echo " STAGE 1 — Contrastive AE training"
echo "======================================================================"

echo "--- [1/1] contrastive AE ---"
$PYTHON scripts/run_ae_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/ae_contrastive.yaml

echo ""
echo "======================================================================"
echo " STAGE 2 — Analysis"
echo "======================================================================"

echo "--- [1/1] contrastive analysis ---"
$PYTHON scripts/run_analysis_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/analysis_contrastive.yaml

echo ""
echo "======================================================================"
echo " STAGE 3 — Classification  (4 runs: 2 targets × 2 feature sets)"
echo "======================================================================"

echo "--- [1/4] contrastive | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/cls_contrastive_fa_lat8.yaml

echo "--- [2/4] contrastive | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/cls_contrastive_pos_lat8.yaml

echo "--- [3/4] contrastive | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/cls_contrastive_fa_lat8dist8.yaml

echo "--- [4/4] contrastive | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/cls_contrastive_pos_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " STAGE 4 — Cross-classification visualization  (2 feature sets)"
echo "======================================================================"

echo "--- [1/2] contrastive | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/vis_contrastive_lat8.yaml

echo "--- [2/2] contrastive | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py \
    --root_folder "$ROOT_FOLDER" \
    $CFG/vis_contrastive_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " ALL DONE — results in: $ROOT_FOLDER/ae_results/contrastive_run/"
echo "======================================================================"
