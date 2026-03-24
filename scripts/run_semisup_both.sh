#!/usr/bin/env bash
# =============================================================================
# run_semisup_both.sh
# Single-variant pipeline: semisup_both only  (1 AE + 1 analysis + 4 cls + 2 vis)
#
# Usage:
#   cd /mnt/d/lding/CLS_GitHub/SubCellAE
#   bash scripts/run_semisup_both.sh
#   bash scripts/run_semisup_both.sh 2>&1 | tee logs/semisup_both.log
# =============================================================================

set -euo pipefail
PYTHON="conda run -n subcellae python"
CFG="config/test_config"

echo "--- [1/8] semisup_both AE ---"
$PYTHON scripts/run_ae_from_config.py          $CFG/ae_semisup_both.yaml

echo "--- [2/8] semisup_both analysis ---"
$PYTHON scripts/run_analysis_from_config.py    $CFG/analysis_semisup_both.yaml

echo "--- [3/8] semisup_both | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_semisup_both_fa_lat8.yaml

echo "--- [4/8] semisup_both | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_semisup_both_pos_lat8.yaml

echo "--- [5/8] semisup_both | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_semisup_both_fa_lat8dist8.yaml

echo "--- [6/8] semisup_both | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_semisup_both_pos_lat8dist8.yaml

echo "--- [7/8] semisup_both | vis lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py    $CFG/vis_semisup_both_lat8.yaml

echo "--- [8/8] semisup_both | vis lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py    $CFG/vis_semisup_both_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " semisup_both DONE"
echo "======================================================================"
