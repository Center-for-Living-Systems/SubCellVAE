#!/usr/bin/env bash
# =============================================================================
# run_dualch.sh
# Full dual-channel (ch1 + ch3) pipeline: patch prep → 4 AEs → 16 cls → 8 vis
#
# Results tree:  ae_results/
#   pax_ch_patch/
#     patch_control_ch1/   patch_control_ch3/
#     patch_ycomp_ch1/     patch_ycomp_ch3/
#   dualch_run/
#     {baseline, semisup_fa, semisup_pos, semisup_both}/
#       latents.csv
#       analysis/
#       fa_cls_lat8/          pos_cls_lat8/
#       fa_cls_lat8dist8/     pos_cls_lat8dist8/
#       vis_lat8/             vis_lat8dist8/
#
# BEFORE RUNNING:
#   1. Check patchprep configs in config/dualch_config/patchprep_*.yaml
#   2. Ensure cell mask dirs exist (same masks used as single-channel runs)
#
# Usage:
#   cd /home/lding/lding/gitcode/SubCellAE
#   bash scripts/run_dualch.sh
#   bash scripts/run_dualch.sh 2>&1 | tee logs/dualch_run.log
# =============================================================================

set -euo pipefail
ROOT_FOLDER="/home/lding/lding/fa_data_analysis"
PYTHON="python"
CFG="config/dualch_config"

mkdir -p logs

# STAGE 1 (patch preparation) already completed — skipped.

echo "======================================================================"
echo " STAGE 2 — Dual-channel AE training  (4 variants)"
echo "======================================================================"

echo "--- [1/4] baseline AE ---"
$PYTHON scripts/run_ae_from_config.py --root_folder "$ROOT_FOLDER" $CFG/ae_baseline.yaml

echo "--- [2/4] semisup_fa AE ---"
$PYTHON scripts/run_ae_from_config.py --root_folder "$ROOT_FOLDER" $CFG/ae_semisup_fa.yaml

echo "--- [3/4] semisup_pos AE ---"
$PYTHON scripts/run_ae_from_config.py --root_folder "$ROOT_FOLDER" $CFG/ae_semisup_pos.yaml

echo "--- [4/4] semisup_both AE ---"
$PYTHON scripts/run_ae_from_config.py --root_folder "$ROOT_FOLDER" $CFG/ae_semisup_both.yaml

echo ""
echo "======================================================================"
echo " STAGE 3 — Analysis  (4 runs)"
echo "======================================================================"

echo "--- [1/4] baseline analysis ---"
$PYTHON scripts/run_analysis_from_config.py --root_folder "$ROOT_FOLDER" $CFG/analysis_baseline.yaml

echo "--- [2/4] semisup_fa analysis ---"
$PYTHON scripts/run_analysis_from_config.py --root_folder "$ROOT_FOLDER" $CFG/analysis_semisup_fa.yaml

echo "--- [3/4] semisup_pos analysis ---"
$PYTHON scripts/run_analysis_from_config.py --root_folder "$ROOT_FOLDER" $CFG/analysis_semisup_pos.yaml

echo "--- [4/4] semisup_both analysis ---"
$PYTHON scripts/run_analysis_from_config.py --root_folder "$ROOT_FOLDER" $CFG/analysis_semisup_both.yaml

echo ""
echo "======================================================================"
echo " STAGE 4 — Classification  (16 runs: 4 AE × 2 targets × 2 feat sets)"
echo "======================================================================"

# ── baseline ──────────────────────────────────────────────────────────────
echo "--- baseline | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_fa_lat8.yaml

echo "--- baseline | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_pos_lat8.yaml

echo "--- baseline | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_fa_lat8dist8.yaml

echo "--- baseline | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_pos_lat8dist8.yaml

# ── semisup_fa ────────────────────────────────────────────────────────────
echo "--- semisup_fa | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_fa_fa_lat8.yaml

echo "--- semisup_fa | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_fa_pos_lat8.yaml

echo "--- semisup_fa | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_fa_fa_lat8dist8.yaml

echo "--- semisup_fa | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_fa_pos_lat8dist8.yaml

# ── semisup_pos ───────────────────────────────────────────────────────────
echo "--- semisup_pos | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_pos_fa_lat8.yaml

echo "--- semisup_pos | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_pos_pos_lat8.yaml

echo "--- semisup_pos | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_pos_fa_lat8dist8.yaml

echo "--- semisup_pos | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_pos_pos_lat8dist8.yaml

# ── semisup_both ──────────────────────────────────────────────────────────
echo "--- semisup_both | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_both_fa_lat8.yaml

echo "--- semisup_both | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_both_pos_lat8.yaml

echo "--- semisup_both | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_both_fa_lat8dist8.yaml

echo "--- semisup_both | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_both_pos_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " STAGE 5 — Cross-classification visualization  (8 runs: 4 AE × 2 feat)"
echo "======================================================================"

echo "--- baseline | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_baseline_lat8.yaml

echo "--- baseline | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_baseline_lat8dist8.yaml

echo "--- semisup_fa | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_fa_lat8.yaml

echo "--- semisup_fa | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_fa_lat8dist8.yaml

echo "--- semisup_pos | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_pos_lat8.yaml

echo "--- semisup_pos | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_pos_lat8dist8.yaml

echo "--- semisup_both | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_both_lat8.yaml

echo "--- semisup_both | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_both_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " ALL DONE"
echo "======================================================================"
