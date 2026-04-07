#!/usr/bin/env bash
# =============================================================================
# run_dualch_lat16.sh
# Dual-channel (ch1 + ch3) pipeline with latent_dim=16: 4 AEs → 16 cls → 8 vis
#
# Results tree:  ae_results/
#   dualch_run_lat16/
#     {baseline, semisup_fa, semisup_pos, semisup_both}/
#       latents.csv
#       analysis/
#       fa_cls_lat16/          pos_cls_lat16/
#       fa_cls_lat16dist8/     pos_cls_lat16dist8/
#       vis_lat16/             vis_lat16dist8/
#
# BEFORE RUNNING:
#   Patch prep (Stage 1) assumed already done by run_dualch.sh.
#
# Usage:
#   cd /home/lding/lding/gitcode/SubCellAE
#   bash scripts/run_dualch_lat16.sh
#   bash scripts/run_dualch_lat16.sh 2>&1 | tee logs/dualch_lat16_run.log
# =============================================================================

set -euo pipefail
ROOT_FOLDER="/home/lding/lding/fa_data_analysis"
PYTHON="python"
CFG="config/dualch_lat16_config"

mkdir -p logs

# STAGE 1 (patch preparation) already completed — skipped.

echo "======================================================================"
echo " STAGE 2 — Dual-channel AE training  (4 variants, latent_dim=16)"
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
echo "--- baseline | FA type  | lat16 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_fa_lat16.yaml

echo "--- baseline | Position | lat16 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_pos_lat16.yaml

echo "--- baseline | FA type  | lat16+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_fa_lat16dist8.yaml

echo "--- baseline | Position | lat16+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_pos_lat16dist8.yaml

# ── semisup_fa ────────────────────────────────────────────────────────────
echo "--- semisup_fa | FA type  | lat16 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_fa_fa_lat16.yaml

echo "--- semisup_fa | Position | lat16 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_fa_pos_lat16.yaml

echo "--- semisup_fa | FA type  | lat16+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_fa_fa_lat16dist8.yaml

echo "--- semisup_fa | Position | lat16+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_fa_pos_lat16dist8.yaml

# ── semisup_pos ───────────────────────────────────────────────────────────
echo "--- semisup_pos | FA type  | lat16 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_pos_fa_lat16.yaml

echo "--- semisup_pos | Position | lat16 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_pos_pos_lat16.yaml

echo "--- semisup_pos | FA type  | lat16+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_pos_fa_lat16dist8.yaml

echo "--- semisup_pos | Position | lat16+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_pos_pos_lat16dist8.yaml

# ── semisup_both ──────────────────────────────────────────────────────────
echo "--- semisup_both | FA type  | lat16 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_both_fa_lat16.yaml

echo "--- semisup_both | Position | lat16 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_both_pos_lat16.yaml

echo "--- semisup_both | FA type  | lat16+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_both_fa_lat16dist8.yaml

echo "--- semisup_both | Position | lat16+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_both_pos_lat16dist8.yaml

echo ""
echo "======================================================================"
echo " STAGE 5 — Cross-classification visualization  (8 runs: 4 AE × 2 feat)"
echo "======================================================================"

echo "--- baseline | lat16 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_baseline_lat16.yaml

echo "--- baseline | lat16+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_baseline_lat16dist8.yaml

echo "--- semisup_fa | lat16 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_fa_lat16.yaml

echo "--- semisup_fa | lat16+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_fa_lat16dist8.yaml

echo "--- semisup_pos | lat16 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_pos_lat16.yaml

echo "--- semisup_pos | lat16+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_pos_lat16dist8.yaml

echo "--- semisup_both | lat16 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_both_lat16.yaml

echo "--- semisup_both | lat16+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_both_lat16dist8.yaml

echo ""
echo "======================================================================"
echo " ALL DONE"
echo "======================================================================"
