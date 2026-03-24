#!/usr/bin/env bash
# =============================================================================
# run_test_configs.sh
# Full pipeline: 4 AEs → 16 classifications → 8 visualizations
#
# Results tree:  results/test_run/
#   {baseline, semisup_fa, semisup_pos, semisup_both}/
#     latents.csv
#     fa_cls_lat8/          pos_cls_lat8/
#     fa_cls_lat8dist8/     pos_cls_lat8dist8/
#     vis_lat8/             vis_lat8dist8/
#
# Usage:
#   cd /mnt/d/lding/CLS_GitHub/SubCellAE
#   bash scripts/run_test_configs.sh
#   bash scripts/run_test_configs.sh 2>&1 | tee logs/test_run.log
# =============================================================================

set -euo pipefail
PYTHON="conda run -n subcellae python"
CFG="config/test_config"

echo "======================================================================"
echo " STAGE 1 — Autoencoder training  (4 variants)"
echo "======================================================================"

echo "--- [1/4] baseline AE ---"
$PYTHON scripts/run_ae_from_config.py        $CFG/ae_baseline.yaml

echo "--- [2/4] semisup_fa AE ---"
$PYTHON scripts/run_ae_from_config.py        $CFG/ae_semisup_fa.yaml

echo "--- [3/4] semisup_pos AE ---"
$PYTHON scripts/run_ae_from_config.py        $CFG/ae_semisup_pos.yaml

echo "--- [4/4] semisup_both AE ---"
$PYTHON scripts/run_ae_from_config.py        $CFG/ae_semisup_both.yaml

echo ""
echo "======================================================================"
echo " STAGE 2 — Analysis  (4 runs: one per AE variant)"
echo "======================================================================"

echo "--- [1/4] baseline analysis ---"
$PYTHON scripts/run_analysis_from_config.py    $CFG/analysis_baseline.yaml

echo "--- [2/4] semisup_fa analysis ---"
$PYTHON scripts/run_analysis_from_config.py    $CFG/analysis_semisup_fa.yaml

echo "--- [3/4] semisup_pos analysis ---"
$PYTHON scripts/run_analysis_from_config.py    $CFG/analysis_semisup_pos.yaml

echo "--- [4/4] semisup_both analysis ---"
$PYTHON scripts/run_analysis_from_config.py    $CFG/analysis_semisup_both.yaml

echo ""
echo "======================================================================"
echo " STAGE 3 — Classification  (16 runs: 4 AE × 2 targets × 2 feature sets)"
echo "======================================================================"

# ── baseline ──────────────────────────────────────────────────────────────
echo "--- baseline | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_baseline_fa_lat8.yaml

echo "--- baseline | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_baseline_pos_lat8.yaml

echo "--- baseline | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_baseline_fa_lat8dist8.yaml

echo "--- baseline | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_baseline_pos_lat8dist8.yaml

# ── semisup_fa ────────────────────────────────────────────────────────────
echo "--- semisup_fa | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_semisup_fa_fa_lat8.yaml

echo "--- semisup_fa | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_semisup_fa_pos_lat8.yaml

echo "--- semisup_fa | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_semisup_fa_fa_lat8dist8.yaml

echo "--- semisup_fa | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_semisup_fa_pos_lat8dist8.yaml

# ── semisup_pos ───────────────────────────────────────────────────────────
echo "--- semisup_pos | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_semisup_pos_fa_lat8.yaml

echo "--- semisup_pos | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_semisup_pos_pos_lat8.yaml

echo "--- semisup_pos | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_semisup_pos_fa_lat8dist8.yaml

echo "--- semisup_pos | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_semisup_pos_pos_lat8dist8.yaml

# ── semisup_both ──────────────────────────────────────────────────────────
echo "--- semisup_both | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_semisup_both_fa_lat8.yaml

echo "--- semisup_both | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_semisup_both_pos_lat8.yaml

echo "--- semisup_both | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_semisup_both_fa_lat8dist8.yaml

echo "--- semisup_both | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  $CFG/cls_semisup_both_pos_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " STAGE 4 — Cross-classification visualization  (8 runs: 4 AE × 2 feature sets)"
echo "======================================================================"

echo "--- baseline | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py    $CFG/vis_baseline_lat8.yaml

echo "--- baseline | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py    $CFG/vis_baseline_lat8dist8.yaml

echo "--- semisup_fa | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py    $CFG/vis_semisup_fa_lat8.yaml

echo "--- semisup_fa | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py    $CFG/vis_semisup_fa_lat8dist8.yaml

echo "--- semisup_pos | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py    $CFG/vis_semisup_pos_lat8.yaml

echo "--- semisup_pos | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py    $CFG/vis_semisup_pos_lat8dist8.yaml

echo "--- semisup_both | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py    $CFG/vis_semisup_both_lat8.yaml

echo "--- semisup_both | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py    $CFG/vis_semisup_both_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " ALL DONE"
echo "======================================================================"
