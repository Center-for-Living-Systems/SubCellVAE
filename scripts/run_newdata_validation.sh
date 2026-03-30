#!/usr/bin/env bash
# =============================================================================
# run_newdata_validation.sh
# Full newdata validation pipeline:
#   STAGE 1 – Patch prep  (2 conditions)
#   STAGE 2 – AE apply    (4 AE variants → latents_newdata.csv)
#   STAGE 3 – Analysis    (4 runs: one per AE variant)
#   STAGE 4 – Cls apply   (16 runs: 4 AE × 2 targets × 2 feature sets)
#   STAGE 5 – Cross-classification vis  (8 runs: 4 AE × 2 feature sets)
#
# Results tree:  results/newdata_validation/
#   patches/
#     control/tiff_patches32/   plot_patches32/
#     ycomp/tiff_patches32/     plot_patches32/
#   {baseline, semisup_fa, semisup_pos, semisup_both}/
#     latents_newdata.csv
#     analysis/
#     fa_cls_lat8/          pos_cls_lat8/
#     fa_cls_lat8dist8/     pos_cls_lat8dist8/
#     vis_lat8/             vis_lat8dist8/
#
# BEFORE RUNNING:
#   1. Edit config/newdata_config/patchprep_control.yaml  – set image_folder
#   2. Edit config/newdata_config/patchprep_ycomp.yaml    – set image_folder
#   3. Edit config/newdata_config/patchprep_*.yaml        – set end_ind
#   4. Check that trained models exist in results/test_run/*/model_final.pt
#
# Usage:
#   cd /home/lding/lding/gitcode/SubCellAE
#   bash scripts/run_newdata_validation.sh
#   bash scripts/run_newdata_validation.sh 2>&1 | tee logs/newdata_validation.log
# =============================================================================

set -euo pipefail
# ---------------------------------------------------------------------------
# Change ROOT_FOLDER when working on a different computer.
# ---------------------------------------------------------------------------
ROOT_FOLDER="/home/lding/lding/fa_data_analysis"
PYTHON="conda run -n subcellae-cuda python"
CFG="config/newdata_config"

mkdir -p logs

echo "======================================================================"
echo " STAGE 1 — Patch preparation  (2 conditions)"
echo "======================================================================"

echo "--- [1/2] control patchprep ---"
$PYTHON scripts/run_patchprep_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/patchprep_control.yaml

echo "--- [2/2] ycomp patchprep ---"
$PYTHON scripts/run_patchprep_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/patchprep_ycomp.yaml

echo ""
echo "======================================================================"
echo " STAGE 2 — AE apply (inference)  (4 variants)"
echo "======================================================================"

echo "--- [1/4] baseline AE apply ---"
$PYTHON scripts/run_ae_apply_from_config.py   --root_folder "$ROOT_FOLDER" $CFG/ae_apply_baseline.yaml

echo "--- [2/4] semisup_fa AE apply ---"
$PYTHON scripts/run_ae_apply_from_config.py   --root_folder "$ROOT_FOLDER" $CFG/ae_apply_semisup_fa.yaml

echo "--- [3/4] semisup_pos AE apply ---"
$PYTHON scripts/run_ae_apply_from_config.py   --root_folder "$ROOT_FOLDER" $CFG/ae_apply_semisup_pos.yaml

echo "--- [4/4] semisup_both AE apply ---"
$PYTHON scripts/run_ae_apply_from_config.py   --root_folder "$ROOT_FOLDER" $CFG/ae_apply_semisup_both.yaml

echo ""
echo "======================================================================"
echo " STAGE 3 — Analysis  (4 runs: one per AE variant)"
echo "======================================================================"

echo "--- [1/4] baseline analysis ---"
$PYTHON scripts/run_analysis_from_config.py   --root_folder "$ROOT_FOLDER" $CFG/analysis_baseline.yaml

echo "--- [2/4] semisup_fa analysis ---"
$PYTHON scripts/run_analysis_from_config.py   --root_folder "$ROOT_FOLDER" $CFG/analysis_semisup_fa.yaml

echo "--- [3/4] semisup_pos analysis ---"
$PYTHON scripts/run_analysis_from_config.py   --root_folder "$ROOT_FOLDER" $CFG/analysis_semisup_pos.yaml

echo "--- [4/4] semisup_both analysis ---"
$PYTHON scripts/run_analysis_from_config.py   --root_folder "$ROOT_FOLDER" $CFG/analysis_semisup_both.yaml

echo ""
echo "======================================================================"
echo " STAGE 4 — Classifier apply  (16 runs: 4 AE × 2 targets × 2 feat sets)"
echo "======================================================================"

# ── baseline ──────────────────────────────────────────────────────────────
echo "--- baseline | FA type  | lat8 ---"
$PYTHON scripts/run_cls_apply_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_apply_baseline_fa_lat8.yaml

echo "--- baseline | Position | lat8 ---"
$PYTHON scripts/run_cls_apply_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_apply_baseline_pos_lat8.yaml

echo "--- baseline | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_cls_apply_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_apply_baseline_fa_lat8dist8.yaml

echo "--- baseline | Position | lat8+dist8 ---"
$PYTHON scripts/run_cls_apply_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_apply_baseline_pos_lat8dist8.yaml

# ── semisup_fa ────────────────────────────────────────────────────────────
echo "--- semisup_fa | FA type  | lat8 ---"
$PYTHON scripts/run_cls_apply_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_apply_semisup_fa_fa_lat8.yaml

echo "--- semisup_fa | Position | lat8 ---"
$PYTHON scripts/run_cls_apply_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_apply_semisup_fa_pos_lat8.yaml

echo "--- semisup_fa | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_cls_apply_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_apply_semisup_fa_fa_lat8dist8.yaml

echo "--- semisup_fa | Position | lat8+dist8 ---"
$PYTHON scripts/run_cls_apply_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_apply_semisup_fa_pos_lat8dist8.yaml

# ── semisup_pos ───────────────────────────────────────────────────────────
echo "--- semisup_pos | FA type  | lat8 ---"
$PYTHON scripts/run_cls_apply_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_apply_semisup_pos_fa_lat8.yaml

echo "--- semisup_pos | Position | lat8 ---"
$PYTHON scripts/run_cls_apply_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_apply_semisup_pos_pos_lat8.yaml

echo "--- semisup_pos | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_cls_apply_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_apply_semisup_pos_fa_lat8dist8.yaml

echo "--- semisup_pos | Position | lat8+dist8 ---"
$PYTHON scripts/run_cls_apply_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_apply_semisup_pos_pos_lat8dist8.yaml

# ── semisup_both ──────────────────────────────────────────────────────────
echo "--- semisup_both | FA type  | lat8 ---"
$PYTHON scripts/run_cls_apply_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_apply_semisup_both_fa_lat8.yaml

echo "--- semisup_both | Position | lat8 ---"
$PYTHON scripts/run_cls_apply_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_apply_semisup_both_pos_lat8.yaml

echo "--- semisup_both | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_cls_apply_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_apply_semisup_both_fa_lat8dist8.yaml

echo "--- semisup_both | Position | lat8+dist8 ---"
$PYTHON scripts/run_cls_apply_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_apply_semisup_both_pos_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " STAGE 5 — Cross-classification visualization  (8 runs: 4 AE × 2 feat)"
echo "======================================================================"

echo "--- baseline | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py  --root_folder "$ROOT_FOLDER" $CFG/vis_baseline_lat8.yaml

echo "--- baseline | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py  --root_folder "$ROOT_FOLDER" $CFG/vis_baseline_lat8dist8.yaml

echo "--- semisup_fa | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py  --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_fa_lat8.yaml

echo "--- semisup_fa | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py  --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_fa_lat8dist8.yaml

echo "--- semisup_pos | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py  --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_pos_lat8.yaml

echo "--- semisup_pos | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py  --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_pos_lat8dist8.yaml

echo "--- semisup_both | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py  --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_both_lat8.yaml

echo "--- semisup_both | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py  --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_both_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " ALL DONE"
echo "======================================================================"
