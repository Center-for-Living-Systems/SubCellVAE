#!/usr/bin/env bash
# =============================================================================
# run_ciorb_configs.sh
# Full pipeline on CIO-RB vinc patches: 4 AEs → 16 classifications → 8 vis
# Results: ae_results/test_run_ciorb/
#
# Usage:
#   bash scripts/run_ciorb_configs.sh
#   bash scripts/run_ciorb_configs.sh 2>&1 | tee logs/test_run_ciorb.log
# =============================================================================

set -euo pipefail
CFG="config/test_config_ciorb"

echo "======================================================================"
echo " STAGE 1 — Autoencoder training  (4 variants)"
echo "======================================================================"

echo "--- [1/4] baseline AE ---"
python scripts/run_ae_from_config.py $CFG/ae_baseline.yaml

echo "--- [2/4] semisup_fa AE ---"
python scripts/run_ae_from_config.py $CFG/ae_semisup_fa.yaml

echo "--- [3/4] semisup_pos AE ---"
python scripts/run_ae_from_config.py $CFG/ae_semisup_pos.yaml

echo "--- [4/4] semisup_both AE ---"
python scripts/run_ae_from_config.py $CFG/ae_semisup_both.yaml

echo ""
echo "======================================================================"
echo " STAGE 2 — Classification  (16 runs: 4 AE × 2 targets × 2 feature sets)"
echo "======================================================================"

# ── baseline ──────────────────────────────────────────────────────────────
echo "--- baseline | FA type  | lat8 ---"
python scripts/run_classification_from_config.py $CFG/cls_baseline_fa_lat8.yaml

echo "--- baseline | Position | lat8 ---"
python scripts/run_classification_from_config.py $CFG/cls_baseline_pos_lat8.yaml

echo "--- baseline | FA type  | lat8+dist8 ---"
python scripts/run_classification_from_config.py $CFG/cls_baseline_fa_lat8dist8.yaml

echo "--- baseline | Position | lat8+dist8 ---"
python scripts/run_classification_from_config.py $CFG/cls_baseline_pos_lat8dist8.yaml

# ── semisup_fa ────────────────────────────────────────────────────────────
echo "--- semisup_fa | FA type  | lat8 ---"
python scripts/run_classification_from_config.py $CFG/cls_semisup_fa_fa_lat8.yaml

echo "--- semisup_fa | Position | lat8 ---"
python scripts/run_classification_from_config.py $CFG/cls_semisup_fa_pos_lat8.yaml

echo "--- semisup_fa | FA type  | lat8+dist8 ---"
python scripts/run_classification_from_config.py $CFG/cls_semisup_fa_fa_lat8dist8.yaml

echo "--- semisup_fa | Position | lat8+dist8 ---"
python scripts/run_classification_from_config.py $CFG/cls_semisup_fa_pos_lat8dist8.yaml

# ── semisup_pos ───────────────────────────────────────────────────────────
echo "--- semisup_pos | FA type  | lat8 ---"
python scripts/run_classification_from_config.py $CFG/cls_semisup_pos_fa_lat8.yaml

echo "--- semisup_pos | Position | lat8 ---"
python scripts/run_classification_from_config.py $CFG/cls_semisup_pos_pos_lat8.yaml

echo "--- semisup_pos | FA type  | lat8+dist8 ---"
python scripts/run_classification_from_config.py $CFG/cls_semisup_pos_fa_lat8dist8.yaml

echo "--- semisup_pos | Position | lat8+dist8 ---"
python scripts/run_classification_from_config.py $CFG/cls_semisup_pos_pos_lat8dist8.yaml

# ── semisup_both ──────────────────────────────────────────────────────────
echo "--- semisup_both | FA type  | lat8 ---"
python scripts/run_classification_from_config.py $CFG/cls_semisup_both_fa_lat8.yaml

echo "--- semisup_both | Position | lat8 ---"
python scripts/run_classification_from_config.py $CFG/cls_semisup_both_pos_lat8.yaml

echo "--- semisup_both | FA type  | lat8+dist8 ---"
python scripts/run_classification_from_config.py $CFG/cls_semisup_both_fa_lat8dist8.yaml

echo "--- semisup_both | Position | lat8+dist8 ---"
python scripts/run_classification_from_config.py $CFG/cls_semisup_both_pos_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " STAGE 3 — Cross-classification visualization  (8 runs: 4 AE × 2 feature sets)"
echo "======================================================================"

echo "--- baseline | lat8 ---"
python scripts/run_cross_classification_vis.py $CFG/vis_baseline_lat8.yaml

echo "--- baseline | lat8+dist8 ---"
python scripts/run_cross_classification_vis.py $CFG/vis_baseline_lat8dist8.yaml

echo "--- semisup_fa | lat8 ---"
python scripts/run_cross_classification_vis.py $CFG/vis_semisup_fa_lat8.yaml

echo "--- semisup_fa | lat8+dist8 ---"
python scripts/run_cross_classification_vis.py $CFG/vis_semisup_fa_lat8dist8.yaml

echo "--- semisup_pos | lat8 ---"
python scripts/run_cross_classification_vis.py $CFG/vis_semisup_pos_lat8.yaml

echo "--- semisup_pos | lat8+dist8 ---"
python scripts/run_cross_classification_vis.py $CFG/vis_semisup_pos_lat8dist8.yaml

echo "--- semisup_both | lat8 ---"
python scripts/run_cross_classification_vis.py $CFG/vis_semisup_both_lat8.yaml

echo "--- semisup_both | lat8+dist8 ---"
python scripts/run_cross_classification_vis.py $CFG/vis_semisup_both_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " ALL DONE"
echo "======================================================================"
