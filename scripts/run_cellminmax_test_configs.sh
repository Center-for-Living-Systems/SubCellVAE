#!/usr/bin/env bash
# =============================================================================
# run_cellminmax_test_configs.sh
# Full pipeline: 4 AEs → 4 analyses → 16 classifications → 8 visualizations
# Normalization: cell_minmax
#
# Results land in a timestamped directory, e.g.:
#   ae_results/test_run_minmax_26041416/{baseline,...}/
#
# Usage:
#   cd /home/lding/lding/gitcode/SubCellAE
#   bash scripts/run_cellminmax_test_configs.sh
#   bash scripts/run_cellminmax_test_configs.sh 2>&1 | tee logs/test_run_minmax.log
# =============================================================================

set -euo pipefail

ROOT_FOLDER="/home/lding/lding/fa_data_analysis"
PYTHON="python"

# Timestamp: YYMMDDhh  (e.g. 26041416 for 2026-04-14 16:xx)
TS=$(date +%y%m%d%H)
RUN_NAME="test_run_minmax_${TS}"
RUN_DIR="$ROOT_FOLDER/ae_results/$RUN_NAME"

echo "Run name : $RUN_NAME"
echo "Run dir  : $RUN_DIR"

# ── Stamp the timestamp into a temporary config directory ──────────────────
TMP_CFG=$(mktemp -d)
# AE configs: test_run_cell_minmax → RUN_NAME
for f in config/test_config/ae_*.yaml; do
    sed "s|test_run_cell_minmax|$RUN_NAME|g" "$f" > "$TMP_CFG/$(basename $f)"
done
# Analysis / cls / vis: ae_results/test_run/ → ae_results/RUN_NAME/
for f in config/test_config/analysis_*.yaml config/test_config/cls_*.yaml config/test_config/vis_*.yaml; do
    sed "s|ae_results/test_run/|ae_results/$RUN_NAME/|g" "$f" > "$TMP_CFG/$(basename $f)"
done
CFG="$TMP_CFG"
trap 'rm -rf "$TMP_CFG"' EXIT   # clean up on exit

pack() {
    local src="$1" out="$2"
    if [ -d "$src" ]; then
        echo "  [pack] $src → $out"
        $PYTHON scripts/pack_run_to_h5.py "$src" --out "$out"
    else
        echo "  [pack] SKIP (not found): $src"
    fi
}

echo "======================================================================"
echo " STAGE 1 — Autoencoder training  (4 variants)"
echo "======================================================================"

echo "--- [1/4] baseline AE ---"
$PYTHON scripts/run_ae_from_config.py        --root_folder "$ROOT_FOLDER" $CFG/ae_baseline.yaml
pack "$RUN_DIR/baseline"     "$RUN_DIR/baseline_ae.h5"

echo "--- [2/4] semisup_fa AE ---"
$PYTHON scripts/run_ae_from_config.py        --root_folder "$ROOT_FOLDER" $CFG/ae_semisup_fa.yaml
pack "$RUN_DIR/semisup_fa"   "$RUN_DIR/semisup_fa_ae.h5"

echo "--- [3/4] semisup_pos AE ---"
$PYTHON scripts/run_ae_from_config.py        --root_folder "$ROOT_FOLDER" $CFG/ae_semisup_pos.yaml
pack "$RUN_DIR/semisup_pos"  "$RUN_DIR/semisup_pos_ae.h5"

echo "--- [4/4] semisup_both AE ---"
$PYTHON scripts/run_ae_from_config.py        --root_folder "$ROOT_FOLDER" $CFG/ae_semisup_both.yaml
pack "$RUN_DIR/semisup_both" "$RUN_DIR/semisup_both_ae.h5"

echo ""
echo "======================================================================"
echo " STAGE 2 — Analysis  (4 runs: one per AE variant)"
echo "======================================================================"

echo "--- [1/4] baseline analysis ---"
$PYTHON scripts/run_analysis_from_config.py    --root_folder "$ROOT_FOLDER" $CFG/analysis_baseline.yaml

echo "--- [2/4] semisup_fa analysis ---"
$PYTHON scripts/run_analysis_from_config.py    --root_folder "$ROOT_FOLDER" $CFG/analysis_semisup_fa.yaml

echo "--- [3/4] semisup_pos analysis ---"
$PYTHON scripts/run_analysis_from_config.py    --root_folder "$ROOT_FOLDER" $CFG/analysis_semisup_pos.yaml

echo "--- [4/4] semisup_both analysis ---"
$PYTHON scripts/run_analysis_from_config.py    --root_folder "$ROOT_FOLDER" $CFG/analysis_semisup_both.yaml

echo ""
echo "======================================================================"
echo " STAGE 3 — Classification  (16 runs: 4 AE × 2 targets × 2 feature sets)"
echo "======================================================================"

# ── baseline ──────────────────────────────────────────────────────────────
echo "--- baseline | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_fa_lat8.yaml

echo "--- baseline | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_pos_lat8.yaml

echo "--- baseline | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_fa_lat8dist8.yaml

echo "--- baseline | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_pos_lat8dist8.yaml

# ── semisup_fa ────────────────────────────────────────────────────────────
echo "--- semisup_fa | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_fa_fa_lat8.yaml

echo "--- semisup_fa | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_fa_pos_lat8.yaml

echo "--- semisup_fa | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_fa_fa_lat8dist8.yaml

echo "--- semisup_fa | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_fa_pos_lat8dist8.yaml

# ── semisup_pos ───────────────────────────────────────────────────────────
echo "--- semisup_pos | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_pos_fa_lat8.yaml

echo "--- semisup_pos | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_pos_pos_lat8.yaml

echo "--- semisup_pos | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_pos_fa_lat8dist8.yaml

echo "--- semisup_pos | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_pos_pos_lat8dist8.yaml

# ── semisup_both ──────────────────────────────────────────────────────────
echo "--- semisup_both | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_both_fa_lat8.yaml

echo "--- semisup_both | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_both_pos_lat8.yaml

echo "--- semisup_both | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_both_fa_lat8dist8.yaml

echo "--- semisup_both | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py  --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_both_pos_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " STAGE 4 — Cross-classification visualization  (8 runs: 4 AE × 2 feature sets)"
echo "======================================================================"

echo "--- baseline | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py    --root_folder "$ROOT_FOLDER" $CFG/vis_baseline_lat8.yaml

echo "--- baseline | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py    --root_folder "$ROOT_FOLDER" $CFG/vis_baseline_lat8dist8.yaml

echo "--- semisup_fa | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py    --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_fa_lat8.yaml

echo "--- semisup_fa | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py    --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_fa_lat8dist8.yaml

echo "--- semisup_pos | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py    --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_pos_lat8.yaml

echo "--- semisup_pos | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py    --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_pos_lat8dist8.yaml

echo "--- semisup_both | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py    --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_both_lat8.yaml

echo "--- semisup_both | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py    --root_folder "$ROOT_FOLDER" $CFG/vis_semisup_both_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " STAGE 5 — Pack all outputs into HDF5  (one file per variant)"
echo "======================================================================"

for VARIANT in baseline semisup_fa semisup_pos semisup_both; do
    echo "--- pack $VARIANT ---"
    pack "$RUN_DIR/$VARIANT" "$RUN_DIR/${VARIANT}_results.h5"
done

echo ""
echo "======================================================================"
echo " Download targets (scp from cluster):"
echo "   AE reconstruction QC (available after Stage 1):"
for VARIANT in baseline semisup_fa semisup_pos semisup_both; do
    echo "     $RUN_DIR/${VARIANT}_ae.h5"
done
echo "   Full results (analysis + cls + vis, available after Stage 5):"
for VARIANT in baseline semisup_fa semisup_pos semisup_both; do
    echo "     $RUN_DIR/${VARIANT}_results.h5"
done
echo "======================================================================"
echo " ALL DONE"
echo "======================================================================"
