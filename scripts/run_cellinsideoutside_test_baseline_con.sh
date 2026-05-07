#!/usr/bin/env bash
# =============================================================================
# run_cellinsideoutside_test_baseline_con.sh
# Pipeline: patchprep → baseline + conae → analysis → classification → vis
# Normalization: cell_insideoutside
# Variants: baseline, conae
#
# Usage:
#   cd /home/lding/lding/gitcode/SubCellAE
#   bash scripts/run_cellinsideoutside_test_baseline_con.sh
#   bash scripts/run_cellinsideoutside_test_baseline_con.sh 2>&1 | tee logs/test_run_cio_bc.log
# =============================================================================

set -euo pipefail

ROOT_FOLDER="/home/lding/lding/fa_data_analysis"
PYTHON="python"

# Timestamp: YYMMDDhh  (e.g. 26041416 for 2026-04-14 16:xx)
TS=$(date +%y%m%d%H)
RUN_NAME="test_run_cio_${TS}"
RUN_DIR="$ROOT_FOLDER/ae_results/$RUN_NAME"

echo "Run name : $RUN_NAME"
echo "Run dir  : $RUN_DIR"

# ── Stamp the timestamp into a temporary config directory ──────────────────
TMP_CFG=$(mktemp -d)
for f in config/test_config_cio/*.yaml; do
    sed "s|test_run_cio|$RUN_NAME|g" "$f" > "$TMP_CFG/$(basename $f)"
done
CFG="$TMP_CFG"
trap 'rm -rf "$TMP_CFG"' EXIT

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
echo " STAGE 0 — Patch preparation  (cell_insideoutside normalization)"
echo "======================================================================"

echo "--- [1/2] control patchprep ---"
$PYTHON scripts/run_patchprep_from_config.py config/config_control_czi_cell_insideoutside.yaml

echo "--- [2/2] ycomp patchprep ---"
$PYTHON scripts/run_patchprep_from_config.py config/config_ycomp_czi_cell_insideoutside.yaml

echo ""
echo "======================================================================"
echo " STAGE 1 — Autoencoder training  (2 variants)"
echo "======================================================================"

echo "--- [1/2] baseline AE ---"
$PYTHON scripts/run_ae_from_config.py --root_folder "$ROOT_FOLDER" $CFG/ae_baseline.yaml
pack "$RUN_DIR/baseline" "$RUN_DIR/baseline_ae.h5"

echo "--- [2/2] conae AE (NT-Xent contrastive) ---"
$PYTHON scripts/run_ae_from_config.py --root_folder "$ROOT_FOLDER" $CFG/ae_conae.yaml
pack "$RUN_DIR/conae"    "$RUN_DIR/conae_ae.h5"

echo ""
echo "======================================================================"
echo " STAGE 2 — Analysis  (2 runs)"
echo "======================================================================"

echo "--- [1/2] baseline ---"
$PYTHON scripts/run_analysis_from_config.py --root_folder "$ROOT_FOLDER" $CFG/analysis_baseline.yaml

echo "--- [2/2] conae ---"
$PYTHON scripts/run_analysis_from_config.py --root_folder "$ROOT_FOLDER" $CFG/analysis_conae.yaml

echo ""
echo "======================================================================"
echo " STAGE 3 — Classification  (8 runs: 2 AE × 2 targets × 2 feature sets)"
echo "======================================================================"

echo "--- baseline | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_fa_lat8.yaml

echo "--- baseline | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_pos_lat8.yaml

echo "--- baseline | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_fa_lat8dist8.yaml

echo "--- baseline | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_pos_lat8dist8.yaml

echo "--- conae | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_conae_fa_lat8.yaml

echo "--- conae | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_conae_pos_lat8.yaml

echo "--- conae | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_conae_fa_lat8dist8.yaml

echo "--- conae | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_conae_pos_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " STAGE 4 — Cross-classification visualization  (4 runs)"
echo "======================================================================"

echo "--- baseline | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_baseline_lat8.yaml

echo "--- baseline | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_baseline_lat8dist8.yaml

echo "--- conae | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_conae_lat8.yaml

echo "--- conae | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_conae_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " STAGE 5 — Pack all outputs"
echo "======================================================================"

for VARIANT in baseline conae; do
    echo "--- pack $VARIANT ---"
    pack "$RUN_DIR/$VARIANT" "$RUN_DIR/${VARIANT}_results.h5"
done

echo ""
echo "======================================================================"
echo " Download targets (scp from cluster):"
echo "   AE reconstruction QC (available after Stage 1):"
for VARIANT in baseline conae; do
    echo "     $RUN_DIR/${VARIANT}_ae.h5"
done
echo "   Full results (analysis + cls + vis, available after Stage 5):"
for VARIANT in baseline conae; do
    echo "     $RUN_DIR/${VARIANT}_results.h5"
done
echo "======================================================================"
echo " ALL DONE"
echo "======================================================================"
