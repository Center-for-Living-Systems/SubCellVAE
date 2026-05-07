#!/usr/bin/env bash
# =============================================================================
# run_cellinsideoutside_test_configs.sh
# Full pipeline: patchprep → 8 AEs → 8 analyses → 32 classifications → 16 vis
# Normalization: cell_insideoutside (median BG sub → cell mean div → fixed scale)
# Variants: baseline, semisup_{fa,pos,both}, conae, semicon_{fa,pos,both}
#
# Results land in a timestamped directory, e.g.:
#   ae_results/test_run_cio_26041416/{baseline,...}/
#
# Usage:
#   cd /home/lding/lding/gitcode/SubCellAE
#   bash scripts/run_cellinsideoutside_test_configs.sh
#   bash scripts/run_cellinsideoutside_test_configs.sh 2>&1 | tee logs/test_run_cio.log
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
echo " STAGE 0 — Patch preparation  (cell_insideoutside normalization)"
echo "======================================================================"

if [ "${SKIP_PATCHPREP:-0}" = "1" ]; then
    echo "  [SKIP] SKIP_PATCHPREP=1 — using existing patches"
else
    echo "--- [1/2] control patchprep ---"
    $PYTHON scripts/run_patchprep_from_config.py config/config_control_czi_cell_insideoutside.yaml

    echo "--- [2/2] ycomp patchprep ---"
    $PYTHON scripts/run_patchprep_from_config.py config/config_ycomp_czi_cell_insideoutside.yaml
fi

echo ""
echo "======================================================================"
echo " STAGE 1 — Autoencoder training  (8 variants)"
echo "======================================================================"

echo "--- [1/8] baseline AE ---"
$PYTHON scripts/run_ae_from_config.py --root_folder "$ROOT_FOLDER" $CFG/ae_baseline.yaml
pack "$RUN_DIR/baseline"      "$RUN_DIR/baseline_ae.h5"

echo "--- [2/8] semisup_fa AE ---"
$PYTHON scripts/run_ae_from_config.py --root_folder "$ROOT_FOLDER" $CFG/ae_semisup_fa.yaml
pack "$RUN_DIR/semisup_fa"    "$RUN_DIR/semisup_fa_ae.h5"

echo "--- [3/8] semisup_pos AE ---"
$PYTHON scripts/run_ae_from_config.py --root_folder "$ROOT_FOLDER" $CFG/ae_semisup_pos.yaml
pack "$RUN_DIR/semisup_pos"   "$RUN_DIR/semisup_pos_ae.h5"

echo "--- [4/8] semisup_both AE ---"
$PYTHON scripts/run_ae_from_config.py --root_folder "$ROOT_FOLDER" $CFG/ae_semisup_both.yaml
pack "$RUN_DIR/semisup_both"  "$RUN_DIR/semisup_both_ae.h5"

echo "--- [5/8] conae AE (NT-Xent contrastive) ---"
$PYTHON scripts/run_ae_from_config.py --root_folder "$ROOT_FOLDER" $CFG/ae_conae.yaml
pack "$RUN_DIR/conae"         "$RUN_DIR/conae_ae.h5"

echo "--- [6/8] semicon_fa AE (SupCon, FA labels) ---"
$PYTHON scripts/run_ae_from_config.py --root_folder "$ROOT_FOLDER" $CFG/ae_semicon_fa.yaml
pack "$RUN_DIR/semicon_fa"    "$RUN_DIR/semicon_fa_ae.h5"

echo "--- [7/8] semicon_pos AE (SupCon, Position labels) ---"
$PYTHON scripts/run_ae_from_config.py --root_folder "$ROOT_FOLDER" $CFG/ae_semicon_pos.yaml
pack "$RUN_DIR/semicon_pos"   "$RUN_DIR/semicon_pos_ae.h5"

echo "--- [8/8] semicon_both AE (SupCon, FA+Position labels) ---"
$PYTHON scripts/run_ae_from_config.py --root_folder "$ROOT_FOLDER" $CFG/ae_semicon_both.yaml
pack "$RUN_DIR/semicon_both"  "$RUN_DIR/semicon_both_ae.h5"

echo ""
echo "======================================================================"
echo " STAGE 2 — Analysis  (8 runs)"
echo "======================================================================"

echo "--- [1/8] baseline ---"
$PYTHON scripts/run_analysis_from_config.py --root_folder "$ROOT_FOLDER" $CFG/analysis_baseline.yaml

echo "--- [2/8] semisup_fa ---"
$PYTHON scripts/run_analysis_from_config.py --root_folder "$ROOT_FOLDER" $CFG/analysis_semisup_fa.yaml

echo "--- [3/8] semisup_pos ---"
$PYTHON scripts/run_analysis_from_config.py --root_folder "$ROOT_FOLDER" $CFG/analysis_semisup_pos.yaml

echo "--- [4/8] semisup_both ---"
$PYTHON scripts/run_analysis_from_config.py --root_folder "$ROOT_FOLDER" $CFG/analysis_semisup_both.yaml

echo "--- [5/8] conae ---"
$PYTHON scripts/run_analysis_from_config.py --root_folder "$ROOT_FOLDER" $CFG/analysis_conae.yaml

echo "--- [6/8] semicon_fa ---"
$PYTHON scripts/run_analysis_from_config.py --root_folder "$ROOT_FOLDER" $CFG/analysis_semicon_fa.yaml

echo "--- [7/8] semicon_pos ---"
$PYTHON scripts/run_analysis_from_config.py --root_folder "$ROOT_FOLDER" $CFG/analysis_semicon_pos.yaml

echo "--- [8/8] semicon_both ---"
$PYTHON scripts/run_analysis_from_config.py --root_folder "$ROOT_FOLDER" $CFG/analysis_semicon_both.yaml

echo ""
echo "======================================================================"
echo " STAGE 3 — Classification  (32 runs: 8 AE × 2 targets × 2 feature sets)"
echo "======================================================================"

echo "--- baseline | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_fa_lat8.yaml

echo "--- baseline | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_pos_lat8.yaml

echo "--- baseline | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_fa_lat8dist8.yaml

echo "--- baseline | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_baseline_pos_lat8dist8.yaml

echo "--- semisup_fa | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_fa_fa_lat8.yaml

echo "--- semisup_fa | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_fa_pos_lat8.yaml

echo "--- semisup_fa | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_fa_fa_lat8dist8.yaml

echo "--- semisup_fa | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_fa_pos_lat8dist8.yaml

echo "--- semisup_pos | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_pos_fa_lat8.yaml

echo "--- semisup_pos | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_pos_pos_lat8.yaml

echo "--- semisup_pos | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_pos_fa_lat8dist8.yaml

echo "--- semisup_pos | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_pos_pos_lat8dist8.yaml

echo "--- semisup_both | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_both_fa_lat8.yaml

echo "--- semisup_both | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_both_pos_lat8.yaml

echo "--- semisup_both | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_both_fa_lat8dist8.yaml

echo "--- semisup_both | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semisup_both_pos_lat8dist8.yaml

echo "--- conae | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_conae_fa_lat8.yaml

echo "--- conae | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_conae_pos_lat8.yaml

echo "--- conae | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_conae_fa_lat8dist8.yaml

echo "--- conae | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_conae_pos_lat8dist8.yaml

echo "--- semicon_fa | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semicon_fa_fa_lat8.yaml

echo "--- semicon_fa | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semicon_fa_pos_lat8.yaml

echo "--- semicon_fa | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semicon_fa_fa_lat8dist8.yaml

echo "--- semicon_fa | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semicon_fa_pos_lat8dist8.yaml

echo "--- semicon_pos | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semicon_pos_fa_lat8.yaml

echo "--- semicon_pos | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semicon_pos_pos_lat8.yaml

echo "--- semicon_pos | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semicon_pos_fa_lat8dist8.yaml

echo "--- semicon_pos | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semicon_pos_pos_lat8dist8.yaml

echo "--- semicon_both | FA type  | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semicon_both_fa_lat8.yaml

echo "--- semicon_both | Position | lat8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semicon_both_pos_lat8.yaml

echo "--- semicon_both | FA type  | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semicon_both_fa_lat8dist8.yaml

echo "--- semicon_both | Position | lat8+dist8 ---"
$PYTHON scripts/run_classification_from_config.py --root_folder "$ROOT_FOLDER" $CFG/cls_semicon_both_pos_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " STAGE 4 — Cross-classification visualization  (16 runs)"
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

echo "--- conae | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_conae_lat8.yaml

echo "--- conae | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_conae_lat8dist8.yaml

echo "--- semicon_fa | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semicon_fa_lat8.yaml

echo "--- semicon_fa | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semicon_fa_lat8dist8.yaml

echo "--- semicon_pos | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semicon_pos_lat8.yaml

echo "--- semicon_pos | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semicon_pos_lat8dist8.yaml

echo "--- semicon_both | lat8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semicon_both_lat8.yaml

echo "--- semicon_both | lat8+dist8 ---"
$PYTHON scripts/run_cross_classification_vis.py --root_folder "$ROOT_FOLDER" $CFG/vis_semicon_both_lat8dist8.yaml

echo ""
echo "======================================================================"
echo " STAGE 5 — Pack all outputs into TIFF stacks"
echo "======================================================================"

for VARIANT in baseline semisup_fa semisup_pos semisup_both conae semicon_fa semicon_pos semicon_both; do
    echo "--- pack $VARIANT ---"
    pack "$RUN_DIR/$VARIANT" "$RUN_DIR/${VARIANT}_results.h5"
done

echo ""
echo "======================================================================"
echo " Download targets (scp from cluster):"
echo "   AE reconstruction QC (available after Stage 1):"
for VARIANT in baseline semisup_fa semisup_pos semisup_both conae semicon_fa semicon_pos semicon_both; do
    echo "     $RUN_DIR/${VARIANT}_ae.h5"
done
echo "   Full results (analysis + cls + vis, available after Stage 5):"
for VARIANT in baseline semisup_fa semisup_pos semisup_both conae semicon_fa semicon_pos semicon_both; do
    echo "     $RUN_DIR/${VARIANT}_results.h5"
done
echo "======================================================================"
echo " ALL DONE"
echo "======================================================================"
