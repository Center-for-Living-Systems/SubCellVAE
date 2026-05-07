#!/usr/bin/env bash
# =============================================================================
# run_variant_pax.sh
# Apply trained models to all variant-paxillin datasets (full pipeline).
#
# Datasets (4 × 2 conditions = 8 patchprep runs):
#   vinc   – 20250311  Vinculin/rb647 + paxillin/m568  [= training experiment]
#   ppax   – 20250721  pPaxy118/rb647 + paxillin/m568  [= previously "newdata"]
#   pfak   – 20250720  pFAK/rb647 + paxillin/m568      [truly new]
#   nih3t3 – 20260227  NIH3T3, Vinc/rb647 + pax/m555   [truly new, diff cell line]
#
# Arguments:
#   MODEL_RUN  — training run directory (e.g. test_run_cio_26041923)
#   NORM       — normalization used by that model: cio | dataset
#                selects config folder (variant_pax_{NORM}_config) and
#                patch subdir (patches_{NORM}/) — keeps model and patches consistent
#   [variants] — optional subset of the 8 variants (default: all)
#
# Stages:
#   0: patchprep   (4 datasets × 2 conditions)           — skippable
#   1: ae_apply    (variants × 4 datasets)
#   2: analysis    (variants × 4 datasets)
#   3: cls_apply   (variants × 4 datasets × 2 targets)
#   4: vis         (variants × 4 datasets)
#   5: overlay     (variants × 4 datasets × 2 targets)
#   6: MSE comparison plots
#
# Usage:
#   cd /home/lding/lding/gitcode/SubCellAE
#   bash scripts/run_variant_pax.sh <MODEL_RUN> <NORM> [variant1 variant2 ...]
#
#   # CIO-trained model, all 8 variants:
#   bash scripts/run_variant_pax.sh test_run_cio_26041923 cio
#
#   # Dataset-trained model, baseline + conae only:
#   bash scripts/run_variant_pax.sh test_run_20260322 dataset baseline conae
#
#   # Skip patchprep (patches already exist):
#   SKIP_PATCHPREP=1 bash scripts/run_variant_pax.sh test_run_cio_26041923 cio
#
#   # With log:
#   bash scripts/run_variant_pax.sh test_run_cio_26041923 cio 2>&1 | tee logs/variant_pax_cio.log
# =============================================================================

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <MODEL_RUN> <NORM> [variant1 variant2 ...]"
    echo "  MODEL_RUN : training run name, e.g. test_run_cio_26041923"
    echo "  NORM      : cio | dataset"
    echo "  variants  : optional subset (default: all 8)"
    exit 1
fi

MODEL_RUN="$1"
NORM="$2"
shift 2

if [[ "$NORM" != "cio" && "$NORM" != "dataset" ]]; then
    echo "Error: NORM must be 'cio' or 'dataset', got '$NORM'"
    exit 1
fi

ROOT_FOLDER="/home/lding/lding/fa_data_analysis"
PYTHON="python"
CFG_DIR="config/variant_pax_${NORM}_config"

ALL_VARIANTS=(baseline semisup_fa semisup_pos semisup_both conae semicon_fa semicon_pos semicon_both)
DATASETS=(vinc ppax pfak nih3t3)

if [ $# -gt 0 ]; then
    VARIANTS=("$@")
else
    VARIANTS=("${ALL_VARIANTS[@]}")
fi

TRAIN_DIR="$ROOT_FOLDER/ae_results/$MODEL_RUN"
APPLY_BASE="$ROOT_FOLDER/ae_results/variant_pax/$MODEL_RUN"

mkdir -p logs

echo "======================================================================"
echo " Model run : $MODEL_RUN"
echo " Norm      : $NORM  (configs: $CFG_DIR)"
echo " Variants  : ${VARIANTS[*]}"
echo " Train dir : $TRAIN_DIR"
echo " Apply dir : $APPLY_BASE"
echo "======================================================================"

# ── Stamp MODEL_RUN into a temporary config directory ─────────────────────
TMP_CFG=$(mktemp -d)
for f in "$CFG_DIR"/*.yaml; do
    sed "s|MODEL_RUN|$MODEL_RUN|g" "$f" > "$TMP_CFG/$(basename $f)"
done
trap 'rm -rf "$TMP_CFG"' EXIT

echo ""
echo "======================================================================"
echo " STAGE 0 — Patch preparation (${NORM} norm, 4 × 2 = 8 runs)"
echo "======================================================================"

if [ "${SKIP_PATCHPREP:-0}" = "1" ]; then
    echo "  SKIP_PATCHPREP=1 — skipping"
else
    for DS in "${DATASETS[@]}"; do
        echo "--- [$DS] control ---"
        $PYTHON scripts/run_patchprep_from_config.py "$TMP_CFG/patchprep_${DS}_control.yaml"
        echo "--- [$DS] ycomp ---"
        $PYTHON scripts/run_patchprep_from_config.py "$TMP_CFG/patchprep_${DS}_ycomp.yaml"
    done
fi

echo ""
echo "======================================================================"
echo " STAGE 1 — AE apply  (${#VARIANTS[@]} × ${#DATASETS[@]})"
echo "======================================================================"

for VARIANT in "${VARIANTS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "--- ae_apply | $VARIANT | $DS ---"
        $PYTHON scripts/run_ae_apply_from_config.py \
            --root_folder "$ROOT_FOLDER" \
            "$TMP_CFG/ae_apply_${VARIANT}_${DS}.yaml"
    done
done

echo ""
echo "======================================================================"
echo " STAGE 2 — Analysis / UMAP  (${#VARIANTS[@]} × ${#DATASETS[@]})"
echo "======================================================================"

for VARIANT in "${VARIANTS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "--- analysis | $VARIANT | $DS ---"
        $PYTHON scripts/run_analysis_from_config.py \
            --root_folder "$ROOT_FOLDER" \
            "$TMP_CFG/analysis_${VARIANT}_${DS}.yaml"
    done
done

echo ""
echo "======================================================================"
echo " STAGE 3 — Classifier apply  (${#VARIANTS[@]} × ${#DATASETS[@]} × 2 targets)"
echo "======================================================================"

for VARIANT in "${VARIANTS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "--- cls FA  | $VARIANT | $DS ---"
        $PYTHON scripts/run_classification_apply_from_config.py \
            --root_folder "$ROOT_FOLDER" \
            "$TMP_CFG/cls_apply_${VARIANT}_fa_lat8_${DS}.yaml"
        echo "--- cls Pos | $VARIANT | $DS ---"
        $PYTHON scripts/run_classification_apply_from_config.py \
            --root_folder "$ROOT_FOLDER" \
            "$TMP_CFG/cls_apply_${VARIANT}_pos_lat8_${DS}.yaml"
    done
done

echo ""
echo "======================================================================"
echo " STAGE 4 — Cross-classification visualization  (${#VARIANTS[@]} × ${#DATASETS[@]})"
echo "======================================================================"

for VARIANT in "${VARIANTS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "--- vis | $VARIANT | $DS ---"
        $PYTHON scripts/run_cross_classification_vis.py \
            --root_folder "$ROOT_FOLDER" \
            "$TMP_CFG/vis_${VARIANT}_lat8_${DS}.yaml"
    done
done

echo ""
echo "======================================================================"
echo " STAGE 5 — Overlay  (${#VARIANTS[@]} × ${#DATASETS[@]} × 2 targets)"
echo "======================================================================"

for VARIANT in "${VARIANTS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "--- overlay FA  | $VARIANT | $DS ---"
        $PYTHON scripts/run_overlay_from_config.py \
            --root_folder "$ROOT_FOLDER" \
            "$TMP_CFG/overlay_${VARIANT}_fa_lat8_${DS}.yaml"
        echo "--- overlay Pos | $VARIANT | $DS ---"
        $PYTHON scripts/run_overlay_from_config.py \
            --root_folder "$ROOT_FOLDER" \
            "$TMP_CFG/overlay_${VARIANT}_pos_lat8_${DS}.yaml"
    done
done

echo ""
echo "======================================================================"
echo " STAGE 6 — MSE comparison plots"
echo "======================================================================"

$PYTHON scripts/plot_mse_comparison.py \
    --train-run-dir "$TRAIN_DIR" \
    --apply-dir     "$APPLY_BASE" \
    --out-dir       "$APPLY_BASE/mse_comparison" \
    --variants "${VARIANTS[@]}"

echo ""
echo "======================================================================"
echo " ALL DONE — results at: $APPLY_BASE"
echo "======================================================================"
