#!/usr/bin/env bash
# =============================================================================
# run_variant_pax_cio.sh
# Apply CIO-trained models to all variant-paxillin datasets.
#
# Datasets (4 × 2 conditions = 8 patchprep runs):
#   vinc   – 20250311  Vinculin/rb647 + paxillin/m568  [= training experiment]
#   ppax   – 20250721  pPaxy118/rb647 + paxillin/m568  [= previously "newdata"]
#   pfak   – 20250720  pFAK/rb647 + paxillin/m568      [truly new]
#   nih3t3 – 20260227  NIH3T3, Vinc/rb647 + pax/m555   [truly new, diff cell line]
#
# Stages:
#   0: patchprep (4 datasets × 2 conditions = 8 runs)  — skippable
#   1: ae_apply  (VARIANTS × 4 datasets)
#   2: analysis  (VARIANTS × 4 datasets)
#   3: MSE comparison plots across train/val + all test datasets
#
# Usage:
#   cd /home/lding/lding/gitcode/SubCellAE
#   bash scripts/run_variant_pax_cio.sh <MODEL_RUN> [variant1 variant2 ...]
#   bash scripts/run_variant_pax_cio.sh test_run_cio_26041923
#   bash scripts/run_variant_pax_cio.sh test_run_cio_26041923 baseline conae
#   bash scripts/run_variant_pax_cio.sh test_run_cio_26041923 2>&1 | tee logs/variant_pax_cio.log
#
# Skip patchprep if patches already exist:
#   SKIP_PATCHPREP=1 bash scripts/run_variant_pax_cio.sh test_run_cio_26041923
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <MODEL_RUN> [variant1 variant2 ...]"
    echo "  MODEL_RUN: training run directory name, e.g. test_run_cio_26041923"
    echo "  variants:  optional subset, default = all 8"
    exit 1
fi

MODEL_RUN="$1"
shift

ROOT_FOLDER="/home/lding/lding/fa_data_analysis"
PYTHON="python"
CFG_DIR="config/variant_pax_cio_config"

ALL_VARIANTS=(baseline semisup_fa semisup_pos semisup_both conae semicon_fa semicon_pos semicon_both)
DATASETS=(vinc ppax pfak nih3t3)

# Use caller-supplied variants or fall back to all
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
echo " STAGE 0 — Patch preparation (cell_insideoutside, 4 × 2 = 8 runs)"
echo "======================================================================"

if [ "${SKIP_PATCHPREP:-0}" = "1" ]; then
    echo "  SKIP_PATCHPREP=1 — skipping patch preparation"
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
echo " STAGE 1 — AE apply  (${#VARIANTS[@]} variants × ${#DATASETS[@]} datasets)"
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
echo " STAGE 2 — Analysis / UMAP  (${#VARIANTS[@]} variants × ${#DATASETS[@]} datasets)"
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
echo " STAGE 3 — MSE comparison plots"
echo "======================================================================"

$PYTHON scripts/plot_mse_comparison.py \
    --train-run-dir "$TRAIN_DIR" \
    --apply-dir     "$APPLY_BASE" \
    --out-dir       "$APPLY_BASE/mse_comparison" \
    --variants "${VARIANTS[@]}"

echo ""
echo "======================================================================"
echo " Download targets (scp from cluster):"
echo "   MSE plots : $APPLY_BASE/mse_comparison/"
for VARIANT in "${VARIANTS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "   Analysis  : $APPLY_BASE/$VARIANT/$DS/analysis/"
    done
done
echo "======================================================================"
echo " ALL DONE"
echo "======================================================================"
