#!/usr/bin/env bash
# =============================================================================
# run_other_paxillin_cio.sh
# Apply CIO-trained models to 4 other-paxillin datasets (cell_insideoutside norm).
#
# Stages:
#   0: patchprep (4 datasets × 2 conditions = 8 runs)  — run once per dataset
#   1: ae_apply  (8 variants × 4 datasets = 32 runs)
#   2: analysis  (8 variants × 4 datasets = 32 runs)
#   3: MSE comparison plots across train/val/test (per variant)
#
# Usage:
#   cd /home/lding/lding/gitcode/SubCellAE
#   bash scripts/run_other_paxillin_cio.sh <MODEL_RUN>
#   bash scripts/run_other_paxillin_cio.sh test_run_cio_26041416
#   bash scripts/run_other_paxillin_cio.sh test_run_cio_26041416 2>&1 | tee logs/run_other_paxillin_cio.log
#
# Optional: skip patchprep (if patches_cio already exist):
#   SKIP_PATCHPREP=1 bash scripts/run_other_paxillin_cio.sh test_run_cio_26041416
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <MODEL_RUN>"
    echo "  MODEL_RUN: training run directory name, e.g. test_run_cio_26041923"
    exit 1
fi

MODEL_RUN="$1"
ROOT_FOLDER="/home/lding/lding/fa_data_analysis"
PYTHON="python"
CFG_DIR="config/other_paxillin_cio_config"

VARIANTS=(baseline semisup_fa semisup_pos semisup_both conae semicon_fa semicon_pos semicon_both)
VARIANTS=(conae)
DATASETS=(vinc pfak ppax nih3t3)

TRAIN_DIR="$ROOT_FOLDER/ae_results/$MODEL_RUN"
APPLY_BASE="$ROOT_FOLDER/ae_results/other_paxillin_cio/$MODEL_RUN"

echo "======================================================================"
echo " Model run : $MODEL_RUN"
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
echo " STAGE 1 — AE apply  (8 variants × 4 datasets = 32 runs)"
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
echo " STAGE 2 — Analysis / UMAP  (8 variants × 4 datasets = 32 runs)"
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
echo " STAGE 3 — MSE comparison plots  (1 per variant = 8 plots)"
echo "======================================================================"

$PYTHON scripts/plot_mse_comparison.py \
    --train-run-dir "$TRAIN_DIR" \
    --apply-dir     "$APPLY_BASE" \
    --out-dir       "$APPLY_BASE/mse_comparison" \
    --variants "${VARIANTS[@]}"

echo ""
echo "======================================================================"
echo " Download targets (scp from cluster):"
echo "   MSE comparison plots:"
echo "     $APPLY_BASE/mse_comparison/"
echo "   Per-dataset analysis (UMAP, MSE dist, etc.):"
for VARIANT in "${VARIANTS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "     $APPLY_BASE/$VARIANT/$DS/analysis/"
    done
done
echo "======================================================================"
echo " ALL DONE"
echo "======================================================================"
