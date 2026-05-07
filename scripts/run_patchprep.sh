#!/usr/bin/env bash
# =============================================================================
# run_patchprep.sh
# Run patch preparation for variant-paxillin datasets.
#
# Configs live in config/patchprep_config/{dataset}_{condition}_{norm}.yaml
# Output:  ae_results/patches/{norm}/{dataset}/{condition}/tiff_patches32/
#
# Datasets  : vinc  ppax  pfak  nih3t3
# Conditions: control  ycomp
# Norms     : cio (cell_insideoutside)  |  dataset (percentile stretch)
#           : cio_rb / dataset_rb  — same norms with rolling ball radius 10
# All runs  : on-the-fly cell segmentation by default.
#
# Usage:
#   bash scripts/run_patchprep.sh [--norm NORM] [--dataset DS] [--condition COND]
#
#   # All 16 combinations:
#   bash scripts/run_patchprep.sh
#
#   # Only CIO normalization, all datasets:
#   bash scripts/run_patchprep.sh --norm cio
#
#   # One dataset, both norms:
#   bash scripts/run_patchprep.sh --dataset vinc
#
#   # One dataset, one norm, one condition:
#   bash scripts/run_patchprep.sh --dataset nih3t3 --norm dataset --condition ycomp
#
#   # With log:
#   bash scripts/run_patchprep.sh --norm cio 2>&1 | tee logs/patchprep_cio.log
# =============================================================================

set -euo pipefail

PYTHON="python"
CFG="config/patchprep_config"

ALL_DATASETS=(vinc ppax pfak nih3t3)
ALL_CONDITIONS=(control ycomp)
ALL_NORMS=(cio dataset cio_rb dataset_rb)

# ── argument parsing ──────────────────────────────────────────────────────────
FILTER_NORM=""
FILTER_DS=""
FILTER_COND=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --norm)      FILTER_NORM="$2";  shift 2 ;;
        --dataset)   FILTER_DS="$2";    shift 2 ;;
        --condition) FILTER_COND="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1  ;;
    esac
done

# Resolve effective lists
NORMS=(      "${ALL_NORMS[@]}" )
DATASETS=(   "${ALL_DATASETS[@]}" )
CONDITIONS=( "${ALL_CONDITIONS[@]}" )

[[ -n "$FILTER_NORM" ]]  && NORMS=("$FILTER_NORM")
[[ -n "$FILTER_DS" ]]    && DATASETS=("$FILTER_DS")
[[ -n "$FILTER_COND" ]]  && CONDITIONS=("$FILTER_COND")

mkdir -p logs

echo "======================================================================"
echo " Patchprep"
echo "   norms      : ${NORMS[*]}"
echo "   datasets   : ${DATASETS[*]}"
echo "   conditions : ${CONDITIONS[*]}"
echo "   total runs : $(( ${#NORMS[@]} * ${#DATASETS[@]} * ${#CONDITIONS[@]} ))"
echo "======================================================================"

# ── run ───────────────────────────────────────────────────────────────────────
for NORM in "${NORMS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        for COND in "${CONDITIONS[@]}"; do
            CFG_FILE="$CFG/${DS}_${COND}_${NORM}.yaml"
            if [[ ! -f "$CFG_FILE" ]]; then
                echo "[WARN] config not found, skipping: $CFG_FILE"
                continue
            fi
            echo "--- [$NORM] $DS / $COND ---"
            $PYTHON scripts/run_patchprep_from_config.py "$CFG_FILE"
        done
    done
done

echo ""
echo "======================================================================"
echo " HISTOGRAMS — per-image + control vs ycomp comparison"
echo "======================================================================"

ROOT_FOLDER="/home/lding/lding/fa_data_analysis"
PATCH_BASE="$ROOT_FOLDER/ae_results/patches"

for NORM in "${NORMS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        CTRL_DIR="$PATCH_BASE/$NORM/$DS/control/tiff_patches32"
        YCOMP_DIR="$PATCH_BASE/$NORM/$DS/ycomp/tiff_patches32"
        HIST_DIR="$PATCH_BASE/$NORM/$DS/histograms"

        # Determine which dirs actually exist (may have run only one condition)
        EXISTING_DIRS=()
        [[ -d "$CTRL_DIR" ]]  && EXISTING_DIRS+=("$CTRL_DIR")
        [[ -d "$YCOMP_DIR" ]] && EXISTING_DIRS+=("$YCOMP_DIR")

        if [[ ${#EXISTING_DIRS[@]} -eq 0 ]]; then
            echo "  [SKIP] no patch dirs found for $NORM/$DS"
            continue
        fi

        echo "--- [$NORM] $DS ---"

        # Per-condition: overlaid (per-image lines) + grid
        $PYTHON scripts/plot_patch_intensity_histogram.py \
            "${EXISTING_DIRS[@]}" \
            --out-dir "$HIST_DIR"

        # Control vs ycomp on the same axes (only when both exist)
        if [[ ${#EXISTING_DIRS[@]} -eq 2 ]]; then
            $PYTHON scripts/plot_patch_intensity_histogram.py \
                "${EXISTING_DIRS[@]}" \
                --compare \
                --out-dir "$HIST_DIR"
        fi
    done
done

echo ""
echo "======================================================================"
echo " Done. Patches and histograms written to:"
echo "   ae_results/patches/{norm}/{dataset}/{condition}/tiff_patches32/"
echo "   ae_results/patches/{norm}/{dataset}/histograms/"
echo "======================================================================"
