#!/usr/bin/env bash
# =============================================================================
# run_other_paxillin.sh
# Apply pre-trained models to 4 other_paxillin datasets:
#   vinc (vinculin), pfak (phospho-paxillin), ppax (paxillin), nih3t3
#
# For each dataset × 4 AE variants:
#   Stage 0: patch prep (8 runs: 4 datasets × 2 conditions)
#   Stage 1: ae_apply  → latents_newdata.csv
#   Stage 2: analysis  → UMAP (reuses pre-trained UMAP model)
#   Stage 3: cls_apply → predictions_all.csv (FA type + Position)
#   Stage 4: vis       → cross-classification plots
#   Stage 5: overlay   → whole-image coloured bounding-box TIFF
#
# Models from: ae_results/test_run_overfit_20260322/
#
# Usage:
#   cd /home/lding/lding/gitcode/SubCellAE
#   bash scripts/run_other_paxillin.sh
#   bash scripts/run_other_paxillin.sh 2>&1 | tee logs/run_other_paxillin.log
# =============================================================================

set -euo pipefail

ROOT_FOLDER="/home/lding/lding/fa_data_analysis"
PYTHON="python"
CFG="config/other_paxillin_config"

DATASETS=(vinc pfak ppax nih3t3)
VARIANTS=(baseline semisup_fa semisup_pos semisup_both)

echo "======================================================================"
echo " STAGE 0 — Patch preparation (4 datasets × 2 conditions = 8 runs)"
echo "======================================================================"

for DS in "${DATASETS[@]}"; do
    echo "--- [$DS] control ---"
    $PYTHON scripts/run_patchprep_from_config.py "$CFG/patchprep_${DS}_control.yaml"

    echo "--- [$DS] ycomp ---"
    $PYTHON scripts/run_patchprep_from_config.py "$CFG/patchprep_${DS}_ycomp.yaml"
done

echo ""
echo "======================================================================"
echo " STAGE 1 — AE apply  (4 variants × 4 datasets = 16 runs)"
echo "======================================================================"

for VARIANT in "${VARIANTS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "--- ae_apply | $VARIANT | $DS ---"
        $PYTHON scripts/run_ae_apply_from_config.py \
            --root_folder "$ROOT_FOLDER" \
            "$CFG/ae_apply_${VARIANT}_${DS}.yaml"
    done
done

echo ""
echo "======================================================================"
echo " STAGE 2 — Analysis / UMAP  (4 variants × 4 datasets = 16 runs)"
echo "======================================================================"

for VARIANT in "${VARIANTS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "--- analysis | $VARIANT | $DS ---"
        $PYTHON scripts/run_analysis_from_config.py \
            --root_folder "$ROOT_FOLDER" \
            "$CFG/analysis_${VARIANT}_${DS}.yaml"
    done
done

echo ""
echo "======================================================================"
echo " STAGE 3 — Classifier apply  (4 variants × 4 datasets × 2 targets = 32 runs)"
echo "======================================================================"

for VARIANT in "${VARIANTS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "--- cls_apply | $VARIANT | FA type | $DS ---"
        $PYTHON scripts/run_cls_apply_from_config.py \
            --root_folder "$ROOT_FOLDER" \
            "$CFG/cls_apply_${VARIANT}_fa_lat8_${DS}.yaml"

        echo "--- cls_apply | $VARIANT | Position | $DS ---"
        $PYTHON scripts/run_cls_apply_from_config.py \
            --root_folder "$ROOT_FOLDER" \
            "$CFG/cls_apply_${VARIANT}_pos_lat8_${DS}.yaml"
    done
done

echo ""
echo "======================================================================"
echo " STAGE 4 — Cross-classification vis  (4 variants × 4 datasets = 16 runs)"
echo "======================================================================"

for VARIANT in "${VARIANTS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "--- vis | $VARIANT | $DS ---"
        $PYTHON scripts/run_cross_classification_vis.py \
            --root_folder "$ROOT_FOLDER" \
            "$CFG/vis_${VARIANT}_lat8_${DS}.yaml"
    done
done

echo ""
echo "======================================================================"
echo " STAGE 5 — Whole-image overlay  (4 variants × 4 datasets × 2 targets = 32 runs)"
echo "======================================================================"

for VARIANT in "${VARIANTS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "--- overlay | $VARIANT | FA type | $DS ---"
        $PYTHON scripts/run_whole_image_overlay.py \
            --root_folder "$ROOT_FOLDER" \
            "$CFG/overlay_${VARIANT}_fa_lat8_${DS}.yaml"

        echo "--- overlay | $VARIANT | Position | $DS ---"
        $PYTHON scripts/run_whole_image_overlay.py \
            --root_folder "$ROOT_FOLDER" \
            "$CFG/overlay_${VARIANT}_pos_lat8_${DS}.yaml"
    done
done

echo ""
echo "======================================================================"
echo " STAGE 6 — Pack interactive HDF5  (4 variants × 4 datasets = 16 files)"
echo "======================================================================"

for VARIANT in "${VARIANTS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        OUT="$ROOT_FOLDER/ae_results/other_paxillin/$VARIANT/$DS"
        echo "--- pack $VARIANT/$DS ---"
        $PYTHON scripts/pack_interactive_h5.py "$OUT" --out "$OUT/interactive.h5"
    done
done

echo ""
echo "======================================================================"
echo " Download targets (scp from cluster):"
for VARIANT in "${VARIANTS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        echo "     $ROOT_FOLDER/ae_results/other_paxillin/$VARIANT/$DS/interactive.h5"
    done
done
echo "======================================================================"
echo " ALL DONE"
echo " View locally:  python scripts/view_interactive.py interactive.h5"
echo "======================================================================"
