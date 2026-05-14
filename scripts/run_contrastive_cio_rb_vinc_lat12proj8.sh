#!/usr/bin/env bash
# =============================================================================
# Full pipeline: contrastive AE on vinc CIO-RB patches, latent=12 proj=8
#   Stage 1: AE training (vinc control + ycomp, 500 epochs)
#   Stage 2: Analysis (UMAP on z_recon and z_proj)
#   Stage 3: Classification (fa + pos, z_recon and z_proj)
#   Stage 4: Cross-classification visualization (z_recon, z_proj)
#
# Results: ae_results/contrastive_run/contrastive_cio_rb_vinc_lat12proj8/
#
# Usage:
#   conda activate subcellae-cuda
#   cd /path/to/SubCellAE-contrastive
#   bash scripts/run_contrastive_cio_rb_vinc_lat12proj8.sh [2>&1 | tee logs/vinc_run.log]
# =============================================================================

set -euo pipefail

export PYTHONPATH="/home/lding/lding/gitcode/SubCellAE-contrastive"
PYTHON="python"
CFG="config/contrastive_config"

mkdir -p logs

echo "======================================================================"
echo " STAGE 1 — AE training (vinc CIO-RB, latent=12, proj=8)"
echo "======================================================================"
$PYTHON scripts/run_ae_from_config.py \
    $CFG/ae_contrastive_cio_rb_vinc_lat12proj8.yaml

echo ""
echo "======================================================================"
echo " STAGE 2 — Analysis (UMAP z_recon + z_proj)"
echo "======================================================================"
$PYTHON scripts/run_analysis_from_config.py \
    $CFG/analysis_contrastive_cio_rb_vinc_lat12proj8.yaml

echo ""
echo "======================================================================"
echo " STAGE 3 — Classification  (4 runs: 2 targets × 2 feature sets)"
echo "======================================================================"

echo "--- [1/4] FA type  | z_recon ---"
$PYTHON scripts/run_classification_from_config.py \
    $CFG/cls_contrastive_cio_rb_vinc_lat12proj8_fa_zrecon.yaml

echo "--- [2/4] FA type  | z_proj  ---"
$PYTHON scripts/run_classification_from_config.py \
    $CFG/cls_contrastive_cio_rb_vinc_lat12proj8_fa_zproj.yaml

echo "--- [3/4] Position | z_recon ---"
$PYTHON scripts/run_classification_from_config.py \
    $CFG/cls_contrastive_cio_rb_vinc_lat12proj8_pos_zrecon.yaml

echo "--- [4/4] Position | z_proj  ---"
$PYTHON scripts/run_classification_from_config.py \
    $CFG/cls_contrastive_cio_rb_vinc_lat12proj8_pos_zproj.yaml

echo ""
echo "======================================================================"
echo " STAGE 4 — Visualization  (2 runs: z_recon, z_proj)"
echo "======================================================================"

echo "--- [1/2] z_recon ---"
$PYTHON scripts/run_cross_classification_vis.py \
    $CFG/vis_contrastive_cio_rb_vinc_lat12proj8_zrecon.yaml

echo "--- [2/2] z_proj  ---"
$PYTHON scripts/run_cross_classification_vis.py \
    $CFG/vis_contrastive_cio_rb_vinc_lat12proj8_zproj.yaml

echo ""
echo "======================================================================"
echo " ALL DONE — results in ae_results/contrastive_run/contrastive_cio_rb_vinc_lat12proj8/"
echo "======================================================================"
