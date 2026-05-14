#!/usr/bin/env bash
# =============================================================================
# Classification + visualization for contrastive_cio_rb_lat12proj8
# Runs 4 cls (fa+pos × z_recon+z_proj) and 2 vis (z_recon, z_proj)
#
# Usage:
#   conda activate subcellae-cuda
#   cd /path/to/SubCellAE-contrastive
#   bash scripts/run_contrastive_cio_rb_lat12proj8_cls_vis.sh
# =============================================================================

set -euo pipefail

PYTHON="python"
CFG="config/contrastive_config"
export PYTHONPATH="/home/lding/lding/gitcode/SubCellAE-contrastive"

echo "======================================================================"
echo " Classification  (4 runs: 2 targets × 2 feature sets)"
echo "======================================================================"

echo "--- [1/4] FA type  | z_recon ---"
$PYTHON scripts/run_classification_from_config.py $CFG/cls_contrastive_cio_rb_lat12proj8_fa_zrecon.yaml

echo "--- [2/4] FA type  | z_proj  ---"
$PYTHON scripts/run_classification_from_config.py $CFG/cls_contrastive_cio_rb_lat12proj8_fa_zproj.yaml

echo "--- [3/4] Position | z_recon ---"
$PYTHON scripts/run_classification_from_config.py $CFG/cls_contrastive_cio_rb_lat12proj8_pos_zrecon.yaml

echo "--- [4/4] Position | z_proj  ---"
$PYTHON scripts/run_classification_from_config.py $CFG/cls_contrastive_cio_rb_lat12proj8_pos_zproj.yaml

echo ""
echo "======================================================================"
echo " Visualization  (2 runs: z_recon, z_proj)"
echo "======================================================================"

echo "--- [1/2] z_recon ---"
$PYTHON scripts/run_cross_classification_vis.py $CFG/vis_contrastive_cio_rb_lat12proj8_zrecon.yaml

echo "--- [2/2] z_proj  ---"
$PYTHON scripts/run_cross_classification_vis.py $CFG/vis_contrastive_cio_rb_lat12proj8_zproj.yaml

echo ""
echo "======================================================================"
echo " ALL DONE"
echo "======================================================================"
