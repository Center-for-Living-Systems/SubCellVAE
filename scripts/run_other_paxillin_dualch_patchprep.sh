#!/usr/bin/env bash
# =============================================================================
# run_other_paxillin_dualch_patchprep.sh
# Patch preparation for 4 other-paxillin datasets — dual-channel (ch1+ch3),
# cell_insideoutside normalization, no rolling ball.
#
# Produces per dataset/condition:
#   ae_results/other_paxillin/patches_dualch_cio/{ds}/{cond}/tiff_patches32_ch1/
#   ae_results/other_paxillin/patches_dualch_cio/{ds}/{cond}/tiff_patches32_ch3/
#
# Usage:
#   cd /home/lding/lding/gitcode/SubCellAE
#   bash scripts/run_other_paxillin_dualch_patchprep.sh
#   bash scripts/run_other_paxillin_dualch_patchprep.sh 2>&1 | tee logs/dualch_patchprep.log
# =============================================================================

set -euo pipefail
PYTHON="python"
CFG="config/other_paxillin_dualch_cio_config"

mkdir -p logs

echo "======================================================================"
echo " Dual-channel patchprep — cell_insideoutside, no rolling ball"
echo " 4 datasets × 2 conditions = 8 runs"
echo "======================================================================"

echo "--- [vinc] control ---"
$PYTHON scripts/run_patchprep_from_config.py "$CFG/patchprep_vinc_control.yaml"

echo "--- [vinc] ycomp ---"
$PYTHON scripts/run_patchprep_from_config.py "$CFG/patchprep_vinc_ycomp.yaml"

echo "--- [pfak] control ---"
$PYTHON scripts/run_patchprep_from_config.py "$CFG/patchprep_pfak_control.yaml"

echo "--- [pfak] ycomp ---"
$PYTHON scripts/run_patchprep_from_config.py "$CFG/patchprep_pfak_ycomp.yaml"

echo "--- [ppax] control ---"
$PYTHON scripts/run_patchprep_from_config.py "$CFG/patchprep_ppax_control.yaml"

echo "--- [ppax] ycomp ---"
$PYTHON scripts/run_patchprep_from_config.py "$CFG/patchprep_ppax_ycomp.yaml"

echo "--- [nih3t3] control ---"
$PYTHON scripts/run_patchprep_from_config.py "$CFG/patchprep_nih3t3_control.yaml"

echo "--- [nih3t3] ycomp ---"
$PYTHON scripts/run_patchprep_from_config.py "$CFG/patchprep_nih3t3_ycomp.yaml"

echo ""
echo "======================================================================"
echo " ALL DONE — patches written to:"
echo "   ae_results/other_paxillin/patches_dualch_cio/{ds}/{cond}/tiff_patches32_ch{1,3}/"
echo "======================================================================"
