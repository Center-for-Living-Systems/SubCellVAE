#!/usr/bin/env bash
# =============================================================================
# run_strategy_sweep.sh
# Full pipeline for the 4 training strategy sweep (semisup_both model)
#
# Strategies:
#   0322  — 200 ep, no weight decay, no warmup, no LR scheduler
#   0324  — 500 ep, weight_decay=1e-4, no warmup, no scheduler
#   mar30 — 500 ep, weight_decay=1e-4, warmup 200 ep, no scheduler
#   final — 500 ep, weight_decay=1e-4, warmup 200 ep, cosine LR → 1e-5
#
# Results tree:  ae_results/strategy_sweep/strategy_{0322,0324,mar30,final}/semisup_both/
#   latents.csv
#   fa_cls_lat8/          pos_cls_lat8/
#   fa_cls_lat8dist8/     pos_cls_lat8dist8/
#   vis_lat8/             vis_lat8dist8/
#
# Usage:
#   bash scripts/run_strategy_sweep.sh
#   bash scripts/run_strategy_sweep.sh 2>&1 | tee logs/strategy_sweep.log
# =============================================================================

set -euo pipefail
AE_CFG="config/training_strategies"
CLS_CFG="config/strategy_sweep"

echo "======================================================================"
echo " STAGE 1 — AE training  (4 strategies)"
echo "======================================================================"

echo "--- [1/4] strategy_0322 ---"
python scripts/run_ae_from_config.py $AE_CFG/strategy_0322.yaml

echo "--- [2/4] strategy_0324 ---"
python scripts/run_ae_from_config.py $AE_CFG/strategy_0324.yaml

echo "--- [3/4] strategy_mar30 ---"
python scripts/run_ae_from_config.py $AE_CFG/strategy_mar30.yaml

echo "--- [4/4] strategy_final ---"
python scripts/run_ae_from_config.py $AE_CFG/strategy_final.yaml

echo ""
echo "======================================================================"
echo " STAGE 2 — Classification  (16 runs: 4 strategies × 2 targets × 2 features)"
echo "======================================================================"

for strategy in 0322 0324 mar30 final; do
  echo "--- ${strategy} | FA type  | lat8 ---"
  python scripts/run_classification_from_config.py $CLS_CFG/cls_${strategy}_fa_lat8.yaml

  echo "--- ${strategy} | Position | lat8 ---"
  python scripts/run_classification_from_config.py $CLS_CFG/cls_${strategy}_pos_lat8.yaml

  echo "--- ${strategy} | FA type  | lat8+dist8 ---"
  python scripts/run_classification_from_config.py $CLS_CFG/cls_${strategy}_fa_lat8dist8.yaml

  echo "--- ${strategy} | Position | lat8+dist8 ---"
  python scripts/run_classification_from_config.py $CLS_CFG/cls_${strategy}_pos_lat8dist8.yaml
done

echo ""
echo "======================================================================"
echo " STAGE 3 — Visualization  (8 runs: 4 strategies × 2 feature sets)"
echo "======================================================================"

for strategy in 0322 0324 mar30 final; do
  echo "--- vis ${strategy} | lat8 ---"
  python scripts/run_cross_classification_vis.py $CLS_CFG/vis_${strategy}_lat8.yaml

  echo "--- vis ${strategy} | lat8+dist8 ---"
  python scripts/run_cross_classification_vis.py $CLS_CFG/vis_${strategy}_lat8dist8.yaml
done

echo ""
echo "======================================================================"
echo " ALL DONE"
echo "======================================================================"
