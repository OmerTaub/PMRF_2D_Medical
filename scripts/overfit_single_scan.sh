#!/usr/bin/env bash
set -e
source /storage/omer/PMRF_2D_Medical/scripts/wandb_key.sh

# Sanity check: overfit on a SINGLE scan (same file for train + val).
# Expected: PSNR should climb rapidly to 40+ dB within minutes.
SINGLE_SCAN_DIR="/storage/omer/data/fastmri/overfit_single_scan"

EXP_NAME="OVERFIT_1scan_SWIN_L_trainDC_residual"
OUTPUT_DIR="experiments_mmse/${EXP_NAME}"

MASK_TYPE="random"
VAL_MASK_TYPE="equispaced"
CENTER_FRACTION=0.04
ACCEL=4

SCALE_MODE="none"
SCALE_PERCENTILE=100.0

python train_fastmri_pmrf.py \
  --phase "train" \
  --train_dataset "$SINGLE_SCAN_DIR" \
  --val_dataset "$SINGLE_SCAN_DIR" \
  --challenge "singlecoil" \
  --stage "mmse" \
  --arch "swinir_L" \
  --num_gpus 1 \
  --train_batch_size 4 \
  --val_batch_size 4 \
  --mask_type "$MASK_TYPE" \
  --val_mask_type "$VAL_MASK_TYPE" \
  --accelerations $ACCEL \
  --center_fractions $CENTER_FRACTION \
  --scale_mode "$SCALE_MODE" \
  --scale_percentile $SCALE_PERCENTILE \
  --resolution 320 \
  --num_workers 4 \
  --max_epochs 500 \
  --ema_decay -1 \
  --eps 0.0 \
  --t_schedule "stratified_uniform" \
  --weight_decay 0.0 \
  --lr 1e-3 \
  --wandb_project_name "PMRF_fastmri" \
  --wandb_group "fastmri_mmse_overfit" \
  --wandb_run_name "$EXP_NAME" \
  --output_dir "$OUTPUT_DIR"
