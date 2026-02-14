#!/usr/bin/env bash
set -e
source /storage/omer/PMRF_2D_Medical/scripts/wandb_key.sh

# Overfitting sanity check: train on a single .h5 file to verify
# the model (with residual learning) can actually learn.
# Expected: PSNR should rapidly climb to 40+ dB within minutes.

TRAIN_DATA_ROOT="/storage/omer/data/fastmri/singlecoil_train"
VAL_DATA_ROOT="/storage/omer/data/fastmri/singlecoil_val"

EXP_NAME="OVERFIT_TEST_residual_swinir_M"
OUTPUT_DIR="experiments_mmse/${EXP_NAME}"

MASK_TYPE="random"
VAL_MASK_TYPE="equispaced"
CENTER_FRACTION=0.04
ACCEL=4

python train_fastmri_pmrf.py \
  --phase "train" \
  --train_dataset "$TRAIN_DATA_ROOT" \
  --val_dataset "$VAL_DATA_ROOT" \
  --challenge "singlecoil" \
  --stage "mmse" \
  --arch "swinir_M" \
  --num_gpus 1 \
  --train_batch_size 4 \
  --val_batch_size 4 \
  --mask_type "$MASK_TYPE" \
  --val_mask_type "$VAL_MASK_TYPE" \
  --accelerations $ACCEL \
  --center_fractions $CENTER_FRACTION \
  --scale_mode "none" \
  --scale_percentile 100.0 \
  --resolution 320 \
  --num_workers 4 \
  --max_epochs 200 \
  --ema_decay -1 \
  --eps 0.0 \
  --t_schedule "stratified_uniform" \
  --weight_decay 0.0 \
  --lr 1e-3 \
  --wandb_project_name "PMRF_fastmri" \
  --wandb_group "fastmri_mmse_overfit" \
  --wandb_run_name "$EXP_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --overfit_train_file_name "file1000001.h5"
