#!/usr/bin/env bash
set -e

# Edit these paths and mask settings for your setup
TRAIN_DATA_ROOT="/home/omertaub/data/knee_demo/singlecoil_train"
VAL_DATA_ROOT="/home/omertaub/data/knee_demo/singlecoil_val"

# Undersampling mask parameters
MASK_TYPE="random"       # "random" or "cartesian"
CENTER_FRACTION=0.08     # fraction of fully-sampled low-frequency k-space
ACCEL=4                  # acceleration factor (e.g., 4, 8)

python scripts/train_fastmri_pmrf.py \
  --train_data_root "$TRAIN_DATA_ROOT" \
  --val_data_root "$VAL_DATA_ROOT" \
  --challenge "singlecoil" \
  --stage "mmse" \
  --arch "swinir_M" \
  --num_gpus 1 \
  --train_batch_size 1 \
  --val_batch_size 1 \
  --synthetic_mask_type "$MASK_TYPE" \
  --synthetic_accel $ACCEL \
  --synthetic_center_fraction $CENTER_FRACTION \
  --intensity_norm "global_percentile" \
  --intensity_norm_percentile 100.0 \
  --num_workers 1 \
  --max_epochs 100000 \
  --ema_decay 0.9999 \
  --eps 0.0 \
  --t_schedule "stratified_uniform" \
  --weight_decay 1e-4 \
  --lr 1e-4 \
  --wandb_project_name "PMRF_fastmri" \
  --wandb_group "fastmri_mmse" \
#   --overfit_single_slice_index 15 # if this option is on - the num_gpus and batch_size must be 1