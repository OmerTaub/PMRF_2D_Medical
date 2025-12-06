#!/usr/bin/env bash
set -e

# Edit these paths and mask settings for your setup
TRAIN_DATA_ROOT="/path/to/fastmri/singlecoil_train"
VAL_DATA_ROOT="/path/to/fastmri/singlecoil_val"
MMSE_CKPT="/path/to/mmse_checkpoint.ckpt"   # output of train_mmse.sh

# Undersampling mask parameters
MASK_TYPE="random"       # "random" or "cartesian"
CENTER_FRACTION=0.08     # fraction of fully-sampled low-frequency k-space
ACCEL=4                  # acceleration factor (e.g., 4, 8)

python scripts/train_fastmri_pmrf.py \
  --train_data_root "$TRAIN_DATA_ROOT" \
  --val_data_root "$VAL_DATA_ROOT" \
  --challenge "singlecoil" \
  --sample_rate 0.1 \
  --val_sample_rate 0.1 \
  --synthetic_mask_type "$MASK_TYPE" \
  --synthetic_accel $ACCEL \
  --synthetic_center_fraction $CENTER_FRACTION \
  --intensity_norm "global_percentile" \
  --intensity_norm_percentile 99.5 \
  --precision "bf16-mixed" \
  --stage "flow" \
  --conditional \
  --arch "hdit_XL2" \
  --mmse_model_ckpt_path "$MMSE_CKPT" \
  --mmse_model_arch "hdit_XL2" \
  --source_noise_std 0.0 \
  --num_flow_steps 16 \
  --num_gpus 1 \
  --train_batch_size 8 \
  --val_batch_size 4 \
  --num_workers 4 \
  --max_epochs 100 \
  --ema_decay 0.9999 \
  --eps 0.0 \
  --t_schedule "stratified_uniform" \
  --weight_decay 1e-4 \
  --lr 1e-4 \
  --wandb_project_name "PMRF" \
  --wandb_group "fastmri_flow"