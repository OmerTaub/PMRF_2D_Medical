#!/usr/bin/env bash
set -e

# Edit these paths and mask settings for your setup
TRAIN_DATA_ROOT="/home/omertaub/data/knee_demo/singlecoil_train"
VAL_DATA_ROOT="/home/omertaub/data/knee_demo/singlecoil_val"
MMSE_CKPT="/home/omertaub/projects/PMRF_2D_Medical/experiments/fastmri_mmse_experiment_accel_4_center_fraction_04/epoch=3-step=140.ckpt"   # output of train_mmse.sh

# Experiment naming and output directory
# Change EXP_NAME and OUTPUT_DIR per experiment to control run name and
# where checkpoints/logs are written.
EXP_NAME="fastmri_flow_experiment_accel_4_center_fraction_04"
OUTPUT_DIR="experiments/${EXP_NAME}"

# Undersampling mask parameters
MASK_TYPE="cartesian"       # "random" or "cartesian"
CENTER_FRACTION=0.08     # fraction of fully-sampled low-frequency k-space
ACCEL=4                  # acceleration factor (e.g., 4, 8)

python scripts/train_fastmri_pmrf.py \
  --train_data_root "$TRAIN_DATA_ROOT" \
  --val_data_root "$VAL_DATA_ROOT" \
  --challenge "singlecoil" \
  --sample_rate 1.0 \
  --val_sample_rate 1.0 \
  --synthetic_mask_type "$MASK_TYPE" \
  --synthetic_accel $ACCEL \
  --synthetic_center_fraction $CENTER_FRACTION \
  --intensity_norm "global_percentile" \
  --intensity_norm_percentile 100.0 \
  --precision "bf16-mixed" \
  --stage "flow" \
  --arch "hdit_XL2" \
  --mmse_model_ckpt_path "$MMSE_CKPT" \
  --mmse_model_arch "swinir_M" \
  --source_noise_std 0.0 \
  --num_flow_steps 16 \
  --num_gpus 1 \
  --train_batch_size 2 \
  --val_batch_size 2 \
  --num_workers 1 \
  --max_epochs 100 \
  --ema_decay 0.9999 \
  --eps 0.0 \
  --t_schedule "stratified_uniform" \
  --weight_decay 1e-4 \
  --lr 1e-4 \
  --wandb_project_name "PMRF_fastmri" \
  --wandb_group "fastmri_flow" \
  --wandb_run_name "$EXP_NAME" \
  --output_dir "$OUTPUT_DIR"