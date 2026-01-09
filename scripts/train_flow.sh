#!/usr/bin/env bash
set -e
source /storage/omer/PMRF_2D_Medical/scripts/wandb_key.sh


# Edit these paths and mask settings for your setup
TRAIN_DATA_ROOT="/storage/omer/data/fastmri/singlecoil_train"
VAL_DATA_ROOT="/storage/omer/data/fastmri/singlecoil_val"
MMSE_CKPT="/storage/omer/PMRF_2D_Medical/experiments/fastmri_mmse_experiment_accel_4_center_fraction_04_reconformer_norm_SWIN_L_continue/epoch=111-step=833856.ckpt"   # output of train_mmse.sh

# Experiment naming and output directory
# Change EXP_NAME and OUTPUT_DIR per experiment to control run name and
# where checkpoints/logs are written.
EXP_NAME="fastmri_flow_experiment_accel_4_center_fraction_04_reconformer_norm_SWIN_L_1channel"
OUTPUT_DIR="experiments_flow/${EXP_NAME}"

# Undersampling mask parameters
MASK_TYPE="random"       # "random" or "cartesian"
CENTER_FRACTION=0.04     # fraction of fully-sampled low-frequency k-space
ACCEL=4                  # acceleration factor (e.g., 4, 8)

SCALE_MODE="volume_subsample_max"        # "none", "subsample_max", or "subsample_percentile"
SCALE_PERCENTILE=100.0

python train_fastmri_pmrf.py \
  --phase "train" \
  --train_dataset "$TRAIN_DATA_ROOT" \
  --val_dataset "$VAL_DATA_ROOT" \
  --challenge "singlecoil" \
  --sample_rate 1.0 \
  --val_sample_rate 1.0 \
  --mask_type "$MASK_TYPE" \
  --accelerations $ACCEL \
  --center_fractions $CENTER_FRACTION \
  --scale_mode "$SCALE_MODE" \
  --scale_percentile $SCALE_PERCENTILE \
  --resolution 320 \
  --precision "bf16-mixed" \
  --stage "flow" \
  --arch "hdit_XL2" \
  --mmse_model_ckpt_path "$MMSE_CKPT" \
  --mmse_model_arch "swinir_L" \
  --num_flow_steps 32 \
  --num_gpus 1 \
  --train_batch_size 8 \
  --val_batch_size 8 \
  --num_workers 8 \
  --max_epochs 1000 \
  --ema_decay -1 \
  --eps 0.0 \
  --t_schedule "stratified_uniform" \
  --weight_decay 0\
  --lr 1e-4 \
  --wandb_project_name "PMRF_fastmri" \
  --wandb_group "fastmri_flow" \
  --wandb_run_name "$EXP_NAME" \
  --output_dir "$OUTPUT_DIR"