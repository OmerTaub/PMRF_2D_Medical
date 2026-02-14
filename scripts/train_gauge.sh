#!/usr/bin/env bash
set -e
source /storage/omer/PMRF_2D_Medical/scripts/wandb_key.sh


# Edit these paths and mask settings for your setup
TRAIN_DATA_ROOT="/storage/omer/data/fastmri/singlecoil_train"
VAL_DATA_ROOT="/storage/omer/data/fastmri/singlecoil_val"
MMSE_CKPT="/storage/omer/PMRF_2D_Medical/experiments/MMSE_SWIN_L_16batch_1e-4lr_complex_dc/epoch=190-step=755699.ckpt"

# Pre-trained flow checkpoint to initialise the velocity network.
# Train a standard flow model first (scripts/train_flow.sh), then point here.
FLOW_CKPT="" 

# Experiment naming and output directory
EXP_NAME="gauge-flow-4acc-complex-dc"
OUTPUT_DIR="experiments_gauge/${EXP_NAME}"

# Undersampling mask parameters (same as flow training)
MASK_TYPE="random"           # training mask type: "random" or "equispaced"
VAL_MASK_TYPE="equispaced"   # validation mask type: "random" or "equispaced"
CENTER_FRACTION=0.04     # fraction of fully-sampled low-frequency k-space
ACCEL=4                  # acceleration factor (e.g., 4, 8)

SCALE_MODE="volume_subsample_max"
SCALE_PERCENTILE=100.0

# Source noise std (sigma_s) - same as flow stage
SOURCE_NOISE_STD=0.0
SOURCE_NOISE_STD_MAX=""

# --- ENGRF gauge parameters ---
GAUGE_STRENGTH=0.1       # bump magnitude: alpha(t) = GAUGE_STRENGTH * sin^2(pi*t)
GAUGE_BASE_CHANNELS=32   # base U-Net width (doubles per level)
GAUGE_NUM_LEVELS=3       # number of encoder/decoder levels

python train_fastmri_pmrf.py \
  --phase "train" \
  --train_dataset "$TRAIN_DATA_ROOT" \
  --val_dataset "$VAL_DATA_ROOT" \
  --challenge "singlecoil" \
  --sample_rate 1.0 \
  --val_sample_rate 1.0 \
  --mask_type "$MASK_TYPE" \
  --val_mask_type "$VAL_MASK_TYPE" \
  --accelerations $ACCEL \
  --center_fractions $CENTER_FRACTION \
  --scale_mode "$SCALE_MODE" \
  --scale_percentile $SCALE_PERCENTILE \
  --resolution 320 \
  --precision "bf16-mixed" \
  --stage "gauge_flow" \
  --arch "hdit_L2" \
  --mmse_model_ckpt_path "$MMSE_CKPT" \
  --mmse_model_arch "swinir_L" \
  --source_noise_std $SOURCE_NOISE_STD \
  ${SOURCE_NOISE_STD_MAX:+--source_noise_std_max $SOURCE_NOISE_STD_MAX} \
  --gauge_strength $GAUGE_STRENGTH \
  --gauge_base_channels $GAUGE_BASE_CHANNELS \
  --gauge_num_levels $GAUGE_NUM_LEVELS \
  --use_gauge_jvp \
  --num_flow_steps 16 \
  --num_gpus 1 \
  --train_batch_size 32 \
  --val_batch_size 32 \
  --num_workers 16 \
  --max_epochs 1000 \
  --ema_decay -1 \
  --eps 0.0 \
  --t_schedule "stratified_uniform" \
  --weight_decay 0 \
  --lr 1e-5 \
  --wandb_project_name "PMRF_fastmri" \
  --wandb_group "fastmri_gauge_flow" \
  --wandb_run_name "$EXP_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --apply_dc_to_source \
  ${FLOW_CKPT:+--load_pretrained "$FLOW_CKPT"} 
    # --conditional \

