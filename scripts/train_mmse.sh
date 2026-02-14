#!/usr/bin/env bash
set -e
source /storage/omer/PMRF_2D_Medical/scripts/wandb_key.sh

# Edit these paths and mask settings for your setup
TRAIN_DATA_ROOT="/storage/omer/data/fastmri/singlecoil_train"
VAL_DATA_ROOT="/storage/omer/data/fastmri/singlecoil_val"

# Experiment naming and output directory
# Change EXP_NAME and OUTPUT_DIR per experiment to control run name and
# where checkpoints/logs are written.
EXP_NAME="test"
OUTPUT_DIR="experiments_mmse/${EXP_NAME}"

# Undersampling mask parameters
MASK_TYPE="random"           # training mask type: "random" or "equispaced"
VAL_MASK_TYPE="equispaced"   # validation mask type: "random" or "equispaced"
CENTER_FRACTION=0.04     # fraction of fully-sampled low-frequency k-space
ACCEL=4                 # acceleration factor (e.g., 4, 8)

# Extra scaling options for x and y (see DataTransform in data/mri_data.py)
SCALE_MODE="none"        # "none", "subsample_max", or "subsample_percentile" "volume_subsample_max", "volume_subsample_percentile"
SCALE_PERCENTILE=100.0            # used only when SCALE_MODE="subsample_percentile" "volume_subsample_percentile"

python train_fastmri_pmrf.py \
  --phase "train" \
  --train_dataset "$TRAIN_DATA_ROOT" \
  --val_dataset "$VAL_DATA_ROOT" \
  --challenge "singlecoil" \
  --stage "mmse" \
  --arch "reconformer_S" \
  --num_gpus 1 \
  --train_batch_size 1 \
  --val_batch_size 1 \
  --mask_type "$MASK_TYPE" \
  --val_mask_type "$VAL_MASK_TYPE" \
  --accelerations $ACCEL \
  --center_fractions $CENTER_FRACTION \
  --scale_mode "$SCALE_MODE" \
  --scale_percentile $SCALE_PERCENTILE \
  --resolution 320 \
  --num_workers 16 \
  --max_epochs 1000 \
  --ema_decay 0.999 \
  --eps 0.0 \
  --t_schedule "stratified_uniform" \
  --weight_decay 1e-5 \
  --lr 1e-4 \
  --wandb_project_name "PMRF_fastmri" \
  --wandb_group "fastmri_mmse" \
  --wandb_run_name "$EXP_NAME" \
  --output_dir "$OUTPUT_DIR" \
  # --load_pretrained "/storage/omer/PMRF_2D_Medical/experiments/MMSE_SWIN_L_16batch_1e-4lr_complex_dc_8acc/epoch=37-step=82536.ckpt"
  # --overfit_train_file_name "file1000001.h5" \
  # --resume_from_ckpt "/storage/omer/PMRF_2D_Medical/experiments/MMSE_SWIN_L_16batch_1e-4lr_complex_dc/epoch=33-step=73848.ckpt" 




