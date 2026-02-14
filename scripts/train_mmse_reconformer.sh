#!/usr/bin/env bash
set -e
source /storage/omer/PMRF_2D_Medical/scripts/wandb_key.sh

# Fine-tune ReconFormer (F_X4_checkpoint.pth) as a posterior mean (MMSE)
# estimator for 4x acceleration on fastMRI singlecoil.

TRAIN_DATA_ROOT="/storage/omer/data/fastmri/singlecoil_train"
VAL_DATA_ROOT="/storage/omer/data/fastmri/singlecoil_val"

# Pretrained ReconFormer checkpoint (4x acceleration)
PRETRAINED_CKPT="/storage/omer/PMRF_2D_Medical/ReconFormer/F_X8_checkpoint.pth"

# Experiment naming and output directory
EXP_NAME="mmse_reconformer_x16"
OUTPUT_DIR="experiments_mmse/${EXP_NAME}"

# Undersampling mask parameters (matching original ReconFormer X4 training)
MASK_TYPE="random"           # training mask type
VAL_MASK_TYPE="equispaced"   # validation mask type
CENTER_FRACTION=0.04         
ACCEL=16                      

# Scaling options
SCALE_MODE="none"
SCALE_PERCENTILE=100.0

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
  --load_pretrained "$PRETRAINED_CKPT"
