# PMRF-MRI: Posterior-Mean Rectified Flow for MRI Reconstruction

A deep learning framework for accelerated MRI reconstruction using **Posterior-Mean Rectified Flow (PMRF)**, adapted for 2D medical imaging on the [fastMRI](https://fastmri.org/) dataset.

> **PMRF** approximates the optimal estimator that minimizes Mean Squared Error (MSE) under a perfect perceptual quality constraint. This implementation adapts the original PMRF framework for MRI reconstruction from undersampled k-space data.

---

## Table of Contents
1. [Overview](#overview)
2. [Method Pipeline](#method-pipeline)
3. [Data Preprocessing & Masking](#1-data-preprocessing--masking)
4. [Model Architecture](#2-model-architecture)
5. [Training Stages](#3-training-stages)
6. [Inference](#4-inference-per-slice-reconstruction)
7. [Evaluation Metrics](#5-evaluation-metrics-per-scan)
8. [Installation](#installation)
9. [Quick Start](#quick-start)
10. [Project Structure](#project-structure)
11. [Citation](#citation)

---

## Overview

MRI reconstruction from undersampled k-space is a classic inverse problem. The goal is to recover a high-quality image from sparse frequency-domain measurements, enabling faster scan times while maintaining diagnostic quality.

This framework implements a **two-stage approach**:

| Stage | Purpose | Output |
|-------|---------|--------|
| **MMSE Stage** | Train a posterior-mean estimator | Blurry but MSE-optimal reconstruction |
| **Flow Stage** | Train a rectified flow to add realistic details | Photo-realistic reconstruction with preserved fidelity |

The key insight of PMRF is that the posterior-mean (MMSE) estimate provides a strong starting point, and the rectified flow learns to transport samples from this distribution to the true posterior—achieving both low MSE and high perceptual quality.

---

## Method Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PMRF-MRI RECONSTRUCTION PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────────────────────────┐
                         │        FULLY-SAMPLED K-SPACE         │
                         │           (Ground Truth)             │
                         └──────────────────┬───────────────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    ▼                       ▼                       ▼
        ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
        │   1. CROP TO      │   │   2. APPLY        │   │   3. NORMALIZE    │
        │   RESOLUTION      │   │   UNDERSAMPLING   │   │   INTENSITY       │
        │   (320×320)       │   │   MASK            │   │   (÷ mean|y|)     │
        └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
                  │                       │                       │
                  └───────────────────────┼───────────────────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │     TRAINING DATA     │
                              │   x: ground truth     │
                              │   y: zero-filled      │
                              │      reconstruction   │
                              └───────────┬───────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
        ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
        │  STAGE 1: MMSE    │ │  STAGE 2: FLOW    │ │  INFERENCE        │
        │  ────────────────  │ │  ────────────────  │ │  ────────────────  │
        │  Input:  y        │ │  Input:  y, ẑ₀    │ │  Input:  y        │
        │  Target: x        │ │  Target: x        │ │  Output: x̂        │
        │  Loss:   MSE      │ │  Loss:   Flow     │ │                   │
        │                   │ │                   │ │  1. MMSE → ẑ₀     │
        │  Output: ẑ₀       │ │  Output: x̂        │ │  2. Flow → x̂      │
        │  (posterior mean) │ │  (refined)        │ │  3. DC (optional) │
        └───────────────────┘ └───────────────────┘ └─────────┬─────────┘
                                                              │
                                                              ▼
                                                  ┌───────────────────────┐
                                                  │     EVALUATION        │
                                                  │  ──────────────────   │
                                                  │  • PSNR per scan      │
                                                  │  • SSIM per scan      │
                                                  │  • LPIPS, DISTS       │
                                                  │  • FID (distribution) │
                                                  └───────────────────────┘
```

---

## 1. Data Preprocessing & Masking

The data pipeline transforms raw k-space measurements into normalized image pairs suitable for training.

### 1.1 Input Data Format

fastMRI data is stored in HDF5 (`.h5`) files with the following structure:

```
file.h5
├── kspace              # Complex k-space data: (num_slices, H, W)
├── reconstruction_esc  # Ground truth reconstruction (singlecoil)
└── metadata            # Acquisition parameters
```

### 1.2 Preprocessing Steps

```python
# Step-by-step preprocessing (from data/mri_data.py)

1. Load k-space:           kspace = data['kspace'][slice]     # Complex (H_orig, W_orig)
2. IFFT to image domain:   image = ifft2(kspace)              # Full-resolution image
3. Center crop:            image = crop(image, 320×320)       # Target resolution
4. FFT back to k-space:    kspace = fft2(image)               # K-space at target res
5. Apply mask:             kspace_masked = kspace × mask      # Undersampling
6. IFFT for zero-filled:   y = ifft2(kspace_masked)           # Zero-filled recon
7. Normalize:              y, x = y / mean(|y|)               # Intensity normalization
```

### 1.3 Undersampling Masks

Two mask types are supported (defined in `data/subsample.py`):

| Mask Type | Description | Use Case |
|-----------|-------------|----------|
| **Random** | Randomly selected k-space columns + fully-sampled center | Training (data augmentation) |
| **Equispaced** | Uniformly spaced columns + fully-sampled center | Evaluation (reproducible) |

**Key Parameters:**
- `acceleration` (e.g., 4×): Ratio of k-space columns sampled (4× = 25% of columns)
- `center_fraction` (e.g., 0.04): Fraction of low-frequency columns always sampled

```
K-space Mask Example (4× acceleration, 4% center fraction):
┌────────────────────────────────────────────────────────────┐
│░░░█░░░░░█░░░░░░█░████████████░░░░█░░░░░░░█░░░█░░░░░░░█░░░░│
└────────────────────────────────────────────────────────────┘
         │                │                    │
    Random samples    Fully-sampled        Random samples
                        center
```

### 1.4 Complex Data Representation

The model operates on **complex-valued images** represented as 2-channel tensors:

```
Complex image: (H, W) complex64
      ↓
Tensor representation: (2, H, W) float32
      └── Channel 0: Real part
      └── Channel 1: Imaginary part
```

This preserves phase information, which is critical for data consistency and accurate reconstruction.

### 1.5 Normalization Strategy

Intensity normalization ensures consistent input ranges across different scans:

```python
# Reconformer-style normalization
norm_factor = mean(|y|)  # Mean magnitude of zero-filled reconstruction
x_normalized = x / norm_factor
y_normalized = y / norm_factor
```

**Volume-level variants** compute a single normalization factor per scan (all slices share the same factor), ensuring consistent intensity across slices.

---

## 2. Model Architecture

### 2.1 SwinIR (Recommended for MMSE)

The MMSE stage uses **SwinIR** (Shifted Window Image Restoration Transformer), a powerful transformer-based architecture for image restoration.

```
SwinIR Architecture:
┌─────────────────────────────────────────────────────────────────┐
│ Input (2, 320, 320)                                             │
│      ↓                                                          │
│ Shallow Feature Extraction (Conv)                               │
│      ↓                                                          │
│ Deep Feature Extraction:                                        │
│   ├── Residual Swin Transformer Block (RSTB) × 6-8              │
│   │     ├── Swin Transformer Layer × 6                          │
│   │     │     ├── Window Multi-head Self-Attention (W-MSA)      │
│   │     │     ├── Shifted Window MSA (SW-MSA)                   │
│   │     │     └── MLP                                           │
│   │     └── Conv                                                │
│   └── Conv                                                      │
│      ↓                                                          │
│ Image Reconstruction (Conv)                                     │
│      ↓                                                          │
│ Output (2, 320, 320)                                            │
└─────────────────────────────────────────────────────────────────┘
```

**Configuration variants** (from `PMRF/utils/create_arch.py`):

| Variant | Embed Dim | Depth | Params |
|---------|-----------|-------|--------|
| `swinir_S` | 32 | 4×4 | ~1M |
| `swinir_M` | 120 | 6×5 | ~12M |
| `swinir_L` | 180 | 6×8 | ~28M |

### 2.2 HDiT (For Flow Stage)

The flow stage can use **HDiT** (Hourglass Diffusion Transformer), designed for high-resolution image generation:

```
HDiT Architecture:
┌─────────────────────────────────────────────────────────────────┐
│ Input: x_t (noisy), t (timestep), [condition]                   │
│      ↓                                                          │
│ Patch Embedding (4×4 patches)                                   │
│      ↓                                                          │
│ Hourglass Structure:                                            │
│   Level 1: 256 dim, Neighborhood Attention, 2 blocks            │
│      ↓ downsample                                               │
│   Level 2: 512 dim, Neighborhood Attention, 2 blocks            │
│      ↓ downsample                                               │
│   Level 3: 1024 dim, Global Attention, 8 blocks (bottleneck)    │
│      ↑ upsample                                                 │
│   Level 2: Skip connection + 2 blocks                           │
│      ↑ upsample                                                 │
│   Level 1: Skip connection + 2 blocks                           │
│      ↓                                                          │
│ Output projection                                               │
│      ↓                                                          │
│ Output: v_t (velocity field)                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Training Stages

### 3.1 Stage 1: MMSE Estimator

**Objective:** Train a network to predict the posterior mean E[x|y]

```
Loss = MSE(model(y), x) = ||f_θ(y) - x||²
```

The MMSE estimator learns to predict the expected value of the fully-sampled image given the undersampled observation. This produces **blurry but artifact-free** reconstructions that minimize MSE.

**Training command:**

```bash
python train_fastmri_pmrf.py \
    --stage mmse \
    --arch swinir_L \
    --train_dataset /path/to/singlecoil_train \
    --val_dataset /path/to/singlecoil_val \
    --accelerations 4 \
    --center_fractions 0.04 \
    --resolution 320 \
    --train_batch_size 16 \
    --lr 1e-4
```

### 3.2 Stage 2: Rectified Flow (PMRF)

**Objective:** Learn a velocity field that transports samples from the MMSE output to the true posterior

```
Rectified Flow Training:
┌─────────────────────────────────────────────────────────────────┐
│ 1. Sample t ~ Uniform[0, 1]                                     │
│ 2. Get MMSE estimate: ẑ₀ = MMSE_model(y)                        │
│ 3. Add noise: z₀ = ẑ₀ + σ·ε,  ε ~ N(0, I)                       │
│ 4. Interpolate: x_t = t·x + (1-t)·z₀                            │
│ 5. Predict velocity: v_θ(x_t, t, y)                             │
│ 6. Loss = ||v_θ(x_t, t, y) - (x - z₀)||²                        │
└─────────────────────────────────────────────────────────────────┘
```

The flow model learns to "push" samples from the MMSE distribution toward realistic samples from the posterior, adding fine details while preserving the low-frequency structure.

**Training command:**

```bash
python train_fastmri_pmrf.py \
    --stage flow \
    --arch hdit_XL2 \
    --mmse_model_ckpt_path /path/to/mmse_checkpoint.ckpt \
    --mmse_model_arch swinir_L \
    --source_noise_std 0.1 \
    --num_flow_steps 16 \
    --conditional
```

---

## 4. Inference (Per-Slice Reconstruction)

### 4.1 Inference Pipeline

```
Inference Flow:
┌─────────────────────────────────────────────────────────────────┐
│ Input: Undersampled k-space                                     │
│      ↓                                                          │
│ Preprocessing (same as training):                               │
│   - Crop, mask, normalize                                       │
│      ↓                                                          │
│ Zero-filled reconstruction: y                                   │
│      ↓                                                          │
│ MMSE model: ẑ₀ = f_MMSE(y)                                      │
│      ↓                                                          │
│ Add noise: z₀ = ẑ₀ + σ·ε                                        │
│      ↓                                                          │
│ ODE solve (Euler method, K steps):                              │
│   for k = 0, 1, ..., K-1:                                       │
│       t_k = k/K                                                 │
│       v_k = f_flow(x_k, t_k, y)                                 │
│       x_{k+1} = x_k + v_k · (1/K)                               │
│      ↓                                                          │
│ Final reconstruction: x̂ = x_K                                   │
│      ↓                                                          │
│ (Optional) Data Consistency:                                    │
│   x̂_dc = IFFT(mask·FFT(y) + (1-mask)·FFT(x̂))                   │
│      ↓                                                          │
│ Convert to magnitude: |x̂|                                       │
│      ↓                                                          │
│ Unnormalize: x̂_raw = |x̂| × norm_factor                          │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Data Consistency (DC)

Data consistency enforces that the reconstruction matches the acquired k-space measurements:

```python
def apply_data_consistency(xhat, kspace, mask):
    """
    At measured locations (mask=1): use original k-space
    At unmeasured locations (mask=0): use predicted k-space
    """
    kspace_pred = fft2(xhat)
    kspace_dc = mask * kspace + (1 - mask) * kspace_pred
    return ifft2(kspace_dc)
```

### 4.3 Running Inference

```bash
python scripts/inference_pmrf.py \
    --checkpoint /path/to/flow_model.ckpt \
    --data_dir /path/to/singlecoil_val \
    --output_dir /path/to/results \
    --num_flow_steps 16 \
    --apply_dc \
    --batch_size 8
```

---

## 5. Evaluation Metrics (Per-Scan)

Metrics are computed **per-scan** (volume), then averaged across all scans. This provides clinically meaningful evaluation since MRI scans are typically analyzed as 3D volumes.

### 5.1 Metric Definitions

| Metric | Description | Range | Better |
|--------|-------------|-------|--------|
| **PSNR** | Peak Signal-to-Noise Ratio | [0, ∞) dB | Higher |
| **SSIM** | Structural Similarity Index | [0, 1] | Higher |
| **LPIPS** | Learned Perceptual Image Patch Similarity | [0, 1] | Lower |
| **DISTS** | Deep Image Structure and Texture Similarity | [0, 1] | Lower |
| **FID** | Fréchet Inception Distance | [0, ∞) | Lower |

### 5.2 Per-Scan Computation

```python
# Per-scan PSNR computation
for each scan in dataset:
    # Compute per-slice MSE
    for each slice in scan:
        mse_slice = mean((pred - gt)²)
    
    # Scan-level metrics
    scan_mse = mean(mse_slice for all slices)
    data_range = max(gt over all slices)
    scan_psnr = 10 × log10(data_range² / scan_mse)

# Final metric = mean(scan_psnr for all scans)
```

### 5.3 Running Evaluation

```bash
python scripts/evaluate_reconformer.py \
    --pred_dir /path/to/predictions \
    --gt_dir /path/to/singlecoil_val \
    --output_csv metrics.csv
```

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.3+ with CUDA
- PyTorch Lightning 2.3+
- CUDA-capable GPU (recommended: 24GB+ VRAM)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/PMRF_2D_Medical.git
cd PMRF_2D_Medical

# Create conda environment
conda create -n pmrf python=3.10
conda activate pmrf

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install pytorch-lightning wandb h5py einops timm piq torch-ema
pip install natten  # For HDiT architecture (see PMRF/README.md for details)
```

---

## Quick Start

### 1. Prepare Data

Download fastMRI data and organize as:
```
/path/to/data/
├── singlecoil_train/
│   ├── file1000000.h5
│   ├── file1000001.h5
│   └── ...
└── singlecoil_val/
    ├── file1000425.h5
    └── ...
```

### 2. Train MMSE Model

```bash
bash scripts/train_mmse.sh
```

### 3. Train Flow Model

```bash
# Edit scripts/train_flow.sh to point to MMSE checkpoint
bash scripts/train_flow.sh
```

### 4. Run Inference

```bash
python scripts/inference_pmrf.py \
    --checkpoint experiments/your_experiment/last.ckpt \
    --data_dir /path/to/singlecoil_val \
    --output_dir results/
```

---

## Project Structure

```
PMRF_2D_Medical/
├── data/                          # Data loading and preprocessing
│   ├── mri_data.py               # SliceData dataset and DataTransform
│   ├── subsample.py              # Undersampling mask generation
│   └── transforms.py             # FFT, IFFT, normalization utilities
│
├── PMRF/                          # Core PMRF implementation
│   ├── arch/                     # Model architectures
│   │   ├── swinir/              # SwinIR transformer
│   │   └── hourglass/           # HDiT transformer
│   ├── lightning_models/         # PyTorch Lightning modules
│   │   ├── mmse_rectified_flow.py  # Main model class
│   │   ├── rf_metrics.py        # Per-scan metric computation
│   │   └── rf_vis.py            # Visualization utilities
│   └── utils/
│       └── create_arch.py       # Architecture factory
│
├── scripts/                       # Training and evaluation scripts
│   ├── train_mmse.sh            # MMSE training script
│   ├── train_flow.sh            # Flow training script
│   ├── inference_pmrf.py        # Inference script
│   └── evaluate_reconformer.py  # Evaluation script
│
├── train_fastmri_pmrf.py         # Main training entry point
└── experiments/                   # Saved checkpoints and logs
```

---

## Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--stage` | Training stage: `mmse`, `flow`, `naive_flow` | Required |
| `--arch` | Model architecture: `swinir_S/M/L`, `hdit_XL2` | Required |
| `--accelerations` | K-space acceleration factor(s) | `[4]` |
| `--center_fractions` | Fraction of center k-space to sample | `[0.04]` |
| `--resolution` | Image resolution (square) | `320` |
| `--scale_mode` | Intensity scaling: `none`, `subsample_max`, `volume_subsample_max` | `subsample_max` |
| `--num_flow_steps` | Number of ODE solver steps | `15` |
| `--source_noise_std` | Noise added to MMSE output for flow | `0.0` |

---

## Citation

If you use this code, please cite the original PMRF paper:

```bibtex
@inproceedings{ohayon2025posteriormean,
    title={Posterior-Mean Rectified Flow: Towards Minimum {MSE} Photo-Realistic Image Restoration},
    author={Guy Ohayon and Tomer Michaeli and Michael Elad},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=hPOt3yUXii}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

- [PMRF](https://github.com/ohayonguy/PMRF) - Original implementation
- [fastMRI](https://fastmri.org/) - Dataset
- [SwinIR](https://github.com/JingyunLiang/SwinIR) - Architecture
- [k-diffusion](https://github.com/crowsonkb/k-diffusion) - HDiT architecture
