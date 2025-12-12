## PMRF 2D Medical (fastMRI)

This repository wraps the PMRF implementation of *Posterior-Mean Rectified Flow (PMRF)* and adapts it to 2D medical image reconstruction on the fastMRI knee dataset.
It provides:

- **MMSE estimator training** on undersampled fastMRI slices.
- **Rectified flow (PMRF) training** conditioned on the MMSE estimator.
- Utilities for **inspecting and debugging** the fastMRI dataset.

The core logic lives in the upstream `PMRF` submodule (Lightning module `MMSERectifiedFlow` and `fastmri_dataset`), while this repo provides training scripts and configuration for the fastMRI setting.

---

### Project structure (high level)

- **`PMRF/`**: Git submodule containing the official PMRF implementation and generic image restoration code.
- **`scripts/train_fastmri_pmrf.py`**: General fastMRI training entry point (MMSE and flow stages).
- **`scripts/train_mmse.sh`**: Convenience script to train the MMSE estimator on fastMRI.
- **`scripts/train_flow.sh`**: Convenience script to train the rectified flow stage on fastMRI, using a trained MMSE checkpoint.
- **`scripts/check_fastmri_dataset.py`**: Utility for inspecting fastMRI `.h5` files and verifying dataset + normalization settings.

---

### Requirements

- Python 3.10+ (recommended)
- PyTorch and PyTorch Lightning (matching the versions used by the `PMRF` submodule)
- CUDA-capable GPU(s) for training
- Access to the **fastMRI** dataset (knee or brain, singlecoil or multicoil)
- Optional: [Weights & Biases](https://wandb.ai) account for logging (used by default in the training scripts)


### Data preparation (fastMRI)

1. Download the fastMRI data (e.g. knee singlecoil) following the official instructions.
2. Organize it into train/val folders, for example:

```bash
/home/<user>/data/knee_demo/
  ├─ singlecoil_train/
  └─ singlecoil_val/
```

3. Update the paths in the shell scripts under `scripts/` to point to your local data.

You can use `scripts/check_fastmri_dataset.py` to sanity-check that your dataset loads correctly and that normalization behaves as expected, e.g.:

```bash
python scripts/check_fastmri_dataset.py \
  --data_root /home/<user>/data/knee_demo/singlecoil_train \
  --challenge singlecoil \
  --image_pair_mode \
  --intensity_norm global_percentile
```

---

### Training the MMSE estimator

The MMSE estimator is trained to predict the fully-sampled image from an undersampled observation; in the PMRF paper, this is the *posterior-mean* predictor used both as a baseline and as a condition for the rectified flow.

The script `scripts/train_mmse.sh` is a simple wrapper around `scripts/train_fastmri_pmrf.py` with `--stage "mmse"`:

```bash
#!/usr/bin/env bash

TRAIN_DATA_ROOT="/home/<user>/data/knee_demo/singlecoil_train"
VAL_DATA_ROOT="/home/<user>/data/knee_demo/singlecoil_val"

MASK_TYPE="random"         # "random" or "cartesian"
CENTER_FRACTION=0.08       # fraction of fully-sampled low-frequency k-space
ACCEL=4                    # acceleration factor (e.g., 4, 8)

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
  --wandb_group "fastmri_mmse"
```

To launch MMSE training:

```bash
chmod +x scripts/train_mmse.sh
./scripts/train_mmse.sh
```

Checkpoints (e.g. `last.ckpt`) will be saved under a run-specific directory (see W&B run configs and Lightning checkpoints).

---

### Training the rectified flow (PMRF stage)

Once the MMSE estimator is trained, the rectified flow stage is trained to learn a flow between a simple source distribution and the posterior distribution, often **conditioned on the MMSE estimator output** (or directly on the measurement).

`scripts/train_flow.sh` is a wrapper that:

- Points to the same fastMRI train/val roots.
- Loads the MMSE checkpoint via `--mmse_model_ckpt_path`.
- Switches to `--stage "flow"` and an appropriate architecture (e.g. `hdit_XL2`).

Example (adapted from `scripts/train_flow.sh`):

```bash
#!/usr/bin/env bash

TRAIN_DATA_ROOT="/home/<user>/data/knee_demo/singlecoil_train"
VAL_DATA_ROOT="/home/<user>/data/knee_demo/singlecoil_val"
MMSE_CKPT="/path/to/mmse_checkpoint.ckpt"   # output of train_mmse.sh

MASK_TYPE="random"
CENTER_FRACTION=0.08
ACCEL=4

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
```

Run:

```bash
chmod +x scripts/train_flow.sh
./scripts/train_flow.sh
```

This will train the rectified flow model conditioned on the MMSE estimator and log metrics/images to Weights & Biases.

---

### How MMSE and rectified flow fit together

- The **MMSE stage** (`--stage mmse`) trains a supervised reconstruction network that approximates the posterior mean \( \mathbb{E}[x \mid y] \) for the fully-sampled image \(x\) given the undersampled measurement \(y\).
- The **flow stage** (`--stage flow`) then learns a time-dependent vector field that transports a simple source distribution to the posterior, typically **conditioned on the MMSE predictor** (or on \(y\)), implementing the *Posterior-Mean Rectified Flow* idea from the PMRF paper.
- At inference, the rectified flow can generate high-fidelity samples and reconstructions that aim to achieve **minimum MSE** with realistic texture and detail, improving over the pure MMSE baseline.


