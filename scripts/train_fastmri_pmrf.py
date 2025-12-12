import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset, Subset

# Ensure project root and PMRF subdir are on sys.path so that
# `import PMRF...` and the internal `from utils...` imports used by
# `MMSERectifiedFlow` resolve correctly when this script is executed
# from the `scripts` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PMRF_ROOT = PROJECT_ROOT / "PMRF"
if str(PMRF_ROOT) not in sys.path:
    sys.path.insert(0, str(PMRF_ROOT))

from PMRF.lightning_models.mmse_rectified_flow import MMSERectifiedFlow
from PMRF.torch_datasets.fastmri_dataset import ImagePairSliceDataset


torch.set_float32_matmul_precision("high")


class FastMRIImagePairDictDataset(Dataset):
    """
    Thin wrapper over `ImagePairSliceDataset` that returns dict batches compatible
    with `MMSERectifiedFlow`:

        {'x': full_image, 'y': subsampled_image}

    where both tensors are shaped as (C, H, W) with C=1 for single-channel MR.
    """

    def __init__(
        self,
        root: str,
        challenge: str,
        sample_rate: Optional[float],
        volume_sample_rate: Optional[float],
        use_dataset_cache: bool,
        dataset_cache_file: str,
        num_cols: Optional[Sequence[int]],
        synthetic_mask_type: str,
        synthetic_accel: int,
        synthetic_center_fraction: float,
        synthetic_rng_seed: Optional[int],
        intensity_norm: str,
        intensity_norm_percentile: float,
    ):
        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "Specify at most one of sample_rate or volume_sample_rate, not both."
            )

        self.base = ImagePairSliceDataset(
            root=root,
            challenge=challenge,
            sample_rate=sample_rate,
            volume_sample_rate=volume_sample_rate,
            use_dataset_cache=use_dataset_cache,
            dataset_cache_file=dataset_cache_file,
            num_cols=tuple(num_cols) if num_cols is not None else None,
            synthetic_mask_type=synthetic_mask_type,
            synthetic_accel=synthetic_accel,
            synthetic_center_fraction=synthetic_center_fraction,
            synthetic_rng_seed=synthetic_rng_seed,
            intensity_norm=intensity_norm,
            intensity_norm_percentile=intensity_norm_percentile,
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        full_img, sub_img, _kspace_full, _kspace_sub, _mask = self.base[idx]

        # Ensure channel dimension (C, H, W) with C=1 for grayscale MR images.
        if full_img.ndim == 2:
            full_img = full_img.unsqueeze(0)
        if sub_img.ndim == 2:
            sub_img = sub_img.unsqueeze(0)

        return {"x": full_img, "y": sub_img}


def create_datasets(args):
    train_ds = FastMRIImagePairDictDataset(
        root=args.train_data_root,
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        volume_sample_rate=args.volume_sample_rate,
        use_dataset_cache=args.use_dataset_cache,
        dataset_cache_file=args.dataset_cache_file,
        num_cols=args.num_cols,
        synthetic_mask_type=args.synthetic_mask_type,
        synthetic_accel=args.synthetic_accel,
        synthetic_center_fraction=args.synthetic_center_fraction,
        synthetic_rng_seed=args.synthetic_rng_seed,
        intensity_norm=args.intensity_norm,
        intensity_norm_percentile=args.intensity_norm_percentile,
    )

    val_ds = FastMRIImagePairDictDataset(
        root=args.val_data_root,
        challenge=args.challenge,
        sample_rate=args.val_sample_rate,
        volume_sample_rate=args.val_volume_sample_rate,
        use_dataset_cache=args.use_dataset_cache,
        dataset_cache_file=args.dataset_cache_file,
        num_cols=args.num_cols,
        synthetic_mask_type=args.synthetic_mask_type,
        synthetic_accel=args.synthetic_accel,
        synthetic_center_fraction=args.synthetic_center_fraction,
        synthetic_rng_seed=args.synthetic_rng_seed,
        intensity_norm=args.intensity_norm,
        intensity_norm_percentile=args.intensity_norm_percentile,
    )

    return train_ds, val_ds


def main(args):
    if args.train_batch_size % args.num_gpus != 0:
        raise ValueError(
            "train_batch_size must be divisible by num_gpus when using DDP."
        )

    if args.sample_rate is not None and args.volume_sample_rate is not None:
        raise ValueError(
            "Specify at most one of --sample_rate or --volume_sample_rate, not both."
        )

    if args.val_sample_rate is not None and args.val_volume_sample_rate is not None:
        raise ValueError(
            "Specify at most one of --val_sample_rate or --val_volume_sample_rate, not both."
        )

    output_dir: Optional[Path] = None
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logger = WandbLogger(
        project=args.wandb_project_name,
        group=args.wandb_group,
        id=args.wandb_id,
        name=args.wandb_run_name,
    )
    logger.log_hyperparams(vars(args))

    # Save a copy of the run configuration to the experiment directory, if provided.
    if output_dir is not None:
        config_path = output_dir / "config.json"
        with config_path.open("w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

    train_data, val_data = create_datasets(args)

    # Optional debug mode: overfit on a single slice for both training and
    # validation without changing dataset code. This wraps the datasets in a
    # Subset. For best results, also run with --num_gpus 1 and
    # --train_batch_size 1.
    if args.overfit_single_slice_index is not None:
        idx = args.overfit_single_slice_index
        if not (0 <= idx < len(train_data)):
            raise ValueError(
                f"--overfit_single_slice_index={idx} is out of bounds for "
                f"train dataset of length {len(train_data)}."
            )
        train_data = Subset(train_data, [idx])

        if not (0 <= idx < len(val_data)):
            raise ValueError(
                f"--overfit_single_slice_index={idx} is out of bounds for "
                f"val dataset of length {len(val_data)}."
            )
        val_data = Subset(val_data, [idx])

    per_device_train_batch = args.train_batch_size // args.num_gpus

    train_loader = DataLoader(
        train_data,
        batch_size=per_device_train_batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers // args.num_gpus,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers // args.num_gpus,
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=str(output_dir) if output_dir is not None else None,
        save_last=True,
    )
    lr_monitor_callback = LearningRateMonitor()
    trainer = Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        accelerator="gpu",
        strategy="ddp",
        devices=args.num_gpus,
        callbacks=[ckpt_callback, lr_monitor_callback],
        precision=args.precision,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )

    with trainer.init_module():
        model = MMSERectifiedFlow(
            stage=args.stage,
            arch=args.arch,
            conditional=args.conditional,
            mmse_model_ckpt_path=args.mmse_model_ckpt_path,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=args.betas,
            mmse_noise_std=args.source_noise_std,
            mmse_model_arch=args.mmse_model_arch,
            num_flow_steps=args.num_flow_steps,
            ema_decay=args.ema_decay,
            eps=args.eps,
            t_schedule=args.t_schedule,
            colorization=False,
        )

    torch.compile(model, mode="max-autotune")
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from_ckpt,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train PMRF (MMSERectifiedFlow) on fastMRI slices using ImagePairSliceDataset."
        )
    )

    # ---------------- fastMRI dataset options ----------------
    parser.add_argument(
        "--train_data_root",
        type=str,
        required=True,
        help="Path to training fastMRI .h5 files directory.",
    )
    parser.add_argument(
        "--val_data_root",
        type=str,
        required=True,
        help="Path to validation fastMRI .h5 files directory.",
    )
    parser.add_argument(
        "--challenge",
        type=str,
        choices=["singlecoil", "multicoil"],
        default="singlecoil",
        help='fastMRI challenge type (default: "singlecoil").',
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=None,
        help="Fraction of slices to load for training (mutually exclusive with volume_sample_rate).",
    )
    parser.add_argument(
        "--volume_sample_rate",
        type=float,
        default=None,
        help="Fraction of volumes to load for training (mutually exclusive with sample_rate).",
    )
    parser.add_argument(
        "--val_sample_rate",
        type=float,
        default=None,
        help="Fraction of slices to load for validation (mutually exclusive with val_volume_sample_rate).",
    )
    parser.add_argument(
        "--val_volume_sample_rate",
        type=float,
        default=None,
        help="Fraction of volumes to load for validation (mutually exclusive with val_sample_rate).",
    )
    parser.add_argument(
        "--use_dataset_cache",
        action="store_true",
        help="Use on-disk dataset cache (see fastmri_dataset.SliceDataset).",
    )
    parser.add_argument(
        "--dataset_cache_file",
        type=str,
        default="dataset_cache.pkl",
        help="Path to dataset cache file (default: dataset_cache.pkl).",
    )
    parser.add_argument(
        "--num_cols",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of allowed encoding sizes in k-space dimension 1.",
    )
    parser.add_argument(
        "--synthetic_mask_type",
        type=str,
        default="random",
        choices=["random", "cartesian"],
        help="Type of synthetic undersampling mask if file mask is not present.",
    )
    parser.add_argument(
        "--synthetic_accel",
        type=int,
        default=4,
        help="Acceleration factor for synthetic undersampling mask.",
    )
    parser.add_argument(
        "--synthetic_center_fraction",
        type=float,
        default=0.08,
        help="Center fraction for synthetic undersampling mask.",
    )
    parser.add_argument(
        "--synthetic_rng_seed",
        type=int,
        default=None,
        help="Optional RNG seed for synthetic random masks.",
    )

    # ---------------- intensity normalization options ----------------
    parser.add_argument(
        "--intensity_norm",
        type=str,
        default="global_percentile",
        choices=[
            "none",
            "per_slice_max",
            "per_slice_percentile",
            "per_volume_percentile",
            "global_percentile",
        ],
        help=(
            "Intensity normalization mode for ImagePairSliceDataset: "
            "'none', 'per_slice_max' (original), 'per_slice_percentile', "
            "'per_volume_percentile' (per-volume), or 'global_percentile' "
            "(dataset-level robust scale)."
        ),
    )
    parser.add_argument(
        "--intensity_norm_percentile",
        type=float,
        default=100.0,
        help=(
            "Percentile used when intensity_norm is a percentile-based mode "
            "(per_slice_percentile or per_volume_percentile)."
        ),
    )

    # ---------------- PMRF / flow model options (mirroring PMRF/train.py) ----
    parser.add_argument(
        "--precision",
        type=str,
        required=False,
        choices=["bf16-mixed", "32"],
        help="The precision used for training.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["mmse", "flow", "naive_flow"],
        help="The stage of the model.",
    )
    parser.add_argument(
        "--conditional",
        action="store_true",
        help=(
            "If set, the flow model is conditioned on either y or the posterior "
            "mean predictor. Applies only to the stage 'flow'."
        ),
    )
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["hdit_XL2", "hdit_ImageNet256Sp4", "swinir_M", "swinir_L", "swinir_S"],
        help="Architecture name and size.",
    )
    parser.add_argument(
        "--mmse_model_ckpt_path",
        type=str,
        required=False,
        default=None,
        help=(
            "Checkpoint path to a pre-trained MMSE model. Relevant only for the "
            "stage 'flow'. If --conditional is set, the outputs of this model "
            "will be the input condition of the flow. Otherwise, if "
            "--conditional is not set, PMRF will be trained."
        ),
    )
    parser.add_argument(
        "--mmse_model_arch",
        type=str,
        required=False,
        default=None,
        help=(
            "The architecture of the pre-trained MMSE model. Only relevant for "
            "the stage 'flow'."
        ),
    )
    parser.add_argument(
        "--source_noise_std",
        type=float,
        required=False,
        default=0.0,
        help=(
            "Noise std to add to the samples from the source distribution "
            "(sigma_s in the paper). Applies only to PMRF and naive flow."
        ),
    )
    parser.add_argument(
        "--num_flow_steps",
        type=int,
        required=False,
        default=16,
        help="Number of flow steps for evaluation.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        required=False,
        default=4,
        help="Number of gpus to use.",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        required=False,
        default=1,
        help="Check validation every n epochs.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        required=False,
        default=256,
        help=(
            "Training batch size (on DDP, will be the total batch size on all GPUs)."
        ),
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        required=False,
        default=32,
        help="Validation batch size (on DDP, will be the batch size on each GPU).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=1,
        help="Number of workers on all GPUs.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        required=False,
        default=100000,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        required=False,
        default=0.9999,
        help="Exponential moving average decay.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        required=False,
        default=0,
        help="Starting time of the flow.",
    )
    parser.add_argument(
        "--t_schedule",
        type=str,
        required=False,
        default="stratified_uniform",
        choices=["uniform", "logit-normal", "stratified_uniform"],
        help=(
            "Flow time scheduler (sampler) for training. "
            "We found stratified_uniform to work best."
        ),
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        required=False,
        default=1e-4,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        required=False,
        default=1e-4,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--betas",
        type=tuple,
        required=False,
        default=(0.9, 0.95),
        help="Betas for the AdamW optimizer.",
    )
    parser.add_argument(
        "--wandb_project_name",
        type=str,
        required=True,
        default="PMRF",
        help="Project name for Weights & Biases logger.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        required=False,
        default=None,
        help="Group of wandb experiment.",
    )
    parser.add_argument(
        "--wandb_id",
        type=str,
        required=False,
        default=None,
        help="Optional W&B run ID (for resuming runs).",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        required=False,
        default=None,
        help="Optional W&B run display name.",
    )
    parser.add_argument(
        "--resume_from_ckpt",
        type=str,
        required=False,
        default=None,
        help="Optional checkpoint path to resume training.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default=None,
        help="Optional directory to save checkpoints and logs.",
    )

    # ---------------- debugging / overfitting options ----------------
    parser.add_argument(
        "--overfit_single_slice_index",
        type=int,
        default=None,
        help=(
            "If set, restrict the *training* dataset to a single slice at this "
            "index (using torch.utils.data.Subset). Useful for debugging / "
            "checking that the model can overfit one example. For best "
            "results, also use --num_gpus 1 and --train_batch_size 1."
        ),
    )

    main(parser.parse_args())


