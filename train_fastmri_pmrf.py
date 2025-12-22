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
import torch
import os
from data import SliceData, VanillaSliceData, DataTransform, create_mask_for_mask_type
import pathlib

# Ensure project root and PMRF subdir are on sys.path so that
# `import PMRF...` and the internal `from utils...` imports used by
# `MMSERectifiedFlow` resolve correctly when this script is executed
# from the `scripts` directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PMRF_ROOT = PROJECT_ROOT / "PMRF"
if str(PMRF_ROOT) not in sys.path:
    sys.path.insert(0, str(PMRF_ROOT))

from PMRF.lightning_models.mmse_rectified_flow import MMSERectifiedFlow


torch.set_float32_matmul_precision("high")



from pathlib import Path
from torch.utils.data import DataLoader

# assumes you have: from fastmri.data.mri_data import SliceData   (or your local SliceData import)
# and that you have data transforms already (data_transform objects)

def _create_dataset(
    args,
    data_path,
    data_transform,
    data_partition,  # unused, kept for backward compatibility
    sequence,        # unused by current SliceData implementation
    bs,
    shuffle,
    sample_rate=None,
    display=False,
    mask_func=None,
):
    sample_rate = sample_rate if sample_rate is not None else args.sample_rate

    # In our fastMRI setup, data_path already points directly to the directory
    # containing the .h5 files (e.g. /.../singlecoil_train), so we should not
    # append an extra subdirectory such as "train" or "val" here.
    # dataset = SliceData(
    #     root=Path(data_path),
    #     transform=data_transform,
    #     challenge=args.challenge,
    #     sequence=sequence,
    #     sample_rate=sample_rate,
    # )
    dataset = VanillaSliceData(
        root=data_path,
        challenge=args.challenge,
        sample_rate=args.sample_rate,
        mask_func=mask_func,
    )
    if display:
        dataset = [dataset[i] for i in range(100, 108)]

    # ---------------- debugging / overfitting helpers ----------------
    # These are intentionally applied only at the DataLoader construction level,
    # so the core SliceData implementation stays untouched.
    if getattr(args, "overfit_train_file_name", None):
        wanted = args.overfit_train_file_name
        keep = [
            i
            for i, (fname, _slice, _pl, _pr) in enumerate(getattr(dataset, "examples", []))
            if getattr(fname, "name", str(fname)) == wanted
        ]
        if len(keep) == 0:
            raise ValueError(
                f"--overfit_train_file_name='{wanted}' did not match any files in dataset. "
                f"Example file: {dataset.examples[0][0].name if len(dataset) else '<empty dataset>'}"
            )
        dataset = Subset(dataset, keep)
        shuffle = False

    if getattr(args, "overfit_first_n_slices", None) is not None:
        n = int(args.overfit_first_n_slices)
        if n <= 0:
            raise ValueError("--overfit_first_n_slices must be > 0")
        n = min(n, len(dataset))
        dataset = Subset(dataset, list(range(n)))
        shuffle = False

    if getattr(args, "overfit_single_slice_index", None) is not None:
        idx = int(args.overfit_single_slice_index)
        if idx < 0 or idx >= len(dataset):
            raise ValueError(
                f"--overfit_single_slice_index={idx} out of range for dataset length {len(dataset)}"
            )
        dataset = Subset(dataset, [idx])
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=args.num_workers,
    )



def main(args):
    path_dict = {
        'train': args.train_dataset,
        'val': args.val_dataset,
    }
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


    if args.challenge == 'singlecoil':
        mask = create_mask_for_mask_type(args.mask_type, args.center_fractions,
                                         args.accelerations)
        train_data_transform = DataTransform(
            args.resolution,
            args.challenge,
            mask,
            use_seed=True,
            scale_mode=args.scale_mode,
            scale_percentile=args.scale_percentile,
        )
        val_data_transform = DataTransform(
            args.resolution,
            args.challenge,
            mask,
            use_seed=True,
            scale_mode=args.scale_mode,
            scale_percentile=args.scale_percentile,
        )

        if args.phase == 'train':
            train_loader = _create_dataset(
                args=args,
                data_path=path_dict["train"],
                data_transform=train_data_transform,
                data_partition="singlecoil_train",
                sequence=None,
                bs=args.train_batch_size,
                shuffle=True,
                sample_rate=args.sample_rate,
                mask_func=mask,
            )
            val_loader = _create_dataset(
                args=args,
                data_path=path_dict["val"],
                data_transform=val_data_transform,
                data_partition="singlecoil_val",
                sequence=None,
                bs=args.val_batch_size,
                shuffle=True,
                sample_rate=args.val_sample_rate,
                mask_func=mask,
            )
    else:
        exit('Error: unrecognized challenge')

    ckpt_callback = ModelCheckpoint(
        dirpath=str(output_dir) if output_dir is not None else None,
        save_last=True,
    )
    lr_monitor_callback = LearningRateMonitor()
    trainer = Trainer(
        logger=logger,
        max_epochs=args.max_epochs,
        accelerator="gpu",
        strategy="auto",
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

    # model = torch.compile(model, mode="default") # TODO OMER ADDED THIS
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
        "--phase",
        type=str,
        required=True,
        choices=["train", "val"],
        help="Phase of the dataset.",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        required=True,
        help="Path to training fastMRI .h5 files directory.",
    )
    parser.add_argument(
        "--val_dataset",
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
        default=1,
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
        "--mask_type",
        type=str,
        default="random",
        choices=["random", "equispaced"],
        help="Type of synthetic undersampling mask.",
    )
    parser.add_argument('--accelerations', nargs='+', default=[4], type=int,
                      help='Ratio of k-space columns to be sampled. If multiple values are '
                           'provided, then one of those is chosen uniformly at random for '
                           'each volume.')
                           
    parser.add_argument('--center_fractions', nargs='+', default=[0.08], type=float,
                      help='Fraction of low-frequency k-space columns to be sampled. Should '
                           'have the same length as accelerations')

    parser.add_argument(
        "--resolution",
        type=int,
        default=320,
        help="Resolution of the image.",
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=None,
        help="Optional RNG seed for random masks.",
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

    # ---------------- additional scaling options for x/y ----------------
    parser.add_argument(
        "--scale_mode",
        type=str,
        default="subsample_max",
        choices=[
            "none",
            "subsample_max",
            "subsample_percentile",
            # volume-level variants (computed from y over the whole volume / .h5 file)
            "volume_subsample_max",
            "volume_subsample_percentile",
        ],
        help=(
            "Optional extra scaling for x and y based on the subsampled image y: "
            "'none', 'subsample_max', or 'subsample_percentile'. "
            "Volume-level variants 'volume_subsample_max'/'volume_subsample_percentile' "
            "compute one scale per volume (cached per .h5) using only y."
        ),
    )
    parser.add_argument(
        "--scale_percentile",
        type=float,
        default=100.0,
        help=(
            "Percentile used when --scale_mode=subsample_percentile "
            "(e.g., 99.0 for 99th percentile of y)."
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

    parser.add_argument(
        "--overfit_first_n_slices",
        type=int,
        default=None,
        help=(
            "If set, restrict the *training* dataset to the first N slices/examples. "
            "Useful for quickly validating that the model can overfit a tiny set."
        ),
    )

    parser.add_argument(
        "--overfit_train_file_name",
        type=str,
        default=None,
        help=(
            "If set, restrict the *training* dataset to only slices coming from the given .h5 "
            "file name (basename match), e.g. 'file1000425.h5'."
        ),
    )

    main(parser.parse_args())


