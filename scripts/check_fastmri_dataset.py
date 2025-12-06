import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
from torch.utils.data import DataLoader

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional
    plt = None

# Ensure project root is on sys.path so that `import PMRF...` works when this
# script is executed from the `scripts` directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PMRF.torch_datasets.fastmri_dataset import (
    SliceDataset,
    ImagePairSliceDataset,
    visualize_dataset_index,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Utility to inspect a fastMRI dataset using SliceDataset / "
            "AnnotatedSliceDataset from PMRF.torch_datasets.fastmri_dataset."
        )
    )

    # Core dataset options
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Directory containing fastMRI .h5 files.",
    )
    parser.add_argument(
        "--challenge",
        type=str,
        choices=["singlecoil", "multicoil"],
        default="singlecoil",
        help='fastMRI challenge type (default: "singlecoil").',
    )

    parser.add_argument(
        "--subsplit",
        type=str,
        choices=["knee", "brain"],
        default="knee",
        help="Subsplit for fastMRI+ annotations (knee or brain).",
    )
    parser.add_argument(
        "--multiple_annotation_policy",
        type=str,
        choices=["first", "random", "all"],
        default="first",
        help=(
            "Policy when multiple annotations exist for a slice: "
            '"first", "random", or "all".'
        ),
    )
    parser.add_argument(
        "--annotation_version",
        type=str,
        default=None,
        help="Optional git hash / version for fastMRI+ annotations CSV.",
    )

    # Sampling / caching options
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=None,
        help="Fraction of slices to load (mutually exclusive with volume_sample_rate).",
    )
    parser.add_argument(
        "--volume_sample_rate",
        type=float,
        default=None,
        help="Fraction of volumes to load (mutually exclusive with sample_rate).",
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

    # Inspection / loader options
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for DataLoader when inspecting batches (default: 1).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=30,
        help="Number of individual samples to inspect from the dataset.",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=2,
        help="Number of batches to inspect from the DataLoader.",
    )
    parser.add_argument(
        "--image_pair_mode",
        action="store_true",
        help=(
            "If set, use ImagePairSliceDataset whose __getitem__ returns "
            "(full_image, subsampled_image) tensors instead of raw k-space tuples."
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "If set, visualize raw and subsampled k-space and images for each "
            "inspected individual sample using matplotlib."
        ),
    )

    # Intensity normalization options for ImagePairSliceDataset
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
            "Intensity normalization mode for ImagePairSliceDataset when "
            "--image_pair_mode is set: 'none', 'per_slice_max' (original), "
            "'per_slice_percentile', 'per_volume_percentile' (per-volume), "
            "or 'global_percentile' (dataset-level robust scale)."
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

    return parser.parse_args()


def _format_shape(x) -> str:
    if x is None:
        return "None"
    if torch.is_tensor(x):
        return f"tensor{tuple(x.shape)}, dtype={x.dtype}"
    return str(type(x))


def inspect_sample(sample, idx: int):
    """
    Print information about a single sample from the dataset.

    - For SliceDataset:
        (kspace, mask, target, attrs, fname, dataslice)
    - For ImagePairSliceDataset:
        (full_image, subsampled_image, kspace_full, kspace_sub, mask)
    """

    # Case 1: ImagePairSliceDataset -> image + k-space pair
    if isinstance(sample, (list, tuple)) and len(sample) == 5 and all(
        torch.is_tensor(x) for x in sample
    ):
        full_img, sub_img, kspace_full, kspace_sub, mask = sample
        print(f"Sample {idx}: ImagePairSliceDataset output")
        print(f"  full_image:       {_format_shape(full_img)}")
        print(f"  subsampled_image: {_format_shape(sub_img)}")
        print(f"  kspace_full:      {_format_shape(kspace_full)}")
        print(f"  kspace_sub:       {_format_shape(kspace_sub)}")
        print(f"  mask:             {_format_shape(mask)}")
        return

    # Case 2: SliceDataset default structure
    kspace, mask, target, attrs, fname, dataslice = sample

    print(f"Sample {idx}: file={fname}, slice={dataslice}")

    # --- Type / tensor checks -------------------------------------------------
    print("  [__getitem__ output types]")
    print(f"    kspace type: {type(kspace)}, is_tensor={torch.is_tensor(kspace)}")
    print(f"    mask   type: {type(mask)}, is_tensor={torch.is_tensor(mask)}")
    print(f"    target type: {type(target)}, is_tensor={torch.is_tensor(target)}")

    # Always convert to tensor view for consistent shape / dtype reporting
    kspace_t = torch.as_tensor(kspace)
    print(f"  kspace: {_format_shape(kspace_t)}")
    if mask is not None:
        mask_t = torch.as_tensor(mask)
        print(f"  mask:   {_format_shape(mask_t)}")
    else:
        print("  mask:   None")
    if target is not None:
        target_t = torch.as_tensor(target)
        print(f"  target: {_format_shape(target_t)}")
    else:
        print("  target: None")

    # Basic attrs / metadata summary
    if isinstance(attrs, dict):
        keys_to_show = [
            "padding_left",
            "padding_right",
            "encoding_size",
            "recon_size",
            "acquisition",
            "patient_id",
            "annotation",
        ]
        present_keys = [k for k in keys_to_show if k in attrs]
        print(f"  attrs keys (subset): {present_keys}")
        for k in present_keys:
            print(f"    {k}: {attrs[k]}")
    else:
        print(f"  attrs type: {type(attrs)}")


def simple_collate(batch):
    """
    Collate function that returns the list of samples as-is.

    This avoids PyTorch's default_collate trying to stack elements like
    `mask=None` or arbitrary dicts, which would otherwise raise errors.
    """
    return batch


def build_dataset(
    data_root: str,
    challenge: str,
    subsplit: str,
    multiple_annotation_policy: str,
    annotation_version: Optional[str],
    sample_rate: Optional[float],
    volume_sample_rate: Optional[float],
    use_dataset_cache: bool,
    dataset_cache_file: str,
    num_cols: Optional[Sequence[int]],
    image_pair_mode: bool,
    intensity_norm: str,
    intensity_norm_percentile: float,
):
    if image_pair_mode:
        dataset = ImagePairSliceDataset(
            root=data_root,
            challenge=challenge,
            transform=None,
            use_dataset_cache=use_dataset_cache,
            sample_rate=sample_rate,
            volume_sample_rate=volume_sample_rate,
            dataset_cache_file=dataset_cache_file,
            num_cols=tuple(num_cols) if num_cols is not None else None,
            intensity_norm=intensity_norm,
            intensity_norm_percentile=intensity_norm_percentile,
        )
    else:
        dataset = SliceDataset(
            root=data_root,
            challenge=challenge,
            transform=None,
            use_dataset_cache=use_dataset_cache,
            sample_rate=sample_rate,
            volume_sample_rate=volume_sample_rate,
            dataset_cache_file=dataset_cache_file,
            num_cols=tuple(num_cols) if num_cols is not None else None,
        )

    return dataset


def main():
    args = parse_args()

    if args.sample_rate is not None and args.volume_sample_rate is not None:
        raise ValueError(
            "Specify at most one of --sample_rate or --volume_sample_rate, not both."
        )

    print("Initializing dataset...")
    dataset = build_dataset(
        data_root=args.data_root,
        challenge=args.challenge,
        subsplit=args.subsplit,
        multiple_annotation_policy=args.multiple_annotation_policy,
        annotation_version=args.annotation_version,
        sample_rate=args.sample_rate,
        volume_sample_rate=args.volume_sample_rate,
        use_dataset_cache=args.use_dataset_cache,
        dataset_cache_file=args.dataset_cache_file,
        num_cols=args.num_cols,
        image_pair_mode=args.image_pair_mode,
        intensity_norm=args.intensity_norm,
        intensity_norm_percentile=args.intensity_norm_percentile,
    )

    print(
        f"Dataset type: {type(dataset).__name__}, "
        f"num_samples={len(dataset)}"
    )

    # Inspect a few individual samples
    num_to_show = min(args.num_samples, len(dataset))
    print(f"\nInspecting {num_to_show} individual samples...\n")
    for i in range(num_to_show):
        try:
            sample = dataset[i]
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR loading sample {i}: {exc}")
            continue
        inspect_sample(sample, i)
        print()

        if args.visualize:
            if isinstance(dataset, ImagePairSliceDataset):
                if plt is None:
                    print("  [visualize] matplotlib not available; cannot plot image pairs.")
                else:
                    full_img, sub_img, kspace_full_t, kspace_sub_t, mask_t = sample
                    full_np = full_img.squeeze().cpu().numpy()
                    sub_np = sub_img.squeeze().cpu().numpy()

                    # Prepare k-space magnitude views from raw tensors
                    def kspace_mag_display(k):
                        k_np = k.cpu().numpy()
                        if k_np.ndim >= 1 and k_np.shape[-1] == 2:
                            k_c = k_np[..., 0] + 1j * k_np[..., 1]
                        else:
                            k_c = k_np.astype(np.complex64)
                        mag = np.abs(k_c)
                        while mag.ndim > 2:
                            mag = mag.sum(axis=0)
                        return np.log1p(mag)

                    k_full = kspace_mag_display(kspace_full_t)
                    k_sub = kspace_mag_display(kspace_sub_t)

                    # Binary mask from mask tensor (collapse non-spatial dims)
                    mask_np = mask_t.cpu().numpy()
                    while mask_np.ndim > 2:
                        mask_np = mask_np[0]
                    mask = (mask_np > 0.5).astype(float)

                    fig, axes = plt.subplots(2, 3, figsize=(10, 7))

                    # Use a fixed intensity range so that different normalization
                    # schemes (e.g. per-slice max vs percentile) produce visually
                    # different results. Without vmin/vmax, matplotlib rescales
                    # each image independently, making pure rescalings look
                    # identical.
                    axes[0, 0].imshow(full_np, cmap="gray", vmin=0.0, vmax=1.0)
                    axes[0, 0].set_title("Full image (fixed [0,1] display range)")
                    axes[0, 0].axis("off")

                    axes[0, 1].imshow(sub_np, cmap="gray", vmin=0.0, vmax=1.0)
                    axes[0, 1].set_title("Subsampled image (fixed [0,1] range)")
                    axes[0, 1].axis("off")

                    axes[0, 2].imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
                    axes[0, 2].set_title("Sampling mask (binary)")
                    axes[0, 2].axis("off")

                    axes[1, 0].imshow(k_full, cmap="gray")
                    axes[1, 0].set_title("Full k-space (log |k|)")
                    axes[1, 0].axis("off")

                    axes[1, 1].imshow(k_sub, cmap="gray")
                    axes[1, 1].set_title("Subsampled k-space (log |k|)")
                    axes[1, 1].axis("off")

                    axes[1, 2].axis("off")

                    fig.suptitle(f"ImagePairSliceDataset sample {i}")
                    fig.tight_layout()
                    plt.show()
            else:
                # This will open a matplotlib window per sample (for debugging / inspection).
                visualize_dataset_index(
                    dataset,
                    i,
                    title_prefix="check_fastmri_dataset",
                )

    # Optionally inspect DataLoader batches
    if args.batch_size > 0:
        print(
            f"\nBuilding DataLoader with batch_size={args.batch_size}, "
            f"num_workers={args.num_workers}..."
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=simple_collate,
        )

        print(f"Inspecting up to {args.num_batches} batches...\n")
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.num_batches:
                break

            print(f"Batch {batch_idx}: type={type(batch)}")
            # Case 1: our simple_collate -> list of samples
            if isinstance(batch, list) and len(batch) > 0 and isinstance(
                batch[0], (list, tuple)
            ):
                first_sample = batch[0]
                kspace_b = torch.as_tensor(first_sample[0])
                target_b = (
                    torch.as_tensor(first_sample[2])
                    if first_sample[2] is not None
                    else None
                )
                print(
                    "  Using first sample in batch to summarize shapes "
                    f"(batch_size={len(batch)})."
                )
                print(f"  kspace: {_format_shape(kspace_b)}")
                print(f"  target: {_format_shape(target_b)}")
            # Case 2: some other collated structure (fallback)
            else:
                print("  Batch structure not recognized; full batch repr:")
                print(f"  {batch!r}")
            print()


if __name__ == "__main__":
    main()


