#!/usr/bin/env python3
"""
PMRF Inference Script - Save predictions as .pt volume files.

Runs inference on fastMRI data using a trained PMRF checkpoint and saves
predictions as .pt files per volume (same format as reconformer).

Usage:
    python scripts/inference_pmrf.py \
        --checkpoint /path/to/model.ckpt \
        --data_dir /storage/omer/data/fastmri/singlecoil_val \
        --output_dir /storage/omer/pmrf_results/recon_pt \
        --num_flow_steps 16 \
        --batch_size 8
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root and PMRF subdir are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PMRF_ROOT = PROJECT_ROOT / "PMRF"
if str(PMRF_ROOT) not in sys.path:
    sys.path.insert(0, str(PMRF_ROOT))

from data import SliceData, DataTransform, create_mask_for_mask_type
from data.transforms import apply_data_consistency
from PMRF.lightning_models.mmse_rectified_flow import MMSERectifiedFlow


def get_fnames_and_slices(batch: Dict, batch_size: int) -> tuple:
    """Extract fname and slice index from batch."""
    # Get fnames
    fnames_raw = batch.get("fname", None)
    if fnames_raw is None:
        fnames = ["unknown"] * batch_size
    elif isinstance(fnames_raw, (list, tuple)):
        fnames = [f if isinstance(f, str) else str(f) for f in fnames_raw]
    else:
        fnames = [str(fnames_raw)] * batch_size
    
    # Get slice indices
    slices_raw = batch.get("slice", None)
    if slices_raw is None:
        slices = list(range(batch_size))
    elif torch.is_tensor(slices_raw):
        slices = slices_raw.detach().cpu().tolist()
    elif isinstance(slices_raw, (list, tuple)):
        slices = [int(s) for s in slices_raw]
    else:
        slices = [int(slices_raw)] * batch_size
    
    return fnames, slices


def get_normalization_factors(batch: Dict, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract normalization factors from batch for unnormalization.
    
    Returns:
        norm_std: (B,) tensor - mean(|y|) or vol_abs_mean per slice
        norm_scale: (B,) tensor - vol_scale or slice-level scale per slice
    """
    # Get norm_std (mean(|y|) or vol_abs_mean)
    norm_std_raw = batch.get("norm_std", None)
    if norm_std_raw is None:
        norm_std = torch.ones(batch_size)
    elif torch.is_tensor(norm_std_raw):
        norm_std = norm_std_raw.detach().cpu().float()
        if norm_std.ndim == 0:
            norm_std = norm_std.repeat(batch_size)
    else:
        norm_std = torch.tensor([float(norm_std_raw)] * batch_size)
    
    # Get norm_scale (vol_scale or slice-level scale)
    norm_scale_raw = batch.get("norm_scale", None)
    if norm_scale_raw is None:
        norm_scale = torch.ones(batch_size)
    elif torch.is_tensor(norm_scale_raw):
        norm_scale = norm_scale_raw.detach().cpu().float()
        if norm_scale.ndim == 0:
            norm_scale = norm_scale.repeat(batch_size)
    else:
        norm_scale = torch.tensor([float(norm_scale_raw)] * batch_size)
    
    return norm_std, norm_scale


def has_dc_data(batch: Dict) -> bool:
    """Check if the batch contains data for data consistency."""
    required_keys = ['kspace', 'mask', 'norm_std', 'norm_scale']
    return all(k in batch for k in required_keys)


def get_dc_data(batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Extract data consistency fields from batch and move to device.
    """
    batch_size = batch['x'].shape[0]
    
    # Get kspace - shape (B, H, W, 2)
    kspace = batch['kspace']
    if kspace.ndim == 3:
        kspace = kspace.unsqueeze(0)
    kspace = kspace.to(device)
    
    # Get mask
    mask = batch['mask']
    if mask.ndim == 3:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
    elif mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask.expand(batch_size, -1, -1, -1)
    mask = mask.to(device)
    
    # Get normalization factors
    norm_std = batch['norm_std']
    if torch.is_tensor(norm_std):
        norm_std = norm_std.to(device).float()
    else:
        norm_std = torch.tensor(norm_std, device=device, dtype=torch.float32)
    if norm_std.ndim == 0:
        norm_std = norm_std.unsqueeze(0).expand(batch_size)
    
    norm_scale = batch['norm_scale']
    if torch.is_tensor(norm_scale):
        norm_scale = norm_scale.to(device).float()
    else:
        norm_scale = torch.tensor(norm_scale, device=device, dtype=torch.float32)
    if norm_scale.ndim == 0:
        norm_scale = norm_scale.unsqueeze(0).expand(batch_size)
    
    resolution = batch['x'].shape[-1]
    
    return {
        'kspace': kspace,
        'mask': mask,
        'norm_std': norm_std,
        'norm_scale': norm_scale,
        'resolution': resolution,
    }


def create_dataloader(
    data_dir: str,
    challenge: str,
    mask_type: str,
    center_fractions: List[float],
    accelerations: List[int],
    resolution: int,
    scale_mode: str,
    scale_percentile: float,
    batch_size: int,
    num_workers: int,
    max_volumes: Optional[int] = None,
    include_dc_data: bool = False,
) -> Tuple[DataLoader, int]:
    """
    Create dataloader for inference.
    
    Args:
        include_dc_data: If True, include kspace and mask for data consistency.
    
    Returns:
        dataloader: The DataLoader instance
        num_volumes: Number of unique volumes in the dataset
    """
    from torch.utils.data import Subset
    
    mask_func = create_mask_for_mask_type(mask_type, center_fractions, accelerations)
    
    data_transform = DataTransform(
        resolution=resolution,
        which_challenge=challenge,
        mask_func=mask_func,
        use_seed=True,  # Use seed for reproducible masks
        scale_mode=scale_mode,
        scale_percentile=scale_percentile,
        include_dc_data=include_dc_data,  # Include DC data if requested
    )
    
    dataset = SliceData(
        root=Path(data_dir),
        transform=data_transform,
        challenge=challenge,
        sequence=None,
        sample_rate=1.0,
    )
    
    # Count unique volumes
    all_fnames = [ex[0].name if hasattr(ex[0], 'name') else str(ex[0]) for ex in dataset.examples]
    unique_fnames = sorted(set(all_fnames))
    num_volumes = len(unique_fnames)
    
    # Filter to max_volumes if specified
    if max_volumes is not None and max_volumes > 0:
        selected_fnames = set(unique_fnames[:max_volumes])
        selected_indices = [
            i for i, ex in enumerate(dataset.examples)
            if (ex[0].name if hasattr(ex[0], 'name') else str(ex[0])) in selected_fnames
        ]
        dataset = Subset(dataset, selected_indices)
        num_volumes = min(max_volumes, num_volumes)
        print(f"Limited to {max_volumes} volume(s): {list(selected_fnames)[:3]}{'...' if max_volumes > 3 else ''}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    
    return dataloader, num_volumes


def main():
    parser = argparse.ArgumentParser(
        description="Run PMRF inference and save predictions as .pt volume files."
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained PMRF checkpoint (.ckpt file).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing fastMRI .h5 files for inference.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save .pt prediction files.",
    )
    
    # Inference parameters
    parser.add_argument(
        "--num_flow_steps",
        type=int,
        default=None,
        help="Number of flow steps for inference. If not set, uses checkpoint default.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference (cuda or cpu).",
    )
    
    # Data parameters (should match training)
    parser.add_argument(
        "--challenge",
        type=str,
        default="singlecoil",
        choices=["singlecoil", "multicoil"],
        help="fastMRI challenge type.",
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        default="random",
        choices=["random", "equispaced"],
        help="Type of undersampling mask.",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        type=float,
        default=[0.04],
        help="Fraction of low-frequency k-space columns to sample.",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        type=int,
        default=[4],
        help="Acceleration factors.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=320,
        help="Resolution of output images.",
    )
    parser.add_argument(
        "--scale_mode",
        type=str,
        default="volume_subsample_max",
        choices=["none", "subsample_max", "subsample_percentile", 
                 "volume_subsample_max", "volume_subsample_percentile"],
        help="Intensity scaling mode (should match training).",
    )
    parser.add_argument(
        "--scale_percentile",
        type=float,
        default=100.0,
        help="Percentile for scaling (if using percentile mode).",
    )
    parser.add_argument(
        "--max_volumes",
        type=int,
        default=None,
        help="Limit inference to first N volumes (for testing). Default: all volumes.",
    )
    
    # Data Consistency arguments
    parser.add_argument(
        "--apply_dc",
        action="store_true",
        help="Apply data consistency to final predictions before saving.",
    )
    parser.add_argument(
        "--apply_dc_to_source",
        action="store_true",
        help=(
            "Apply data consistency (DC) to the MMSE posterior mean before using it "
            "as the source distribution input to the flow model. This enforces the "
            "measured k-space frequencies from the subsampled y data on the MMSE output."
        ),
    )
    parser.add_argument(
        "--save_both",
        action="store_true",
        help="Save both DC and non-DC predictions (implies --apply_dc).",
    )
    
    args = parser.parse_args()
    
    # If save_both is set, also set apply_dc
    if args.save_both:
        args.apply_dc = True
    
    # Need DC data for either apply_dc or apply_dc_to_source
    need_dc_data = args.apply_dc or args.apply_dc_to_source
    
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print()
    
    # Load model from checkpoint
    print("Loading model from checkpoint...")
    model = MMSERectifiedFlow.load_from_checkpoint(
        args.checkpoint,
        map_location=device,
    )
    model = model.to(device)
    model.eval()
    
    # Get num_flow_steps from args or model
    num_flow_steps = args.num_flow_steps
    if num_flow_steps is None:
        num_flow_steps = model.hparams.num_flow_steps
    print(f"Using num_flow_steps: {num_flow_steps}")
    print()
    
    # Create dataloader
    print("Creating dataloader...")
    dataloader, num_volumes = create_dataloader(
        data_dir=args.data_dir,
        challenge=args.challenge,
        mask_type=args.mask_type,
        center_fractions=args.center_fractions,
        accelerations=args.accelerations,
        resolution=args.resolution,
        scale_mode=args.scale_mode,
        scale_percentile=args.scale_percentile,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_volumes=args.max_volumes,
        include_dc_data=need_dc_data,  # Include DC data if applying data consistency to source or output
    )
    print(f"Total volumes: {num_volumes}")
    print(f"Total slices: {len(dataloader.dataset)}")
    print()
    
    # Accumulate predictions by volume (fname)
    # Structure: volume_preds[fname] = {slice_idx: tensor}
    volume_preds: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict)
    volume_preds_dc: Dict[str, Dict[int, torch.Tensor]] = defaultdict(dict) if args.apply_dc else None
    
    print("Running inference...")
    print("Note: Complex output is converted to magnitude and unnormalized to match raw GT scale.")
    if args.apply_dc_to_source:
        print("Data consistency (DC) will be applied to MMSE output before flow.")
    if args.apply_dc:
        print("Data consistency (DC) will be applied to final predictions.")
    if args.save_both:
        print("Both DC and non-DC predictions will be saved.")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            non_noisy_z0 = batch.get("non_noisy_z0", None)
            if non_noisy_z0 is not None:
                non_noisy_z0 = non_noisy_z0.to(device)
            
            batch_size = x.shape[0]
            fnames, slices = get_fnames_and_slices(batch, batch_size)
            
            # Get normalization factors for unnormalization
            norm_std, norm_scale = get_normalization_factors(batch, batch_size)
            
            # Get DC data for source distribution if apply_dc_to_source is enabled
            dc_data_for_source = None
            if args.apply_dc_to_source and has_dc_data(batch):
                dc_data_for_source = get_dc_data(batch, device)
            
            # Generate reconstructions (returns complex 2-channel output)
            with model.maybe_ema():
                xhat, _, _ = model.generate_reconstructions(
                    x, y, non_noisy_z0, num_flow_steps, torch.device("cpu"), dc_data=dc_data_for_source
                )
            
            # Apply data consistency if requested
            xhat_dc = None
            if args.apply_dc and has_dc_data(batch):
                dc_data = get_dc_data(batch, device)
                xhat_dc = apply_data_consistency(
                    xhat=xhat.to(device),
                    kspace=dc_data['kspace'],
                    mask=dc_data['mask'],
                    norm_std=dc_data['norm_std'],
                    norm_scale=dc_data['norm_scale'],
                    resolution=dc_data['resolution'],
                )
                xhat_dc = xhat_dc.cpu()
            
            # Convert complex (2-ch) to magnitude (1-ch) for saving
            # xhat: (B, 2, H, W) -> (B, 1, H, W)
            xhat_mag = torch.sqrt(xhat[:, 0:1, :, :] ** 2 + xhat[:, 1:2, :, :] ** 2)
            xhat_dc_mag = None
            if xhat_dc is not None:
                xhat_dc_mag = torch.sqrt(xhat_dc[:, 0:1, :, :] ** 2 + xhat_dc[:, 1:2, :, :] ** 2)
            
            # Store predictions by fname and slice (unnormalized)
            for i in range(batch_size):
                fname = fnames[i]
                slice_idx = slices[i]
                # xhat_mag is (B, 1, H, W), we want (H, W) for each slice
                pred_normalized = xhat_mag[i].squeeze(0).cpu()  # Remove channel dim -> (H, W)
                
                # Unnormalize: raw = normalized * norm_scale * norm_std
                # This reverses the DataTransform normalization
                unnorm_factor = norm_scale[i].item() * norm_std[i].item()
                pred_raw = pred_normalized * unnorm_factor
                
                # Save non-DC predictions if save_both or not applying DC
                if args.save_both or not args.apply_dc:
                    volume_preds[fname][slice_idx] = pred_raw
                
                # Save DC predictions
                if xhat_dc_mag is not None:
                    pred_dc_normalized = xhat_dc_mag[i].squeeze(0).cpu()
                    pred_dc_raw = pred_dc_normalized * unnorm_factor
                    if args.save_both:
                        volume_preds_dc[fname][slice_idx] = pred_dc_raw
                    elif args.apply_dc:
                        # Only DC, store in main dict
                        volume_preds[fname][slice_idx] = pred_dc_raw
    
    # Save volumes as .pt files
    print()
    print("Saving volumes...")
    
    def save_volume_dict(vol_dict: Dict[str, Dict[int, torch.Tensor]], suffix: str = ""):
        """Save a dictionary of volumes to .pt files."""
        for fname, slices_dict in tqdm(vol_dict.items(), desc=f"Saving volumes{suffix}"):
            # Sort slices by index and stack
            sorted_indices = sorted(slices_dict.keys())
            slices_list = [slices_dict[idx] for idx in sorted_indices]
            volume_tensor = torch.stack(slices_list, dim=0)  # (num_slices, H, W)
            
            # Remove .h5 extension if present in fname
            fname_clean = fname.replace(".h5", "").replace(".hdf5", "")
            output_path = output_dir / f"{fname_clean}{suffix}.pt"
            torch.save(volume_tensor, output_path)
    
    # Save main predictions (non-DC or DC depending on args)
    if volume_preds:
        if args.save_both:
            save_volume_dict(volume_preds, "")  # Non-DC without suffix
        elif args.apply_dc:
            save_volume_dict(volume_preds, "_dc")  # DC with suffix
        else:
            save_volume_dict(volume_preds, "")  # Non-DC without suffix
    
    # Save DC predictions separately if save_both
    if args.save_both and volume_preds_dc:
        save_volume_dict(volume_preds_dc, "_dc")
    
    print()
    print("=" * 50)
    print("Inference Complete")
    print("=" * 50)
    print(f"Volumes saved: {len(volume_preds)}")
    if args.save_both and volume_preds_dc:
        print(f"DC volumes saved: {len(volume_preds_dc)}")
    print(f"Output directory: {output_dir}")
    
    # Print some statistics
    total_slices = sum(len(v) for v in volume_preds.values())
    print(f"Total slices processed: {total_slices}")
    
    # Print DC mode info
    if args.apply_dc_to_source:
        print(f"DC applied to source (MMSE output): Yes")
    else:
        print(f"DC applied to source (MMSE output): No")
    
    if args.apply_dc:
        print(f"DC applied to final output: Yes")
        if args.save_both:
            print(f"  - Non-DC files: <volume_name>.pt")
            print(f"  - DC files: <volume_name>_dc.pt")
        else:
            print(f"  - DC files: <volume_name>_dc.pt")
    else:
        print(f"DC applied to final output: No")
    
    # Sample output file info
    if volume_preds:
        sample_fname = next(iter(volume_preds.keys()))
        suffix = "" if args.save_both or not args.apply_dc else "_dc"
        sample_path = output_dir / f"{sample_fname.replace('.h5', '').replace('.hdf5', '')}{suffix}.pt"
        if sample_path.exists():
            sample_tensor = torch.load(sample_path, map_location="cpu")
            print(f"Sample output shape: {sample_tensor.shape}")
            print(f"Sample output dtype: {sample_tensor.dtype}")
            print(f"Sample output range: [{sample_tensor.min():.6f}, {sample_tensor.max():.6f}]")


if __name__ == "__main__":
    main()

