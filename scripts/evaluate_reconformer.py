#!/usr/bin/env python3
"""
Evaluate Reconformer predictions against fastMRI ground truth.

Computes:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - LPIPS (Learned Perceptual Image Patch Similarity)
  - DISTS (Deep Image Structure and Texture Similarity)

Usage:
    python evaluate_reconformer.py \
        --pred_dir /storage/omer/reconformer_results/X4/fastmri/recon_pt \
        --gt_dir /storage/omer/data/fastmri/singlecoil_val \
        --output_csv reconformer_metrics.csv
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

# PIQ perceptual metrics
from piq import LPIPS, DISTS


def load_gt_volume(gt_path: Path) -> np.ndarray:
    """Load ground truth volume from HDF5 file."""
    with h5py.File(gt_path, "r") as f:
        # fastMRI singlecoil uses 'reconstruction_esc'
        if "reconstruction_esc" in f:
            gt = f["reconstruction_esc"][:]
        elif "reconstruction_rss" in f:
            gt = f["reconstruction_rss"][:]
        else:
            raise KeyError(f"No reconstruction key found in {gt_path}")
    return gt.astype(np.float32)


def load_pred_volume(pred_path: Path) -> np.ndarray:
    """Load prediction volume from .pt file."""
    pred = torch.load(pred_path, map_location="cpu")
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    return pred.astype(np.float32)


def compute_slice_metrics(
    pred_slice: np.ndarray,
    gt_slice: np.ndarray,
    data_range: float,
) -> Dict[str, float]:
    """Compute PSNR and SSIM for a single slice."""
    # Ensure 2D
    pred_slice = pred_slice.squeeze()
    gt_slice = gt_slice.squeeze()
    
    # PSNR
    psnr = peak_signal_noise_ratio(gt_slice, pred_slice, data_range=data_range)
    
    # SSIM
    ssim = structural_similarity(gt_slice, pred_slice, data_range=data_range)
    
    return {"psnr": psnr, "ssim": ssim}


def compute_perceptual_metrics_batch(
    pred_batch: torch.Tensor,
    gt_batch: torch.Tensor,
    lpips_model: LPIPS,
    dists_model: DISTS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute LPIPS and DISTS for a batch of images.
    
    Args:
        pred_batch: (B, 1, H, W) tensor
        gt_batch: (B, 1, H, W) tensor
        lpips_model: LPIPS model instance
        dists_model: DISTS model instance
    
    Returns:
        lpips_scores: (B,) tensor
        dists_scores: (B,) tensor
    """
    # LPIPS and DISTS expect 3-channel images, tile grayscale to RGB
    if pred_batch.shape[1] == 1:
        pred_3ch = pred_batch.repeat(1, 3, 1, 1)
        gt_3ch = gt_batch.repeat(1, 3, 1, 1)
    else:
        pred_3ch = pred_batch
        gt_3ch = gt_batch
    
    # Clamp to [0, 1] for perceptual metrics
    pred_3ch = pred_3ch.clamp(0, 1)
    gt_3ch = gt_3ch.clamp(0, 1)
    
    with torch.no_grad():
        lpips_scores = lpips_model(pred_3ch, gt_3ch)
        dists_scores = dists_model(pred_3ch, gt_3ch)
    
    return lpips_scores, dists_scores


def evaluate_volume(
    pred: np.ndarray,
    gt: np.ndarray,
    lpips_model: LPIPS,
    dists_model: DISTS,
    device: torch.device,
    batch_size: int = 8,
) -> Dict[str, List[float]]:
    """
    Evaluate all slices in a volume.
    
    Returns dict with lists of per-slice metrics.
    """
    # Normalize both by GT max (fastMRI convention)
    gt_max = gt.max()
    if gt_max < 0:
        gt_max = 1.0  # Avoid division by zero for blank volumes
    
    pred_norm = pred / gt_max
    gt_norm = gt / gt_max
    
    num_slices = gt.shape[0]
    
    # Handle slice count mismatch
    if pred.shape[0] != num_slices:
        print(f"  Warning: slice mismatch pred={pred.shape[0]} vs gt={num_slices}")
        num_slices = min(pred.shape[0], gt.shape[0])
    
    metrics = {"psnr": [], "ssim": [], "lpips": [], "dists": []}
    
    # Compute PSNR and SSIM per slice
    for s in range(num_slices):
        slice_metrics = compute_slice_metrics(
            pred_norm[s], gt_norm[s], data_range=1.0
        )
        metrics["psnr"].append(slice_metrics["psnr"])
        metrics["ssim"].append(slice_metrics["ssim"])
    
    # Compute perceptual metrics in batches
    pred_tensor = torch.from_numpy(pred_norm[:num_slices]).unsqueeze(1).float()
    gt_tensor = torch.from_numpy(gt_norm[:num_slices]).unsqueeze(1).float()
    
    for start in range(0, num_slices, batch_size):
        end = min(start + batch_size, num_slices)
        pred_batch = pred_tensor[start:end].to(device)
        gt_batch = gt_tensor[start:end].to(device)
        
        lpips_scores, dists_scores = compute_perceptual_metrics_batch(
            pred_batch, gt_batch, lpips_model, dists_model
        )
        
        metrics["lpips"].extend(lpips_scores.cpu().tolist())
        metrics["dists"].extend(dists_scores.cpu().tolist())
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Reconformer predictions against fastMRI ground truth."
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default="/storage/omer/reconformer_results/X4/fastmri/recon_pt",
        help="Directory containing .pt prediction files.",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/storage/omer/data/fastmri/singlecoil_val",
        help="Directory containing .h5 ground truth files.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="reconformer_metrics.csv",
        help="Output CSV file for per-volume results.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for perceptual metrics computation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for perceptual metrics (cuda or cpu).",
    )
    args = parser.parse_args()
    
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    device = torch.device(args.device)
    
    print(f"Prediction directory: {pred_dir}")
    print(f"Ground truth directory: {gt_dir}")
    print(f"Device: {device}")
    print()
    
    # Find matching files
    pred_files = sorted([f for f in pred_dir.iterdir() if f.suffix == ".pt"])
    gt_files = {f.stem: f for f in gt_dir.iterdir() if f.suffix in (".h5", ".hdf5")}
    
    matched_pairs = []
    for pred_file in pred_files:
        fname = pred_file.stem
        if fname in gt_files:
            matched_pairs.append((pred_file, gt_files[fname], fname))
    
    print(f"Found {len(matched_pairs)} matching prediction/GT pairs")
    print()
    
    if len(matched_pairs) == 0:
        print("No matching files found. Exiting.")
        return
    
    # Initialize perceptual metrics models
    print("Loading perceptual metric models...")
    lpips_model = LPIPS(replace_pooling=True, reduction="none").to(device).eval()
    dists_model = DISTS(reduction="none").to(device).eval()
    
    for param in lpips_model.parameters():
        param.requires_grad = False
    for param in dists_model.parameters():
        param.requires_grad = False
    
    print("Models loaded.\n")
    
    # Evaluate all volumes
    all_results = []
    all_psnr = []
    all_ssim = []
    all_lpips = []
    all_dists = []
    
    for pred_path, gt_path, fname in tqdm(matched_pairs, desc="Evaluating volumes"):
        # Load data
        pred = load_pred_volume(pred_path)
        gt = load_gt_volume(gt_path)
        
        # Compute metrics
        metrics = evaluate_volume(
            pred, gt, lpips_model, dists_model, device, args.batch_size
        )
        
        # Per-volume averages
        vol_psnr = np.mean(metrics["psnr"])
        vol_ssim = np.mean(metrics["ssim"])
        vol_lpips = np.mean(metrics["lpips"])
        vol_dists = np.mean(metrics["dists"])
        
        all_results.append({
            "filename": fname,
            "num_slices": len(metrics["psnr"]),
            "psnr": vol_psnr,
            "ssim": vol_ssim,
            "lpips": vol_lpips,
            "dists": vol_dists,
        })
        
        # Collect all slice metrics for overall statistics
        all_psnr.extend(metrics["psnr"])
        all_ssim.extend(metrics["ssim"])
        all_lpips.extend(metrics["lpips"])
        all_dists.extend(metrics["dists"])
    
    # Save per-volume results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(args.output_csv, index=False)
    
    # Compute overall statistics
    psnr_mean = np.mean(all_psnr)
    psnr_std = np.std(all_psnr)
    ssim_mean = np.mean(all_ssim)
    ssim_std = np.std(all_ssim)
    lpips_mean = np.mean(all_lpips)
    lpips_std = np.std(all_lpips)
    dists_mean = np.mean(all_dists)
    dists_std = np.std(all_dists)
    
    # Also compute per-volume averages then averaged (scan-level metric)
    vol_psnrs = [r["psnr"] for r in all_results]
    vol_ssims = [r["ssim"] for r in all_results]
    vol_lpips = [r["lpips"] for r in all_results]
    vol_dists = [r["dists"] for r in all_results]
    
    print()
    print("=" * 50)
    print("Reconformer Evaluation Results")
    print("=" * 50)
    print()
    print("Per-Slice Statistics (mean +/- std):")
    print(f"  PSNR:  {psnr_mean:.4f} +/- {psnr_std:.4f} dB")
    print(f"  SSIM:  {ssim_mean:.6f} +/- {ssim_std:.6f}")
    print(f"  LPIPS: {lpips_mean:.6f} +/- {lpips_std:.6f}")
    print(f"  DISTS: {dists_mean:.6f} +/- {dists_std:.6f}")
    print()
    print("Per-Volume Averages (mean +/- std across volumes):")
    print(f"  PSNR:  {np.mean(vol_psnrs):.4f} +/- {np.std(vol_psnrs):.4f} dB")
    print(f"  SSIM:  {np.mean(vol_ssims):.6f} +/- {np.std(vol_ssims):.6f}")
    print(f"  LPIPS: {np.mean(vol_lpips):.6f} +/- {np.std(vol_lpips):.6f}")
    print(f"  DISTS: {np.mean(vol_dists):.6f} +/- {np.std(vol_dists):.6f}")
    print()
    print(f"Results saved to: {args.output_csv}")
    print(f"Total volumes evaluated: {len(all_results)}")
    print(f"Total slices evaluated: {len(all_psnr)}")


if __name__ == "__main__":
    main()

