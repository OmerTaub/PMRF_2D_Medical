#!/usr/bin/env python3
"""
Visualize and compare two MRI undersampling strategies:

1. Center-only masking: Retains ONLY the low-frequency center fraction of k-space
2. Full random masking: Standard fastMRI approach - center fraction + random high-frequency lines

For each sample, creates a side-by-side visualization showing:
- Undersampling masks
- K-space magnitude (log scale)
- Zero-filled reconstructions (y)
- Ground truth (x)
- PSNR and SSIM metrics

Usage:
    python scripts/visualize_mask_comparison.py \
        --data_path /storage/omer/data/fastmri/singlecoil_train \
        --resolution 320 \
        --center_fraction 0.04 \
        --acceleration 4 \
        --num_samples 5 \
        --save_dir ./mask_comparison_results
"""

import argparse
import sys
import os
import csv
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from data.mri_data import SliceData, DataTransform
from data.subsample import create_mask_for_mask_type, RandomMaskFunc, CenterOnlyMaskFunc
from data import transforms


def complex_to_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor (2, H, W) or (H, W, 2) to magnitude (H, W)."""
    if x.ndim == 3:
        if x.shape[0] == 2:  # (2, H, W)
            return torch.sqrt(x[0] ** 2 + x[1] ** 2)
        elif x.shape[-1] == 2:  # (H, W, 2)
            return torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
    raise ValueError(f"Unexpected shape: {x.shape}")


def normalize_for_display(img: torch.Tensor, percentile: float = 99) -> np.ndarray:
    """Normalize image for display, clipping at percentile."""
    img_np = img.numpy() if isinstance(img, torch.Tensor) else img
    vmax = np.percentile(img_np, percentile)
    vmin = img_np.min()
    if vmax > vmin:
        img_np = (img_np - vmin) / (vmax - vmin)
    return np.clip(img_np, 0, 1)


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    """
    Compute PSNR and SSIM between ground truth and prediction.
    
    Args:
        gt: Ground truth image (H, W)
        pred: Prediction/reconstruction image (H, W)
    
    Returns:
        Dictionary with 'psnr' and 'ssim' values
    """
    # Ensure 2D
    gt = gt.squeeze()
    pred = pred.squeeze()
    
    # Use GT max as data range (fastMRI convention)
    data_range = float(gt.max())
    if data_range < 0:
        data_range = 1.0
    
    psnr = peak_signal_noise_ratio(gt, pred, data_range=data_range)
    ssim = structural_similarity(gt, pred, data_range=data_range)
    
    return {'psnr': psnr, 'ssim': ssim}


def visualize_mask_comparison(
    sample_center: dict,
    sample_random: dict,
    idx: int,
    center_fraction: float,
    acceleration: int,
    save_path: str = None
):
    """
    Create a comprehensive visualization comparing center-only vs random masking.
    
    Args:
        sample_center: Sample dict from center-only mask transform
        sample_random: Sample dict from random mask transform
        idx: Sample index
        center_fraction: Center fraction used
        acceleration: Acceleration factor (for random mask)
        save_path: If provided, save figure to this path
    """
    # Extract data
    x_center = sample_center['x']  # (2, H, W) - ground truth (same for both)
    y_center = sample_center['y']  # (2, H, W) - zero-filled (center-only)
    mask_center = sample_center.get('mask', None)
    
    x_random = sample_random['x']
    y_random = sample_random['y']
    mask_random = sample_random.get('mask', None)
    
    fname = sample_center['fname']
    slice_idx = sample_center['slice']
    
    # Convert to magnitude
    x_mag = complex_to_magnitude(x_center).numpy()  # Ground truth (same for both)
    y_center_mag = complex_to_magnitude(y_center).numpy()
    y_random_mag = complex_to_magnitude(y_random).numpy()
    
    # Compute metrics
    metrics_center = compute_metrics(x_mag, y_center_mag)
    metrics_random = compute_metrics(x_mag, y_random_mag)
    
    # Get k-space for visualization
    y_center_complex = y_center.permute(1, 2, 0)  # (H, W, 2)
    y_random_complex = y_random.permute(1, 2, 0)
    
    y_center_kspace = transforms.fft2(y_center_complex)
    y_random_kspace = transforms.fft2(y_random_complex)
    
    y_center_kspace_mag = complex_to_magnitude(y_center_kspace).numpy()
    y_random_kspace_mag = complex_to_magnitude(y_random_kspace).numpy()
    
    # Log scale for k-space visualization
    y_center_kspace_log = np.log1p(y_center_kspace_mag)
    y_random_kspace_log = np.log1p(y_random_kspace_mag)
    
    # Create figure
    fig, axes = plt.subplots(4, 3, figsize=(15, 18))
    
    H, W = x_mag.shape
    
    # Row 0: Title and Ground Truth
    axes[0, 0].axis('off')
    axes[0, 0].text(0.5, 0.5, 
                    f"Sample: {fname}\nSlice: {slice_idx}\nResolution: {H}x{W}",
                    ha='center', va='center', fontsize=12, transform=axes[0, 0].transAxes)
    
    axes[0, 1].imshow(normalize_for_display(torch.from_numpy(x_mag)), cmap='gray')
    axes[0, 1].set_title('Ground Truth (x)', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[0, 2].axis('off')
    axes[0, 2].text(0.5, 0.5, 
                    f"Center Fraction: {center_fraction:.2%}\nAcceleration: {acceleration}x",
                    ha='center', va='center', fontsize=12, transform=axes[0, 2].transAxes)
    
    # Row 1: Masks
    axes[1, 0].set_title('Mask Comparison', fontsize=11)
    axes[1, 0].axis('off')
    
    if mask_center is not None:
        mask_center_1d = mask_center.squeeze().numpy()
        mask_center_2d = np.tile(mask_center_1d, (H, 1))
        axes[1, 1].imshow(mask_center_2d, cmap='gray', vmin=0, vmax=1, aspect='auto')
        n_sampled_center = mask_center_1d.sum()
        axes[1, 1].set_title(f'Center-Only Mask\n{int(n_sampled_center)}/{len(mask_center_1d)} lines ({100*n_sampled_center/len(mask_center_1d):.1f}%)', fontsize=10)
    else:
        axes[1, 1].text(0.5, 0.5, 'No mask', ha='center', va='center')
    axes[1, 1].axis('off')
    
    if mask_random is not None:
        mask_random_1d = mask_random.squeeze().numpy()
        mask_random_2d = np.tile(mask_random_1d, (H, 1))
        axes[1, 2].imshow(mask_random_2d, cmap='gray', vmin=0, vmax=1, aspect='auto')
        n_sampled_random = mask_random_1d.sum()
        axes[1, 2].set_title(f'Random Mask\n{int(n_sampled_random)}/{len(mask_random_1d)} lines ({100*n_sampled_random/len(mask_random_1d):.1f}%)', fontsize=10)
    else:
        axes[1, 2].text(0.5, 0.5, 'No mask', ha='center', va='center')
    axes[1, 2].axis('off')
    
    # Row 2: K-space (log magnitude)
    axes[2, 0].set_title('K-space (log mag)', fontsize=11)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(normalize_for_display(torch.from_numpy(y_center_kspace_log)), cmap='gray')
    axes[2, 1].set_title('Center-Only K-space', fontsize=10)
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(normalize_for_display(torch.from_numpy(y_random_kspace_log)), cmap='gray')
    axes[2, 2].set_title('Random K-space', fontsize=10)
    axes[2, 2].axis('off')
    
    # Row 3: Zero-filled reconstructions (y) with metrics
    axes[3, 0].set_title('Zero-filled Recon (y)', fontsize=11)
    axes[3, 0].axis('off')
    
    # Use same vmax for fair comparison
    vmax = max(np.percentile(y_center_mag, 99), np.percentile(y_random_mag, 99))
    
    axes[3, 1].imshow(y_center_mag, cmap='gray', vmin=0, vmax=vmax)
    axes[3, 1].set_title(f'Center-Only\nPSNR: {metrics_center["psnr"]:.2f} dB\nSSIM: {metrics_center["ssim"]:.4f}', fontsize=10)
    axes[3, 1].axis('off')
    
    axes[3, 2].imshow(y_random_mag, cmap='gray', vmin=0, vmax=vmax)
    axes[3, 2].set_title(f'Random\nPSNR: {metrics_random["psnr"]:.2f} dB\nSSIM: {metrics_random["ssim"]:.4f}', fontsize=10)
    axes[3, 2].axis('off')
    
    # Overall title
    fig.suptitle(f'Mask Comparison: Center-Only vs Random Sampling\nSample {idx}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()
    
    return metrics_center, metrics_random


def main():
    parser = argparse.ArgumentParser(
        description='Compare center-only vs random MRI undersampling masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data_path', type=str, 
                        default='/storage/omer/data/fastmri/singlecoil_train',
                        help='Path to the data directory')
    parser.add_argument('--resolution', type=int, default=320,
                        help='Target resolution')
    parser.add_argument('--center_fraction', type=float, default=0.04,
                        help='Fraction of center k-space to keep')
    parser.add_argument('--acceleration', type=int, default=4,
                        help='Acceleration factor for random mask')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None,
                        help='Specific sample indices to visualize')
    parser.add_argument('--save_dir', type=str, default='./mask_comparison_results',
                        help='Directory to save visualizations')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sample selection')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("MRI Mask Comparison: Center-Only vs Random Sampling")
    print("=" * 70)
    print(f"Data path: {args.data_path}")
    print(f"Resolution: {args.resolution}")
    print(f"Center fraction: {args.center_fraction:.2%}")
    print(f"Acceleration: {args.acceleration}x")
    print(f"Num samples: {args.num_samples}")
    print("=" * 70)
    
    # Create mask functions
    # 1. Center-only mask (no acceleration, just center fraction)
    mask_func_center = CenterOnlyMaskFunc([args.center_fraction])
    
    # 2. Random mask (center + random high-frequency lines)
    mask_func_random = RandomMaskFunc([args.center_fraction], [args.acceleration])
    
    # Create transforms
    transform_center = DataTransform(
        resolution=args.resolution,
        which_challenge='singlecoil',
        mask_func=mask_func_center,
        use_seed=True,
        scale_mode='none',
        include_dc_data=True,
    )
    
    transform_random = DataTransform(
        resolution=args.resolution,
        which_challenge='singlecoil',
        mask_func=mask_func_random,
        use_seed=True,
        scale_mode='none',
        include_dc_data=True,
    )
    
    # Create datasets
    print("\nLoading datasets...")
    dataset_center = SliceData(
        root=args.data_path,
        transform=transform_center,
        challenge='singlecoil',
        sequence=None,
        sample_rate=1.0,
    )
    
    dataset_random = SliceData(
        root=args.data_path,
        transform=transform_random,
        challenge='singlecoil',
        sequence=None,
        sample_rate=1.0,
    )
    
    print(f"Dataset size: {len(dataset_center)} slices")
    
    # Determine which samples to visualize
    if args.sample_indices is not None:
        indices = args.sample_indices
    else:
        np.random.seed(args.seed)
        indices = np.random.choice(len(dataset_center), 
                                   min(args.num_samples, len(dataset_center)), 
                                   replace=False)
    
    print(f"Visualizing samples: {list(indices)}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect metrics for summary
    all_metrics = []
    
    print("\nProcessing samples...")
    for i, idx in enumerate(indices):
        print(f"\n[{i+1}/{len(indices)}] Sample {idx}")
        
        sample_center = dataset_center[idx]
        sample_random = dataset_random[idx]
        
        save_path = save_dir / f"comparison_{idx:05d}.png"
        
        metrics_center, metrics_random = visualize_mask_comparison(
            sample_center,
            sample_random,
            idx=idx,
            center_fraction=args.center_fraction,
            acceleration=args.acceleration,
            save_path=str(save_path)
        )
        
        all_metrics.append({
            'idx': idx,
            'fname': sample_center['fname'],
            'slice': sample_center['slice'],
            'center_psnr': metrics_center['psnr'],
            'center_ssim': metrics_center['ssim'],
            'random_psnr': metrics_random['psnr'],
            'random_ssim': metrics_random['ssim'],
        })
        
        print(f"  Center-only: PSNR={metrics_center['psnr']:.2f} dB, SSIM={metrics_center['ssim']:.4f}")
        print(f"  Random:      PSNR={metrics_random['psnr']:.2f} dB, SSIM={metrics_random['ssim']:.4f}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    center_psnrs = [m['center_psnr'] for m in all_metrics]
    center_ssims = [m['center_ssim'] for m in all_metrics]
    random_psnrs = [m['random_psnr'] for m in all_metrics]
    random_ssims = [m['random_ssim'] for m in all_metrics]
    
    print(f"\nCenter-Only Mask (only {args.center_fraction:.1%} low-frequency):")
    print(f"  PSNR: {np.mean(center_psnrs):.2f} +/- {np.std(center_psnrs):.2f} dB")
    print(f"  SSIM: {np.mean(center_ssims):.4f} +/- {np.std(center_ssims):.4f}")
    
    print(f"\nRandom Mask ({args.center_fraction:.1%} center + random, {args.acceleration}x accel):")
    print(f"  PSNR: {np.mean(random_psnrs):.2f} +/- {np.std(random_psnrs):.2f} dB")
    print(f"  SSIM: {np.mean(random_ssims):.4f} +/- {np.std(random_ssims):.4f}")
    
    print(f"\nImprovement from adding high-frequency samples:")
    print(f"  PSNR: +{np.mean(random_psnrs) - np.mean(center_psnrs):.2f} dB")
    print(f"  SSIM: +{np.mean(random_ssims) - np.mean(center_ssims):.4f}")
    
    # Create summary plot
    create_summary_plot(all_metrics, args, save_dir)
    
    # Export metrics to CSV
    export_metrics_csv(all_metrics, args, save_dir)
    
    print(f"\nAll results saved to: {save_dir}")
    print("=" * 70)
    print("Done!")


def export_metrics_csv(all_metrics: list, args, save_dir: Path):
    """Export metrics to CSV file."""
    csv_path = save_dir / "metrics_comparison.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'sample_idx', 'fname', 'slice',
            'center_psnr_db', 'center_ssim',
            'random_psnr_db', 'random_ssim',
            'psnr_improvement_db', 'ssim_improvement'
        ])
        
        # Data rows
        for m in all_metrics:
            psnr_improvement = m['random_psnr'] - m['center_psnr']
            ssim_improvement = m['random_ssim'] - m['center_ssim']
            
            writer.writerow([
                m['idx'], m['fname'], m['slice'],
                f"{m['center_psnr']:.4f}", f"{m['center_ssim']:.6f}",
                f"{m['random_psnr']:.4f}", f"{m['random_ssim']:.6f}",
                f"{psnr_improvement:.4f}", f"{ssim_improvement:.6f}"
            ])
        
        # Summary row
        writer.writerow([])
        writer.writerow(['SUMMARY', '', '',
                         f"Mean: {np.mean([m['center_psnr'] for m in all_metrics]):.4f}",
                         f"Mean: {np.mean([m['center_ssim'] for m in all_metrics]):.6f}",
                         f"Mean: {np.mean([m['random_psnr'] for m in all_metrics]):.4f}",
                         f"Mean: {np.mean([m['random_ssim'] for m in all_metrics]):.6f}",
                         f"Mean: {np.mean([m['random_psnr'] - m['center_psnr'] for m in all_metrics]):.4f}",
                         f"Mean: {np.mean([m['random_ssim'] - m['center_ssim'] for m in all_metrics]):.6f}"])
        
        writer.writerow(['', '', '',
                         f"Std: {np.std([m['center_psnr'] for m in all_metrics]):.4f}",
                         f"Std: {np.std([m['center_ssim'] for m in all_metrics]):.6f}",
                         f"Std: {np.std([m['random_psnr'] for m in all_metrics]):.4f}",
                         f"Std: {np.std([m['random_ssim'] for m in all_metrics]):.6f}",
                         '', ''])
    
    print(f"\nMetrics exported to: {csv_path}")
    return csv_path


def create_summary_plot(all_metrics: list, args, save_dir: Path):
    """Create a summary bar plot comparing metrics across all samples."""
    n_samples = len(all_metrics)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    indices = np.arange(n_samples)
    width = 0.35
    
    # PSNR plot
    ax = axes[0]
    center_psnrs = [m['center_psnr'] for m in all_metrics]
    random_psnrs = [m['random_psnr'] for m in all_metrics]
    
    bars1 = ax.bar(indices - width/2, center_psnrs, width, label='Center-Only', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(indices + width/2, random_psnrs, width, label='Random', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('PSNR Comparison')
    ax.set_xticks(indices)
    ax.set_xticklabels([m['idx'] for m in all_metrics])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean lines
    ax.axhline(y=np.mean(center_psnrs), color='#27ae60', linestyle='--', linewidth=2, 
               label=f'Center Mean: {np.mean(center_psnrs):.1f}')
    ax.axhline(y=np.mean(random_psnrs), color='#2980b9', linestyle='--', linewidth=2,
               label=f'Random Mean: {np.mean(random_psnrs):.1f}')
    ax.legend()
    
    # SSIM plot
    ax = axes[1]
    center_ssims = [m['center_ssim'] for m in all_metrics]
    random_ssims = [m['random_ssim'] for m in all_metrics]
    
    bars1 = ax.bar(indices - width/2, center_ssims, width, label='Center-Only', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(indices + width/2, random_ssims, width, label='Random', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('SSIM')
    ax.set_title('SSIM Comparison')
    ax.set_xticks(indices)
    ax.set_xticklabels([m['idx'] for m in all_metrics])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add mean lines
    ax.axhline(y=np.mean(center_ssims), color='#27ae60', linestyle='--', linewidth=2,
               label=f'Center Mean: {np.mean(center_ssims):.3f}')
    ax.axhline(y=np.mean(random_ssims), color='#2980b9', linestyle='--', linewidth=2,
               label=f'Random Mean: {np.mean(random_ssims):.3f}')
    ax.legend()
    
    fig.suptitle(f'Metrics Summary: Center-Only ({args.center_fraction:.1%}) vs Random ({args.acceleration}x)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_path = save_dir / "metrics_summary.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSummary plot saved: {save_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()

