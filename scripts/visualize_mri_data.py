#!/usr/bin/env python
"""
Visualization script to validate MRI data pipeline.

This script loads slices using SliceData and DataTransform, then visualizes:
- The undersampling mask
- K-space magnitude (before and after masking)
- Zero-filled reconstruction (y)
- Fully-sampled target (x)
- IFFT of k-space

All visualizations are at the same resolution to validate the pipeline.
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from data.mri_data import SliceData, DataTransform
from data.subsample import create_mask_for_mask_type
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


def visualize_sample(sample: dict, title_prefix: str = "", save_path: str = None):
    """
    Visualize a single sample from the data pipeline.
    
    Args:
        sample: Dictionary returned by DataTransform
        title_prefix: Prefix for the plot title
        save_path: If provided, save the figure to this path
    """
    # Extract data from sample
    x = sample['x']  # (2, H, W) - fully sampled target
    y = sample['y']  # (2, H, W) - zero-filled reconstruction
    fname = sample['fname']
    slice_idx = sample['slice']
    norm_std = sample.get('norm_std', torch.tensor(1.0))
    norm_scale = sample.get('norm_scale', torch.tensor(1.0))
    
    # DC data (if available)
    kspace = sample.get('kspace', None)  # (H, W, 2) - k-space at target resolution
    mask = sample.get('mask', None)  # (1, W, 1) - undersampling mask
    
    # Convert to magnitude images
    x_mag = complex_to_magnitude(x)  # (H, W)
    y_mag = complex_to_magnitude(y)  # (H, W)
    
    # Get resolution
    H, W = x_mag.shape
    
    print(f"\n{'='*60}")
    print(f"Sample: {fname} - Slice {slice_idx}")
    print(f"{'='*60}")
    print(f"Resolution: {H} x {W}")
    print(f"x (target) shape: {x.shape}, dtype: {x.dtype}")
    print(f"y (input) shape: {y.shape}, dtype: {y.dtype}")
    print(f"norm_std: {norm_std.item():.6f}")
    print(f"norm_scale: {norm_scale.item():.6f}")
    
    # Determine number of subplots based on available data
    n_plots = 4  # x_mag, y_mag, y_kspace_mag, ifft_y
    if kspace is not None:
        n_plots += 2  # original kspace, masked kspace
    if mask is not None:
        n_plots += 1  # mask
    
    # Create figure
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_plots == 1 else axes.flatten()
    
    plot_idx = 0
    
    # 1. Fully sampled target (x) - magnitude
    ax = axes[plot_idx]
    ax.imshow(normalize_for_display(x_mag), cmap='gray')
    ax.set_title(f'Target (x)\nshape: {x_mag.shape}')
    ax.axis('off')
    plot_idx += 1
    
    # 2. Zero-filled reconstruction (y) - magnitude
    ax = axes[plot_idx]
    ax.imshow(normalize_for_display(y_mag), cmap='gray')
    ax.set_title(f'Zero-filled (y)\nshape: {y_mag.shape}')
    ax.axis('off')
    plot_idx += 1
    
    # 3. K-space of y (FFT of y) - log magnitude
    y_complex = y.permute(1, 2, 0)  # (H, W, 2)
    y_kspace = transforms.fft2(y_complex)  # (H, W, 2)
    y_kspace_mag = complex_to_magnitude(y_kspace)
    y_kspace_log = torch.log1p(y_kspace_mag)
    
    ax = axes[plot_idx]
    ax.imshow(normalize_for_display(y_kspace_log), cmap='gray')
    ax.set_title(f'K-space of y (log mag)\nshape: {y_kspace.shape}')
    ax.axis('off')
    plot_idx += 1
    
    # 4. IFFT of y's k-space (should match y)
    y_reconstructed = transforms.ifft2(y_kspace)  # (H, W, 2)
    y_recon_mag = complex_to_magnitude(y_reconstructed)
    
    ax = axes[plot_idx]
    ax.imshow(normalize_for_display(y_recon_mag), cmap='gray')
    ax.set_title(f'IFFT of y kspace\nshape: {y_recon_mag.shape}')
    ax.axis('off')
    plot_idx += 1
    
    # 5. Original k-space (if available)
    if kspace is not None:
        print(f"kspace shape: {kspace.shape}, dtype: {kspace.dtype}")
        kspace_mag = complex_to_magnitude(kspace)
        kspace_log = torch.log1p(kspace_mag)
        
        ax = axes[plot_idx]
        ax.imshow(normalize_for_display(kspace_log), cmap='gray')
        ax.set_title(f'Original K-space (log mag)\nshape: {kspace.shape}')
        ax.axis('off')
        plot_idx += 1
        
        # 6. IFFT of original kspace
        kspace_ifft = transforms.ifft2(kspace)
        kspace_ifft_mag = complex_to_magnitude(kspace_ifft)
        
        ax = axes[plot_idx]
        ax.imshow(normalize_for_display(kspace_ifft_mag), cmap='gray')
        ax.set_title(f'IFFT of original kspace\nshape: {kspace_ifft_mag.shape}')
        ax.axis('off')
        plot_idx += 1
    
    # 7. Mask (if available)
    if mask is not None:
        print(f"mask shape: {mask.shape}, dtype: {mask.dtype}")
        # Expand mask for visualization (1, W, 1) -> (H, W)
        mask_expanded = mask.squeeze()  # (W,)
        if mask_expanded.ndim == 1:
            mask_2d = mask_expanded.unsqueeze(0).expand(H, -1)  # (H, W)
        else:
            mask_2d = mask_expanded
        
        ax = axes[plot_idx]
        ax.imshow(mask_2d.numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Undersampling Mask\nshape: {mask.shape} -> {mask_2d.shape}')
        ax.axis('off')
        plot_idx += 1
        
        # Print mask statistics
        mask_flat = mask.flatten()
        n_sampled = mask_flat.sum().item()
        n_total = mask_flat.numel()
        accel = n_total / n_sampled if n_sampled > 0 else float('inf')
        print(f"Mask: {n_sampled}/{n_total} lines sampled ({100*n_sampled/n_total:.1f}%)")
        print(f"Effective acceleration: {accel:.2f}x")
    
    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    # Add overall title
    fig.suptitle(f'{title_prefix}{fname} - Slice {slice_idx}\nResolution: {H}x{W}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()
    
    # Validation checks
    print(f"\n--- Validation Checks ---")
    print(f"x and y same resolution: {x.shape == y.shape}")
    if kspace is not None:
        print(f"kspace resolution matches: {kspace.shape[0]} x {kspace.shape[1]} == {H} x {W}: {kspace.shape[0] == H and kspace.shape[1] == W}")
    if mask is not None:
        mask_w = mask.shape[1] if mask.ndim >= 2 else mask.shape[0]
        print(f"mask width matches image: {mask_w} == {W}: {mask_w == W}")
    
    # Check if IFFT roundtrip is correct
    diff = torch.abs(y_recon_mag - y_mag).max().item()
    print(f"IFFT(FFT(y)) roundtrip error (max): {diff:.2e}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize MRI data pipeline')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the data directory (e.g., singlecoil_val)')
    parser.add_argument('--resolution', type=int, default=320,
                        help='Target resolution (default: 320)')
    parser.add_argument('--mask_type', type=str, default='random',
                        choices=['random', 'equispaced'],
                        help='Type of undersampling mask')
    parser.add_argument('--center_fraction', type=float, default=0.04,
                        help='Fraction of center k-space to keep')
    parser.add_argument('--acceleration', type=int, default=4,
                        help='Acceleration factor')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to visualize')
    parser.add_argument('--sample_indices', type=int, nargs='+', default=None,
                        help='Specific sample indices to visualize')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save visualizations')
    parser.add_argument('--no_dc_data', action='store_true',
                        help='Disable DC data (kspace, mask) in transform')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MRI Data Pipeline Visualization")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Resolution: {args.resolution}")
    print(f"Mask type: {args.mask_type}")
    print(f"Center fraction: {args.center_fraction}")
    print(f"Acceleration: {args.acceleration}x")
    print(f"Include DC data: {not args.no_dc_data}")
    
    # Create mask function
    mask_func = create_mask_for_mask_type(
        args.mask_type,
        [args.center_fraction],
        [args.acceleration]
    )
    
    # Create data transform
    transform = DataTransform(
        resolution=args.resolution,
        which_challenge='singlecoil',
        mask_func=mask_func,
        use_seed=True,  # Reproducible masks
        scale_mode='none',
        scale_percentile=100.0,
        include_dc_data=not args.no_dc_data,
    )
    
    # Create dataset
    dataset = SliceData(
        root=args.data_path,
        transform=transform,
        challenge='singlecoil',
        sequence=None,  # Not used but required by constructor
        sample_rate=1.0,
    )
    
    print(f"\nDataset size: {len(dataset)} slices")
    
    # Determine which samples to visualize
    if args.sample_indices is not None:
        indices = args.sample_indices
    else:
        # Random samples
        np.random.seed(42)
        indices = np.random.choice(len(dataset), min(args.num_samples, len(dataset)), replace=False)
    
    print(f"Visualizing samples: {indices}")
    
    # Create save directory if needed
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Visualize each sample
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        save_path = None
        if args.save_dir:
            save_path = os.path.join(args.save_dir, f"sample_{idx:05d}.png")
        
        visualize_sample(
            sample,
            title_prefix=f"[{i+1}/{len(indices)}] ",
            save_path=save_path
        )
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

