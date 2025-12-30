#!/usr/bin/env python3
"""
Compare visualization for ComplexDataTransform vs DataTransform samples.

ComplexDataTransform expected sample dict keys:
  - "x": target (complex, 2xHxW, real/imag) [normalized]
  - "y": zero-filled (complex, 2xHxW, real/imag) [normalized]
  - "mean": scalar (expected 0.0)
  - "std": scalar (computed as mean(|y_orig|) per slice)

DataTransform expected sample dict keys:
  - "x": target magnitude (1xHxW) [normalized]
  - "y": zero-filled magnitude (1xHxW) [normalized]

This script:
  - Loads fastMRI SliceDataset twice (same ordering), once per transform
  - For selected indices, plots:
      * y_norm: Complex |y_norm| vs DataTransform y vs diff
      * x_norm: Complex |x_norm| vs DataTransform x vs diff
  - Prints sanity checks (e.g., mean(y_norm) ≈ 1).
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence, Any, Dict, Tuple

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


# -----------------------------
# Helpers: complex ops (2xHxW)
# -----------------------------
def complex_abs_2ch(z: torch.Tensor) -> torch.Tensor:
    """z: (2,H,W) or (B,2,H,W) -> magnitude: (H,W) or (B,H,W)"""
    if z.ndim == 3 and z.shape[0] == 2:
        return torch.sqrt(z[0] ** 2 + z[1] ** 2)
    if z.ndim == 4 and z.shape[1] == 2:
        return torch.sqrt(z[:, 0] ** 2 + z[:, 1] ** 2)
    raise ValueError(f"Expected (2,H,W) or (B,2,H,W), got {tuple(z.shape)}")


def complex_phase_2ch(z: torch.Tensor) -> torch.Tensor:
    """z: (2,H,W) or (B,2,H,W) -> phase (atan2(imag, real))"""
    if z.ndim == 3 and z.shape[0] == 2:
        return torch.atan2(z[1], z[0])
    if z.ndim == 4 and z.shape[1] == 2:
        return torch.atan2(z[:, 1], z[:, 0])
    raise ValueError(f"Expected (2,H,W) or (B,2,H,W), got {tuple(z.shape)}")


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def robust_vmax(imgs: Sequence[np.ndarray], p: float = 99.5) -> float:
    vals = np.concatenate([im.reshape(-1) for im in imgs], axis=0)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    return float(np.percentile(vals, p))


# -----------------------------
# Mask function (optional)
# -----------------------------
def build_mask_func(
    mask_type: str,
    center_fractions: Sequence[float],
    accelerations: Sequence[int],
):
    """
    Tries to build a fastMRI-style mask func.

    Supports:
      - random  -> RandomMaskFunc
      - cartesian -> EquispacedMaskFunc
    """
    mask_type = mask_type.lower()
    # Try local project first, then fastmri.
    tried = []

    # Local
    try:
        from data.subsample import RandomMaskFunc, EquispacedMaskFunc  # type: ignore
        if mask_type == "random":
            return RandomMaskFunc(center_fractions, accelerations)
        if mask_type in ("cartesian", "equispaced"):
            return EquispacedMaskFunc(center_fractions, accelerations)
        raise ValueError(f"Unknown mask_type={mask_type}")
    except Exception as e:
        tried.append(f"data.subsample: {e}")

    # fastmri
    try:
        from fastmri.data.subsample import RandomMaskFunc, EquispacedMaskFunc  # type: ignore
        if mask_type == "random":
            return RandomMaskFunc(center_fractions, accelerations)
        if mask_type in ("cartesian", "equispaced"):
            return EquispacedMaskFunc(center_fractions, accelerations)
        raise ValueError(f"Unknown mask_type={mask_type}")
    except Exception as e:
        tried.append(f"fastmri.data.subsample: {e}")

    raise RuntimeError(
        "Could not import a MaskFunc implementation. Tried:\n  - " + "\n  - ".join(tried)
    )


# -----------------------------
# Load dataset + transform
# -----------------------------
def import_slice_dataset():
    # Prefer your project dataset (as in your inspiration)
    try:
        from PMRF.torch_datasets.fastmri_dataset import SliceDataset  # type: ignore
        return SliceDataset
    except Exception:
        pass

    # Fallback: fastmri official
    try:
        from fastmri.data.mri_data import SliceDataset  # type: ignore
        return SliceDataset
    except Exception as e:
        raise RuntimeError(
            "Could not import SliceDataset from either PMRF.torch_datasets.fastmri_dataset "
            "or fastmri.data.mri_data."
        ) from e


def import_transform():
    """
    Import your ComplexDataTransform and DataTransform from wherever they live.
    Adjust the import below if needed.
    """
    tried = []

    # Try common local locations
    for mod in [
        "data.mri_data",
        "data",
        "data.transforms",
        "PMRF.data.transforms",
        "PMRF.torch_datasets.transforms",
    ]:
        try:
            m = __import__(mod, fromlist=["ComplexDataTransform", "DataTransform"])
            ComplexDataTransform = getattr(m, "ComplexDataTransform")
            DataTransform = getattr(m, "DataTransform")
            return ComplexDataTransform, DataTransform
        except Exception as e:
            tried.append(f"{mod}: {e}")

    raise RuntimeError(
        "Could not import ComplexDataTransform/DataTransform. Adjust import_transform() to match your codebase.\n"
        "Tried:\n  - " + "\n  - ".join(tried)
    )


# -----------------------------
# Visualization
# -----------------------------
def _unwrap_sample(sample: Any) -> Any:
    # Some dataset wrappers return (sample, fname, slice) etc.
    if isinstance(sample, (tuple, list)) and len(sample) in (2, 3) and isinstance(sample[0], dict):
        return sample[0]
    return sample


def _as_2d_mag(sample: Dict[str, Any], key: str) -> np.ndarray:
    """
    Convert a sample[key] to a 2D numpy magnitude image.
    - ComplexDataTransform: (2,H,W) -> |.| -> (H,W)
    - DataTransform: (1,H,W) -> squeeze -> (H,W)
    """
    t = sample[key]
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)

    if t.ndim == 3 and t.shape[0] == 2:
        return to_numpy(complex_abs_2ch(t))
    if t.ndim == 3 and t.shape[0] == 1:
        return to_numpy(t.squeeze(0))
    if t.ndim == 2:
        return to_numpy(t)
    raise ValueError(f"Unsupported tensor shape for sample['{key}']: {tuple(t.shape)}")


def _center_crop_np(img: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Center-crop a 2D numpy array to (H, W)."""
    if img.ndim != 2:
        raise ValueError(f"_center_crop_np expects 2D array, got shape {img.shape}")
    h, w = img.shape
    out_h, out_w = out_hw
    if out_h > h or out_w > w:
        raise ValueError(f"Requested crop {(out_h, out_w)} larger than input {(h, w)}")
    top = (h - out_h) // 2
    left = (w - out_w) // 2
    return img[top : top + out_h, left : left + out_w]


def _match_crop(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Center-crop two 2D arrays to their common overlap shape (min(H), min(W)).
    Returns (a_cropped, b_cropped).
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"_match_crop expects 2D arrays, got {a.shape} and {b.shape}")
    out_h = min(a.shape[0], b.shape[0])
    out_w = min(a.shape[1], b.shape[1])
    if a.shape != (out_h, out_w):
        a = _center_crop_np(a, (out_h, out_w))
    if b.shape != (out_h, out_w):
        b = _center_crop_np(b, (out_h, out_w))
    return a, b


def visualize_comparison(
    sample_complex: Dict[str, Any],
    sample_data: Dict[str, Any],
    idx: int,
    save_path: Optional[Path] = None,
):
    if plt is None:
        raise RuntimeError("matplotlib is not available in this environment.")

    y_c = _as_2d_mag(sample_complex, "y")
    x_c = _as_2d_mag(sample_complex, "x")
    y_d = _as_2d_mag(sample_data, "y")
    x_d = _as_2d_mag(sample_data, "x")

    # DataTransform may return x and y at different spatial sizes (e.g., x from file recon
    # at 320x320 while y from IFFT at 640x372). For fair visualization and diffs, crop
    # each comparison pair to a shared overlap shape.
    y_c, y_d = _match_crop(y_c, y_d)
    x_c, x_d = _match_crop(x_c, x_d)

    vmax = robust_vmax([y_c, y_d, x_c, x_d], p=99.5)
    vmax = max(vmax, 1e-8)

    y_diff = y_c - y_d
    x_diff = x_c - x_d
    diff_v = robust_vmax([np.abs(y_diff), np.abs(x_diff)], p=99.5)
    diff_v = max(diff_v, 1e-8)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    # y comparison
    axes[0, 0].imshow(y_c, cmap="gray", vmin=0.0, vmax=vmax)
    axes[0, 0].set_title("y (Complex |y_norm|)")
    axes[0, 1].imshow(y_d, cmap="gray", vmin=0.0, vmax=vmax)
    axes[0, 1].set_title("y (DataTransform)")
    axes[0, 2].imshow(y_diff, cmap="gray", vmin=-diff_v, vmax=diff_v)
    axes[0, 2].set_title("y diff (Complex - Data)")

    # x comparison
    axes[1, 0].imshow(x_c, cmap="gray", vmin=0.0, vmax=vmax)
    axes[1, 0].set_title("x (Complex |x_norm|)")
    axes[1, 1].imshow(x_d, cmap="gray", vmin=0.0, vmax=vmax)
    axes[1, 1].set_title("x (DataTransform)")
    axes[1, 2].imshow(x_diff, cmap="gray", vmin=-diff_v, vmax=diff_v)
    axes[1, 2].set_title("x diff (Complex - Data)")

    for ax in axes.reshape(-1):
        ax.axis("off")

    std = sample_complex.get("std", torch.tensor(float("nan")))
    if not torch.is_tensor(std):
        std = torch.tensor(float(std))
    mean = sample_complex.get("mean", torch.tensor(float("nan")))
    if not torch.is_tensor(mean):
        mean = torch.tensor(float(mean))

    fig.suptitle(
        f"idx={idx} | Complex std={float(std):.6g} mean={float(mean):.6g} | "
        f"y shape {tuple(y_c.shape)} x shape {tuple(x_c.shape)} | "
        f"mean(y_c)={float(np.mean(y_c)):.4f} mean(y_d)={float(np.mean(y_d)):.4f}",
        fontsize=12,
    )
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def print_sanity_complex(sample: Dict[str, Any], idx: int):
    x = sample["x"]
    y = sample["y"]
    std = sample.get("std", torch.tensor(1.0))
    if not torch.is_tensor(std):
        std = torch.tensor(float(std))

    y_norm_mag_mean = float(complex_abs_2ch(y).mean())
    x_norm_mag_mean = float(complex_abs_2ch(x).mean())

    print(f"[idx={idx}] ComplexDataTransform: x shape={tuple(x.shape)} y shape={tuple(y.shape)} std={float(std):.6g}")
    print(f"         mean(|y_norm|)={y_norm_mag_mean:.6f}  (expected ~1.0)")
    print(f"         mean(|x_norm|)={x_norm_mag_mean:.6f}")
    y_dn_mag_mean = float((complex_abs_2ch(y) * std).mean())
    print(f"         mean(|y_dn|)={y_dn_mag_mean:.6f}  (≈ std)")


def print_sanity_data(sample: Dict[str, Any], idx: int):
    x = sample["x"]
    y = sample["y"]
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    x2 = x.squeeze(0) if x.ndim == 3 and x.shape[0] == 1 else x
    y2 = y.squeeze(0) if y.ndim == 3 and y.shape[0] == 1 else y
    print(f"[idx={idx}] DataTransform: x shape={tuple(x.shape)} y shape={tuple(y.shape)}")
    print(f"         mean(y)={float(y2.mean()):.6f}  mean(x)={float(x2.mean()):.6f}")


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Compare ComplexDataTransform vs DataTransform (x/y normalized)")

    p.add_argument("--data_root", type=str, required=False, default="/storage/omer/data/fastmri/singlecoil_train", help="Directory with fastMRI .h5 files.")
    p.add_argument("--challenge", type=str, default="singlecoil", choices=["singlecoil", "multicoil"])
    p.add_argument("--resolution", type=int, default=None, help="Optional center crop resolution.")
    p.add_argument("--num_samples", type=int, default=20, help="How many indices to inspect.")
    p.add_argument("--start", type=int, default=0, help="Start index.")
    p.add_argument("--stride", type=int, default=1, help="Stride between inspected indices.")

    p.add_argument("--use_dataset_cache", action="store_true")
    p.add_argument("--dataset_cache_file", type=str, default="dataset_cache.pkl")

    # Masking (optional)
    p.add_argument("--mask_type", type=str, default=None, choices=["random", "cartesian", "equispaced"])
    p.add_argument("--center_fraction", type=float, default=0.04)
    p.add_argument("--accel", type=int, default=4)

    # DataTransform optional scaling
    p.add_argument(
        "--scale_mode",
        type=str,
        default="none",
        choices=[
            "none",
            "subsample_max",
            "subsample_percentile",
            "volume_subsample_max",
            "volume_subsample_percentile",
        ],
        help="Extra DataTransform scaling (applies to both x and y after normalization).",
    )
    p.add_argument(
        "--scale_percentile",
        type=float,
        default=100.0,
        help='Percentile (0-100] for scale_mode="*_percentile".',
    )

    # Output
    p.add_argument("--save_dir", type=str, default="/storage/omer/projects/PMRF_2D_Medical/check_complex_dataset", help="If set, save PNGs here instead of showing.")
    return p.parse_args()


def main():
    args = parse_args()

    # Ensure project root on path (common pattern)
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    SliceDataset = import_slice_dataset()
    ComplexDataTransform, DataTransform = import_transform()

    # DataTransform in this codebase expects a mask_func (otherwise masked_kspace is undefined).
    # To keep comparison simple, if --mask_type is not provided we default to "random".
    mask_type = args.mask_type or "random"
    mask_func = build_mask_func(
        mask_type=mask_type,
        center_fractions=[args.center_fraction],
        accelerations=[args.accel],
    )

    transform_complex = ComplexDataTransform(
        resolution=args.resolution,
        which_challenge=args.challenge,
        mask_func=mask_func,
        use_seed=True,
    )

    transform_data = DataTransform(
        resolution=args.resolution,
        which_challenge=args.challenge,
        mask_func=mask_func,
        use_seed=True,
        scale_mode=args.scale_mode,
        scale_percentile=args.scale_percentile,
    )

    dataset_complex = SliceDataset(
        root=args.data_root,
        challenge=args.challenge,
        transform=transform_complex,
        use_dataset_cache=args.use_dataset_cache,
        dataset_cache_file=args.dataset_cache_file,
    )

    dataset_data = SliceDataset(
        root=args.data_root,
        challenge=args.challenge,
        transform=transform_data,
        use_dataset_cache=args.use_dataset_cache,
        dataset_cache_file=args.dataset_cache_file,
    )

    if len(dataset_complex) != len(dataset_data):
        raise RuntimeError(
            f"Dataset length mismatch: complex={len(dataset_complex)} vs data={len(dataset_data)}"
        )

    print(
        f"Dataset: {type(dataset_complex).__name__} | n={len(dataset_complex)} | "
        f"mask_type={mask_type} cf={args.center_fraction} accel={args.accel} | "
        f"scale_mode={args.scale_mode} p={args.scale_percentile}"
    )
    save_dir = Path(args.save_dir)

    shown = 0
    i = args.start
    while shown < args.num_samples and i < len(dataset_complex):
        sample_c = _unwrap_sample(dataset_complex[i])
        sample_d = _unwrap_sample(dataset_data[i])

        if not isinstance(sample_c, dict) or "x" not in sample_c or "y" not in sample_c:
            print(f"[idx={i}] Unexpected ComplexDataTransform sample type/structure: {type(sample_c)}")
            i += args.stride
            shown += 1
            continue

        if not isinstance(sample_d, dict) or "x" not in sample_d or "y" not in sample_d:
            print(f"[idx={i}] Unexpected DataTransform sample type/structure: {type(sample_d)}")
            i += args.stride
            shown += 1
            continue

        print_sanity_complex(sample_c, i)
        print_sanity_data(sample_d, i)

        out_path = None
        if save_dir is not None:
            out_path = save_dir / f"compare_{i:06d}.png"

        visualize_comparison(sample_c, sample_d, i, save_path=out_path)

        i += args.stride
        shown += 1

    print("Done.")


if __name__ == "__main__":
    main()
