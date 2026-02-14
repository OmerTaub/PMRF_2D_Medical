from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pure-torch PSNR (replaces skimage for-loop → fully vectorised on GPU)
# ---------------------------------------------------------------------------

def compute_psnr_torch(
    gt: torch.Tensor,
    pred: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure-torch per-sample PSNR.  data_range = max(gt_sample).

    Args:
        gt: Ground truth tensor of shape (B, C, H, W).
        pred: Prediction tensor of shape (B, C, H, W).

    Returns:
        psnr_per_sample: Tensor of shape (B,) with PSNR values (on same device as gt).
        slice_max: Tensor of shape (B,) with max GT value per sample.
    """
    gt_f32 = gt.detach().to(torch.float32)
    pred_f32 = pred.detach().to(torch.float32)

    # Per-sample data range (max of GT)
    slice_max = gt_f32.flatten(1).max(dim=1).values          # (B,)

    # Per-sample MSE
    mse = (gt_f32 - pred_f32).pow(2).flatten(1).mean(dim=1)  # (B,)

    # PSNR = 10 * log10(data_range^2 / mse)
    valid = (slice_max > 0) & (mse > 0)
    psnr = torch.where(
        valid,
        10.0 * torch.log10(slice_max.pow(2) / mse.clamp(min=1e-10)),
        torch.zeros_like(mse),
    )
    return psnr, slice_max


def slice_sse_mse_psnr(
    xhat: torch.Tensor,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Compute per-sample SSE/MSE/PSNR for batched images.

    PSNR uses data_range = max(GT slice) — fully on-device (no CPU round-trip).
    """
    x_f32 = x.detach().to(torch.float32)
    xhat_f32 = xhat.detach().to(torch.float32)
    diff = xhat_f32 - x_f32
    flat = diff.flatten(1)
    count = int(flat.shape[1])
    sse = (flat ** 2).sum(dim=1)  # (B,)
    mse = sse / float(count)

    psnr, slice_max = compute_psnr_torch(x, xhat)

    return sse, mse, psnr, slice_max, count


# ---------------------------------------------------------------------------
# Pure-torch vectorised SSIM (replaces skimage for-loop → batched GPU conv)
# ---------------------------------------------------------------------------

def _gaussian_kernel_2d(
    size: int = 11,
    sigma: float = 1.5,
    channels: int = 1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a 2-D Gaussian kernel for depthwise convolution."""
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g1d = torch.exp(-coords.pow(2) / (2 * sigma * sigma))
    g1d = g1d / g1d.sum()
    g2d = g1d.unsqueeze(-1) @ g1d.unsqueeze(0)          # (size, size)
    return g2d.expand(channels, 1, size, size).contiguous()


def compute_ssim_per_sample(
    gt: torch.Tensor,
    pred: torch.Tensor,
    data_range_per_sample: torch.Tensor,
    kernel_size: int = 7,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """
    Vectorised per-sample SSIM entirely on the current device (GPU).

    Each sample uses its own ``data_range`` for the C1/C2 constants, matching
    the behaviour of the previous skimage implementation but running as a
    single batched convolution instead of a Python for-loop on CPU.

    Args:
        gt:   Ground truth  (B, C, H, W).
        pred: Prediction    (B, C, H, W).
        data_range_per_sample: Per-sample data range (B,).
        kernel_size: Size of the Gaussian window (default 7 — skimage default).
        sigma: Gaussian std  (default 1.5 — skimage default).
        k1, k2: SSIM stability constants.

    Returns:
        ssim_per_sample: (B,) tensor of mean SSIM per image.
    """
    gt_f32 = gt.detach().to(torch.float32)
    pred_f32 = pred.detach().to(torch.float32)
    dr = data_range_per_sample.detach().to(torch.float32)

    C = gt_f32.shape[1]
    pad = kernel_size // 2
    kernel = _gaussian_kernel_2d(kernel_size, sigma, C,
                                 device=gt.device, dtype=torch.float32)

    mu_x = F.conv2d(gt_f32, kernel, padding=pad, groups=C)
    mu_y = F.conv2d(pred_f32, kernel, padding=pad, groups=C)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(gt_f32.pow(2), kernel, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(pred_f32.pow(2), kernel, padding=pad, groups=C) - mu_y_sq
    sigma_xy = F.conv2d(gt_f32 * pred_f32, kernel, padding=pad, groups=C) - mu_xy

    # Per-sample C1, C2  →  (B, 1, 1, 1)
    dr_4d = dr.view(-1, 1, 1, 1)
    C1 = (k1 * dr_4d).pow(2)
    C2 = (k2 * dr_4d).pow(2)

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

    # Handle blank slices (data_range ≈ 0)
    blank = (dr < 1e-11).view(-1, 1, 1, 1).expand_as(ssim_map)
    ssim_map = ssim_map.masked_fill(blank, 0.0)

    # Mean over C, H, W → (B,)
    if torch.isnan(ssim_map).any():
        print(f"GT: {gt.shape}, Pred: {pred.shape}, Data Range: {dr.shape}")
        print(f"GT: {gt.min()}, {gt.max()}, Pred: {pred.min()}, {pred.max()}, Data Range: {dr.min()}, {dr.max()}")
        
    return ssim_map.flatten(1).mean(dim=1)


def get_fnames_list(batch: Any, batch_size: int) -> List[str]:
    """
    Return a list of scan identifiers (`fname`) of length `batch_size`.

    fastMRI-style datasets typically provide `batch["fname"]` as a list of strings,
    one per slice, where the same `fname` groups slices belonging to the same scan.
    """
    if not isinstance(batch, dict):
        return ["_global"] * batch_size
    fnames_raw = batch.get("fname", None)
    if fnames_raw is None:
        return ["_global"] * batch_size
    if isinstance(fnames_raw, (list, tuple)):
        return [f.name if hasattr(f, "name") else str(f) for f in fnames_raw]
    key = fnames_raw.name if hasattr(fnames_raw, "name") else str(fnames_raw)
    return [key] * batch_size


def _as_1d_float_tensor(
    value: Any,
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Best-effort conversion of an arbitrary batch field into a float32 tensor of shape (B,).
    """
    try:
        t = value if torch.is_tensor(value) else torch.as_tensor(value)
    except Exception:
        return None
    t = t.to(device=device, dtype=torch.float32)
    if t.ndim == 0:
        return t.repeat(batch_size)
    t = t.reshape(-1)
    if int(t.numel()) != int(batch_size):
        return None
    return t


def get_optional_scan_max(
    batch: Any,
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Optional per-sample scan-level GT max values provided by the dataloader.

    If absent, callers should fall back to computing max from the GT tensor.
    """
    if not isinstance(batch, dict):
        return None
    # NOTE: We intentionally do NOT treat generic fields like `max_value` as a GT scan max.
    # Some pipelines use `max_value` as an unnormalization factor, which would inflate PSNR
    # if x/y tensors are already normalized.
    for key in ("gt_scan_max", "scan_max"):
        if key in batch:
            t = _as_1d_float_tensor(batch[key], batch_size, device)
            if t is not None:
                return t
    return None


@dataclass
class ScanStatsAccumulator:
    """
    Streaming per-scan accumulators to compute epoch-end PSNR using GT scan max.

    Per-scan aggregation matches:
      1) Compute MSE per slice (mean over pixels), then
      2) Compute scan MSE = mean(MSE_slice over all slices in the scan), and
      3) Compute scan PSNR using data_range = max(GT over the entire scan).

    stats[fname] = {"mse_sum": float, "n_slices": int, "max": float}
    """

    stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def reset(self) -> None:
        self.stats.clear()

    def update(
        self,
        fnames_list: List[str],
        mse_per_sample: torch.Tensor,
        max_per_sample: torch.Tensor,
    ) -> None:
        mse_list = mse_per_sample.detach().cpu().tolist()
        max_list = max_per_sample.detach().cpu().tolist()
        for i, fname in enumerate(fnames_list):
            stats = self.stats.get(fname)
            if stats is None:
                self.stats[fname] = {
                    "mse_sum": float(mse_list[i]),
                    "n_slices": 1,
                    "max": float(max_list[i]),
                }
            else:
                stats["mse_sum"] += float(mse_list[i])
                stats["n_slices"] += 1
                stats["max"] = max(float(stats["max"]), float(max_list[i]))

    @staticmethod
    def merge(stats_list: List[Optional[Dict[str, Dict[str, float]]]]) -> Dict[str, Dict[str, float]]:
        merged: Dict[str, Dict[str, float]] = {}
        for stats in stats_list:
            if not stats:
                continue
            for fname, s in stats.items():
                mse_sum = float(s.get("mse_sum", 0.0))
                n_slices = int(s.get("n_slices", 0))
                mx = float(s.get("max", 0.0))
                if fname not in merged:
                    merged[fname] = {"mse_sum": mse_sum, "n_slices": n_slices, "max": mx}
                else:
                    merged[fname]["mse_sum"] += mse_sum
                    merged[fname]["n_slices"] += n_slices
                    merged[fname]["max"] = max(merged[fname]["max"], mx)
        return merged

    def gathered(self) -> Dict[str, Dict[str, float]]:
        """
        Gather per-scan stats across DDP ranks and merge (correct per-scan MSE/max).
        Single-process short-circuit avoids expensive pickle round-trip.
        """
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size <= 1:
                return self.stats
            gathered: List[Optional[Dict[str, Dict[str, float]]]] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, self.stats)
            return self.merge(gathered)
        return self.stats


@dataclass
class ScanMeanAccumulator:
    """
    Streaming per-scan accumulator for scalar slice-level metrics.

    Intended for metrics like SSIM:
      - compute metric per slice -> mean over slices in a scan -> mean over scans.

    stats[fname] = {"sum": float, "n_slices": int}
    """

    stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def reset(self) -> None:
        self.stats.clear()

    def update(self, fnames_list: List[str], value_per_sample: torch.Tensor) -> None:
        values = value_per_sample.detach().cpu().tolist()
        for i, fname in enumerate(fnames_list):
            stats = self.stats.get(fname)
            if stats is None:
                self.stats[fname] = {"sum": float(values[i]), "n_slices": 1}
            else:
                stats["sum"] += float(values[i])
                stats["n_slices"] += 1

    @staticmethod
    def merge(stats_list: List[Optional[Dict[str, Dict[str, float]]]]) -> Dict[str, Dict[str, float]]:
        merged: Dict[str, Dict[str, float]] = {}
        for stats in stats_list:
            if not stats:
                continue
            for fname, s in stats.items():
                sm = float(s.get("sum", 0.0))
                n = int(s.get("n_slices", 0))
                if fname not in merged:
                    merged[fname] = {"sum": sm, "n_slices": n}
                else:
                    merged[fname]["sum"] += sm
                    merged[fname]["n_slices"] += n
        return merged

    def gathered(self) -> Dict[str, Dict[str, float]]:
        """
        Gather per-scan stats across DDP ranks and merge (correct per-scan means).
        Single-process short-circuit avoids expensive pickle round-trip.
        """
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size <= 1:
                return self.stats
            gathered: List[Optional[Dict[str, Dict[str, float]]]] = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, self.stats)
            return self.merge(gathered)
        return self.stats


def summarize_scan_psnr_mse(
    merged_stats: Dict[str, Dict[str, float]],
) -> Tuple[float, float, Dict[str, float]]:
    """
    Return (avg_psnr_per_scan, avg_mse_per_scan, scan_max_by_fname).

    PSNR is computed as:
        psnr = 10 * log10(data_range^2 / mse)

    For each scan:
      - data_range = max(GT over all slices in scan)
      - mse = mean(MSE per slice) over all slices in scan
    """
    scan_psnr_values: List[float] = []
    scan_mse_values: List[float] = []
    scan_max_by_fname = {k: float(v.get("max", 0.0)) for k, v in merged_stats.items()}

    for _fname, s in merged_stats.items():
        data_range = max(float(s.get("max", 0.0)), 1e-8)
        n_slices = max(int(s.get("n_slices", 0)), 1)
        mse = float(s.get("mse_sum", 0.0)) / float(n_slices)
        mse = max(mse, 1e-10)
        psnr = 10.0 * np.log10((data_range ** 2) / mse)
        scan_psnr_values.append(psnr)
        scan_mse_values.append(mse)

    avg_scan_psnr = sum(scan_psnr_values) / len(scan_psnr_values) if scan_psnr_values else 0.0
    avg_scan_mse = sum(scan_mse_values) / len(scan_mse_values) if scan_mse_values else 0.0
    return avg_scan_psnr, avg_scan_mse, scan_max_by_fname


def summarize_scan_mean(
    merged_stats: Dict[str, Dict[str, float]],
) -> Tuple[float, Dict[str, float]]:
    """
    Return (avg_metric_per_scan, metric_by_fname) where:
      - metric_per_scan(fname) = mean(metric(slice) for slices in scan)
      - avg_metric_per_scan = mean(metric_per_scan(fname) for all scans)
    """
    scan_values: List[float] = []
    metric_by_fname: Dict[str, float] = {}
    for fname, s in merged_stats.items():
        n = max(int(s.get("n_slices", 0)), 1)
        val = float(s.get("sum", 0.0)) / float(n)
        metric_by_fname[fname] = val
        scan_values.append(val)
    avg = sum(scan_values) / len(scan_values) if scan_values else 0.0
    return avg, metric_by_fname


@dataclass
class ScanSSIMImageAccumulator:
    """
    Stores GT and pred magnitude slices per scan on CPU for epoch-end SSIM
    computation with per-scan data_range = max(GT_scan).

    Unlike ``ScanMeanAccumulator`` (which averages per-slice SSIM values
    computed with per-slice data_range), this accumulator defers the SSIM
    computation until all slices are collected so that the correct per-scan
    max can be used as data_range for the C1/C2 constants.

    Memory note: images are stored as float32 on CPU.  For a typical
    validation set (100 scans × 30 slices × 1×320×320) this is ~1.2 GB
    for GT+pred combined, which is acceptable.
    """

    gt_by_scan: Dict[str, List[torch.Tensor]] = field(default_factory=dict)
    pred_by_scan: Dict[str, List[torch.Tensor]] = field(default_factory=dict)

    def reset(self) -> None:
        self.gt_by_scan.clear()
        self.pred_by_scan.clear()

    def update(
        self,
        fnames_list: List[str],
        gt: torch.Tensor,
        pred: torch.Tensor,
    ) -> None:
        """Store per-slice GT and pred magnitude images on CPU."""
        gt_cpu = gt.detach().cpu().to(torch.float32)
        pred_cpu = pred.detach().cpu().to(torch.float32)
        for i, fname in enumerate(fnames_list):
            if fname not in self.gt_by_scan:
                self.gt_by_scan[fname] = []
                self.pred_by_scan[fname] = []
            self.gt_by_scan[fname].append(gt_cpu[i : i + 1])
            self.pred_by_scan[fname].append(pred_cpu[i : i + 1])

    def gathered(self) -> "ScanSSIMImageAccumulator":
        """Gather per-scan image lists across DDP ranks and merge by fname.

        Single-process short-circuit avoids pickle-serialising thousands of
        image tensors (which can be multiple GB and take minutes).
        """
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            if world_size <= 1:
                return self
            all_gt: List[Optional[Dict]] = [None] * world_size
            all_pred: List[Optional[Dict]] = [None] * world_size
            dist.all_gather_object(all_gt, self.gt_by_scan)
            dist.all_gather_object(all_pred, self.pred_by_scan)
            merged = ScanSSIMImageAccumulator()
            for gt_dict, pred_dict in zip(all_gt, all_pred):
                if gt_dict is None:
                    continue
                for fname in gt_dict:
                    if fname not in merged.gt_by_scan:
                        merged.gt_by_scan[fname] = []
                        merged.pred_by_scan[fname] = []
                    merged.gt_by_scan[fname].extend(gt_dict[fname])
                    merged.pred_by_scan[fname].extend(pred_dict[fname])
            return merged
        return self

    def compute_per_scan_ssim(
        self,
        device: Optional[torch.device] = None,
        num_workers: int = 0,
        chunk_size: int = 256,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute per-scan SSIM with data_range = max(GT_scan).

        When *device* is provided (e.g. a CUDA device), all slices across
        every scan are concatenated into a single tensor and processed in
        large batched GPU calls (chunked to *chunk_size* to bound VRAM).
        This is vastly faster than a one-scan-at-a-time loop because it
        amortises kernel-launch and CPU↔GPU transfer overhead.

        When *num_workers* > 0 **and** *device* is ``None`` (CPU mode), a
        :class:`concurrent.futures.ThreadPoolExecutor` is used so that
        multiple scans are computed in parallel across CPU threads (PyTorch
        releases the GIL during tensor ops, so threads scale well here).

        Args:
            device: Device for SSIM convolutions (e.g. ``torch.device("cuda:0")``).
                When ``None`` the computation stays on CPU.
            num_workers: Number of CPU threads for parallel per-scan SSIM.
                Only used when ``device is None``.  0 → sequential / batched.
            chunk_size: Max slices per GPU batch (default 256).

        Returns:
            (avg_ssim_across_scans, ssim_by_fname)
        """
        if not self.gt_by_scan:
            return 0.0, {}

        # ---- CPU multi-threaded path (for CPU-only environments) ----
        if num_workers > 0 and device is None:
            return self._compute_ssim_threaded(num_workers)

        # ---- Batched path (GPU or single-thread CPU) ----
        # Concatenate all slices from every scan into one tensor so we can
        # compute SSIM for all of them in a few large, efficient GPU calls
        # instead of hundreds of tiny sequential ones.
        fnames: List[str] = []
        scan_boundaries: List[Tuple[int, int]] = []   # (start, end) per scan
        all_gt_parts: List[torch.Tensor] = []
        all_pred_parts: List[torch.Tensor] = []
        all_dr_values: List[float] = []

        idx = 0
        for fname in self.gt_by_scan:
            gt_stack = torch.cat(self.gt_by_scan[fname], dim=0)   # (N_slices, C, H, W)
            pred_stack = torch.cat(self.pred_by_scan[fname], dim=0)
            n = gt_stack.shape[0]
            scan_max = max(float(gt_stack.max().item()), 1e-8)

            fnames.append(fname)
            scan_boundaries.append((idx, idx + n))
            all_gt_parts.append(gt_stack)
            all_pred_parts.append(pred_stack)
            all_dr_values.extend([scan_max] * n)
            idx += n

        all_gt = torch.cat(all_gt_parts, dim=0)        # (T, C, H, W)
        all_pred = torch.cat(all_pred_parts, dim=0)
        all_dr = torch.tensor(all_dr_values, dtype=torch.float32)

        if device is not None:
            all_gt = all_gt.to(device)
            all_pred = all_pred.to(device)
            all_dr = all_dr.to(device)

        # Compute SSIM in chunks to bound peak memory
        total = all_gt.shape[0]
        ssim_parts: List[torch.Tensor] = []
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            ssim_chunk = compute_ssim_per_sample(
                all_gt[start:end],
                all_pred[start:end],
                all_dr[start:end],
            )
            ssim_parts.append(ssim_chunk.detach().cpu())

        all_ssim = torch.cat(ssim_parts, dim=0)  # (T,)

        # Aggregate per scan
        ssim_by_fname: Dict[str, float] = {}
        for i, fname in enumerate(fnames):
            s, e = scan_boundaries[i]
            ssim_by_fname[fname] = float(all_ssim[s:e].mean().item())

        avg = sum(ssim_by_fname.values()) / len(ssim_by_fname)
        return avg, ssim_by_fname

    def _compute_ssim_threaded(
        self,
        num_workers: int,
    ) -> Tuple[float, Dict[str, float]]:
        """CPU multi-threaded per-scan SSIM (GIL released by PyTorch C++ ops)."""
        from concurrent.futures import ThreadPoolExecutor

        def _one_scan(fname: str) -> Tuple[str, float]:
            gt_stack = torch.cat(self.gt_by_scan[fname], dim=0)
            pred_stack = torch.cat(self.pred_by_scan[fname], dim=0)
            scan_max = max(float(gt_stack.max().item()), 1e-8)
            dr = torch.full((gt_stack.shape[0],), scan_max)
            ssim_vals = compute_ssim_per_sample(gt_stack, pred_stack, dr)
            return fname, float(ssim_vals.mean().item())

        ssim_by_fname: Dict[str, float] = {}
        fnames = list(self.gt_by_scan.keys())
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for fname, val in pool.map(_one_scan, fnames):
                ssim_by_fname[fname] = val

        avg = sum(ssim_by_fname.values()) / len(ssim_by_fname) if ssim_by_fname else 0.0
        return avg, ssim_by_fname


def all_gather_list(local_list: List[Any]) -> List[Any]:
    """
    Gather a Python list across DDP ranks and concatenate.
    Single-process short-circuit avoids expensive pickle round-trip.
    """
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        if world_size <= 1:
            return local_list
        gathered: List[Any] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_list)
        merged: List[Any] = []
        for g in gathered:
            if g:
                merged.extend(g)
        return merged
    return local_list
