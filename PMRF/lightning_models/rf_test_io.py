from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

import torch
from torchvision.utils import save_image

from .rf_metrics import slice_sse_mse_psnr


METRICS_FIELDNAMES = [
    "id",
    "fname",
    "slice",
    "img_file_name",
    "variant",
    "num_flow_steps",
    "mse",
    "psnr",
    "gt_slice_max",
]


def get_slice_list(batch: Any, batch_size: int) -> List[int]:
    if not isinstance(batch, dict):
        return list(range(batch_size))
    slices_raw = batch.get("slice", None)
    if slices_raw is None:
        return list(range(batch_size))
    if torch.is_tensor(slices_raw):
        return [int(v) for v in slices_raw.detach().cpu().tolist()]
    if isinstance(slices_raw, (list, tuple)):
        return [int(v) for v in slices_raw]
    try:
        return [int(slices_raw)] * batch_size
    except Exception:
        return list(range(batch_size))


def resolve_img_file_names(
    batch: Any,
    batch_size: int,
    fnames_list: List[str],
    slice_list: List[int],
) -> List[str]:
    """
    Prefer dataset-provided `img_file_name` (ImageFolderDataset).
    Fall back to `{fname}_slice{slice}.png` (fastMRI-style).
    """
    if not isinstance(batch, dict):
        return [f"sample_{i}.png" for i in range(batch_size)]
    img_file_names = batch.get("img_file_name", None)
    if isinstance(img_file_names, str):
        return [img_file_names] * batch_size
    if img_file_names is None:
        return [f"{fnames_list[i]}_slice{slice_list[i]}.png" for i in range(batch_size)]
    return [str(n) for n in img_file_names]


def save_image_batch(images: torch.Tensor, folder: str, image_file_names: List[str]) -> None:
    os.makedirs(folder, exist_ok=True)
    for i, _name in enumerate(image_file_names):
        save_image(images[i].clip(0, 1), os.path.join(folder, image_file_names[i]))


def append_per_slice_metrics(
    records: List[Dict[str, Any]],
    *,
    xhat_cpu: torch.Tensor,
    x_cpu: torch.Tensor,
    img_file_names: List[str],
    fnames_list: List[str],
    slice_list: List[int],
    variant: str,
    num_flow_steps: Optional[int] = None,
) -> None:
    """
    Record per-slice PSNR/MSE where:
      - data_range = max(GT slice)
      - mse        = MSE(pred slice, GT slice)
    """
    _sse, mse, psnr, gt_slice_max, _count = slice_sse_mse_psnr(xhat_cpu, x_cpu)
    mse_list = mse.detach().cpu().tolist()
    psnr_list = psnr.detach().cpu().tolist()
    max_list = gt_slice_max.detach().cpu().tolist()

    bs = len(img_file_names)
    for i in range(bs):
        records.append(
            {
                "id": str(img_file_names[i]),
                "fname": str(fnames_list[i]) if fnames_list else "",
                "slice": int(slice_list[i]) if slice_list else "",
                "img_file_name": str(img_file_names[i]),
                "variant": str(variant),
                "num_flow_steps": int(num_flow_steps) if num_flow_steps is not None else "",
                "mse": float(mse_list[i]),
                "psnr": float(psnr_list[i]),
                "gt_slice_max": float(max_list[i]),
            }
        )


def write_metrics_csv(records: List[Dict[str, Any]], out_dir: str, file_name: str = "metrics_per_slice.csv") -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, file_name)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_FIELDNAMES)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    return out_path


