from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Union

import torch

try:  # Optional dependency; training scripts use W&B but keep import-time robust.
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None  # type: ignore

from torchvision.transforms.functional import to_pil_image

from utils.img_utils import create_grid


def normalize_for_visualization_by_scan_max(
    img: torch.Tensor,
    scan_max: Union[torch.Tensor, float],
    eps: float = 0,
) -> torch.Tensor:
    """
    Normalize images to [0, 1] for visualization by dividing by a GT scan-level max.

    Intended for fastMRI-style "scan == one volume / fname" normalization: every slice
    from the same scan is scaled by the same max(x) computed over the full GT volume.
    """
    img = img.to(torch.float32)

    if torch.is_tensor(scan_max):
        scale = scan_max.to(dtype=torch.float32, device=img.device)
    else:
        scale = torch.tensor(scan_max, dtype=torch.float32, device=img.device)

    if img.ndim == 4:
        if scale.ndim == 0:
            scale = scale.view(1, 1, 1, 1)
        elif scale.ndim == 1:
            scale = scale.view(-1, 1, 1, 1)
    elif img.ndim == 3:
        if scale.ndim != 0:
            scale = scale.reshape(-1)[0]
    else:
        raise ValueError(
            "normalize_for_visualization_by_scan_max expects (B, C, H, W) or (C, H, W), "
            f"got shape {tuple(img.shape)}"
        )

    img = img / torch.clamp(scale, min=eps)
    return torch.clamp(img, 0.0, 1.0)


def _is_global_zero(pl_module: Any) -> bool:
    return not hasattr(pl_module, "trainer") or bool(getattr(pl_module.trainer, "is_global_zero", True))


def get_wandb_logger(pl_module: Any) -> Optional[Any]:
    """
    Return the underlying W&B experiment object if available; otherwise None.
    """
    if wandb is None:
        return None
    if not _is_global_zero(pl_module):
        return None
    try:
        logger = pl_module.logger.experiment
    except Exception:
        return None
    if logger is None or not callable(getattr(logger, "log", None)):
        return None
    return logger


def _scale_tensor_from_fnames(
    fnames: Optional[List[str]],
    scan_max_by_fname: Mapping[str, float],
    fallback: float,
    device: torch.device,
) -> torch.Tensor:
    if not fnames:
        return torch.tensor(fallback, dtype=torch.float32, device=device)
    scales = [float(scan_max_by_fname.get(str(f), fallback)) for f in fnames]
    return torch.tensor(scales, dtype=torch.float32, device=device)


def log_image_triplet(
    wandb_logger: Any,
    prefix: str,
    x: torch.Tensor,
    y: torch.Tensor,
    xhat: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    x_vis = normalize_for_visualization_by_scan_max(x, scale)
    y_vis = normalize_for_visualization_by_scan_max(y, scale)
    xhat_vis = normalize_for_visualization_by_scan_max(xhat, scale)

    wandb_logger.log(
        {
            f"{prefix}/x": [wandb.Image(to_pil_image(create_grid(x_vis)))],
            f"{prefix}/y": [wandb.Image(to_pil_image(create_grid(y_vis)))],
            f"{prefix}/xhat": [wandb.Image(to_pil_image(create_grid(xhat_vis)))],
        }
    )


def log_train_epoch_images(
    pl_module: Any,
    vis: Optional[Dict[str, Any]],
    scan_max_by_fname: Mapping[str, float],
) -> None:
    if not vis:
        return
    wandb_logger = get_wandb_logger(pl_module)
    if wandb_logger is None:
        return

    x = vis["x"]
    y = vis["y"]
    xhat = vis["xhat"]
    fnames = vis.get("fnames", None)
    fallback = float(x.max().item())
    scale = _scale_tensor_from_fnames(fnames, scan_max_by_fname, fallback, device=x.device)
    log_image_triplet(wandb_logger, "train_images", x, y, xhat, scale)


def log_val_epoch_images(
    pl_module: Any,
    vis: Optional[Dict[str, Any]],
    scan_max_by_fname: Mapping[str, float],
    mmse_model: Optional[torch.nn.Module] = None,
) -> None:
    if not vis:
        return
    wandb_logger = get_wandb_logger(pl_module)
    if wandb_logger is None:
        return

    fnames_vis = vis["fnames"]
    fallback = float(vis["x"].max().item())
    scale_t = _scale_tensor_from_fnames(fnames_vis, scan_max_by_fname, fallback, device=vis["x"].device)

    log_image_triplet(wandb_logger, "val_images", vis["x"], vis["y"], vis["xhat"], scale_t)

    # Log Data Consistency (DC) corrected predictions if available
    if "xhat_dc" in vis:
        xhat_dc = vis["xhat_dc"]
        xhat_dc_vis = normalize_for_visualization_by_scan_max(xhat_dc, scale_t)
        wandb_logger.log(
            {
                "val_images/xhat_dc": [
                    wandb.Image(to_pil_image(create_grid(xhat_dc_vis)))
                ],
            }
        )

    # Optional trajectory visualization (flow-only).
    if "x_t_seq_first" in vis and "source_first" in vis:
        fname0 = vis.get("fname_first", fnames_vis[0] if fnames_vis else "_global")
        max0 = float(scan_max_by_fname.get(fname0, fallback))
        max0_t = torch.tensor(max0, dtype=torch.float32, device=vis["x"].device)

        x_t_seq_stack = torch.cat(vis["x_t_seq_first"], dim=0)  # (T, C, H, W)
        x_t_seq_vis = normalize_for_visualization_by_scan_max(x_t_seq_stack, max0_t)
        source_vis = normalize_for_visualization_by_scan_max(vis["source_first"], max0_t)

        wandb_logger.log(
            {
                "val_images/x_t_seq": [
                    wandb.Image(
                        to_pil_image(
                            create_grid(
                                x_t_seq_vis,
                                num_images=int(x_t_seq_vis.shape[0]),
                            )
                        )
                    )
                ],
                "val_images/source_distribution_samples": [
                    wandb.Image(to_pil_image(create_grid(source_vis)))
                ],
            }
        )

        if mmse_model is not None and "y_complex" in vis:
            with torch.no_grad():
                y_dev = vis["y_complex"].to(pl_module.device)
                xhat_mmse = mmse_model(y_dev).to(torch.float32).cpu()
                # mmse_model outputs complex (B,2,H,W), convert to magnitude (B,1,H,W)
                xhat_mmse = torch.sqrt(xhat_mmse[:, 0:1] ** 2 + xhat_mmse[:, 1:2] ** 2)
            xhat_mmse_vis = normalize_for_visualization_by_scan_max(xhat_mmse, scale_t)
            wandb_logger.log(
                {
                    "val_images/xhat_mmse": [
                        wandb.Image(to_pil_image(create_grid(xhat_mmse_vis)))
                    ]
                }
            )


