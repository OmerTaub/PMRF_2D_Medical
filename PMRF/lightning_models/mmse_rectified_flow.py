import os
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Optional

import torch
from huggingface_hub import PyTorchModelHubMixin
from pytorch_lightning import LightningModule
from torch.nn.functional import mse_loss, sigmoid
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_ema import ExponentialMovingAverage as EMA
from piq import LPIPS, DISTS

from utils.create_arch import create_arch
from arch.gauge_net import GaugeNet

from .rf_metrics import (
    ScanMeanAccumulator,
    ScanSSIMImageAccumulator,
    ScanStatsAccumulator,
    all_gather_list,
    compute_ssim_per_sample,
    get_fnames_list,
    get_optional_scan_max,
    slice_sse_mse_psnr,
    summarize_scan_mean,
    summarize_scan_psnr_mse,
)
from .rf_test_io import (
    append_per_slice_metrics,
    get_slice_list,
    resolve_img_file_names,
    save_image_batch,
    write_metrics_csv,
)
from .rf_vis import get_wandb_logger, log_train_epoch_images, log_val_epoch_images

# Import data consistency function
import sys
from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
from data.transforms import apply_data_consistency
from arch.reconformer_wrapper import ReconFormer as _ReconFormerCls


def _is_reconformer(model) -> bool:
    """Check whether *model* is a ReconFormer instance."""
    return isinstance(model, _ReconFormerCls)


def _extract_rf_dc(batch: Dict, device: torch.device):
    """
    Extract ReconFormer-compatible DC data from the batch.

    Returns ``(masked_kspace_norm, reconformer_mask)`` or ``(None, None)``
    if the required fields are not present.

    * ``masked_kspace_norm``: ``(B, 2, H, W)`` — normalized masked k-space.
    * ``reconformer_mask``:   ``(B, 1, H, W)`` — binary mask.
    """
    mk = batch.get("masked_kspace_norm")
    rf_mask = batch.get("reconformer_mask")
    if mk is None or rf_mask is None:
        return None, None
    return mk.to(device), rf_mask.to(device)


def complex_to_magnitude(x: torch.Tensor) -> torch.Tensor:
    """
    Convert complex 2-channel tensor to magnitude 1-channel tensor.
    
    Args:
        x: Complex tensor of shape (B, 2, H, W) where [0]=real, [1]=imag
        
    Returns:
        Magnitude tensor of shape (B, 1, H, W)
    """
    return torch.sqrt(x[:, 0:1, :, :] ** 2 + x[:, 1:2, :, :] ** 2)


def has_dc_data(batch: Dict) -> bool:
    """
    Check if the batch contains the necessary data for data consistency.
    
    Required fields: 'kspace', 'mask', 'norm_std', 'norm_scale'
    """
    required_keys = ['kspace', 'mask', 'norm_std', 'norm_scale']
    return all(k in batch for k in required_keys)


def get_dc_data(batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Extract data consistency fields from batch and move to device.
    
    Returns:
        dict with 'kspace', 'mask', 'norm_std', 'norm_scale', 'resolution'
    """
    batch_size = batch['x'].shape[0]
    
    # Get kspace - shape (B, H, W, 2)
    kspace = batch['kspace']
    if kspace.ndim == 3:
        # Single sample: (H, W, 2) -> (1, H, W, 2)
        kspace = kspace.unsqueeze(0)
    kspace = kspace.to(device)
    
    # Get mask - typically (B, 1, W, 1) for column-wise undersampling
    mask = batch['mask']
    if mask.ndim == 3:
        # (1, W, 1) -> (B, 1, W, 1) by repeating for batch
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
    elif mask.ndim == 4 and mask.shape[0] == 1:
        # (1, 1, W, 1) -> (B, 1, W, 1)
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
    
    # Get resolution from original_shape or infer from x
    if 'original_shape' in batch:
        original_shape = batch['original_shape']
        if isinstance(original_shape, (list, tuple)):
            # Already a tuple
            pass
        elif torch.is_tensor(original_shape):
            original_shape = tuple(original_shape.tolist())
    else:
        # Infer from kspace shape
        original_shape = (kspace.shape[1], kspace.shape[2])
    
    # Get crop resolution from x
    resolution = batch['x'].shape[-1]  # Assume square crops
    
    return {
        'kspace': kspace,
        'mask': mask,
        'norm_std': norm_std,
        'norm_scale': norm_scale,
        'resolution': resolution,
        'original_shape': original_shape,
    }


def get_unnorm_factor(batch: Dict, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Per-sample unnormalization factor: norm_std * norm_scale.

    Reverses the DataTransform normalization so that:
        raw_value = normalized_value * unnorm_factor

    Returns:
        Tensor of shape (B,) on *device*.
    """
    # --- norm_std ---
    norm_std_raw = batch.get('norm_std', None)
    if norm_std_raw is None:
        norm_std = torch.ones(batch_size, device=device, dtype=torch.float32)
    elif torch.is_tensor(norm_std_raw):
        norm_std = norm_std_raw.to(device=device, dtype=torch.float32)
    else:
        norm_std = torch.tensor(norm_std_raw, device=device, dtype=torch.float32)
    if norm_std.ndim == 0:
        norm_std = norm_std.expand(batch_size)
    norm_std = norm_std.reshape(-1)

    # --- norm_scale ---
    norm_scale_raw = batch.get('norm_scale', None)
    if norm_scale_raw is None:
        norm_scale = torch.ones(batch_size, device=device, dtype=torch.float32)
    elif torch.is_tensor(norm_scale_raw):
        norm_scale = norm_scale_raw.to(device=device, dtype=torch.float32)
    else:
        norm_scale = torch.tensor(norm_scale_raw, device=device, dtype=torch.float32)
    if norm_scale.ndim == 0:
        norm_scale = norm_scale.expand(batch_size)
    norm_scale = norm_scale.reshape(-1)

    return norm_std * norm_scale  # (B,)


########################################################################
# ENGRF gauge utilities
########################################################################

import math

def gauge_alpha(t: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
    """Smooth bump function alpha(t) = strength * sin^2(pi * t).
    
    Satisfies alpha(0) = alpha(1) = 0 (endpoint neutrality).
    """
    return strength * torch.sin(math.pi * t).pow(2)


def gauge_alpha_prime(t: torch.Tensor, strength: float = 0.1) -> torch.Tensor:
    """Derivative: alpha'(t) = strength * pi * sin(2*pi*t)."""
    return strength * math.pi * torch.sin(2.0 * math.pi * t)


def gauge_jvp_fd(
    gauge_net: torch.nn.Module,
    z: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Approximate Jacobian-vector product J_W(z;y) * v via central finite differences.
    
    J_W * v ≈ [W(z + eps*v, y) - W(z - eps*v, y)] / (2*eps)
    
    This requires 2 extra forward passes through gauge_net (lightweight ~0.5M params).
    """
    inp_plus = torch.cat([z + eps * v, y], dim=1)
    inp_minus = torch.cat([z - eps * v, y], dim=1)
    w_plus = gauge_net(inp_plus)
    w_minus = gauge_net(inp_minus)
    return (w_plus - w_minus) / (2.0 * eps)


class MMSERectifiedFlow(LightningModule,
                        PyTorchModelHubMixin,
                        pipeline_tag="image-to-image",
                        license="mit",
                        ):
    def __init__(self,
                 stage,
                 arch,
                 conditional=True,
                 mmse_model_ckpt_path=None,
                 mmse_model_arch=None,
                 lr=1e-4,
                 weight_decay=0,
                 betas=(0.9, 0.95),
                 mmse_noise_std=0.1,
                 mmse_noise_std_max=None,
                 num_flow_steps=50,
                 ema_decay=0.9999,
                 eps=0.0,
                 t_schedule='stratified_uniform',
                 apply_dc_to_source=False,
                 # --- ENGRF gauge_flow params ---
                 gauge_strength=0.1,
                 gauge_base_channels=32,
                 gauge_num_levels=3,
                 use_gauge_jvp=True,
                 freeze_flow_model=False,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        print(f"hparams.conditional: {self.hparams.conditional}")
        if stage == 'flow' or stage == 'gauge_flow':
            if conditional:
                condition_channels = 2  # Match complex (real+imag) output of MMSE model
            else:
                condition_channels = 0
            if mmse_model_arch is None and 'colorization' in kwargs and kwargs['colorization']:
                condition_channels //= 3
            self.model = create_arch(arch, condition_channels)
            self.mmse_model = create_arch(mmse_model_arch, 0) if mmse_model_arch is not None else None
            if mmse_model_ckpt_path is not None:
                ckpt = torch.load(mmse_model_ckpt_path, map_location="cpu")
                if mmse_model_arch is None:
                    mmse_model_arch = ckpt['hyper_parameters']['arch']
                self.mmse_model = create_arch(mmse_model_arch, 0)
                if 'ema' in ckpt:
                    # ema_decay doesn't affect anything here, because we are doing load_state_dict
                    mmse_ema = EMA(self.mmse_model.parameters(), decay=ema_decay)
                    mmse_ema.load_state_dict(ckpt['ema'])
                    mmse_ema.copy_to()
                elif 'params_ema' in ckpt:
                    self.mmse_model.load_state_dict(ckpt['params_ema'])
                else:
                    state_dict = ckpt['state_dict']
                    state_dict = {layer_name.replace('model.', ''): weights for layer_name, weights in
                                  state_dict.items()}
                    state_dict = {layer_name.replace('module.', ''): weights for layer_name, weights in
                                  state_dict.items()}
                    self.mmse_model.load_state_dict(state_dict)
                for param in self.mmse_model.parameters():
                    param.requires_grad = False
                self.mmse_model.eval()

            # --- ENGRF: create gauge network ---
            if stage == 'gauge_flow':
                self.gauge_net = GaugeNet(
                    in_channels=4,   # concat(Z_t, y): 2 + 2
                    out_channels=2,  # displacement in real+imag
                    base_channels=gauge_base_channels,
                    num_levels=gauge_num_levels,
                )
                # Optionally freeze the velocity network (train gauge only)
                if freeze_flow_model:
                    for param in self.model.parameters():
                        param.requires_grad = False
            else:
                self.gauge_net = None
        else:
            assert stage == 'mmse' or stage == 'naive_flow'
            assert not conditional
            self.model = create_arch(arch, 0)
            self.mmse_model = None
            self.gauge_net = None
        if 'flow' in stage:
            # PIQ full-reference metrics - keep on CPU to save GPU memory.
            # These are stored as private attributes to prevent LightningModule
            # from auto-moving them to GPU.
            # LPIPS/DISTS are kept as private attrs (underscore prefix) so
            # Lightning won't auto-move them.  We lazily move them to the
            # model's device in validation_step for fast GPU inference.
            self._piq_lpips = LPIPS(replace_pooling=True, reduction='none').eval()
            self._piq_dists = DISTS(reduction='none').eval()
            for param in self._piq_lpips.parameters():
                param.requires_grad = False
            for param in self._piq_dists.parameters():
                param.requires_grad = False
            # DC versions for LPIPS/DISTS
            self.val_lpips_stats_dc = ScanMeanAccumulator()
            self.val_dists_stats_dc = ScanMeanAccumulator()
            # Non-DC versions for LPIPS/DISTS
            self.val_lpips_stats = ScanMeanAccumulator()
            self.val_dists_stats = ScanMeanAccumulator()

        # Per-scan stats for PSNR using GT scan max as data range.
        #
        # All metrics are computed on UNNORMALIZED magnitude images:
        #   raw_mag = complex_to_magnitude(x) * norm_std * norm_scale
        # This ensures per-scan PSNR (data_range = max(GT_volume)) is
        # consistent across slices that have different normalization factors.
        self.val_scan_stats = ScanStatsAccumulator()
        # Baseline per-scan stats for y vs x (naive PSNR-per-scan).
        self.train_scan_stats_y = ScanStatsAccumulator()
        self.val_scan_stats_y = ScanStatsAccumulator()

        # Per-scan SSIM: stores GT/pred images per scan on CPU, computes SSIM
        # at epoch end with data_range = max(GT_scan).
        self.val_scan_ssim_stats = ScanSSIMImageAccumulator()
        # Baseline per-scan SSIM stats for y vs x (naive SSIM-per-scan).
        self.train_scan_ssim_stats_y = ScanMeanAccumulator()
        self.val_scan_ssim_stats_y = ScanMeanAccumulator()

        # Data Consistency (DC) specific accumulators for validation metrics
        # These track metrics computed AFTER applying data consistency
        self.val_scan_stats_dc = ScanStatsAccumulator()
        self.val_scan_ssim_stats_dc = ScanSSIMImageAccumulator()

        # Cached batch to log once we know the scan max values.
        self._train_vis_data = None
        self._val_vis_data = None
        # Test/inference per-slice metric records (collected in `test_step`).
        self._test_slice_metrics = []

        self.ema = EMA(self.model.parameters(), decay=ema_decay) if self.ema_wanted else None
        # Separate EMA for gauge_net (if present)
        self.gauge_ema = (
            EMA(self.gauge_net.parameters(), decay=ema_decay)
            if self.ema_wanted and self.gauge_net is not None
            else None
        )
        self.test_results_path = None
        # Temporary per-step storage for ReconFormer DC data (masked_kspace, mask).
        self._rf_dc_data = (None, None)
        # Indices of batches used for visualization; will be randomized each epoch. # TODO OMER ADDED 2 LINES
        self.train_sample_batch_idx = 0
        self.val_sample_batch_idx = 0

    @property
    def ema_wanted(self):
        return self.hparams.ema_decay != -1

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_wanted:
            checkpoint['ema'] = self.ema.state_dict()
        if self.gauge_ema is not None:
            checkpoint['gauge_ema'] = self.gauge_ema.state_dict()
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_wanted:
            self.ema.load_state_dict(checkpoint['ema'])
        if self.gauge_ema is not None and 'gauge_ema' in checkpoint:
            self.gauge_ema.load_state_dict(checkpoint['gauge_ema'])
        # Strip keys from the saved state_dict that no longer exist in the
        # current model (e.g. removed _piq_inception_v3 / piq_fid / piq_pr).
        if "state_dict" in checkpoint:
            current_keys = set(self.state_dict().keys())
            stale = [k for k in checkpoint["state_dict"] if k not in current_keys]
            if stale:
                print(f"[on_load_checkpoint] Dropping {len(stale)} stale key(s) from checkpoint state_dict")
                for k in stale:
                    del checkpoint["state_dict"][k]
        return super().on_load_checkpoint(checkpoint)

    def on_before_zero_grad(self, optimizer) -> None:
        if self.ema_wanted:
            self.ema.update(self.model.parameters())
        if self.gauge_ema is not None:
            self.gauge_ema.update(self.gauge_net.parameters())
        return super().on_before_zero_grad(optimizer)

    def to(self, *args, **kwargs):
        if self.ema_wanted:
            self.ema.to(*args, **kwargs)
        if self.gauge_ema is not None:
            self.gauge_ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def on_train_epoch_start(self) -> None: # TODO OMER ADDED THIS FUNCTION
        """
        Randomize which training batch will be visualized this epoch.
        """
        # Reset per-epoch scan stats + cached vis batch.
        self.train_scan_stats_y.reset()
        self.train_scan_ssim_stats_y.reset()
        self._train_vis_data = None

        num_batches = getattr(self.trainer, "num_training_batches", None)
        train_sample_batch_idx = 0
        return super().on_train_epoch_start()

    def on_validation_epoch_start(self) -> None:
        """
        Randomize which validation batch will be visualized this epoch.
        """
        # Reset per-epoch scan stats + cached vis batch.
        self.val_scan_stats.reset()
        self.val_scan_stats_y.reset()
        self.val_scan_ssim_stats.reset()
        self.val_scan_ssim_stats_y.reset()
        # Reset DC accumulators
        self.val_scan_stats_dc.reset()
        self.val_scan_ssim_stats_dc.reset()
        self._val_vis_data = None

        # Reset PIQ accumulators (flow stage only)
        if 'flow' in self.hparams.stage:
            self.val_lpips_stats_dc.reset()
            self.val_dists_stats_dc.reset()
            self.val_lpips_stats.reset()
            self.val_dists_stats.reset()

        num_val_batches = getattr(self.trainer, "num_val_batches", None)
        # num_val_batches can be a list/tuple when multiple val dataloaders are used.
        if isinstance(num_val_batches, (list, tuple)):
            num_batches = int(num_val_batches[0]) if len(num_val_batches) > 0 else 0
        else:
            num_batches = int(num_val_batches) if num_val_batches is not None else 0

        try:
            if num_batches > 0:
                self.val_sample_batch_idx = int(torch.randint(num_batches, (1,), device=self.device).item())
            else:
                self.val_sample_batch_idx = 0
        except TypeError:
            self.val_sample_batch_idx = 0
        return super().on_validation_epoch_start()

    def on_train_epoch_end(self) -> None:
        """
        Log naive baselines per-scan + one training visualization batch.
        """
        merged_stats_y = self.train_scan_stats_y.gathered()
        naive_scan_psnr, _naive_scan_mse, _scan_max_by_fname_y = summarize_scan_psnr_mse(merged_stats_y)

        merged_ssim_y = self.train_scan_ssim_stats_y.gathered()
        naive_scan_ssim, _naive_ssim_by_fname = summarize_scan_mean(merged_ssim_y)

        if not hasattr(self, "trainer") or getattr(self.trainer, "is_global_zero", True):
            self.log(
                "train_metrics/naive_psnr_per_scan",
                naive_scan_psnr,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                sync_dist=False,
                batch_size=1,
            )
            self.log(
                "train_metrics/naive_ssim_per_scan",
                naive_scan_ssim,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                sync_dist=False,
                batch_size=1,
            )

        # Pass {} so the vis code falls back to per-batch max for [0,1] normalization.
        log_train_epoch_images(self, getattr(self, "_train_vis_data", None), {})

        # Clear cached data for next epoch.
        self._train_vis_data = None
        self.train_scan_stats_y.reset()
        self.train_scan_ssim_stats_y.reset()
        return super().on_train_epoch_end()

    # This will use the contextmanager of ema, to copy the EMA weights to the flow model during validation, and then restore them for training.
    @contextmanager
    def maybe_ema(self):
        if self.ema is None:
            yield
        else:
            with self.ema.average_parameters():
                if self.gauge_ema is not None:
                    with self.gauge_ema.average_parameters():
                        yield
                else:
                    yield

    def forward_mmse(self, y):
        if _is_reconformer(self.model):
            # ReconFormer: direct prediction with internal data consistency.
            k0, mask_rf = getattr(self, '_rf_dc_data', (None, None)) or (None, None)
            return self.model(y, k0=k0, mask=mask_rf)
        return y + self.model(y)  # Residual learning: model predicts correction (x - y)

    def forward_flow(self, x_t, t, y=None):
        if self.hparams.conditional:
            if self.mmse_model is not None:
                with torch.no_grad():
                    self.mmse_model.eval()
                    if _is_reconformer(self.mmse_model):
                        k0, mask_rf = getattr(self, '_rf_dc_data', (None, None)) or (None, None)
                        condition = self.mmse_model(y, k0=k0, mask=mask_rf)
                    else:
                        condition = self.mmse_model(y)
            else:
                condition = y
            x_t = torch.cat((x_t, condition), dim=1)
        return self.model(x_t, t)

    def forward(self, x_t, t, y):
        if 'flow' in self.hparams.stage:
            return self.forward_flow(x_t, t, y)
        else:
            return self.forward_mmse(y)

    def _sample_noise_std(self) -> float:
        """Sample noise std, uniformly from [min, max] if max is provided, else return min."""
        if self.hparams.mmse_noise_std_max is not None:
            return torch.empty(1).uniform_(
                self.hparams.mmse_noise_std,
                self.hparams.mmse_noise_std_max
            ).item()
        return self.hparams.mmse_noise_std

    @torch.no_grad()
    def create_source_distribution_samples(self, x, y, non_noisy_z0, dc_data=None):
        """
        Create source distribution samples for flow training.
        
        Args:
            x: Ground truth tensor (B, C, H, W)
            y: Zero-filled / undersampled input (B, C, H, W)
            non_noisy_z0: Pre-computed MMSE output if available (B, C, H, W)
            dc_data: Optional dict with data consistency fields:
                     {'kspace', 'mask', 'norm_std', 'norm_scale', 'resolution'}
                     If provided and apply_dc_to_source is True, DC is applied to the
                     MMSE output before adding noise.
        
        Returns:
            source_dist_samples: Tensor (B, C, H, W) to be used as the source of the flow.
        """
        with torch.no_grad():
            if self.hparams.conditional:
                source_dist_samples = torch.randn_like(x)
            else:
                # Sample noise std (uniform from range if max is set, else fixed value)
                noise_std = self._sample_noise_std()
                if self.hparams.stage in ('flow', 'gauge_flow'):
                    if non_noisy_z0 is None:
                        self.mmse_model.eval()
                        if _is_reconformer(self.mmse_model):
                            k0, mask_rf = getattr(self, '_rf_dc_data', (None, None)) or (None, None)
                            non_noisy_z0 = self.mmse_model(y, k0=k0, mask=mask_rf)
                        else:
                            non_noisy_z0 = self.mmse_model(y)
                    
                    # Apply data consistency to MMSE output if enabled and DC data available
                    if self.hparams.apply_dc_to_source and dc_data is not None:
                        non_noisy_z0 = apply_data_consistency(
                            xhat=non_noisy_z0,
                            kspace=dc_data['kspace'],
                            mask=dc_data['mask'],
                            norm_std=dc_data['norm_std'],
                            norm_scale=dc_data['norm_scale'],
                            resolution=dc_data['resolution'],
                        )
                    
                    source_dist_samples = non_noisy_z0 + torch.randn_like(non_noisy_z0) * noise_std
                else:
                    assert self.hparams.stage == 'naive_flow'
                    if non_noisy_z0 is not None:
                        source_dist_samples = non_noisy_z0
                    else:
                        source_dist_samples = y
                    if source_dist_samples.shape[1] != x.shape[1]:
                        assert source_dist_samples.shape[1] == 1  # Colorization
                        source_dist_samples = source_dist_samples.expand(-1, x.shape[1], -1, -1)
                    if self.hparams.mmse_noise_std is not None:
                        source_dist_samples = source_dist_samples + torch.randn_like(source_dist_samples) * noise_std
        return source_dist_samples

    @staticmethod
    def stratified_uniform(bs, group=0, groups=1, dtype=None, device=None):
        if groups <= 0:
            raise ValueError(f"groups must be positive, got {groups}")
        if group < 0 or group >= groups:
            raise ValueError(f"group must be in [0, {groups})")
        n = bs * groups
        offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
        u = torch.rand(bs, dtype=dtype, device=device)
        return ((offsets + u) / n).view(bs, 1, 1, 1)

    def generate_random_t(self, bs, dtype=None):
        if self.hparams.t_schedule == 'logit-normal':
            return sigmoid(torch.randn(bs, 1, 1, 1, device=self.device)) * (1.0 - self.hparams.eps) + self.hparams.eps
        elif self.hparams.t_schedule == 'uniform':
            return torch.rand(bs, 1, 1, 1, device=self.device) * (1.0 - self.hparams.eps) + self.hparams.eps
        elif self.hparams.t_schedule == 'stratified_uniform':
            return self.stratified_uniform(bs, self.trainer.global_rank, self.trainer.world_size, dtype=dtype,
                                           device=self.device) * (1.0 - self.hparams.eps) + self.hparams.eps
        else:
            raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y']
        non_noisy_z0 = batch['non_noisy_z0'] if 'non_noisy_z0' in batch else None

        # ReconFormer DC data: set on self so forward methods can access it.
        self._rf_dc_data = _extract_rf_dc(batch, self.device)

        # `fname` groups slices belonging to the same scan/volume.
        fnames_list = get_fnames_list(batch, x.shape[0])

        # Extract DC data if needed for apply_dc_to_source
        dc_data = None
        if self.hparams.apply_dc_to_source and has_dc_data(batch):
            dc_data = get_dc_data(batch, self.device)

        if 'flow' in self.hparams.stage:
            with torch.no_grad():
                t = self.generate_random_t(x.shape[0], dtype=x.dtype)
                source_dist_samples = self.create_source_distribution_samples(x, y, non_noisy_z0, dc_data=dc_data)
                x_t = t * x + (1.0 - t) * source_dist_samples

            if self.hparams.stage == 'gauge_flow':
                # --- ENGRF gauge_flow training ---
                # delta = X - X* (or X - source when using noise)
                delta = x - source_dist_samples  # (B, 2, H, W)

                # Compute gauge displacement W = W_psi(Z_t; Y)
                gauge_input = torch.cat([x_t, y], dim=1)  # (B, 4, H, W)
                W = self.gauge_net(gauge_input)  # (B, 2, H, W)

                # Bump function values
                a = gauge_alpha(t, self.hparams.gauge_strength)      # (B, 1, 1, 1)
                a_prime = gauge_alpha_prime(t, self.hparams.gauge_strength)  # (B, 1, 1, 1)

                # Gauged interpolation: Z_tilde_t = Z_t + alpha(t) * W
                z_tilde_t = x_t + a * W

                # Target velocity: delta + alpha'(t)*W + alpha(t)*JW*delta
                # NOTE: W and jvp are NOT detached so gauge_net gets gradients
                # from the target terms.  Gradients also flow through the gauged
                # input z_tilde_t → v_t once the velocity model's output
                # layer (zero-init at start) becomes non-zero during training.
                target = delta + a_prime * W
                if self.hparams.use_gauge_jvp:
                    # JVP correction: J_W(Z_t; Y) * delta
                    jw_delta = gauge_jvp_fd(self.gauge_net, x_t.detach(), y, delta.detach())
                    target = target + a * jw_delta

                # Forward velocity on the gauged path
                v_t = self(z_tilde_t, t.squeeze(), y)
                loss = mse_loss(v_t, target)
            else:
                # Standard flow training (unchanged)
                v_t = self(x_t, t.squeeze(), y)
                loss = mse_loss(v_t, x - source_dist_samples)
        else:
            # MMSE stage: loss on normalized complex data (same as training convention)
            xhat = self(x_t=None, t=None, y=y)

            loss = mse_loss(xhat, x)

        # ---------------- Per-slice (step) metrics ----------------
        # All metrics are computed on UNNORMALIZED MAGNITUDE images
        # (ReconFormer-style: complex_abs(pred) * norm_std * norm_scale).
        with torch.no_grad():
            unnorm = get_unnorm_factor(batch, x.shape[0], x.device)  # (B,)
            unnorm_4d = unnorm.view(-1, 1, 1, 1)

            if 'flow' in self.hparams.stage:
                # Single-step reconstruction estimate: source + v_t.
                xhat_step = (source_dist_samples + v_t).detach()
            else:
                xhat_step = xhat.detach()

            # Convert complex (2-ch) to unnormalized magnitude (1-ch) for metrics
            x_mag_raw = complex_to_magnitude(x) * unnorm_4d
            y_mag_raw = complex_to_magnitude(y) * unnorm_4d
            xhat_mag_raw = complex_to_magnitude(xhat_step) * unnorm_4d

            sse_per_sample, mse_per_sample, psnr_per_sample, slice_max, count = slice_sse_mse_psnr(
                xhat_mag_raw, x_mag_raw
            )
            psnr_step = psnr_per_sample.mean()

            # Per-step SSIM on unnormalized magnitude
            ssim_per_sample = compute_ssim_per_sample(x_mag_raw, xhat_mag_raw, slice_max)
            ssim_step = ssim_per_sample.mean()

            # Baseline per-scan stats: y vs x (naive PSNR-per-scan).
            _sse_y, mse_y, _psnr_y, _slice_max_y, _count_y = slice_sse_mse_psnr(y_mag_raw, x_mag_raw)
            self.train_scan_stats_y.update(fnames_list, mse_y, slice_max)

            # Baseline per-scan SSIM: y vs x
            ssim_y_per_sample = compute_ssim_per_sample(x_mag_raw, y_mag_raw, slice_max)
            self.train_scan_ssim_stats_y.update(fnames_list, ssim_y_per_sample)

        self.log(
            "train_metrics/psnr_per_step",
            psnr_step,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )
        self.log(
            "train_metrics/ssim_per_step",
            ssim_step,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=x.shape[0],
        )

        # Log training loss (MSE) to both logger and progress bar so it is visible in
        # the tqdm/pbar during training and also aggregated over the epoch.
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=x.shape[0],
        )
        # Visualize a random training sample once per epoch. The batch index is # TODO OMER ADDED THIS PART
        # randomized in `on_train_epoch_start`, so the visualized sample changes
        # every epoch.
        if batch_idx == getattr(self, "train_sample_batch_idx", 0) and get_wandb_logger(self) is not None:
            with torch.no_grad():
                xhat_train, _, _ = self.generate_reconstructions(
                    x,
                    y,
                    non_noisy_z0,
                    self.hparams.num_flow_steps,
                    self.device,
                    dc_data=dc_data,
                )

            # Cache magnitude images for visualization (convert from complex)
            self._train_vis_data = {
                "x": complex_to_magnitude(x).detach().cpu().to(torch.float32),
                "y": complex_to_magnitude(y).detach().cpu().to(torch.float32),
                "xhat": complex_to_magnitude(xhat_train).detach().cpu().to(torch.float32),
                "fnames": fnames_list,
            }

        return loss

    @torch.no_grad()
    def generate_reconstructions(self, x, y, non_noisy_z0, num_flow_steps, result_device, dc_data=None):
        """
        Generate reconstructions using the flow model (or MMSE model).
        
        Args:
            x: Ground truth tensor (B, C, H, W) - used for shape reference
            y: Zero-filled / undersampled input (B, C, H, W)
            non_noisy_z0: Pre-computed MMSE output if available (B, C, H, W)
            num_flow_steps: Number of Euler integration steps
            result_device: Device to move results to
            dc_data: Optional dict with data consistency fields for apply_dc_to_source
        
        Returns:
            xhat: Final reconstruction (B, C, H, W)
            x_t_seq: List of intermediate states (flow only)
            source_dist_samples: Source distribution samples used (flow only)
        """
        with self.maybe_ema():
            if 'flow' in self.hparams.stage:
                source_dist_samples = self.create_source_distribution_samples(x, y, non_noisy_z0, dc_data=dc_data)

                dt = (1.0 / num_flow_steps) * (1.0 - self.hparams.eps)
                x_t_next = source_dist_samples.clone()
                x_t_seq = [x_t_next]
                t_one = torch.ones(x.shape[0], device=self.device)
                for i in range(num_flow_steps):
                    num_t = (i / num_flow_steps) * (1.0 - self.hparams.eps) + self.hparams.eps
                    v_t_next = self(x_t=x_t_next, t=t_one * num_t, y=y).to(x_t_next.dtype)
                    x_t_next = x_t_next.clone() + v_t_next * dt
                    x_t_seq.append(x_t_next.to(result_device))

                # For complex data, don't clip (real/imag can be negative)
                # Clipping will be done on magnitude when needed for visualization/metrics
                xhat = x_t_seq[-1].to(torch.float32)
                source_dist_samples = source_dist_samples.to(result_device)
            else:
                xhat = self(x_t=None, t=None, y=y).to(torch.float32)
                x_t_seq = None
                source_dist_samples = None
            return xhat.to(result_device), x_t_seq, source_dist_samples
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = batch['x']
        y = batch['y']
        non_noisy_z0 = batch['non_noisy_z0'] if 'non_noisy_z0' in batch else None

        # ReconFormer DC data: set on self so forward methods can access it.
        self._rf_dc_data = _extract_rf_dc(batch, self.device)
        
        # Extract DC data if available and apply_dc_to_source is enabled
        dc_data_for_source = None
        if self.hparams.apply_dc_to_source and has_dc_data(batch):
            dc_data_for_source = get_dc_data(batch, self.device)
        
        xhat, x_t_seq, source_dist_samples = self.generate_reconstructions(
            x, y, non_noisy_z0, self.hparams.num_flow_steps, self.device, dc_data=dc_data_for_source
        )
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        xhat = xhat.to(torch.float32)

        # ---- All metrics on UNNORMALIZED MAGNITUDE (ReconFormer-style) ----
        unnorm = get_unnorm_factor(batch, x.shape[0], x.device)  # (B,)
        unnorm_4d = unnorm.view(-1, 1, 1, 1)

        x_mag = complex_to_magnitude(x) * unnorm_4d
        y_mag = complex_to_magnitude(y) * unnorm_4d
        xhat_mag = complex_to_magnitude(xhat) * unnorm_4d

        fnames_list = get_fnames_list(batch, x.shape[0])

        # Per-scan PSNR accumulation (model predictions)
        _sse, mse_per_sample, _psnr, slice_max, _count = slice_sse_mse_psnr(xhat_mag, x_mag)
        self.val_scan_stats.update(fnames_list, mse_per_sample, slice_max)

        # Per-scan SSIM: store images; SSIM computed at epoch end with
        # data_range = max(GT_scan) instead of per-slice max.
        self.val_scan_ssim_stats.update(fnames_list, x_mag, xhat_mag)

        # Baseline per-scan stats: y vs x (naive)
        _sse_y, mse_y, _psnr_y, _max_y, _cnt_y = slice_sse_mse_psnr(y_mag, x_mag)
        self.val_scan_stats_y.update(fnames_list, mse_y, slice_max)

        ssim_y_per_sample = compute_ssim_per_sample(x_mag, y_mag, slice_max)
        self.val_scan_ssim_stats_y.update(fnames_list, ssim_y_per_sample)

        # ---------------- Data Consistency (DC) metrics ----------------
        xhat_dc = None
        xhat_dc_mag = None
        if has_dc_data(batch):
            dc_data = get_dc_data(batch, self.device)
            xhat_dc = apply_data_consistency(
                xhat=xhat,
                kspace=dc_data['kspace'],
                mask=dc_data['mask'],
                norm_std=dc_data['norm_std'],
                norm_scale=dc_data['norm_scale'],
                resolution=dc_data['resolution'],
            )
            xhat_dc_mag = complex_to_magnitude(xhat_dc) * unnorm_4d

            # DC per-scan PSNR
            _sse_dc, mse_dc, _psnr_dc, slice_max_dc, _cnt_dc = slice_sse_mse_psnr(xhat_dc_mag, x_mag)
            self.val_scan_stats_dc.update(fnames_list, mse_dc, slice_max_dc)

            # DC per-scan SSIM: store images; computed at epoch end with
            # data_range = max(GT_scan).
            self.val_scan_ssim_stats_dc.update(fnames_list, x_mag, xhat_dc_mag)

        # ---------------- PIQ perceptual metrics (flow stage only) ----------------
        if 'flow' in self.hparams.stage:
            with torch.no_grad():
                # Normalize to [0,1] for PIQ by dividing by per-sample max
                x_01 = (x_mag / slice_max.view(-1, 1, 1, 1).clamp(min=1e-8)).clamp(0, 1).float()
                xhat_01 = (xhat_mag / slice_max.view(-1, 1, 1, 1).clamp(min=1e-8)).clamp(0, 1).float()

                x_3ch = x_01.repeat(1, 3, 1, 1)
                xhat_3ch = xhat_01.repeat(1, 3, 1, 1)

                # Lazily move metric models to GPU (once)
                self._piq_lpips.to(self.device)
                self._piq_dists.to(self.device)

                # Non-DC LPIPS/DISTS
                lpips_per_sample = self._piq_lpips(xhat_3ch, x_3ch)
                dists_per_sample = self._piq_dists(xhat_3ch, x_3ch)
                self.val_lpips_stats.update(fnames_list, lpips_per_sample)
                self.val_dists_stats.update(fnames_list, dists_per_sample)

                # DC LPIPS/DISTS
                if xhat_dc_mag is not None:
                    xhat_dc_01 = (xhat_dc_mag / slice_max_dc.view(-1, 1, 1, 1).clamp(min=1e-8)).clamp(0, 1).float()
                    xhat_dc_3ch = xhat_dc_01.repeat(1, 3, 1, 1)
                    lpips_dc_per_sample = self._piq_lpips(xhat_dc_3ch, x_3ch)
                    dists_dc_per_sample = self._piq_dists(xhat_dc_3ch, x_3ch)
                    self.val_lpips_stats_dc.update(fnames_list, lpips_dc_per_sample)
                    self.val_dists_stats_dc.update(fnames_list, dists_dc_per_sample)

        # Visualize a random validation sample once per epoch.
        if batch_idx == getattr(self, "val_sample_batch_idx", 0): # TODO OMER ADDED THIS PART
            # Cache magnitude images for visualization (keep normalized for vis)
            x_mag_vis = complex_to_magnitude(x)
            y_mag_vis = complex_to_magnitude(y)
            xhat_mag_vis = complex_to_magnitude(xhat)
            if get_wandb_logger(self) is not None:
                vis = {
                    "x": x_mag_vis.detach().cpu().to(torch.float32),
                    "y": y_mag_vis.detach().cpu().to(torch.float32),
                    "y_complex": y.detach().cpu().to(torch.float32),  # complex y for mmse_model
                    "xhat": xhat_mag_vis.detach().cpu().to(torch.float32),
                    "fnames": fnames_list,
                }

                # Add DC visualization if available
                if xhat_dc_mag is not None:
                    xhat_dc_mag_vis = complex_to_magnitude(xhat_dc)
                    vis["xhat_dc"] = xhat_dc_mag_vis.detach().cpu().to(torch.float32)

                if 'flow' in self.hparams.stage and x_t_seq is not None and source_dist_samples is not None:
                    # Keep only the first element of the trajectory to keep memory bounded.
                    fname0 = fnames_list[0]
                    vis["fname_first"] = fname0
                    vis["x_t_seq_first"] = [complex_to_magnitude(elem[0:1]).detach().cpu().to(torch.float32) for elem in x_t_seq]
                    vis["source_first"] = complex_to_magnitude(source_dist_samples[0:1]).detach().cpu().to(torch.float32)

                self._val_vis_data = vis

    def on_validation_epoch_end(self):
        # Per-scan PSNR (model predictions)
        merged_stats = self.val_scan_stats.gathered()
        avg_scan_psnr, _avg_scan_mse, scan_max_by_fname = summarize_scan_psnr_mse(merged_stats)

        # Per-scan SSIM (model predictions) — data_range = max(GT_scan)
        # Run SSIM convolutions on GPU for speed (images are stored on CPU).
        gathered_ssim = self.val_scan_ssim_stats.gathered()
        avg_scan_ssim, _ssim_by_fname = gathered_ssim.compute_per_scan_ssim(device=self.device)

        # Naive baselines
        merged_stats_y = self.val_scan_stats_y.gathered()
        naive_scan_psnr, _naive_scan_mse, _scan_max_by_fname_y = summarize_scan_psnr_mse(merged_stats_y)
        merged_ssim_y = self.val_scan_ssim_stats_y.gathered()
        naive_scan_ssim, _naive_ssim_by_fname = summarize_scan_mean(merged_ssim_y)

        # Data Consistency (DC) metrics
        merged_stats_dc = self.val_scan_stats_dc.gathered()
        avg_scan_psnr_dc, _avg_scan_mse_dc, _scan_max_dc = summarize_scan_psnr_mse(merged_stats_dc)
        gathered_ssim_dc = self.val_scan_ssim_stats_dc.gathered()
        avg_scan_ssim_dc, _ssim_by_fname_dc = gathered_ssim_dc.compute_per_scan_ssim(device=self.device)

        # Visualization (pass {} so vis code falls back to per-batch max for [0,1] normalization)
        log_val_epoch_images(
            self,
            getattr(self, "_val_vis_data", None),
            {},
            mmse_model=self.mmse_model,
        )

        # Reset accumulators
        self._val_vis_data = None
        self.val_scan_stats.reset()
        self.val_scan_stats_y.reset()
        self.val_scan_ssim_stats.reset()
        self.val_scan_ssim_stats_y.reset()
        self.val_scan_stats_dc.reset()
        self.val_scan_ssim_stats_dc.reset()

        if not hasattr(self, "trainer") or getattr(self.trainer, "is_global_zero", True):
            metrics_dict = {
                "val_metrics/psnr_per_scan": avg_scan_psnr,
                "val_metrics/ssim_per_scan": avg_scan_ssim,
                "val_metrics/psnr_per_scan_dc": avg_scan_psnr_dc,
                "val_metrics/ssim_per_scan_dc": avg_scan_ssim_dc,
                "val_metrics/naive_psnr_per_scan": naive_scan_psnr,
                "val_metrics/naive_ssim_per_scan": naive_scan_ssim,
            }

            self.log_dict(
                metrics_dict,
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                sync_dist=False,
                batch_size=1,
            )

        if 'flow' in self.hparams.stage:
            # Non-DC LPIPS/DISTS
            merged_lpips = self.val_lpips_stats.gathered()
            avg_lpips, _ = summarize_scan_mean(merged_lpips)
            merged_dists = self.val_dists_stats.gathered()
            avg_dists, _ = summarize_scan_mean(merged_dists)

            # DC LPIPS/DISTS
            merged_lpips_dc = self.val_lpips_stats_dc.gathered()
            avg_lpips_dc, _ = summarize_scan_mean(merged_lpips_dc)
            merged_dists_dc = self.val_dists_stats_dc.gathered()
            avg_dists_dc, _ = summarize_scan_mean(merged_dists_dc)

            piq_metrics = {
                'val_metrics/lpips': avg_lpips,
                'val_metrics/dists': avg_dists,
                'val_metrics/lpips_dc': avg_lpips_dc,
                'val_metrics/dists_dc': avg_dists_dc,
            }

            self.log_dict(
                piq_metrics,
                on_epoch=True,
                on_step=False,
                sync_dist=False,
                batch_size=1,
            )
            self.val_lpips_stats.reset()
            self.val_dists_stats.reset()
            self.val_lpips_stats_dc.reset()
            self.val_dists_stats_dc.reset()

    def on_test_epoch_start(self) -> None:
        # Collect per-slice metrics during test/inference (and write once at epoch end).
        self._test_slice_metrics = []
        return super().on_test_epoch_start()

    def on_test_epoch_end(self) -> None:
        records = all_gather_list(self._test_slice_metrics)

        # Write per-slice metrics once (global-zero only) to avoid file corruption under DDP.
        if records and (not hasattr(self, "trainer") or getattr(self.trainer, "is_global_zero", True)):
            out_dir = self.test_results_path or "."
            write_metrics_csv(records, out_dir)

        self._test_slice_metrics = []
        return super().on_test_epoch_end()

    def test_step(self, batch, batch_idx):
        assert self.test_results_path is not None, "Please set test_results_path before testing."
        assert os.path.isdir(self.test_results_path), 'Please make sure the test_result_path dir exists.'

        os.makedirs(self.test_results_path, exist_ok=True)
        x = batch['x']
        y = batch['y']
        non_noisy_z0 = batch['non_noisy_z0'] if 'non_noisy_z0' in batch else None

        # ReconFormer DC data: set on self so forward methods can access it.
        self._rf_dc_data = _extract_rf_dc(batch, self.device)

        batch_size = int(x.shape[0])
        fnames_list = get_fnames_list(batch, batch_size)
        slice_list = get_slice_list(batch, batch_size)
        img_file_names = resolve_img_file_names(batch, batch_size, fnames_list, slice_list)

        # Convert to magnitude for saving images and computing metrics
        x_mag_cpu = complex_to_magnitude(x).detach().cpu().to(torch.float32)
        y_mag = complex_to_magnitude(y)

        y_path = os.path.join(self.test_results_path, 'y')
        save_image_batch(y_mag, y_path, img_file_names)

        # Check if DC data is available
        dc_available = has_dc_data(batch)
        dc_data = get_dc_data(batch, self.device) if dc_available else None
        
        # DC data for source distribution (apply_dc_to_source)
        dc_data_for_source = dc_data if self.hparams.apply_dc_to_source else None

        if 'flow' in self.hparams.stage:
            source_dist_samples_to_save = None

            for num_flow_steps in getattr(self, "num_test_flow_steps", (self.hparams.num_flow_steps,)):
                xhat, x_t_seq, source_dist_samples = self.generate_reconstructions(
                    x, y, non_noisy_z0, num_flow_steps, torch.device("cpu"), dc_data=dc_data_for_source
                )
                # Convert to magnitude for saving and metrics
                xhat_mag = complex_to_magnitude(xhat)
                xhat_path = os.path.join(self.test_results_path, f"num_flow_steps={num_flow_steps}", 'xhat')
                save_image_batch(xhat_mag, xhat_path, img_file_names)
                append_per_slice_metrics(
                    self._test_slice_metrics,
                    xhat_cpu=xhat_mag,
                    x_cpu=x_mag_cpu,
                    img_file_names=img_file_names,
                    fnames_list=fnames_list,
                    slice_list=slice_list,
                    variant="xhat",
                    num_flow_steps=int(num_flow_steps),
                )

                # Data Consistency (DC) metrics and saving
                if dc_available:
                    xhat_dc = apply_data_consistency(
                        xhat=xhat.to(self.device),
                        kspace=dc_data['kspace'],
                        mask=dc_data['mask'],
                        norm_std=dc_data['norm_std'],
                        norm_scale=dc_data['norm_scale'],
                        resolution=dc_data['resolution'],
                    )
                    xhat_dc_mag = complex_to_magnitude(xhat_dc).detach().cpu().to(torch.float32)
                    xhat_dc_path = os.path.join(self.test_results_path, f"num_flow_steps={num_flow_steps}", 'xhat_dc')
                    save_image_batch(xhat_dc_mag, xhat_dc_path, img_file_names)
                    append_per_slice_metrics(
                        self._test_slice_metrics,
                        xhat_cpu=xhat_dc_mag,
                        x_cpu=x_mag_cpu,
                        img_file_names=img_file_names,
                        fnames_list=fnames_list,
                        slice_list=slice_list,
                        variant="xhat_dc",
                        num_flow_steps=int(num_flow_steps),
                    )

                if source_dist_samples_to_save is None:
                    source_dist_samples_to_save = source_dist_samples

            source_distribution_samples_path = os.path.join(self.test_results_path, 'source_distribution_samples')
            save_image_batch(complex_to_magnitude(source_dist_samples_to_save), source_distribution_samples_path, img_file_names)
            if self.mmse_model is not None:
                if _is_reconformer(self.mmse_model):
                    k0, mask_rf = self._rf_dc_data or (None, None)
                    mmse_estimates = self.mmse_model(y, k0=k0, mask=mask_rf)
                else:
                    mmse_estimates = self.mmse_model(y)
                mmse_mag = complex_to_magnitude(mmse_estimates)
                mmse_samples_path = os.path.join(self.test_results_path, 'mmse_samples')
                save_image_batch(mmse_mag, mmse_samples_path, img_file_names)
                append_per_slice_metrics(
                    self._test_slice_metrics,
                    xhat_cpu=mmse_mag.detach().cpu().to(torch.float32),
                    x_cpu=x_mag_cpu,
                    img_file_names=img_file_names,
                    fnames_list=fnames_list,
                    slice_list=slice_list,
                    variant="mmse_model",
                )

                # DC for MMSE model predictions
                if dc_available:
                    mmse_dc = apply_data_consistency(
                        xhat=mmse_estimates,
                        kspace=dc_data['kspace'],
                        mask=dc_data['mask'],
                        norm_std=dc_data['norm_std'],
                        norm_scale=dc_data['norm_scale'],
                        resolution=dc_data['resolution'],
                    )
                    mmse_dc_mag = complex_to_magnitude(mmse_dc).detach().cpu().to(torch.float32)
                    mmse_dc_path = os.path.join(self.test_results_path, 'mmse_samples_dc')
                    save_image_batch(mmse_dc_mag, mmse_dc_path, img_file_names)
                    append_per_slice_metrics(
                        self._test_slice_metrics,
                        xhat_cpu=mmse_dc_mag,
                        x_cpu=x_mag_cpu,
                        img_file_names=img_file_names,
                        fnames_list=fnames_list,
                        slice_list=slice_list,
                        variant="mmse_model_dc",
                    )


        else:
            xhat, _, _ = self.generate_reconstructions(x, y, non_noisy_z0, None, torch.device('cpu'))
            xhat_mag = complex_to_magnitude(xhat)
            xhat_path = os.path.join(self.test_results_path, 'xhat')
            save_image_batch(xhat_mag, xhat_path, img_file_names)
            append_per_slice_metrics(
                self._test_slice_metrics,
                xhat_cpu=xhat_mag,
                x_cpu=x_mag_cpu,
                img_file_names=img_file_names,
                fnames_list=fnames_list,
                slice_list=slice_list,
                variant="xhat",
            )

            # Data Consistency (DC) for MMSE stage
            if dc_available:
                xhat_dc = apply_data_consistency(
                    xhat=xhat.to(self.device),
                    kspace=dc_data['kspace'],
                    mask=dc_data['mask'],
                    norm_std=dc_data['norm_std'],
                    norm_scale=dc_data['norm_scale'],
                    resolution=dc_data['resolution'],
                )
                xhat_dc_mag = complex_to_magnitude(xhat_dc).detach().cpu().to(torch.float32)
                xhat_dc_path = os.path.join(self.test_results_path, 'xhat_dc')
                save_image_batch(xhat_dc_mag, xhat_dc_path, img_file_names)
                append_per_slice_metrics(
                    self._test_slice_metrics,
                    xhat_cpu=xhat_dc_mag,
                    x_cpu=x_mag_cpu,
                    img_file_names=img_file_names,
                    fnames_list=fnames_list,
                    slice_list=slice_list,
                    variant="xhat_dc",
                )

    def configure_optimizers(self):
        # Collect trainable parameters
        if self.hparams.stage == 'gauge_flow':
            # gauge_flow: gauge_net params always included; velocity model
            # params are included unless frozen
            params = list(self.gauge_net.parameters())
            if not self.hparams.freeze_flow_model:
                params = list(self.model.parameters()) + params
        else:
            params = self.model.parameters()

        optimizer = AdamW(params,
                          betas=self.hparams.betas,
                          eps=1e-8,
                          lr=self.hparams.lr,
                          weight_decay=self.hparams.weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=getattr(self.hparams, "lr_scheduler_patience", 30),
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss_epoch",
                "interval": "epoch",
                "frequency": 1,
            },
        }
