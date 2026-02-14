"""
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
import random

import numpy as np
import h5py
from torch.utils.data import Dataset
from data import transforms
import torch


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sequence, sample_rate, seed=42):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' else 'reconstruction_rss'

        self.examples = []
        root = pathlib.Path(root)
        # Only consider actual HDF5 files to avoid errors from directories or
        # non-HDF5 side files in the dataset directory.
        files = [
            f for f in root.iterdir()
            if f.is_file() and f.suffix in (".h5", ".hdf5")
        ]
        print('Loading dataset :', root)
        random.seed(seed)
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            # IMPORTANT: do not keep HDF5 files open in __init__ (can exhaust file handles).
            with h5py.File(fname, "r") as data:
                kspace = data["kspace"]
                num_slices = int(kspace.shape[0])

            # Padding fields are kept for compatibility with existing codepaths,
            # but are unused by the current transform.
            padding_left = None
            padding_right = None
            self.examples += [
                (fname, slice_idx, padding_left, padding_right)
                for slice_idx in range(num_slices)
            ]

        # Cache for per-volume (per .h5) statistics to support volume-level normalization/scaling.
        # Keyed by file path (string) to avoid pathlib/h5py object identity issues.
        self._volume_stats_cache = {}

    def _compute_volume_stats_from_y(self, fname: pathlib.Path):
        """
        Compute per-volume stats using ONLY the subsampled / zero-filled reconstruction y.

        This is inference-safe: at inference you always have access to k-space and the mask.
        The stats are cached per file and reused for all slices.
        """
        # Only needed if the current transform requests volume-level scaling.
        mask_func = getattr(self.transform, "mask_func", None)
        if mask_func is None:
            raise ValueError("volume_* scale_mode requires a mask_func (got None).")

        resolution = getattr(self.transform, "resolution", None)
        use_seed = bool(getattr(self.transform, "use_seed", True))
        scale_mode = getattr(self.transform, "scale_mode", "none")
        scale_percentile = float(getattr(self.transform, "scale_percentile", 100.0))
        eps = 0

        # Collect all y magnitudes across the volume (after the first normalization) so we
        # can compute a stable volume-wide max/percentile. For typical fastMRI volumes this
        # is manageable; for large datasets, prefer a streaming estimator.
        y_all = []
        abs_sum = 0.0
        abs_count = 0

        with h5py.File(fname, "r") as data:
            kspace_ds = data["kspace"]
            num_slices = kspace_ds.shape[0]
            seed_tuple = None
            if use_seed:
                seed_tuple = tuple(map(ord, fname.name))

            # Pass 1: compute mean(|y|) over the whole volume (cropped) to use as denom.
            # Masking is applied at target resolution for consistency with DataTransform.
            for s in range(num_slices):
                kspace = transforms.to_tensor(kspace_ds[s])
                # IFFT to image domain, crop, FFT back to get k-space at target resolution
                image_full = transforms.ifft2(kspace)
                if resolution is not None:
                    image_cropped = transforms.complex_center_crop(image_full, (resolution, resolution))
                else:
                    image_cropped = image_full
                kspace_target_res = transforms.fft2(image_cropped)
                # Apply mask at target resolution
                masked_kspace, _mask = transforms.apply_mask(kspace_target_res, mask_func, seed_tuple)
                image = transforms.ifft2(masked_kspace)
                abs_image = transforms.complex_abs(image)
                abs_sum += float(abs_image.sum().item())
                abs_count += int(abs_image.numel())

            vol_abs_mean = abs_sum / max(abs_count, 1)
            vol_abs_mean = max(vol_abs_mean, eps)

            # Pass 2: collect y magnitudes after dividing by vol_abs_mean.
            for s in range(num_slices):
                kspace = transforms.to_tensor(kspace_ds[s])
                # IFFT to image domain, crop, FFT back to get k-space at target resolution
                image_full = transforms.ifft2(kspace)
                if resolution is not None:
                    image_cropped = transforms.complex_center_crop(image_full, (resolution, resolution))
                else:
                    image_cropped = image_full
                kspace_target_res = transforms.fft2(image_cropped)
                # Apply mask at target resolution
                masked_kspace, _mask = transforms.apply_mask(kspace_target_res, mask_func, seed_tuple)
                image = transforms.ifft2(masked_kspace)
                # normalize complex by volume denom
                image = image.permute(2, 0, 1)
                image = transforms.normalize(image, torch.tensor(0.0), torch.tensor(vol_abs_mean), eps=0.0)
                image = image.permute(1, 2, 0)
                y_mag = transforms.complex_abs(image)
                y_all.append(y_mag.reshape(-1))

        y_all = torch.cat(y_all, dim=0)
        if scale_mode == "volume_subsample_max":
            vol_scale = float(y_all.max().item())
        else:
            # "volume_subsample_percentile"
            q = max(min(scale_percentile / 100.0, 1.0), 0.0)
            vol_scale = float(torch.quantile(y_all, q).item())
        vol_scale = max(vol_scale, eps)
        return {"vol_abs_mean": float(vol_abs_mean), "vol_scale": float(vol_scale)}


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice, padding_left, padding_right = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            mask = np.asarray(data['mask']) if 'mask' in data else None
            target = data[self.recons_key][slice] if self.recons_key in data else None
            attrs = dict(data.attrs)
            attrs['padding_left'] = padding_left
            attrs['padding_right'] = padding_right
            # If requested, compute per-volume stats for this file and pass through attrs.
            scale_mode = getattr(self.transform, "scale_mode", "none")
            if isinstance(scale_mode, str) and scale_mode.startswith("volume_"):
                key = str(fname)
                if key not in self._volume_stats_cache:
                    self._volume_stats_cache[key] = self._compute_volume_stats_from_y(pathlib.Path(fname))
                stats = self._volume_stats_cache[key]
                attrs["vol_abs_mean"] = stats["vol_abs_mean"]
                attrs["vol_scale"] = stats["vol_scale"]
            return self.transform(kspace, mask, target, attrs, fname.name, slice)


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self,
                 resolution,
                 which_challenge,
                 mask_func=None,
                 use_seed=True,
                 scale_mode: str = "none",
                 scale_percentile: float = 100.0,
                 include_dc_data: bool = True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
            scale_mode (str): Optional extra intensity scaling applied to both x and y
                based on the subsampled image y. Options:
                    - "none" (default): no additional scaling.
                    - "subsample_max": divide x and y by max(y).
                    - "subsample_percentile": divide x and y by the given percentile of y.
            scale_percentile (float): Percentile (0–100] used when scale_mode is
                "subsample_percentile". Default: 100.0.
            include_dc_data (bool): If True, include kspace, mask, and original_shape
                in the returned sample for data consistency. Set to False for training
                (where variable k-space sizes prevent batching) and True for
                validation/inference. Default: False.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(
                f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.scale_mode = scale_mode
        self.scale_percentile = scale_percentile
        self.include_dc_data = include_dc_data

    def __call__(self, kspace, mask, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            mask (numpy.array): Mask from the test dataset
            target (numpy.array): Target image reconstructed from *fully sampled* k-space.
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.

        Returns:
            dict with:

            - ``x`` (torch.Tensor): **Fully sampled image** (ground-truth) tensor at target resolution.
            - ``y`` (torch.Tensor): **Subsampled / zero-filled image** tensor at target resolution.
            - ``kspace`` (torch.Tensor): **K-space at target resolution** (resolution, resolution, 2) for data consistency.
            - ``mask`` (torch.Tensor): **Undersampling mask at target resolution** for data consistency.
            - ``original_shape`` (tuple): K-space shape (resolution, resolution) - same as y's spatial dims.

        All of x, y, kspace, and mask are at the same target resolution.
        
        This matches the interface expected by ``MMSERectifiedFlow``, where
        ``x`` is the target image and ``y`` is the degraded observation.
        """
        # Keep the file-provided target (if present) separate from the internally
        # reconstructed complex target.
        target_np = target
        kspace = transforms.to_tensor(kspace)

        # ------------------------------------------------------------------
        # NEW FLOW: Crop to target resolution BEFORE masking.
        # This ensures kspace, mask, and y are all at the same resolution.
        # 
        # Steps:
        # 1. IFFT original k-space to image domain
        # 2. Crop to target resolution
        # 3. FFT back to get k-space at target resolution
        # 4. Apply mask at target resolution
        # 5. IFFT to get zero-filled reconstruction (y)
        # ------------------------------------------------------------------
        
        # Step 1: IFFT to image domain (full resolution)
        image_full = transforms.ifft2(kspace)
        
        # Step 2: Crop to target resolution in image domain
        if self.resolution is not None:
            target_cropped = transforms.complex_center_crop(image_full, (self.resolution, self.resolution))
        else:
            target_cropped = image_full
        
        # Step 3: FFT back to get k-space at target resolution
        kspace_target_res = transforms.fft2(target_cropped)
        
        # Store k-space at target resolution for data consistency (before masking)
        kspace_original = kspace_target_res.clone()
        # Now kspace shape is (resolution, resolution, 2)
        original_shape = (kspace_target_res.shape[-3], kspace_target_res.shape[-2])  # (resolution, resolution)
        
        # Step 4: Apply undersampling at target resolution
        mask = None
        if self.mask_func is not None:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = transforms.apply_mask(kspace_target_res, self.mask_func, seed)
        else:
            masked_kspace = kspace_target_res
        
        # Step 5: IFFT to get zero-filled reconstruction (y) at target resolution
        image = transforms.ifft2(masked_kspace)
        # target is already at target resolution
        target = target_cropped

        # ------------------------------------------------------------------
        # Complex-domain intensity normalization (Reconformer-style).
        #
        # Default: per-slice mean(|y|).
        #
        # IMPORTANT: if we are using a *volume_* scale_mode, SliceData caches
        # `vol_abs_mean` and `vol_scale` computed from y **after dividing by
        # vol_abs_mean**. In that case we must normalize this slice with the
        # same vol_abs_mean, otherwise the cached vol_scale will be inconsistent
        # with the values seen by the model.
        # ------------------------------------------------------------------
        eps = 0

        abs_image = transforms.complex_abs(image)
        mean = torch.tensor(0.0, dtype=abs_image.dtype, device=abs_image.device)

        # Default Reconformer-style normalization: per-slice mean(|y|).
        std = abs_image.mean()

        # For volume_* scale modes, enforce a *per-volume* denom so that every slice
        # in the scan is in a consistent intensity scale and the cached `vol_scale`
        # (computed in SliceData._compute_volume_stats_from_y) matches what the
        # transform produces here.
        if (
            isinstance(self.scale_mode, str)
            and self.scale_mode.startswith("volume_")
            and isinstance(attrs, dict)
            and "vol_abs_mean" in attrs
        ):
            try:
                vol_abs_mean = float(attrs["vol_abs_mean"])
            except Exception:
                vol_abs_mean = None
            if vol_abs_mean is not None and np.isfinite(vol_abs_mean) and vol_abs_mean > 0.0:
                std = torch.tensor(vol_abs_mean, dtype=abs_image.dtype, device=abs_image.device)

        image = image.permute(2, 0, 1)
        image = transforms.normalize(image, mean, std, eps=eps)
        image = image.permute(1, 2, 0)

        target = target.permute(2, 0, 1)
        target = transforms.normalize(target, mean, std, eps=eps)
        target = target.permute(1, 2, 0)

        # Convert from (H, W, 2) to (2, H, W) for complex representation
        # Channel 0 = real, Channel 1 = imaginary
        y_complex = image.permute(2, 0, 1)  # (2, H, W)
        x_complex = target.permute(2, 0, 1)  # (2, H, W)

        # Compute magnitude for scaling (but keep complex for output)
        y_mag = transforms.complex_abs(image)  # (H, W)

        # Track the scale factor for unnormalization (default 1.0 if no scaling)
        norm_scale = torch.tensor(1.0, dtype=x_complex.dtype)
        
        if self.scale_mode in ("subsample_max", "subsample_percentile", "volume_subsample_max", "volume_subsample_percentile"):
            # Slice-level scaling uses stats from this slice's y (magnitude).
            # Volume-level scaling uses cached stats provided in attrs (computed from y over the full volume).
            if self.scale_mode.startswith("volume_"):
                scale = attrs.get("vol_scale", None) if isinstance(attrs, dict) else None
                scale = torch.tensor(float(scale)) if scale is not None else None
            else:
                y_flat = y_mag.view(-1)
                if self.scale_mode == "subsample_max":
                    scale = y_flat.max()
                else:  # "subsample_percentile"
                    q = self.scale_percentile / 100.0
                    q = max(min(q, 1.0), 0.0)
                    scale = torch.quantile(y_flat, q)
            if scale is not None and torch.isfinite(scale) and scale > 0:
                scale = torch.clamp(scale, min=eps)
                # Apply scaling to complex tensors (both real and imag channels)
                x_complex = x_complex / scale
                y_complex = y_complex / scale
                norm_scale = scale


        sample = {
            "x": x_complex,  # fully sampled complex image (2, H, W): [real, imag]
            "y": y_complex,  # subsampled / zero-filled complex image (2, H, W): [real, imag]
            # For per-scan metrics (scan == one .h5 volume in fastMRI).
            # `fname` is the volume identifier; `slice` is the slice index inside it.
            "fname": fname,
            "slice": slice,
            # Normalization factors for unnormalization:
            # raw_value = normalized_value * norm_scale * norm_std
            "norm_std": std,      # mean(|y|) or vol_abs_mean
            "norm_scale": norm_scale,  # vol_scale or slice-level scale (1.0 if none)
        }
        
        # Data consistency fields (optional)
        # All DC fields are at target resolution (same as y):
        # - kspace: (resolution, resolution, 2) - k-space at target resolution, before masking
        # - mask: (1, resolution, 1) - undersampling mask at target resolution
        # - original_shape: (resolution, resolution) - same as kspace spatial dims
        if self.include_dc_data:
            sample["kspace"] = kspace_original  # (resolution, resolution, 2) - k-space at target resolution
            sample["mask"] = mask  # undersampling mask at target resolution
            sample["original_shape"] = original_shape  # (resolution, resolution)

            # ---- ReconFormer-compatible DC fields ----
            # masked_kspace: normalized masked k-space in (2, H, W) channel-first
            #   Same normalization as y_complex (divided by std and norm_scale).
            mk = masked_kspace.permute(2, 0, 1)  # (H, W, 2) -> (2, H, W)
            mk = transforms.normalize(mk, mean, std, eps=eps)
            if norm_scale > 0:
                mk = mk / norm_scale
            sample["masked_kspace_norm"] = mk

            # reconformer_mask: mask in (1, H, W) format.
            # Original mask from apply_mask is (1, W, 1) (column-wise).
            # Expand to (1, H, W) so every row has the same column pattern.
            if mask is not None:
                H = target_cropped.shape[0]
                # (1, W, 1) -> squeeze last -> (1, W) -> expand -> (H, W) -> unsqueeze -> (1, H, W)
                rf_mask = mask.squeeze(-1).expand(H, -1).unsqueeze(0)  # (1, H, W)
                sample["reconformer_mask"] = rf_mask
        
        return sample

