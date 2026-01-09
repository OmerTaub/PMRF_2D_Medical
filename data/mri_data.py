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
        eps = 1e-8

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
            for s in range(num_slices):
                kspace = transforms.to_tensor(kspace_ds[s])
                masked_kspace, _mask = transforms.apply_mask(kspace, mask_func, seed_tuple)
                image = transforms.ifft2(masked_kspace)
                if resolution is not None:
                    image = transforms.complex_center_crop(image, (resolution, resolution))
                abs_image = transforms.complex_abs(image)
                abs_sum += float(abs_image.sum().item())
                abs_count += int(abs_image.numel())

            vol_abs_mean = abs_sum / max(abs_count, 1)
            vol_abs_mean = max(vol_abs_mean, eps)

            # Pass 2: collect y magnitudes after dividing by vol_abs_mean.
            for s in range(num_slices):
                kspace = transforms.to_tensor(kspace_ds[s])
                masked_kspace, _mask = transforms.apply_mask(kspace, mask_func, seed_tuple)
                image = transforms.ifft2(masked_kspace)
                if resolution is not None:
                    image = transforms.complex_center_crop(image, (resolution, resolution))
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
                 scale_percentile: float = 100.0):
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

            - ``x`` (torch.Tensor): **Fully sampled image** (ground-truth) tensor.
            - ``y`` (torch.Tensor): **Subsampled / zero-filled image** tensor.

        This matches the interface expected by ``MMSERectifiedFlow``, where
        ``x`` is the target image and ``y`` is the degraded observation.
        """
        # Keep the file-provided target (if present) separate from the internally
        # reconstructed complex target.
        target_np = target
        kspace = transforms.to_tensor(kspace)

        # Apply undersampling.
        #
        # Common fastMRI practice:
        # - For train/val, we typically generate a synthetic mask (mask_func).
        # - For test sets, a mask may be stored in the file; if mask_func is None,
        #   we apply the provided mask.
        if self.mask_func is not None:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        

        # Inverse Fourier Transform to get zero-filled reconstruction (y) and
        # fully-sampled reconstruction (x) in complex (real/imag) form.
        image = transforms.ifft2(masked_kspace)
        target = transforms.ifft2(kspace)

        # if self.resolution is not None:
        #     crop_shape = (self.resolution, self.resolution)
        #     image = transforms.complex_center_crop(image, crop_shape)
        #     target_cplx = transforms.complex_center_crop(target_cplx, crop_shape)

        # If the dataset provides a precomputed reconstruction target (fastMRI provides
        # `reconstruction_esc`/`reconstruction_rss`), prefer it as ground-truth magnitude.
        # This is the common supervised fastMRI convention and avoids subtle FFT scaling
        # differences across implementations.
        # target_mag = None
        # if target_np is not None:
        #     target_mag = transforms.to_tensor(target_np).to(torch.float32)
        #     if self.resolution is not None:
        #         target_mag = transforms.center_crop(target_mag, (self.resolution, self.resolution))

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
        eps = 1e-8

        abs_image = transforms.complex_abs(image)
        mean = torch.tensor(0.0)

        std = abs_image.mean()
        if isinstance(self.scale_mode, str) and self.scale_mode.startswith("volume_"):
            vol_abs_mean = attrs.get("vol_abs_mean", None) if isinstance(attrs, dict) else None
            if vol_abs_mean is not None:
                std = torch.tensor(float(vol_abs_mean), dtype=abs_image.dtype)
        std = torch.clamp(std, min=eps)

        image = image.permute(2, 0, 1)
        image = transforms.normalize(image, mean, std, eps=eps)
        image = image.permute(1, 2, 0)

        target = target.permute(2, 0, 1)
        target = transforms.normalize(target, mean, std, eps=eps)
        target = target.permute(1, 2, 0)
        # UNTIL HERE IS RECONFORMER
        
        if self.resolution is not None:
            image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
            target = transforms.complex_center_crop(target, (self.resolution, self.resolution))


        y_mag = transforms.complex_abs(image)
        x_mag = transforms.complex_abs(target)
        x_mag = x_mag.unsqueeze(0)
        y_mag = y_mag.unsqueeze(0)


        if self.scale_mode in ("subsample_max", "subsample_percentile", "volume_subsample_max", "volume_subsample_percentile"):
            # Slice-level scaling uses stats from this slice's y.
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
                x_mag = x_mag / scale
                y_mag = y_mag / scale


        sample = {
            "x": x_mag,  # fully sampled magnitude image
            "y": y_mag,  # subsampled / zero-filled magnitude image           
            # For per-scan metrics (scan == one .h5 volume in fastMRI).
            # `fname` is the volume identifier; `slice` is the slice index inside it.
            "fname": fname,
            "slice": slice,
        }
        return sample


class VanillaSliceData(Dataset):
    """
    A minimal PyTorch Dataset for fastMRI-style HDF5 files.

    This dataset returns **raw** (un-normalized, un-cropped, un-scaled) k-space
    for a single slice:
      - Fully-sampled k-space (as stored in the file)
      - Subsampled k-space (masked k-space)

    No image-domain reconstruction, no normalization, no scaling, and no
    volume-level caching/statistics are performed.
    """

    def __init__(
        self,
        root,
        challenge: str,
        sample_rate: float = 1.0,
        mask_func=None,
        use_seed: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            root (pathlib.Path | str): Directory containing `.h5`/`.hdf5` files.
            challenge (str): "singlecoil" or "multicoil". Used only to pick a
                conventional reconstruction key (if present) for convenience.
            sample_rate (float): Fraction of volumes to include in [0, 1].
            mask_func (callable | None): Optional synthetic mask function. If
                provided, we apply it to k-space to form the subsampled data.
                If None, we will use the file-provided `mask` dataset when present.
            use_seed (bool): If True and using `mask_func`, seed mask generation by
                filename to keep masks consistent across slices of a volume.
            seed (int): RNG seed used when sub-sampling the file list by `sample_rate`.
        """
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')
        if not (0 < float(sample_rate) <= 1.0):
            raise ValueError("sample_rate must be in (0, 1].")

        self.mask_func = mask_func
        self.use_seed = bool(use_seed)
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )

        root = pathlib.Path(root)
        files = [f for f in root.iterdir() if f.is_file() and f.suffix in (".h5", ".hdf5")]
        if len(files) == 0:
            raise FileNotFoundError(f"No .h5/.hdf5 files found in: {root}")

        random.seed(seed)
        if sample_rate < 1.0:
            random.shuffle(files)
            files = files[: round(len(files) * sample_rate)]

        self.examples = []
        for fname in sorted(files):
            # Do not keep the file handle open; only query slice count.
            with h5py.File(fname, "r") as data:
                num_slices = int(data["kspace"].shape[0])
            self.examples.extend((fname, slice_idx) for slice_idx in range(num_slices))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, slice_idx = self.examples[i]

        with h5py.File(fname, "r") as data:
            kspace_np = data["kspace"][slice_idx]
            mask_np = np.asarray(data["mask"]) if "mask" in data else None
            attrs = dict(data.attrs)

        kspace_full = transforms.to_tensor(kspace_np)

        # Build subsampled k-space:
        # - Prefer synthetic mask_func if given (train/val typical)
        # - Else use file-provided mask if present (test typical)
        # - Else fall back to "no mask" (identity)
        kspace_sub = kspace_full
        if self.mask_func is not None:
            seed_tuple = None
            if self.use_seed:
                seed_tuple = tuple(map(ord, fname.name))
            kspace_sub, mask = transforms.apply_mask(kspace_full, self.mask_func, seed_tuple)


        # print("kspace_full.min(), kspace_full.max()", kspace_full.min(), kspace_full.max())
        # print("kspace_sub.min(), kspace_sub.max()", kspace_sub.min(), kspace_sub.max())
        # print("shape of kspace_full", kspace_full.shape)
        # print("shape of kspace_sub", kspace_sub.shape)
        # ifft kspaces
        image_full = transforms.ifft2(kspace_full)
        image_sub = transforms.ifft2(kspace_sub)
        # print("image_full.min(), image_full.max()", image_full.min(), image_full.max())
        # print("image_sub.min(), image_sub.max()", image_sub.min(), image_sub.max())

        # abs image
        image_full_abs = transforms.complex_abs(image_full)
        image_sub_abs = transforms.complex_abs(image_sub)
        # print("image_full_abs.min(), image_full_abs.max()", image_full_abs.min(), image_full_abs.max())
        # print("image_sub_abs.min(), image_sub_abs.max()", image_sub_abs.min(), image_sub_abs.max())

        # cetner crop to 320X320
        image_full_abs = transforms.center_crop(image_full_abs, (320, 320))
        image_sub_abs = transforms.center_crop(image_sub_abs, (320, 320))
        # print("image_full_abs.min(), image_full_abs.max()", image_full_abs.min(), image_full_abs.max())
        # print("image_sub_abs.min(), image_sub_abs.max()", image_sub_abs.min(), image_sub_abs.max())
        # print("shape of image_full_abs", image_full_abs.shape)
        # print("shape of image_sub_abs", image_sub_abs.shape)

        # min_max normalization to images
        min_image_full = image_full_abs.min()
        max_image_full = image_full_abs.max()
        min_image_sub = image_sub_abs.min()
        max_image_sub = image_sub_abs.max()
        normalized_image_full = (image_full_abs - min_image_full) / (max_image_full - min_image_full)
        normalized_image_sub = (image_sub_abs - min_image_sub) / (max_image_sub - min_image_sub)
        # print("normalized_image_full.min(), normalized_image_full.max()", normalized_image_full.min(), normalized_image_full.max())
        # print("normalized_image_sub.min(), normalized_image_sub.max()", normalized_image_sub.min(), normalized_image_sub.max())
        # print("shape of normalized_image_full", normalized_image_full.shape)
        # print("shape of normalized_image_sub", normalized_image_sub.shape)

        # unsqueeze to 1, 320, 320
        normalized_image_full = normalized_image_full.unsqueeze(0)
        normalized_image_sub = normalized_image_sub.unsqueeze(0)
        # print("shape of normalized_image_full", normalized_image_full.shape)
        # print("shape of normalized_image_sub", normalized_image_sub.shape)
        # plot
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(normalized_image_full.squeeze(0).cpu().numpy(), cmap="gray")
        # plt.title("normalized_image_full")
        # plt.subplot(1, 2, 2)
        # plt.imshow(normalized_image_sub.squeeze(0).cpu().numpy(), cmap="gray")
        # plt.title("normalized_image_sub")
        # plt.savefig("normalized_image_full_and_sub.png")
        # plt.close()

        sample = {
            "x": normalized_image_full,
            "y": normalized_image_sub,
            # Keep identifiers consistent with SliceData/DataTransform so that
            # downstream metrics can group by scan.
            "fname": fname.name if hasattr(fname, "name") else str(fname),
            "slice": slice_idx,
        }
        return sample
