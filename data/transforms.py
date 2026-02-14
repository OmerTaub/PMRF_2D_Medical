"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.

    Args:
        data (np.array): Input numpy array

    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def to_numpy(data):
    """
    Convert PyTorch tensor to numpy array. For complex tensor with two channels, the complex numpy arrays are used.

    Args:
        data (torch.Tensor): Input torch tensor

    Returns:
        np.array numpy arrays
    """
    if data.shape[-1] == 2:
        out = np.zeros(data.shape[:-1], dtype=np.complex64)
        real = data[..., 0].numpy()
        imag = data[..., 1].numpy()

        out.real = real
        out.imag = imag
    else:
        out = data.numpy()
    return out


def apply_mask(data, mask_func, seed=None):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    return data * mask, mask


def _to_complex(data: torch.Tensor) -> torch.Tensor:
    """
    Convert a real tensor with last dimension = 2 (real, imag) to a complex tensor.
    """
    assert data.size(-1) == 2, "Expected tensor with last dimension = 2 for complex representation."
    return torch.view_as_complex(data.to(torch.float32).contiguous())


def _from_complex(data: torch.Tensor) -> torch.Tensor:
    """
    Convert a complex tensor to a real tensor with last dimension = 2 (real, imag).
    """
    return torch.view_as_real(data)


def fft2(data, normalized=True):
    """
    Apply centered 2 dimensional Fast Fourier Transform using torch.fft API.

    Args:
        data (torch.Tensor): Complex-valued input encoded with last dimension size 2 (real, imag).

    Returns:
        torch.Tensor: The FFT of the input, encoded with last dimension size 2 (real, imag).
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data_c = _to_complex(data)
    norm = "ortho" if normalized else "backward"
    data_c = torch.fft.fft2(data_c, norm=norm)
    data = _from_complex(data_c)
    data = fftshift(data, dim=(-3, -2))
    return data


def rfft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The FFT of the input.
    """
    data = ifftshift(data, dim=(-2, -1))
    data = torch.rfft(data, 2, normalized=True, onesided=False)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data, normalized=True):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform using torch.fft API.

    Args:
        data (torch.Tensor): Complex-valued input encoded with last dimension size 2 (real, imag).

    Returns:
        torch.Tensor: The IFFT of the input, encoded with last dimension size 2 (real, imag).
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data_c = _to_complex(data)
    norm = "ortho" if normalized else "backward"
    data_c = torch.fft.ifft2(data_c, norm=norm)
    data = _from_complex(data_c)
    data = fftshift(data, dim=(-3, -2))
    return data


def irfft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.

    Returns:
        torch.Tensor: The IFFT of the input.
    """
    data = ifftshift(data, dim=(-3, -2))
    data = torch.irfft(data, 2, normalized=True, onesided=False)
    data = fftshift(data, dim=(-2, -1))
    return data


def complex_to_mag_phase(data):
    """
    :param data (torch.Tensor): A complex valued tensor, where the size of the third last dimension should be 2
    :return: Mag and Phase (torch.Tensor): tensor of same size as input
    """

    assert data.size(-3) == 2
    mag = (data ** 2).sum(dim=-3).sqrt()
    phase = torch.atan2(data[:, 1, :, :], data[:, 0, :, :])
    return torch.stack((mag, phase), dim=-3)


def mag_phase_to_complex(data):
    """
    :param data (torch.Tensor): Mag and Phase (torch.Tensor):
    :return: A complex valued tensor, where the size of the third last dimension is 2
    """

    assert data.size(-3) == 2
    real = data[:, 0, :, :] * torch.cos(data[:, 1, :, :])
    imag = data[:, 0, :, :] * torch.sin(data[:, 1, :, :])
    return torch.stack((real, imag), dim=-3)


def partial_fourier(data):
    """
    :param data:
    :return:
    """



def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2 or data.size(-3) == 2
    return (data ** 2).sum(dim=-1).sqrt() if data.size(-1) == 2 else (data ** 2).sum(dim=-3).sqrt()


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


def normalize_volume(data, mean, std, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are provided and computed from volume.

        Args:
            data (torch.Tensor): Input data to be normalized
            mean: mean of whole volume
            std: std of whole volume
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    return normalize(data, mean, std, eps), mean, std


def normalize_complex(data, eps=0.):
    """
        Normalize the given complex tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from magnitude of data.

        Note that data is centered by complex mean so that the result centered data have average zero magnitude.

        Args:
            data (torch.Tensor): Input data to be normalized (*, 2)
            mean: mean of image magnitude
            std: std of image magnitude
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized complex tensor with 2 channels (*, 2)
        """
    mag = complex_abs(data)
    mag_mean = mag.mean()
    mag_std = mag.std()

    temp = mag_mean/mag

    mean_real = data[..., 0] * temp
    mean_imag = data[..., 1] * temp

    mean_complex = torch.stack((mean_real, mean_imag), dim=-1)

    stddev = mag_std

    return (data - mean_complex) / (stddev + eps), mag_mean, stddev


# Helper functions

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def complex_center_pad(data, shape):
    """
    Apply center zero-padding to a complex image tensor.

    Args:
        data (torch.Tensor): Complex input tensor of shape (..., H, W, 2).
        shape (tuple): Target shape (H_target, W_target). Must be >= input shape.

    Returns:
        torch.Tensor: Zero-padded tensor of shape (..., H_target, W_target, 2).
    """
    h_in, w_in = data.shape[-3], data.shape[-2]
    h_out, w_out = shape
    assert h_out >= h_in and w_out >= w_in, "Target shape must be >= input shape"

    if h_out == h_in and w_out == w_in:
        return data

    # Create output tensor filled with zeros
    out_shape = list(data.shape)
    out_shape[-3] = h_out
    out_shape[-2] = w_out
    out = torch.zeros(out_shape, dtype=data.dtype, device=data.device)

    # Compute padding offsets for center placement
    h_start = (h_out - h_in) // 2
    w_start = (w_out - w_in) // 2

    out[..., h_start:h_start + h_in, w_start:w_start + w_in, :] = data
    return out


def apply_data_consistency(
    xhat: torch.Tensor,
    kspace: torch.Tensor,
    mask: torch.Tensor,
    norm_std: torch.Tensor,
    norm_scale: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    """
    Apply data consistency to predictions by enforcing measured k-space values.

    At measured k-space locations (mask=1), the original measurements are preserved.
    At unmeasured locations (mask=0), the predicted k-space values are used.

    NOTE: With the updated DataTransform, kspace and mask are now at target resolution
    (same as xhat), so no padding/cropping is needed. The function handles both cases
    for backwards compatibility.

    Args:
        xhat: Normalized prediction tensor of shape (B, 2, H, W).
              Channel 0 = real, Channel 1 = imaginary.
        kspace: K-space tensor of shape (B, H, W, 2) at target resolution.
        mask: Undersampling mask tensor, broadcastable to kspace shape.
              Typically (B, 1, W, 1) for column-wise undersampling.
              mask=1 means sampled (measured), mask=0 means not sampled.
        norm_std: Per-sample normalization factor (B,) - typically mean(|y|).
        norm_scale: Per-sample additional scaling factor (B,) - typically 1.0 or percentile scale.
        resolution: The target resolution (H = W = resolution).

    Returns:
        torch.Tensor: DC-corrected prediction of shape (B, 2, H, W), normalized.
    """
    batch_size = xhat.shape[0]
    device = xhat.device
    dtype = xhat.dtype

    # Ensure tensors are on the same device and have correct shapes
    kspace = kspace.to(device=device, dtype=dtype)
    mask = mask.to(device=device, dtype=dtype)

    # Reshape norm factors for broadcasting: (B,) -> (B, 1, 1, 1)
    if norm_std.ndim == 0:
        norm_std = norm_std.unsqueeze(0)
    if norm_scale.ndim == 0:
        norm_scale = norm_scale.unsqueeze(0)
    norm_std = norm_std.to(device=device, dtype=dtype).view(-1, 1, 1, 1)
    norm_scale = norm_scale.to(device=device, dtype=dtype).view(-1, 1, 1, 1)

    # Get original k-space dimensions
    h_orig, w_orig = kspace.shape[1], kspace.shape[2]
    h_crop, w_crop = resolution, resolution

    # Step 1: Unnormalize prediction
    # xhat is normalized as: xhat_norm = xhat_raw / norm_std / norm_scale
    # So: xhat_raw = xhat_norm * norm_scale * norm_std
    xhat_unnorm = xhat * norm_scale * norm_std  # (B, 2, H_crop, W_crop)

    # Step 2: Permute from (B, 2, H, W) to (B, H, W, 2) for FFT functions
    xhat_unnorm = xhat_unnorm.permute(0, 2, 3, 1)  # (B, H_crop, W_crop, 2)

    # Step 3: Zero-pad to original k-space size if cropped
    if h_crop != h_orig or w_crop != w_orig:
        xhat_padded = complex_center_pad(xhat_unnorm, (h_orig, w_orig))
    else:
        xhat_padded = xhat_unnorm

    # Step 4: FFT to k-space domain
    kspace_pred = fft2(xhat_padded)  # (B, H_orig, W_orig, 2)

    # Step 5: Apply data consistency
    # DC formula: kspace_dc = mask * kspace_original + (1 - mask) * kspace_pred
    # Where mask=1 means we have measurements, mask=0 means we don't
    kspace_dc = mask * kspace + (1.0 - mask) * kspace_pred

    # Step 6: IFFT back to image domain
    image_dc = ifft2(kspace_dc)  # (B, H_orig, W_orig, 2)

    # Step 7: Center-crop back to resolution
    if h_crop != h_orig or w_crop != w_orig:
        image_dc = complex_center_crop(image_dc, (h_crop, w_crop))

    # Step 8: Renormalize
    # Apply the same normalization that was applied to the original data
    image_dc = image_dc / (norm_std.permute(0, 2, 3, 1) * norm_scale.permute(0, 2, 3, 1))

    # Step 9: Permute back to (B, 2, H, W)
    image_dc = image_dc.permute(0, 3, 1, 2)  # (B, 2, H_crop, W_crop)

    return image_dc


# Alias for complex magnitude computation
complex_to_magnitude = complex_abs