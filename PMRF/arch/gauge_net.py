"""
Lightweight residual U-Net for the ENGRF gauge displacement field W_psi.

The gauge network learns a displacement W_psi(Z_t; Y) such that the
linearized gauge transform is:
    Z_tilde_t = Z_t + alpha(t) * W_psi(Z_t; Y)
with alpha(0) = alpha(1) = 0 (endpoint neutrality).

Design choices:
  - Input: concat(Z_t, y) → 4 channels (complex real+imag for both).
  - Output: 2 channels (displacement in real+imag space).
  - Output conv is **zero-initialized** so the gauge starts as identity.
  - Small parameter count (~0.5M at base_channels=32, num_levels=3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Conv-GroupNorm-SiLU-Conv residual block."""

    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 8):
        super().__init__()
        # Ensure num_groups divides channel count
        gn1 = min(num_groups, in_ch)
        gn2 = min(num_groups, out_ch)

        self.block = nn.Sequential(
            nn.GroupNorm(gn1, in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(gn2, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.block(x)


class GaugeNet(nn.Module):
    """
    Lightweight encoder-bottleneck-decoder U-Net for the gauge displacement.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 4 = 2 for Z_t + 2 for y).
    out_channels : int
        Number of output channels (default 2 = displacement in real+imag).
    base_channels : int
        Channel width of the first level (doubled at each subsequent level).
    num_levels : int
        Number of encoder/decoder levels (including bottleneck).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 2,
        base_channels: int = 32,
        num_levels: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Build channel schedule: [base, base*2, base*4, ...]
        channels = [base_channels * (2 ** i) for i in range(num_levels)]

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_ch = in_channels
        for ch in channels[:-1]:  # all but bottleneck
            self.encoder_blocks.append(ResBlock(prev_ch, ch))
            self.downsamples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            prev_ch = ch

        # --- Bottleneck ---
        self.bottleneck = ResBlock(prev_ch, channels[-1])

        # --- Decoder ---
        self.upsamples = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        prev_ch = channels[-1]
        for ch in reversed(channels[:-1]):
            self.upsamples.append(nn.ConvTranspose2d(prev_ch, ch, 2, stride=2))
            # After concat with skip: ch (upsample) + ch (skip) = 2*ch
            self.decoder_blocks.append(ResBlock(ch * 2, ch))
            prev_ch = ch

        # --- Output (zero-initialized for identity start) ---
        self.out_conv = nn.Conv2d(prev_ch, out_channels, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, H, W)
            Concatenation of Z_t and y along channel dim.

        Returns
        -------
        W : (B, out_channels, H, W)
            Gauge displacement field.
        """
        skips = []
        h = x
        for enc, down in zip(self.encoder_blocks, self.downsamples):
            h = enc(h)
            skips.append(h)
            h = down(h)

        h = self.bottleneck(h)

        for up, dec, skip in zip(self.upsamples, self.decoder_blocks, reversed(skips)):
            h = up(h)
            # Handle potential size mismatch from odd spatial dims
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        return self.out_conv(h)
