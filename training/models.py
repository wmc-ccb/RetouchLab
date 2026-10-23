"""A small residual denoiser for photographic noise.

Deliberately compact (a few hundred thousand parameters) so it trains in
minutes on a single consumer GPU and runs on CPU in a few tens of
milliseconds at preview resolution. The architecture is a standard
residual-prediction U-Net: the network predicts the noise, and the caller
subtracts it, which makes the identity mapping easy to learn.

Two noise models are supported because the two sources behave differently:

* **Gaussian** — sensor read noise, signal-independent.
* **Heteroscedastic** — shot noise, where the standard deviation grows with
  the square root of the signal. This matches real high-ISO files far better
  than a fixed sigma.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.norm = nn.GroupNorm(min(8, cout), cout)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Denoiser(nn.Module):
    """Residual U-Net denoiser.

    Parameters
    ----------
    width : base channel count; the encoder doubles it twice.
    depth : number of encoder stages (2 or 3 are plenty for this task).
    """

    def __init__(self, width: int = 32, depth: int = 3, in_ch: int = 3):
        super().__init__()
        self.depth = depth

        self.stem = ConvBlock(in_ch, width)

        # Encoder: each stage keeps `ch` then halves the resolution into 2*ch.
        # Record the channel count of every skip so the decoder can be built
        # against the exact same numbers instead of re-deriving them.
        self.enc = nn.ModuleList()
        skip_chs = []
        ch = width
        for _ in range(depth):
            skip_chs.append(ch)
            self.enc.append(nn.Sequential(
                ConvBlock(ch, ch),
                ConvBlock(ch, ch * 2, stride=2),
            ))
            ch *= 2

        self.mid = nn.Sequential(
            ConvBlock(ch, ch),
            ConvBlock(ch, ch),
        )

        # Decoder: consume (upsampled ch + skip_ch), emit skip_ch // 2, so the
        # output of the last stage matches the stem's width for the head.
        self.dec = nn.ModuleList()
        for skip_ch in reversed(skip_chs):
            out_ch = max(width, skip_ch // 2)
            self.dec.append(nn.Sequential(
                ConvBlock(ch + skip_ch, skip_ch),
                ConvBlock(skip_ch, out_ch),
            ))
            ch = out_ch

        self.head = nn.Conv2d(ch, in_ch, 3, 1, 1)
        # start as the identity: predicted noise is zero at initialisation
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """Return the denoised image (input minus predicted noise)."""
        h = self.stem(x)
        skips = []
        for enc in self.enc:
            skips.append(h)
            h = enc(h)

        h = self.mid(h)

        for dec, skip in zip(self.dec, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
            h = dec(torch.cat([h, skip], dim=1))

        noise = self.head(h)
        return x - noise

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def add_gaussian_noise(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Signal-independent noise; `sigma` is per-sample, shape (N,1,1,1)."""
    return x + torch.randn_like(x) * sigma


def add_shot_noise(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Signal-dependent noise: sigma = k * sqrt(signal).

    Models photon shot noise, which is what dominates in the mid-tones and
    highlights of a high-ISO capture.
    """
    sigma = k * torch.sqrt(torch.clamp(x, min=1e-4))
    return x + torch.randn_like(x) * sigma


def add_realistic_noise(x: torch.Tensor, read_sigma: torch.Tensor,
                        shot_k: torch.Tensor, chroma_boost: float = 1.6) -> torch.Tensor:
    """Read noise + shot noise, with extra chroma noise as real sensors show.

    Chroma noise is stronger than luma noise in most cameras because the
    colour channels are sampled less densely; modelling that makes the
    denoiser's behaviour match what a retoucher expects to remove.
    """
    y = add_gaussian_noise(x, read_sigma)
    y = add_shot_noise(y, shot_k)

    if chroma_boost != 1.0:
        # split luma/chroma, amplify chroma noise, recombine
        w = torch.tensor([0.2126, 0.7152, 0.0722], device=x.device,
                         dtype=x.dtype).view(1, 3, 1, 1)
        luma = (y * w).sum(dim=1, keepdim=True)
        chroma = y - luma
        chroma = chroma * chroma_boost
        y = luma + chroma
    return y


class FlatFieldLoss(nn.Module):
    """L1 reconstruction plus a gradient term that discourages blotching.

    Plain L1 on noisy targets leaves low-frequency mottling behind. Adding a
    penalty on the difference of gradients keeps flat areas flat, which is
    what makes a denoise result look clean rather than merely smoothed.
    """

    def __init__(self, grad_weight: float = 0.08):
        super().__init__()
        self.grad_weight = grad_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1 = F.l1_loss(pred, target)
        dp = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dt = target[:, :, 1:, :] - target[:, :, :-1, :]
        gv = F.l1_loss(dp, dt)
        dp2 = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dt2 = target[:, :, :, 1:] - target[:, :, :, :-1]
        gh = F.l1_loss(dp2, dt2)
        return l1 + self.grad_weight * (gv + gh)
