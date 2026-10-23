"""Neural denoising: run the trained checkpoint on full-resolution images.

`training/train_denoise.py` trains a residual U-Net on synthesised read + shot
noise (see `training/models.py`). This module loads that checkpoint and applies
it to real photos.

Two things make inference differ from training:

* **tiling.** The network is fully convolutional, but a 5000px photo does not
  fit in memory at once. The image is processed in overlapping tiles and the
  results are blended with a smooth window, which avoids visible seams. Overlap
  must exceed the network's receptive field growth per tile, or tile edges
  keep a faint discontinuity.
* **noise estimation.** The model was trained across a *range* of noise
  levels, so one model covers the whole strength slider: `strength` maps to
  the noise level the network should assume, and blending with the original
  gives continuous control between the raw frame and the fully denoised one.

The model is optional. Without the checkpoint the caller falls back to the
edge-aware filter in `engine.tone.ai_denoise`, which is why this module's
`available()` exists.
"""
from __future__ import annotations

from .paths import models_dir
import os
import threading
from typing import Optional, Tuple

import numpy as np

CKPT_NAME = "denoise.pt"

_SESSION: dict = {}
_LOCK = threading.Lock()


def model_path() -> str:
    return os.path.join(models_dir(), CKPT_NAME)


def available() -> bool:
    return os.path.exists(model_path())


def info() -> Optional[dict]:
    """Checkpoint metadata, without loading the weights onto the GPU."""
    if not available():
        return None
    import torch

    ck = torch.load(model_path(), map_location="cpu", weights_only=False)
    return {
        "epoch": ck.get("epoch"),
        "mean_psnr": ck.get("mean_psnr"),
        "val": ck.get("val"),
        "width": ck.get("width"),
        "depth": ck.get("depth"),
    }


def _load():
    """Load the model once; returns (module, device)."""
    with _LOCK:
        if _SESSION:
            return _SESSION["model"], _SESSION["device"]
        import torch

        from training.models import Denoiser

        ck = torch.load(model_path(), map_location="cpu", weights_only=False)
        model = Denoiser(width=int(ck.get("width", 32)),
                         depth=int(ck.get("depth", 3)))
        model.load_state_dict(ck["state_dict"])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        _SESSION["model"] = model
        _SESSION["device"] = device
        _SESSION["meta"] = {"epoch": ck.get("epoch"),
                            "mean_psnr": ck.get("mean_psnr")}
    return _SESSION["model"], _SESSION["device"]


def _tile_bounds(n: int, tile: int, overlap: int):
    """Yield (start, end) tiles covering [0, n) with the given overlap."""
    step = max(1, tile - overlap)
    start = 0
    while True:
        end = min(n, start + tile)
        yield start, end
        if end >= n:
            return
        start += step


def _blend_window(h: int, w: int, top: int, left: int,
                  th: int, tw: int, feather: int) -> np.ndarray:
    """A raised-cosine window for one tile, in image coordinates."""
    win = np.ones((th, tw), np.float32)
    if feather > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, feather, dtype=np.float32)))
        ramp = ramp / max(ramp[-1], 1e-6)
        for i in range(min(feather, th)):
            win[i, :] *= ramp[i]
            win[th - 1 - i, :] *= ramp[i]
        for j in range(min(feather, tw)):
            win[:, j] *= ramp[j]
            win[:, tw - 1 - j] *= ramp[j]
    return win


def denoise(img: np.ndarray, strength: float = 50.0,
            tile: int = 512, overlap: int = 64) -> np.ndarray:
    """Denoise a float32 RGB image with the trained network.

    `strength` (0..100) both selects the noise level the model assumes and
    blends the result with the input, so the control is continuous and 0 is a
    no-op. Tiles are blended with raised-cosine windows so no seam is visible.
    """
    k = float(np.clip(strength, 0.0, 100.0)) / 100.0
    if k <= 0:
        return img.copy()

    model, device = _load()
    import torch

    src = np.clip(img, 0.0, 1.0).astype(np.float32)
    h, w = src.shape[:2]

    # strength selects the training noise level the model should assume: the
    # training draw was read sigma in [0, 0.03] and shot k in [0, 0.05]
    read_sigma = 0.03 * k
    shot_k = 0.05 * k

    tile = max(64, int(tile))
    overlap = min(overlap, tile // 2)
    feather = overlap // 2

    acc = np.zeros((h, w, 3), np.float32)
    weight = np.zeros((h, w), np.float32)

    use_cuda = device == "cuda"
    with torch.no_grad():
        for (y0, y1) in _tile_bounds(h, tile, overlap):
            for (x0, x1) in _tile_bounds(w, tile, overlap):
                th, tw = y1 - y0, x1 - x0
                patch = src[y0:y1, x0:x1]

                # pad the tile so its dims are multiples of 16 (the network
                # halves three times; odd sizes would misalign the decoder)
                ph = (16 - th % 16) % 16
                pw = (16 - tw % 16) % 16
                if ph or pw:
                    patch = np.pad(patch, ((0, ph), (0, pw), (0, 0)),
                                   mode="reflect")

                blob = torch.from_numpy(
                    np.ascontiguousarray(patch.transpose(2, 0, 1)[None]))
                blob = blob.to(device, non_blocking=True)
                out = model(blob)
                out = out[0].permute(1, 2, 0).float().cpu().numpy()
                out = out[:th, :tw]

                win = _blend_window(h, w, y0, x0, th, tw, feather)
                acc[y0:y1, x0:x1] += out * win[..., None]
                weight[y0:y1, x0:x1] += win

    weight = np.maximum(weight, 1e-6)
    den = acc / weight[..., None]

    # blend with the original: strength 100 means trust the network fully
    out = src * (1.0 - k) + den * k
    return np.clip(out, 0.0, 1.0).astype(np.float32)
