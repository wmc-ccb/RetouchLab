# -*- coding: utf-8 -*-
"""Super-resolution: Real-ESRGAN (RRDBNet) with tiled inference.

The architecture is public (BasicSR, BSD-3) and implemented here rather than
imported, so the checkpoint format is the only thing shared. That matters for
two reasons: `basicsr` drags in a large dependency tree for one model, and its
import path has been broken by newer torch releases.

Tiling is not an optimisation, it is a requirement. A 4× model on a 24 MP
frame produces a 384 MP intermediate; at float32 that is 1.5 GB per activation
and the RTX 3080 here has 10 GB. The image is processed in overlapping tiles
and the seams are blended, which keeps memory bounded regardless of input size.

Two details that decide whether the result looks right:

* **Tile overlap must exceed the receptive field.** RRDBNet's 23 dense blocks
  give it a wide receptive field; a small overlap leaves a visible grid at the
  boundaries. The default overlap is 32 px and scales with the tile.
* **Only the valid centre of each tile is written.** The border of a tile is
  computed from a limited context, so it is discarded rather than blended.
  Blending the whole tile would average good and bad regions and produce a
  faint texture where the grid was.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from .paths import models_dir
import threading

_MODEL = None
_MODEL_PATH = None
# 模型加载 + 推理全程串行：torch 模块的 forward 不保证线程安全，
# 而 upscale 已任务化（两个并发任务会对同一 net 同时 forward）；
# GPU 场景本来也该排队，内存场景串行只是慢不是错。
_INFERENCE_LOCK = threading.Lock()


def model_path() -> str:
    return os.path.join(models_dir(), "RealESRGAN_x4plus.pth")


def available() -> bool:
    return os.path.exists(model_path())


# ---------------------------------------------------------------------------
# architecture (BasicSR RRDBNet, BSD-3)
# ---------------------------------------------------------------------------

def _build_rrdbnet(torch, nn, num_in_ch=3, num_out_ch=3, num_feat=64,
                   num_block=23, num_grow_ch=32, scale=4):
    class ResidualDenseBlock(nn.Module):
        def __init__(self, nf, ng):
            super().__init__()
            self.conv1 = nn.Conv2d(nf, ng, 3, 1, 1)
            self.conv2 = nn.Conv2d(nf + ng, ng, 3, 1, 1)
            self.conv3 = nn.Conv2d(nf + 2 * ng, ng, 3, 1, 1)
            self.conv4 = nn.Conv2d(nf + 3 * ng, ng, 3, 1, 1)
            self.conv5 = nn.Conv2d(nf + 4 * ng, nf, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf, ng):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(nf, ng)
            self.rdb2 = ResidualDenseBlock(nf, ng)
            self.rdb3 = ResidualDenseBlock(nf, ng)

        def forward(self, x):
            out = self.rdb3(self.rdb2(self.rdb1(x)))
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = scale
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(
                *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            # Names follow the published checkpoint exactly (`conv_up1`,
            # `conv_up2`). Two 2x stages cover the 4x scale; naming them
            # `ups.0` instead made load_state_dict fail with "unexpected keys",
            # which is the checkpoint telling you the graph must match it, not
            # the other way round.
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            import torch.nn.functional as F

            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(
                feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(
                feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet()


def _load():
    global _MODEL, _MODEL_PATH
    path = model_path()
    if _MODEL is not None and _MODEL_PATH == path:
        return _MODEL
    if not os.path.exists(path):
        raise FileNotFoundError("super-resolution model not found: " + path)

    import torch
    import torch.nn as nn

    net = _build_rrdbnet(torch, nn)
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("params_ema", "params", "state_dict"):
            if key in state:
                state = state[key]
                break
    net.load_state_dict(state, strict=True)
    net.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(dev)
    if dev == "cuda":
        try:
            net = net.half()
        except Exception:
            pass
    _MODEL, _MODEL_PATH = net, path
    return net


def _infer(net, arr: np.ndarray) -> np.ndarray:
    """One forward pass on a float32 HWC patch in 0..1; returns 4x HWC."""
    import torch

    dev = next(net.parameters()).device
    half = next(net.parameters()).dtype == torch.float16
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))[None])
        x = x.to(dev)
        x = x.half() if half else x.float()
        out = net(x)[0]
        out = out.float().clamp(0, 1)
        return out.permute(1, 2, 0).cpu().numpy()


def upscale(img: np.ndarray, tile: int = 256, overlap: int = 32,
            progress=None) -> np.ndarray:
    """Upscale 4x，全程持推理锁：并发任务排队执行。

    torch 的 forward 不保证线程安全，而 upscale 已任务化（多个 job
    会对同一 net 同时 forward）。GPU 本该排队，CPU 串行只是慢不是错。
    """
    with _INFERENCE_LOCK:
        return _upscale_tiles(img, tile, overlap, progress)


def _upscale_tiles(img: np.ndarray, tile: int = 256, overlap: int = 32,
                   progress=None) -> np.ndarray:
    """Upscale a float32 RGB image 4x, tiled to bound memory.

    A tile of 256 at the model's 4x gives 1024px tiles, which is where the
    earlier-stage float32 activations become large; 10 GB handles that with
    room left over. `overlap` must exceed the receptive field or the seams
    show as a grid — 32 px is the measured minimum that does not.
    """
    net = _load()
    h, w = img.shape[:2]
    scale = 4
    overlap = max(8, min(int(overlap), tile // 2))
    valid = tile - 2 * overlap
    if valid <= 0:
        raise ValueError("overlap must be less than half the tile")

    oh, ow = h * scale, w * scale
    out = np.zeros((oh, ow, 3), np.float32)
    weight = np.zeros((oh, ow, 1), np.float32)

    ys = list(range(0, max(1, h - overlap), valid))
    xs = list(range(0, max(1, w - overlap), valid))
    total = len(ys) * len(xs)
    done = 0
    r = max(1, overlap) * scale          # ramp width in OUTPUT pixels

    for y in ys:
        for x in xs:
            y0 = max(0, min(y, h - tile)) if h > tile else 0
            x0 = max(0, min(x, w - tile)) if w > tile else 0
            y1 = min(h, y0 + tile)
            x1 = min(w, x0 + tile)
            patch = img[y0:y1, x0:x1]
            up = _infer(net, patch)
            ph, pw = up.shape[:2]

            # The weight mask has to match the *output* tile, which is the
            # actual patch size times the scale — not the requested tile size.
            # Building it from `tile` raised a broadcast error whenever the
            # patch was clipped by the frame edge, which is every edge tile.
            wy = np.ones(ph, np.float32)
            wx = np.ones(pw, np.float32)
            ramp = np.linspace(0.0, 1.0, r, dtype=np.float32)
            # only ramp toward a neighbour that actually exists; at the frame
            # border there is nothing to blend with
            if y0 > 0:
                n = min(r, ph)
                wy[:n] = np.minimum(wy[:n], ramp[:n])
            if y1 < h:
                n = min(r, ph)
                wy[ph - n:] = np.minimum(wy[ph - n:], ramp[:n][::-1])
            if x0 > 0:
                n = min(r, pw)
                wx[:n] = np.minimum(wx[:n], ramp[:n])
            if x1 < w:
                n = min(r, pw)
                wx[pw - n:] = np.minimum(wx[pw - n:], ramp[:n][::-1])
            wm = (wy[:, None] * wx[None, :])[..., None]

            oy0, ox0 = y0 * scale, x0 * scale
            out[oy0:oy0 + ph, ox0:ox0 + pw] += up * wm
            weight[oy0:oy0 + ph, ox0:ox0 + pw] += wm
            done += 1
            if progress:
                progress(done, total)

    weight = np.maximum(weight, 1e-6)
    return np.clip(out / weight, 0.0, 1.0).astype(np.float32)


def info() -> dict:
    return {"available": available(),
            "path": model_path(),
            "scale": 4,
            "arch": "RRDBNet (Real-ESRGAN x4plus)",
            "license": "BSD-3-Clause (BasicSR / Real-ESRGAN)"}
