"""Training data: clean crops from a high-resolution image set.

Training patches are drawn from DIV2K's high-resolution images, which are
2K-resolution photographs covering portraits, landscapes, architecture and
texture. Only clean images are needed: noise is synthesised on the fly with a
known model, so the network learns to invert that model exactly.
"""
from __future__ import annotations

import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


def list_images(root: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    out = []
    for dirpath, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(exts):
                out.append(os.path.join(dirpath, n))
    out.sort()
    return out


def load_rgb(path: str) -> np.ndarray:
    """Load as float32 RGB in [0, 1], honouring a possible 16-bit source."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot decode {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32) / 255.0
    return img


class DenoisePatches(Dataset):
    """Random clean patches; noise is applied per sample in `__getitem__`.

    The noise level is drawn per sample from the configured range, so one
    model covers the whole strength slider instead of needing a separate
    network per ISO.
    """

    def __init__(self, root: str, patch: int = 128, samples_per_epoch: int = 2000,
                 read_range: Tuple[float, float] = (0.0, 0.03),
                 shot_range: Tuple[float, float] = (0.0, 0.05),
                 seed: int = 0):
        self.paths = list_images(root)
        if not self.paths:
            raise FileNotFoundError(f"no images under {root}")
        self.patch = patch
        self.n = samples_per_epoch
        self.read_range = read_range
        self.shot_range = shot_range
        self.rng = random.Random(seed)
        # cache a decoded copy of each image; DIV2K images are ~2K, so a full
        # cache would be large, but the working set is small in practice
        self._cache: dict = {}

    def __len__(self) -> int:
        return self.n

    def _image(self, idx: int) -> np.ndarray:
        p = self.paths[idx]
        if p not in self._cache:
            if len(self._cache) > 60:
                self._cache.clear()
            self._cache[p] = load_rgb(p)
        return self._cache[p]

    def __getitem__(self, i: int):
        # pick an image, retrying if the first is smaller than one patch
        for _ in range(10):
            img = self._image(self.rng.randrange(len(self.paths)))
            h, w = img.shape[:2]
            if h >= self.patch and w >= self.patch:
                break
        else:
            img = cv2.resize(img, (self.patch * 2, self.patch * 2),
                             interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]

        y = self.rng.randrange(0, h - self.patch + 1)
        x = self.rng.randrange(0, w - self.patch + 1)
        clean = img[y:y + self.patch, x:x + self.patch]

        # random flips and 90-degree rotations
        if self.rng.random() < 0.5:
            clean = clean[:, ::-1]
        if self.rng.random() < 0.5:
            clean = clean[::-1, :]
        k = self.rng.randrange(4)
        if k:
            clean = np.rot90(clean, k)

        clean = np.ascontiguousarray(clean.transpose(2, 0, 1))
        return torch.from_numpy(clean)


class SyntheticDenoise(Dataset):
    """Patch dataset that returns (noisy, clean) pairs with the noise applied."""

    def __init__(self, patches: DenoisePatches, seed: int = 0):
        self.p = patches
        self.gen = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.p)

    def __getitem__(self, i: int):
        from .models import add_realistic_noise

        clean = self.p[i]
        r = self.p.rng
        read = r.uniform(*self.p.read_range)
        shot = r.uniform(*self.p.shot_range)
        read_t = torch.tensor(read, dtype=torch.float32).view(1, 1, 1)
        shot_t = torch.tensor(shot, dtype=torch.float32).view(1, 1, 1)

        noisy = add_realistic_noise(clean.unsqueeze(0), read_t, shot_t)[0]
        return noisy.clamp(0.0, 1.0), clean
