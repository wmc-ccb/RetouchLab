# -*- coding: utf-8 -*-
"""Validate the trained denoiser on synthesised noise.

The honest test: add noise with the *same model* the network was trained on
but a fresh random draw, then measure PSNR against the clean frame. A gain of
several dB at all three noise levels means the checkpoint really learned the
noise model, not the validation set.
"""
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.imgutil import load_image, save_image, fit_long_edge
from training.models import add_realistic_noise
from engine.nndenoise import denoise


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def main():
    img = fit_long_edge(load_image(
        os.path.join(r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰", "_DSC7550.jpg")), 800)
    clean = np.clip(img, 0, 1).astype(np.float32)
    print("image:", clean.shape[1], "x", clean.shape[0])
    print()
    print("%-6s %10s %10s %8s %8s" % ("level", "noisy PSNR", "denoised", "gain", "ms"))
    print("-" * 48)
    ok = True
    for label, rs, sk in [("low", 0.008, 0.012), ("mid", 0.016, 0.024),
                          ("high", 0.026, 0.042)]:
        t = torch.from_numpy(np.ascontiguousarray(clean.transpose(2, 0, 1)[None]))
        noisy = add_realistic_noise(t, torch.tensor(rs).view(1, 1, 1, 1),
                                    torch.tensor(sk).view(1, 1, 1, 1))
        noisy = noisy[0].permute(1, 2, 0).numpy().clip(0, 1).astype(np.float32)

        t0 = time.perf_counter()
        out = denoise(noisy, strength=90)
        dt = (time.perf_counter() - t0) * 1000

        p0, p1 = psnr(clean, noisy), psnr(clean, out)
        save_image(os.path.join(ROOT, "out", "denoise_%s.jpg" % label), out, 94)
        print("%-6s %10.2f %10.2f %+8.2f %8.0f" % (label, p0, p1, p1 - p0, dt))
        if p1 <= p0:
            ok = False

    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
