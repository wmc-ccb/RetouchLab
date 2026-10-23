"""Train the denoiser.

Usage
-----
python -m training.train_denoise --data <dir-with-hr-images> --epochs 12

Trains on randomly-cropped clean patches with synthesised read + shot noise,
validates on held-out images at fixed noise levels so the reported PSNR is
comparable between epochs, and writes a checkpoint that `engine.nn` loads.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.data import DenoisePatches, SyntheticDenoise, list_images, load_rgb
from training.models import Denoiser, FlatFieldLoss, add_realistic_noise


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


@torch.no_grad()
def validate(model, images, device, levels, crop: int = 256) -> dict:
    """PSNR at fixed noise levels on fixed crops, so epochs are comparable."""
    model.eval()
    out = {}
    for name, (read, shot) in levels.items():
        scores = []
        for img in images:
            h, w = img.shape[:2]
            ch = min(crop, h)
            cw = min(crop, w)
            y = (h - ch) // 2
            x = (w - cw) // 2
            clean = torch.from_numpy(
                np.ascontiguousarray(img[y:y + ch, x:x + cw].transpose(2, 0, 1))
            ).unsqueeze(0).to(device)
            read_t = torch.tensor(read, device=device).view(1, 1, 1, 1)
            shot_t = torch.tensor(shot, device=device).view(1, 1, 1, 1)
            noisy = add_realistic_noise(clean, read_t, shot_t).clamp(0, 1)
            pred = model(noisy).clamp(0, 1)
            scores.append(psnr(pred, clean))
        out[name] = float(np.mean(scores))
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory of clean HR images")
    ap.add_argument("--val", default=None, help="directory of held-out images")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=600, help="steps per epoch")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="models/denoise.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}")

    # ---- data ----------------------------------------------------------
    patches = DenoisePatches(args.data, patch=args.patch,
                             samples_per_epoch=args.steps * args.batch,
                             seed=args.seed)
    ds = SyntheticDenoise(patches, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=(device == "cuda"),
                    drop_last=True, persistent_workers=args.workers > 0)
    print(f"train images: {len(patches.paths)}")

    val_dir = args.val or args.data
    val_paths = list_images(val_dir)[:8]
    val_imgs = [load_rgb(p) for p in val_paths]
    print(f"val images  : {len(val_imgs)}")

    levels = {
        "low":  (0.006, 0.010),
        "mid":  (0.014, 0.024),
        "high": (0.024, 0.042),
    }

    # ---- model ---------------------------------------------------------
    model = Denoiser(width=args.width, depth=args.depth).to(device)
    print(f"parameters: {model.param_count():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(dl),
        pct_start=0.15, anneal_strategy="cos")
    crit = FlatFieldLoss(grad_weight=0.08).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    history = []
    best = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        run = 0.0
        n = 0
        for noisy, clean in dl:
            noisy = noisy.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                pred = model(noisy)
                loss = crit(pred, clean)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()
            n += 1

        v = validate(model, val_imgs, device, levels)
        mean_v = float(np.mean(list(v.values())))
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{args.epochs}  loss={run/max(n,1):.5f}  "
              f"psnr low={v['low']:.2f} mid={v['mid']:.2f} high={v['high']:.2f}  "
              f"mean={mean_v:.2f}  {dt:.0f}s")

        history.append({"epoch": epoch, "loss": run / max(n, 1),
                        "val": v, "mean_psnr": mean_v, "seconds": dt})

        if mean_v > best:
            best = mean_v
            torch.save({
                "state_dict": model.state_dict(),
                "width": args.width,
                "depth": args.depth,
                "epoch": epoch,
                "mean_psnr": mean_v,
                "val": v,
            }, args.out)
            print(f"  saved {args.out} (best mean PSNR {best:.2f})")

    with open(args.out + ".history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=1)
    print(f"done. best mean PSNR {best:.2f}")


if __name__ == "__main__":
    main()
