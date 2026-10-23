"""Smoke-test the denoiser training path on whatever images are available.

Verifies the model runs, the loss decreases, validation PSNR improves over
the untrained network, and the checkpoint round-trips. Uses a handful of
local images so it completes in about a minute.
"""
import os
import sys
import shutil
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.models import Denoiser, FlatFieldLoss, add_realistic_noise
from training.data import DenoisePatches, SyntheticDenoise, load_rgb
from torch.utils.data import DataLoader
import torch.nn.functional as F

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_data_dir() -> str:
    """Collect whatever real images exist locally into one directory."""
    tmp = tempfile.mkdtemp(prefix="denoise-smoke-")
    cands = []
    for sub in ("samples", "out"):
        d = os.path.join(root, sub)
        if os.path.isdir(d):
            for n in os.listdir(d):
                if n.lower().endswith((".jpg", ".png")):
                    cands.append(os.path.join(d, n))
    if len(cands) < 4:
        # synthesise a few textured images so the test is self-contained
        for i in range(6):
            h, w = 512, 512
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            base = 0.5 + 0.25 * np.sin(xx / (17 + i * 3)) * np.cos(yy / (13 + i * 2))
            rng = np.random.default_rng(i)
            fine = rng.standard_normal((h, w)).astype(np.float32)
            fine = np.clip(fine, -3, 3) * 0.05
            img = np.stack([base + fine, base * 0.95 + fine, base * 0.9 + fine], axis=2)
            img = np.clip(img, 0, 1)
            p = os.path.join(tmp, f"syn{i}.png")
            import cv2
            cv2.imwrite(p, cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    else:
        for p in cands:
            shutil.copy(p, os.path.join(tmp, os.path.basename(p)))
    return tmp


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse)


def main():
    data = make_data_dir()
    n_imgs = len([f for f in os.listdir(data) if f.lower().endswith((".jpg", ".png"))])
    print(f"data dir: {data}  ({n_imgs} images)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    patch, batch = 64, 8
    patches = DenoisePatches(data, patch=patch, samples_per_epoch=8 * batch, seed=0)
    ds = SyntheticDenoise(patches, seed=0)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, drop_last=True)

    model = Denoiser(width=16, depth=2).to(device)
    print(f"parameters: {model.param_count():,}")

    # fixed validation set
    val = []
    for f in sorted(os.listdir(data))[:4]:
        if f.lower().endswith((".jpg", ".png")):
            img = load_rgb(os.path.join(data, f))
            h, w = img.shape[:2]
            ch, cw = min(96, h), min(96, w)
            crop = img[(h-ch)//2:(h-ch)//2+ch, (w-cw)//2:(w-cw)//2+cw]
            val.append(torch.from_numpy(np.ascontiguousarray(crop.transpose(2,0,1))).unsqueeze(0).to(device))

    def val_psnr():
        model.eval()
        scores = []
        with torch.no_grad():
            for clean in val:
                rt = torch.tensor(0.014, device=device).view(1,1,1,1)
                st = torch.tensor(0.024, device=device).view(1,1,1,1)
                noisy = add_realistic_noise(clean, rt, st).clamp(0,1)
                pred = model(noisy).clamp(0,1)
                scores.append(psnr(pred, clean))
        model.train()
        return float(np.mean(scores))

    before = val_psnr()
    print(f"untrained PSNR: {before:.2f} dB")

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    crit = FlatFieldLoss()
    steps = 60
    losses = []
    model.train()
    it = iter(dl)
    for i in range(steps):
        try:
            noisy, clean = next(it)
        except StopIteration:
            it = iter(dl); noisy, clean = next(it)
        noisy, clean = noisy.to(device), clean.to(device)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(noisy), clean)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (i+1) % 20 == 0:
            print(f"  step {i+1:3d}/{steps}  loss={np.mean(losses[-20:]):.5f}")

    after = val_psnr()
    print(f"trained PSNR  : {after:.2f} dB  (gain {after-before:+.2f})")

    # checkpoint round-trip
    ck = os.path.join(root, "models", "_smoke_denoise.pt")
    torch.save({"state_dict": model.state_dict(), "width": 16, "depth": 2}, ck)
    m2 = Denoiser(width=16, depth=2).to(device)
    sd = torch.load(ck, map_location=device, weights_only=False)
    m2.load_state_dict(sd["state_dict"])
    m2.eval()
    with torch.no_grad():
        a = model(val[0]).cpu().numpy()
        b = m2(val[0]).cpu().numpy()
    same = float(np.abs(a-b).max())
    os.remove(ck)
    print(f"checkpoint round-trip max diff: {same:.2e}")

    ok = (np.mean(losses[-20:]) < np.mean(losses[:20])) and after > before and same < 1e-5
    print("\nRESULT:", "PASS" if ok else "FAIL")
    shutil.rmtree(data, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
