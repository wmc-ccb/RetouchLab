"""Smoke test: run the pipeline on a real image and verify the stages work."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from engine.imgutil import load_image, save_image
from engine.pipeline import RetouchPipeline, PRESETS, apply_preset
from engine.face import FaceDetector

SRC = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"

os.makedirs(OUT, exist_ok=True)
img = load_image(SRC)
print(f"loaded {SRC}  shape={img.shape}  dtype={img.dtype}  range=[{img.min():.3f},{img.max():.3f}]")

# --- detector -------------------------------------------------------------
det = FaceDetector()
faces = det.detect(img)
print(f"faces detected: {len(faces)}")
for i, f in enumerate(faces):
    print(f"  face {i}: box={[round(v,1) for v in f.box]} score={f.score:.3f}")
    FaceDetector.build_regions(f, img.shape)
    print(f"           eyes={[[round(v,1) for v in e] for e in f.eyes]} mouth={[round(v,1) for v in (f.mouth or [])]}")
    print(f"           skin_mask: max={f.skin_mask.max():.3f} mean={f.skin_mask.mean():.4f}")

# --- presets --------------------------------------------------------------
pipe = RetouchPipeline()
for name in PRESETS:
    t0 = time.perf_counter()
    res, info = pipe.run(img, apply_preset(name))
    dt = (time.perf_counter() - t0) * 1000
    diff = float(np.abs(res - img).mean())
    save_image(os.path.join(OUT, f"preset_{name}.jpg"), res, 92)
    print(f"  {name:10} {dt:7.0f} ms  mean|diff|={diff:.5f}  applied={len(info['applied'])}")

# --- individual stage checks ---------------------------------------------
print("\n--- stage sanity ---")
checks = {
    "exposure+":  {"exposure": 50},
    "exposure-":  {"exposure": -50},
    "contrast+":  {"contrast": 60},
    "shadows+":   {"shadows": 70},
    "highlights-":{"highlights": -70},
    "dehaze+":    {"dehaze": 60},
    "vibrance+":  {"vibrance": 70},
    "saturation-":{"saturation": -80},
    "clarity+":   {"clarity": 60},
    "sharp+":     {"sharpening": 80},
    "grain":      {"grain_amount": 60},
    "vignette":   {"vignetting": 60},
    "smooth":     {"smooth": 70},
    "acne":       {"spotless_acne": 80},
    "slim":       {"slimFace": 50},
    "eyes":       {"enlargeEye": 50},
}
for label, prm in checks.items():
    res, info = pipe.run(img, prm)
    d = float(np.abs(res - img).mean())
    finite = bool(np.isfinite(res).all())
    inrange = bool(res.min() >= 0.0 and res.max() <= 1.0)
    status = "OK " if (finite and inrange and d > 0) else "FAIL"
    print(f"  {status} {label:12} mean|diff|={d:.5f} finite={finite} in_range={inrange}")

# --- HSL bands ------------------------------------------------------------
print("\n--- HSL bands ---")
for band in ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]:
    res, _ = pipe.run(img, {f"hsl_{band}_sat": 60})
    print(f"  {band:8} mean|diff|={float(np.abs(res-img).mean()):.5f}")

# --- empty params must be a no-op ----------------------------------------
res, info = pipe.run(img, {})
d = float(np.abs(res - img).mean())
print(f"\nempty params -> mean|diff|={d:.8f}  (must be ~0)  stages={len(info['stages'])}")

print("\nsmoke test done")
