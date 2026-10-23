"""Verify monotonic response for sliders whose direction matters."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from engine.imgutil import load_image
from engine.face import FaceDetector, skin_tone_mask
from engine import skin as S
from engine import tone as T
from engine import liquify as L

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img = load_image(os.path.join(root, "samples", "synthetic_portrait.jpg"))
det = FaceDetector()
faces = det.detect(img)
f = faces[0]
FaceDetector.build_regions(f, img.shape)
mask = skin_tone_mask(img, f)
print(f"faces={len(faces)} box={[round(v,1) for v in f.box]} mask_px={(mask>0.5).sum()}")

ok = True

print("\n=== blemish sensitivity must be monotonic increasing ===")
prev = -1
for sens in (10, 25, 40, 55, 70, 85, 100):
    bm = S.detect_blemishes(img, mask, sensitivity=sens)
    n = int((bm > 0.35).sum())
    mono = "OK" if n >= prev else "NON-MONOTONIC"
    if n < prev:
        ok = False
    print(f"  sens={sens:3}: detected_px={n:6}  {mono}")
    prev = n

print("\n=== acne strength must be monotonic increasing ===")
prev = -1
for amt in (0, 20, 40, 60, 80, 100):
    r = S.spotless(img, mask, acne=amt)
    d = float(np.abs(r - img).mean())
    mono = "OK" if d >= prev - 1e-9 else "NON-MONOTONIC"
    if d < prev - 1e-9:
        ok = False
    print(f"  acne={amt:3}: mean|diff|={d:.6f}  {mono}")
    prev = d

print("\n=== smooth strength must be monotonic increasing ===")
prev = -1
for amt in (0, 20, 40, 60, 80, 100):
    r = S.smooth_skin(img, mask, amount=amt, texture=50)
    d = float(np.abs(r - img).mean())
    mono = "OK" if d >= prev - 1e-9 else "NON-MONOTONIC"
    if d < prev - 1e-9:
        ok = False
    print(f"  smooth={amt:3}: mean|diff|={d:.6f}  {mono}")
    prev = d

print("\n=== exposure must be monotonic in both directions ===")
vals = []
for amt in (-100, -50, -25, 0, 25, 50, 100):
    r = T.exposure(img, amt)
    vals.append((amt, float(r.mean())))
prev = -1
for amt, m in vals:
    mono = "OK" if m >= prev else "NON-MONOTONIC"
    if m < prev:
        ok = False
    print(f"  exposure={amt:5}: mean={m:.4f}  {mono}")
    prev = m

print("\n=== contrast must increase spread ===")
for amt in (-50, 0, 50, 100):
    r = T.contrast(img, amt)
    print(f"  contrast={amt:5}: std={r.std():.4f} (orig {img.std():.4f})")

print("\n=== eye-bag reduction must lighten the under-eye area ===")
from engine.skin import eye_bag_reduction
for amt in (0, 40, 100):
    r = eye_bag_reduction(img, f, degree=amt, keep_silkworm=0)
    ex, ey, ew, eh = f.eyes[0]
    cx, cy = int(ex + ew/2), int(ey + eh/2 + 20)
    patch_o = img[cy-8:cy+8, cx-12:cx+12].mean()
    patch_r = r[cy-8:cy+8, cx-12:cx+12].mean()
    print(f"  degree={amt:3}: under-eye mean {patch_o:.4f} -> {patch_r:.4f}")

print("\n=== face slimming must narrow the face ===")
for amt in (0, 50, 100):
    r = L.slim_face(img, f, amount=amt)
    x, y, bw, bh = f.box
    row = int(y + bh * 0.62)
    # measure how many skin pixels remain across the face at jaw height
    from engine.face import detect_skin_global
    m = detect_skin_global(r)
    width = int((m[row] > 0.5).sum())
    print(f"  slimFace={amt:3}: jaw-row skin width = {width}")

print("\n=== teeth whitening must lighten the mouth ===")
from engine.skin import teeth_whiten
for amt in (0, 50, 100):
    r = teeth_whiten(img, f, brightness=amt, de_yellow=amt)
    mx, my, mw, mh = f.mouth
    cy, cx = int(my + mh/2), int(mx + mw/2)
    patch_o = img[cy-4:cy+4, cx-10:cx+10].mean()
    patch_r = r[cy-4:cy+4, cx-10:cx+10].mean()
    print(f"  whiten={amt:3}: teeth mean {patch_o:.4f} -> {patch_r:.4f}")

print("\nRESULT:", "ALL MONOTONIC" if ok else "MONOTONICITY VIOLATIONS FOUND")
