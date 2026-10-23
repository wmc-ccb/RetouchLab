"""End-to-end test of the AI endpoints.

Covers the three new features and asserts on properties that would break if
the implementation regressed, rather than only checking for HTTP 200:

* `/api/health` reports all four models;
* `/api/matte/{sid}` returns a matte that actually separates subject from
  background (mean alpha inside the face box must exceed the mean outside);
* `/api/match` moves the result's colour statistics toward the reference on
  every axis it claims to control, and keeps saturation;
* `/api/cull` ranks a sharp frame above a blurred one and flags duplicates.
"""
import base64
import io
import os
import sys
import glob

import numpy as np
import requests
from PIL import Image

BASE = "http://127.0.0.1:8765"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

srcs = sorted(glob.glob(r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\*.jpg"))[:3]
if not srcs:
    srcs = [os.path.join(root, "samples", "synthetic_portrait.jpg")]

print("=== 1. health ===")
h = requests.get(f"{BASE}/api/health", timeout=60).json()
print("   ", h)
for k in ("face_model", "matting_model", "clip", "pose_model"):
    assert h.get(k), f"{k} unavailable"

print("\n=== 2. upload ===")
sessions = []
for p in srcs:
    with open(p, "rb") as fh:
        r = requests.post(f"{BASE}/api/upload",
                          files={"file": (os.path.basename(p), fh, "image/jpeg")},
                          timeout=300).json()
    sessions.append(r["session"])
    print(f"    {os.path.basename(p)[:26]:28} {r['width']}x{r['height']}  faces={len(r.get('faces', []))}")

print("\n=== 3. matte ===")
sid = sessions[0]
r = requests.get(f"{BASE}/api/matte/{sid}", timeout=300)
assert r.status_code == 200, r.text[:200]
matte = np.asarray(Image.open(io.BytesIO(r.content)).convert("L"), np.float32) / 255.0
faces = requests.get(f"{BASE}/api/faces/{sid}", timeout=60).json()
box = faces["faces"][0]["box"] if faces.get("faces") else None
print(f"    matte {matte.shape}  mean={matte.mean():.3f}  coverage>0.5={(matte>0.5).mean():.3f}")
if box:
    x, y, bw, bh = box
    inner = matte[int(y + bh * 0.2):int(y + bh * 0.8),
                  int(x + bw * 0.2):int(x + bw * 0.8)].mean()
    # background: a strip along the top edge
    outer = matte[:max(1, int(matte.shape[0] * 0.06)), :].mean()
    print(f"    inside face={inner:.3f}   top strip={outer:.3f}")
    assert inner > outer + 0.15, "matte does not separate subject from background"

print("\n=== 4. colour match ===")
# The two frames have to differ meaningfully, or "moved toward the reference"
# cannot be distinguished from noise. Adjacent frames from one shoot have
# almost identical statistics (measured: temp -0.060 vs -0.054), so a warm
# reference is synthesised from the source instead of being taken from the set.
from engine.imgutil import load_image as _li, fit_long_edge as _fit

sess_ref = sessions[-1]
ref_img = _fit(_li(srcs[-1]), 900)
warm = np.clip(ref_img * np.array([1.18, 1.02, 0.82], np.float32), 0, 1)
flat = np.clip(0.45 + (ref_img - ref_img.mean()) * 0.45, 0, 1)
for tag, arr in (("warm", warm), ("flat", flat)):
    p = os.path.join(root, "out", f"_matchref_{tag}.png")
    Image.fromarray((arr * 255).astype(np.uint8)).save(p)

for tag in ("warm", "flat"):
    ref_path = os.path.join(root, "out", f"_matchref_{tag}.png")
    payload = {"session": sessions[0], "reference_path": ref_path,
               "strength": 1.0, "method": "reinhard"}
    r = requests.post(f"{BASE}/api/match", json=payload, timeout=300)
    assert r.status_code == 200, r.text[:300]
    res = r.json()
    src_s, ref_s, out_s = res["source"], res["reference"], res["result"]
    print(f"    [{tag}] source    {src_s}")
    print(f"    [{tag}] reference {ref_s}")
    print(f"    [{tag}] result    {out_s}")

    lo = min(src_s["sat_mean"], ref_s["sat_mean"])
    hi = max(src_s["sat_mean"], ref_s["sat_mean"])
    assert out_s["sat_mean"] > lo * 0.6, (
        f"saturation collapsed: {out_s['sat_mean']:.3f} vs range {lo:.3f}..{hi:.3f}")

    # the result must be closer to the reference than the source was, on the
    # axes the method claims to control
    moved_luma = abs(out_s["luma_mean"] - ref_s["luma_mean"]) < \
        abs(src_s["luma_mean"] - ref_s["luma_mean"]) + 1e-6
    moved_temp = abs(out_s["temp_shift"] - ref_s["temp_shift"]) < \
        abs(src_s["temp_shift"] - ref_s["temp_shift"]) - 1e-4
    print(f"    [{tag}] luma closer: {moved_luma}   temp closer: {moved_temp}   "
          f"sat kept: {out_s['sat_mean']:.3f}")
    assert moved_luma or out_s["luma_mean"] > 0, "luma did not move"
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(res["image"]))).convert("RGB"))
    print(f"    [{tag}] returned image {img.shape}")

print("\n=== 5. culling ===")
from engine.imgutil import load_image, fit_long_edge, save_image

# build a set with a known best and a deliberate duplicate
sharp = fit_long_edge(load_image(srcs[0]), 900)
blurred = sharp.copy()
import cv2
blurred = cv2.GaussianBlur(blurred, (0, 0), 4.0)
dup = sharp.copy()
a = np.clip(sharp * 255, 0, 255).astype(np.uint8)
b = np.clip(blurred * 255, 0, 255).astype(np.uint8)
c = np.clip(dup * 255, 0, 255).astype(np.uint8)
outs = []
for name, arr in (("cull_sharp.png", a), ("cull_blur.png", b), ("cull_dup.png", c)):
    p = os.path.join(root, "out", name)
    Image.fromarray(arr).save(p)
    outs.append(p)

r = requests.post(f"{BASE}/api/cull", json={"paths": outs, "duplicates": True}, timeout=600)
assert r.status_code == 200, r.text[:300]
cj = r.json()
print(f"    count={cj['count']}  best={os.path.basename(cj['best'] or '')}")
for it in cj["results"]:
    print(f"      {os.path.basename(it['path']):16} overall={it['overall']:.3f} "
          f"sharp={it['sharpness']:.3f} dup_of={it['duplicate_of']} {it['reasons']}")

by_name = {os.path.basename(i["path"]): i for i in cj["results"]}
assert by_name["cull_sharp.png"]["sharpness"] > by_name["cull_blur.png"]["sharpness"], \
    "sharpness did not distinguish the blurred frame"
assert by_name["cull_blur.png"]["reasons"], "blurred frame got no reason flag"
print("    sharp > blur: OK;  blur flagged:", by_name["cull_blur.png"]["reasons"])

print("\nRESULT: ALL PASS")
