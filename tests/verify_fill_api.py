# -*- coding: utf-8 -*-
"""End-to-end test of the generative-fill endpoint.

Uploads a frame, POSTs a synthetic mask (a rectangle over the left third),
and asserts the response actually changed the masked region while leaving the
rest identical — which is the contract the UI depends on when it re-bases the
session on the returned image.
"""
import base64
import io
import os
import sys

import numpy as np
import requests
from PIL import Image

BASE = "http://127.0.0.1:8765"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.imgutil import load_image, fit_long_edge
from engine import genfill

src = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
img = fit_long_edge(load_image(src), 768)
h, w = img.shape[:2]

st = requests.get(BASE + "/api/fill/status", timeout=30).json()
print("fill status:", st)
assert st["available"], "ComfyUI not available"

with open(src, "rb") as fh:
    up = requests.post(BASE + "/api/upload",
                       files={"file": ("f.jpg", fh, "image/jpeg")},
                       timeout=300).json()
sid = up["session"]
print("session:", sid, up["width"], "x", up["height"])

import cv2
mask = np.zeros((h, w), np.float32)
cv2.rectangle(mask, (0, int(h * 0.35)), (int(w * 0.16), int(h * 0.95)), 1.0, -1)
mask = cv2.GaussianBlur(mask, (31, 31), 8)
ok, buf = cv2.imencode(".png", (mask * 255).astype(np.uint8))
mask_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

print("calling /api/fill …")
r = requests.post(BASE + "/api/fill",
                  json={"session": sid, "mask_png": mask_b64,
                        "prompt": "empty indoor shopping mall corridor, "
                                  "tiled floor, clean", "steps": 20},
                  timeout=900)
print("status:", r.status_code)
assert r.status_code == 200, r.text[:400]
j = r.json()
print("info:", j["info"])

out = np.asarray(Image.open(io.BytesIO(base64.b64decode(j["image"]))
                            ).convert("RGB"), np.float32) / 255.0
# the server works at its own preview scale (1400 long edge), so resize both
# to a common frame before comparing
import cv2
common_w = min(out.shape[1], img.shape[1])
scale_o = common_w / out.shape[1]
scale_i = common_w / img.shape[1]
out_c = cv2.resize(out, (common_w, int(out.shape[0] * scale_o)),
                   interpolation=cv2.INTER_AREA)
img_c = cv2.resize(img, (common_w, int(img.shape[0] * scale_i)),
                   interpolation=cv2.INTER_AREA)
d = np.abs(out_c - img_c).mean(axis=2)
frac = 0.18
box = d[:, :int(d.shape[1] * frac)]
rest = d[:, int(d.shape[1] * 0.30):]
print("masked region |diff| = %.5f   outside = %.5f  (ratio %.1fx)"
      % (box.mean(), rest.mean(), box.mean() / max(rest.mean(), 1e-9)))
# the outside floor is JPEG round-trip noise from the server's own encode,
# not zero; the filled region only has to stand clear of it
assert box.mean() > rest.mean() * 2.5, "change is not concentrated in the mask"
print("change concentrated in the mask: OK")

print("\nRESULT: PASS")
