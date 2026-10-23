# -*- coding: utf-8 -*-
"""End-to-end test of the formula endpoints.

Runs a builtin single-step formula, a parametric one with slots, and the
composed three-step portrait formula, asserting on image change rather than
just HTTP 200. The composed run is the real test: it exercises the library
merge, slot propagation into sub-formulas, and three different runners.
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

src = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"

h = requests.get(BASE + "/api/health", timeout=60).json()
print("health:", h["ok"])

fl = requests.get(BASE + "/api/formulas", timeout=60).json()
print("\nformulas (%d):" % fl["count"])
for f in fl["formulas"]:
    print("  %-16s %2d steps  %s" % (f["name"], f["steps"], f["description"]))

with open(src, "rb") as fh:
    up = requests.post(BASE + "/api/upload",
                       files={"file": ("f.jpg", fh, "image/jpeg")},
                       timeout=300).json()
sid = up["session"]
orig = np.asarray(Image.open(io.BytesIO(base64.b64decode(up["preview"]))
                             ).convert("RGB"), np.float32) / 255.0
print("\nsession:", sid, up["width"], "x", up["height"])

failures = []


def run_formula(name, slots=None, label=None):
    r = requests.post(BASE + "/api/formulas/run",
                      json={"name": name, "session": sid, "slots": slots or {}},
                      timeout=900)
    if r.status_code != 200:
        print("  %-16s FAIL %s" % (label or name, r.text[:200]))
        failures.append(label or name)
        return None
    j = r.json()
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(j["image"]))
                                ).convert("RGB"), np.float32) / 255.0
    d = float(np.abs(img - orig).mean())
    print("  %-16s OK  applied=%s  |diff|=%.5f"
          % (label or name, j["applied"], d))
    return img, d


print("\n=== run ===")
r1 = run_formula("skin_prep")
if r1:
    check_img, d1 = r1
    if d1 < 1e-4:
        failures.append("skin_prep changed nothing")

r2 = run_formula("cinematic")
r3 = run_formula("corner_vignette", slots={"strength": 70}, label="corner_vignette(70)")
r4 = run_formula("portrait_full", label="portrait_full(composed)")
if r4 and r4[1] < 1e-3:
    failures.append("composed formula barely changed the image")

# parametric check: a different strength must give a different result
r5 = run_formula("corner_vignette", slots={"strength": 20}, label="corner_vignette(20)")
if r3 and r5:
    dd = float(np.abs(r3[0] - r5[0]).mean())
    print("  slot sensitivity: strength 70 vs 20 -> |diff| = %.5f" % dd)
    if dd < 1e-4:
        failures.append("slots did not change the result")

# error path: unknown formula
r = requests.post(BASE + "/api/formulas/run",
                  json={"name": "nope", "session": sid}, timeout=60)
print("  unknown formula -> %d (expected 404)" % r.status_code)
if r.status_code != 404:
    failures.append("unknown formula should 404")

print()
if failures:
    print("FAILED:", failures)
    sys.exit(1)
print("RESULT: ALL PASS")
