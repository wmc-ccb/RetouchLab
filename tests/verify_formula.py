# -*- coding: utf-8 -*-
"""Validate the recipe engine.

Covers the four step kinds, slot binding, composition, persistence, and the
error paths that would otherwise surface half-way through a run. The image
math is already tested elsewhere; this is about orchestration behaving as
documented.
"""
import copy
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.formula import (FormulaError, delete, load_library, run,
                            save_library, upsert, validate)
from engine.imgutil import load_image, fit_long_edge

SRC = os.path.join(ROOT, "samples", "synthetic_portrait.jpg")


def main():
    img = fit_long_edge(load_image(SRC), 640)
    base_mean = float(img.mean())
    print("image: %dx%d  mean=%.4f" % (img.shape[1], img.shape[0], base_mean))

    failures = []

    def check(label, cond, detail=""):
        print("%-46s %s %s" % (label, "OK" if cond else "FAIL", detail))
        if not cond:
            failures.append(label)

    # ---- 1. validation catches malformed formulas ----------------------
    print("\n=== validation ===")
    errs = validate({"steps": [{"kind": "nope"}]})
    check("unknown kind rejected", any("unknown kind" in e for e in errs))
    errs = validate({"name": "x", "steps": [{"kind": "params"}]})
    check("params step without params rejected",
          any("needs a params object" in e for e in errs))
    errs = validate({"name": "x", "steps": [{"kind": "fill"}]})
    check("fill step without mask rejected",
          any("needs a mask" in e for e in errs))
    check("valid empty-params formula passes",
          not validate({"name": "x", "steps": [{"kind": "params", "params": {}}]}))

    # ---- 2. a params-only formula equals the pipeline ------------------
    print("\n=== params step ===")
    f1 = {"name": "warm", "steps": [
        {"kind": "params", "params": {"exposure": 10, "temperature": 0}}]}
    out1, info1 = run(f1, img)
    check("params formula changed the image",
          abs(float(out1.mean()) - base_mean) > 1e-4,
          "mean %.4f -> %.4f" % (base_mean, float(out1.mean())))
    check("applied records the step", info1["applied"] == ["params:0"])

    # ---- 3. slot binding -----------------------------------------------
    print("\n=== slots ===")
    f2 = {"name": "grade", "steps": [
        {"kind": "params",
         "params": {"grade_mid_hue": "{{mid_hue}}",
                    "grade_high_hue": "{{high_hue}}"}}]}
    out2, info2 = run(f2, img, slots={"mid_hue": 170, "high_hue": -30})
    try:
        run(f2, img, slots={})
        check("unbound slot raises", False)
    except FormulaError as e:
        check("unbound slot raises", "mid_hue" in str(e))

    # ---- 4. local step -------------------------------------------------
    print("\n=== local step ===")
    h, w = img.shape[:2]
    f3 = {"name": "brighten_face", "steps": [
        {"kind": "local", "strength": 100, "entries": [
            {"type": "radial", "cx": 0.5, "cy": 0.45, "rx": 0.18, "ry": 0.22,
             "exposure": 30}]}]}
    out3, _ = run(f3, img)
    centre = (slice(int(h * 0.38), int(h * 0.52)),
              slice(int(w * 0.42), int(w * 0.58)))
    d = np.abs(out3 - img).mean(axis=2)
    check("local brightens the centre more than the edge",
          d[centre].mean() > d.mean() * 1.5,
          "centre %.4f vs overall %.4f" % (d[centre].mean(), d.mean()))

    # ---- 5. composition ------------------------------------------------
    print("\n=== composition ===")
    lib = {"warm": f1, "grade": f2}
    f4 = {"name": "full_look", "steps": [
        {"kind": "formula", "formula": "warm"},
        {"kind": "formula", "formula": "grade",
         "slots": {"mid_hue": 175, "high_hue": -28}}]}
    out4, info4 = run(f4, img, library=lib)
    check("both sub-formulas ran",
          info4["applied"] == ["formula:warm", "formula:grade"],
          str(info4["applied"]))
    errs = validate({"name": "self", "steps": [{"kind": "formula",
                                                "formula": "self"}]},
                    {"self"})
    check("self-reference rejected", any("itself" in e for e in errs))

    # deep composition guard
    chain = {"name": "c0", "steps": [{"kind": "params", "params": {}}]}
    lib2 = {"c0": chain}
    for i in range(1, 12):
        lib2["c%d" % i] = {"name": "c%d" % i,
                           "steps": [{"kind": "formula", "formula": "c%d" % (i - 1)}]}
    try:
        run(lib2["c11"], img, library=lib2, depth=0)
        check("deep chain raises", False)
    except FormulaError as e:
        check("deep chain raises", "too deep" in str(e))

    # ---- 6. fill step via an injected runner ---------------------------
    print("\n=== fill step ===")
    calls = []

    def fake_fill(image, mask, prompt):
        calls.append((mask.shape, prompt))
        return image * 0.9

    f5 = {"name": "clear_left", "steps": [
        {"kind": "fill", "mask": {"type": "rect", "x": 0, "y": 0.3,
                                  "x2": 0.15, "y2": 0.9},
         "prompt": "tiled floor"}]}
    out5, info5 = run(f5, img, fill_runner=fake_fill)
    check("fill runner called once with a prompt",
          len(calls) == 1 and calls[0][1] == "tiled floor")
    check("fill darkened the frame", abs(float(out5.mean()) - base_mean) > 1e-3)

    # ---- 7. persistence -------------------------------------------------
    print("\n=== persistence ===")
    lib = upsert(copy.deepcopy(f1))
    check("upsert stores", "warm" in load_library())
    errs = validate({"name": "warm", "steps": [{"kind": "formula",
                                                "formula": "warm"}]}, set(lib))
    check("stored library is visible to validation", any("itself" in e for e in errs))
    delete("warm")
    check("delete removes", "warm" not in load_library())

    # ---- 8. JSON round-trip ---------------------------------------------
    print("\n=== serialisation ===")
    text = json.dumps(f4, ensure_ascii=False)
    back = json.loads(text)
    # f4 composes 'warm' and 'grade'; persistence tests above replaced the
    # on-disk library, so pass the in-memory pair explicitly
    out6, _ = run(back, img, library={"warm": f1, "grade": f2})
    check("JSON round-trip runs", float(out6.mean()) != base_mean)

    print()
    if failures:
        print("FAILED: %d checks: %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
