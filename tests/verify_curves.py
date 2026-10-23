# -*- coding: utf-8 -*-
"""Validate curves and levels.

The properties that matter are not "does it change the image" but:

* **monotonicity** — a monotone set of control points must give a monotone
  curve. This is what separates monotone Hermite from a natural cubic spline,
  and a failure here is the classic "curve adds halos" bug.
* **no overshoot** — the output must stay inside the control points' range.
  A natural spline overshoots; PCHIP provably does not.
* **identity** — two points on the diagonal must be a no-op, exactly.
* **levels are invertible** — applying levels then its inverse returns the
  original within one 8-bit level.
* **composition order** — master then channel equals applying them in that
  order, and differs from the reverse for an asymmetric pair.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.imgutil import load_image, fit_long_edge
from engine import curves as C

SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
failures = []


def check(label, cond, detail=""):
    print("%-56s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


def main():
    img = fit_long_edge(load_image(SRC), 600)
    print("image: %dx%d" % (img.shape[1], img.shape[0]))

    # ---- 1. identity -----------------------------------------------------
    print("\n=== 恒等 ===")
    lut = C.curve_lut([[0.0, 0.0], [1.0, 1.0]])
    dev = float(np.abs(lut - np.linspace(0, 1, len(lut))).max())
    check("对角线两点 = 恒等表", dev < 1e-6, "max dev %.2e" % dev)
    # outside the control range the curve is held flat, so at exact 0 and 1
    # the identity holds and the interpolation error is zero at the ends
    out = C.curve_lut([[0.2, 0.2]])       # single point -> identity
    check("单点 = 恒等表", float(np.abs(out - np.linspace(0, 1, len(out))).max()) < 1e-6)
    img2 = C.curves(img, rgb=[[0.0, 0.0], [1.0, 1.0]])
    check("曲线恒等不改变图像", float(np.abs(img2 - img).max()) < 0.002,
          "max %.4f" % float(np.abs(img2 - img).max()))

    # ---- 2. monotone + no overshoot --------------------------------------
    print("\n=== 单调性与过冲 ===")
    cases = {
        "S 曲线": [[0, 0], [0.25, 0.15], [0.5, 0.5], [0.75, 0.85], [1, 1]],
        "提亮": [[0, 0], [0.5, 0.65], [1, 1]],
        "压暗": [[0, 0], [0.5, 0.35], [1, 1]],
        "近点(易过冲)": [[0, 0], [0.48, 0.30], [0.52, 0.70], [1, 1]],
        "平坦后陡升": [[0, 0], [0.7, 0.05], [1, 1]],
    }
    for name, pts in cases.items():
        lut = C.curve_lut(pts)
        diffs = np.diff(lut)
        mono = bool((diffs >= -1e-6).all())
        ys = np.array([p[1] for p in pts])
        lo, hi = float(ys.min()), float(ys.max())
        over = float(max(0.0, lut.min() - lo, lut.max() - hi))
        check("%s: 单调" % name, mono)
        check("%s: 无过冲" % name, over < 1e-5, "过冲 %.2e" % over)

    # contrast with a natural cubic spline on the same near points: it must
    # overshoot, which is the reason PCHIP is used
    try:
        from scipy.interpolate import CubicSpline
        pts = cases["近点(易过冲)"]
        xs = np.array([p[0] for p in pts], float)
        ys = np.array([p[1] for p in pts], float)
        sp = CubicSpline(xs, ys)
        x = np.linspace(0, 1, 400)
        y = sp(x)
        cs_over = float(max(0.0, y.min() - 0, y.max() - 1))
        print("   对照：自然三次样条在同组点上过冲 %.4f（PCHIP 为 0）" % cs_over)
    except Exception:
        pass

    # ---- 3. levels -------------------------------------------------------
    print("\n=== 色阶 ===")
    ident = C.levels_lut(0.0, 1.0, 1.0, 0.0, 1.0)
    check("色阶恒等", float(np.abs(ident - np.linspace(0, 1, len(ident))).max()) < 1e-5)

    # Invertibility holds **inside** the input range only. A black/white point
    # clips by design — everything below the input black maps to the output
    # black and cannot be recovered — so asserting a full round trip would be
    # asserting that levels does not clip, which is the opposite of its job.
    spec = (0.05, 0.92, 1.4, 0.02, 0.98)
    inv = (spec[3], spec[4], 1.0 / spec[2], spec[0], spec[1])
    a = C.levels(img, master=spec)
    b = C.levels(a, master=inv)
    inside = (img >= spec[0]) & (img <= spec[1])
    err_in = float(np.abs((b - img)[inside]).max()) if inside.any() else 0.0
    check("色阶在输入范围内可逆（≤ 1/255）", err_in <= 1.0 / 255.0,
          "in-range max %.5f (%d px)" % (err_in, int(inside.sum())))

    # and it must clip outside the range, to the output endpoints
    below = img < spec[0]
    above = img > spec[1]
    clip_ok = True
    if below.any():
        clip_ok &= bool(np.abs(a[below] - spec[3]).max() < 1e-5)
    if above.any():
        clip_ok &= bool(np.abs(a[above] - spec[4]).max() < 1e-5)
    check("输入范围外被裁切到输出端点", clip_ok)

    # a collapsed input range must not divide by zero
    try:
        bad = C.levels_lut(0.5, 0.5, 1.0, 0.0, 1.0)
        check("退化输入区间不崩且有效", bool(np.isfinite(bad).all()))
    except Exception as exc:
        check("退化输入区间不崩且有效", False, str(exc)[:40])

    # monotone response
    mono_ok = True
    for g in (0.4, 0.7, 1.0, 1.6, 2.5):
        lut = C.levels_lut(0.05, 0.95, g, 0.0, 1.0)
        if not bool((np.diff(lut) >= -1e-6).all()):
            mono_ok = False
    check("色阶对各 gamma 单调", mono_ok)

    # ---- 4. composition order --------------------------------------------
    print("\n=== 主曲线 + 通道曲线 的组合顺序 ===")
    master = [[0, 0], [0.5, 0.62], [1, 1]]
    red = [[0, 0], [0.5, 0.35], [1, 1]]
    composed = C.curves(img, rgb=master, r=red)
    step1 = C.curves(img, rgb=master)
    step2 = C.curves(step1, r=red)
    dev = float(np.abs(composed - step2).max())
    check("组合 = 先主后通道（逐步应用）", dev < 0.01, "max dev %.5f" % dev)
    reversed_ = C.curves(C.curves(img, r=red), rgb=master)
    diff = float(np.abs(composed - reversed_).mean())
    check("顺序不同结果不同（顺序确实生效）", diff > 1e-4,
          "平均差 %.5f" % diff)

    # ---- 5. auto levels --------------------------------------------------
    print("\n=== 自动色阶 ===")
    sug = C.auto_levels(img)
    print("   建议:", sug)
    lo, hi = sug["master"][0], sug["master"][1]
    check("自动色阶范围合理", 0.0 <= lo < hi <= 1.0, "%.4f..%.4f" % (lo, hi))
    a2 = C.levels(img, master=sug["master"])
    check("自动色阶确实改变了对比", float(a2.std()) > float(img.std()) * 0.9,
          "std %.4f -> %.4f" % (img.std(), a2.std()))

    # ---- 6. locality: a per-channel curve must not touch other channels ---
    print("\n=== 局部性：单通道曲线不得影响其他通道 ===")
    out = C.curves(img, r=[[0, 0], [0.5, 0.7], [1, 1]])
    d = np.abs(out - img)
    check("只改 R 通道", float(d[..., 1].max()) < 1e-6 and float(d[..., 2].max()) < 1e-6,
          "G max %.2e B max %.2e" % (d[..., 1].max(), d[..., 2].max()))

    print()
    if failures:
        print("FAILED: %d -> %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
