# -*- coding: utf-8 -*-
"""Validate the colour wheel and grey-point picker.

The first version of this test had three flawed metrics, all of which looked
like engine bugs:

* **family leakage** compared the *mean hue of pixels inside a window* before
  and after a rotation. Rotation moves pixels out of the window, so the
  population changes and the two means are not comparable. The correct check
  is per-pixel: track the same pixels and measure how far each moved.
* **saturation monotonicity** used the whole-frame green-channel mean, which
  shifts as pixels migrate between families. Fixed-pixel measurement is the
  meaningful one.
* **the grey-point picker** selected a patch at the frame edge, where the
  radius clipped to an empty slice. The picker position must be clamped
  inward by the patch radius.
"""
import os
import sys

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.imgutil import load_image, fit_long_edge, save_image, \
    srgb_to_linear, linear_to_srgb
from engine.wheel import HUE_CENTRES, grey_point, select_color

SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"

failures = []


def check(label, cond, detail=""):
    print("%-56s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


def hue_shift_per_pixel(before: np.ndarray, after: np.ndarray,
                        family: str) -> np.ndarray:
    """Per-pixel hue displacement among pixels that start in the family.

    Hue is circular, so the displacement is wrapped to [-180, 180]: a pixel
    that rotates across the 0/360 seam moved +10 degrees, not -350. Forgetting
    the wrap produced shifts of -64000 degrees and made a correct engine look
    broken.
    """
    hb = cv2.cvtColor(np.clip(before, 0, 1).astype(np.float32),
                      cv2.COLOR_RGB2HSV)[..., 0] * 360.0
    ha = cv2.cvtColor(np.clip(after, 0, 1).astype(np.float32),
                      cv2.COLOR_RGB2HSV)[..., 0] * 360.0
    centre = HUE_CENTRES[family]
    d = np.abs(np.mod(hb - centre + 180.0, 360.0) - 180.0)
    sel = (d < 20.0) & (cv2.cvtColor(np.clip(before, 0, 1).astype(np.float32),
                                     cv2.COLOR_RGB2HSV)[..., 1] > 0.08)
    if sel.sum() < 50:
        return np.array([])
    shift = np.mod(ha - hb + 180.0, 360.0) - 180.0
    return shift[sel]


def main():
    img = fit_long_edge(load_image(SRC), 900)
    print("image:", img.shape[1], "x", img.shape[0])

    # ---- 1. no-op -------------------------------------------------------
    print("\n=== colour wheel ===")
    z = select_color(img, "red", 0, 0, 0)
    check("empty call is a no-op", float(np.abs(z - img).max()) == 0.0)

    # ---- 2. family isolation, per-pixel ---------------------------------
    print("\nper-family response to a red rotation:")
    red_rot = select_color(img, "red", hue=40.0)
    own_moved = None
    others_ok = True
    for fam in HUE_CENTRES:
        shifts = hue_shift_per_pixel(img, red_rot, fam)
        if shifts.size == 0:
            print("   %-8s too few pixels" % fam)
            continue
        mean_shift = float(shifts.mean())
        spread = float(np.percentile(np.abs(shifts), 90))
        if fam == "red":
            own_moved = mean_shift
            print("   %-8s mean %+.2f  p90 |shift| %.2f  (n=%d)"
                  % (fam, mean_shift, spread, shifts.size))
        else:
            print("   %-8s mean %+.2f  p90 |shift| %.2f"
                  % (fam, mean_shift, spread))
            if abs(mean_shift) > 4.0 or spread > 10.0:
                others_ok = False
    check("red rotation moves its own family", own_moved is not None
          and own_moved > 3.0, "mean %+.2f" % (own_moved or 0))
    check("other families stay put", others_ok)

    # ---- 3. monotonicity, fixed pixels ----------------------------------
    print("\nmonotonicity (green sat), measured on pixels green at 0:")
    hsv0 = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32),
                        cv2.COLOR_RGB2HSV)
    h0 = hsv0[..., 0] * 360.0
    d0 = np.abs(np.mod(h0 - 120.0 + 180.0, 360.0) - 180.0)
    fixed_sel = (d0 < 20.0) & (hsv0[..., 1] > 0.08)
    check("enough green pixels to test", fixed_sel.sum() > 200,
          "%d" % int(fixed_sel.sum()))

    prev = None
    mono = True
    print("   (measuring HSV saturation of the same pixels, not a channel mean)")
    for k in (0, 25, 50, 75, 100):
        o = select_color(img, "green", sat=k)
        hsv_k = cv2.cvtColor(np.clip(o, 0, 1).astype(np.float32),
                             cv2.COLOR_RGB2HSV)
        val = float(hsv_k[..., 1][fixed_sel].mean())
        if prev is not None and val < prev - 1e-4:
            mono = False
        prev = val
        print("   sat=%3d  HSV-S=%.4f" % (k, val))
    check("saturation slider is monotonic", mono)

    save_image(os.path.join(ROOT, "out", "wheel_red.jpg"),
               select_color(img, "red", hue=35, sat=20), 92)
    save_image(os.path.join(ROOT, "out", "wheel_yellow.jpg"),
               select_color(img, "yellow", hue=-30, sat=25), 92)

    # ---- 4. grey point on a synthetic cast ------------------------------
    print("\n=== grey point ===")
    clean = fit_long_edge(load_image(SRC), 700)
    cast = np.array([1.25, 1.0, 0.78], np.float32)
    tinted = np.clip(linear_to_srgb(srgb_to_linear(clean) * cast[None, None, :]),
                     0, 1).astype(np.float32)

    # picker: a patch that is genuinely neutral in the *clean* frame. The
    # brightest low-saturation pixel is usually a blown highlight or a white
    # border, which cannot demonstrate a correction (it is already at the
    # gamut corner). A mid-luminance, low-saturation block is the case a
    # retoucher actually picks on.
    sat = clean.max(axis=2) - clean.min(axis=2)
    lum = clean @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    from scipy import ndimage
    sat_b = ndimage.uniform_filter(sat, 31)
    lum_b = ndimage.uniform_filter(lum, 31)
    score = -sat_b - 0.8 * np.abs(lum_b - 0.5)
    cy, cx = np.unravel_index(int(np.argmax(score)), score.shape)
    radius = 8
    cx = int(np.clip(cx, radius + 1, clean.shape[1] - radius - 2))
    cy = int(np.clip(cy, radius + 1, clean.shape[0] - radius - 2))
    print("  picker at (%d, %d)  block sat=%.4f lum=%.3f"
          % (cx, cy, sat_b[cy, cx], lum_b[cy, cx]))

    fixed, info = grey_point(tinted, float(cx), float(cy), radius=radius)
    print("  estimated gains:", info["gains"], " cast:", info["cast"])

    def patch_mean(im):
        y0, y1 = max(0, cy - 4), min(im.shape[0], cy + 5)
        x0, x1 = max(0, cx - 4), min(im.shape[1], cx + 5)
        return im[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)

    p_t = patch_mean(tinted)
    p_f = patch_mean(fixed)
    cast_before = float(np.sqrt(((p_t / max(p_t.mean(), 1e-6) - 1) ** 2).sum()))
    cast_after = float(np.sqrt(((p_f / max(p_f.mean(), 1e-6) - 1) ** 2).sum()))
    check("grey point reduced the patch cast", cast_after < cast_before,
          "%.4f -> %.4f" % (cast_before, cast_after))

    m0 = tinted.reshape(-1, 3).mean(axis=0)
    m1 = fixed.reshape(-1, 3).mean(axis=0)
    bal0 = abs(m0[0] - m0[2]) / max(m0.mean(), 1e-6)
    bal1 = abs(m1[0] - m1[2]) / max(m1.mean(), 1e-6)
    check("global red-blue imbalance reduced", bal1 < bal0,
          "%.4f -> %.4f" % (bal0, bal1))

    safe, info2 = grey_point(tinted, float(cx), float(cy), radius=radius,
                             strength=0.2)
    g2 = np.array(info2["gains"], np.float32)
    check("low strength stays near unity",
          abs(g2[0] - 1) < 0.15 and abs(g2[2] - 1) < 0.15, str(info2["gains"]))

    save_image(os.path.join(ROOT, "out", "grey_tinted.jpg"), tinted, 92)
    save_image(os.path.join(ROOT, "out", "grey_fixed.jpg"), fixed, 92)

    print()
    if failures:
        print("FAILED:", failures)
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
