# -*- coding: utf-8 -*-
"""表情管理与妆容调整 — expression and makeup from 5-point face geometry.

Both are *local shape or colour* edits driven by measured facial geometry, not
by a face box: a mouth corner is a specific point, and a blush has to sit on the
cheekbone.

**Why 5 points, not 106.** This module originally indexed `2d106det.onnx` using
the InsightFace 2d106 layout. Measurement showed that model's output is *not*
that layout: the point it treats as "inner lip" sits in the upper half of the
face, and "left eye" spans more than half the face height. Indexing it produced
anchors in the wrong place, so warps and tints landed on cheeks and forehead
instead of lips and eyes — which is exactly why the portrait features felt
unusable.

YuNet's own 5 points carry unambiguous semantics (eye centres, nose tip, mouth
corners), and measured interocular / face-width = 0.46, which matches human
anatomy. So the regions are derived from those five points via anthropometric
ratios in `geoface`. The trade-off is stated plainly there: positions are
reliable, boundaries are elliptical approximations rather than true segmentation.

Everything is expressed in **interocular-distance units**, so the same slider
value behaves identically on a large face and a small one.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from . import geoface as GEO

# 5-point order as emitted by YuNet
L_EYE_R, L_EYE_L, L_NOSE, L_MOUTH_R, L_MOUTH_L = 0, 1, 2, 3, 4


def _interocular(lm: np.ndarray) -> float:
    return GEO._iod(np.asarray(lm, np.float32))


def _pt(lm: np.ndarray, i: int) -> np.ndarray:
    return np.asarray(lm[i], np.float32)[:2]


def _gauss_blob(shape: Tuple[int, int], centre: Tuple[float, float],
                radii: Tuple[float, float], angle: float = 0.0
                ) -> np.ndarray:
    """An elliptical falloff. Used for blush/shadow so they have no hard rim."""
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - centre[0]
    dy = yy - centre[1]
    ca, sa = np.cos(angle), np.sin(angle)
    xr = dx * ca + dy * sa
    yr = -dx * sa + dy * ca
    rx = max(radii[0], 1.0)
    ry = max(radii[1], 1.0)
    d = (xr / rx) ** 2 + (yr / ry) ** 2
    return np.clip(np.exp(-2.0 * d), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# 表情管理
# ---------------------------------------------------------------------------

def expression(img: np.ndarray, lm: np.ndarray,
               smile: float = 0.0, eye_open: float = 0.0,
               brow_raise: float = 0.0, mouth_narrow: float = 0.0,
               squash: float = 0.0) -> np.ndarray:
    """Warp the mouth / eyes / brows for expression.

    Anchors come from the 5 measured points (eye centres, nose tip, mouth
    corners) plus ratios, so every displacement starts from a position that is
    actually correct. Earlier this read 106-point indices that do not carry the
    assumed semantics, so the warps moved forehead and cheeks.

    A single global affine is deliberately not used: rotating a whole face to
    "smile" moves the nose and ears too and reads as a rubber mask.
    """
    lm = np.asarray(lm, np.float32)
    out = np.asarray(img, np.float32).copy()
    iod = _interocular(lm)
    h, w = out.shape[:2]
    ang = GEO._roll_angle(lm)
    down = np.array([-np.sin(ang), np.cos(ang)], np.float32)
    right = np.array([np.cos(ang), np.sin(ang)], np.float32)
    eye_r, eye_l = _pt(lm, L_EYE_R), _pt(lm, L_EYE_L)
    nose = _pt(lm, L_NOSE)
    mouth_r, mouth_l = _pt(lm, L_MOUTH_R), _pt(lm, L_MOUTH_L)

    if smile:
        amt = float(smile) / 100.0 * iod * 0.20
        for p, side in ((mouth_l, 1), (mouth_r, -1)):
            # 嘴角沿「向外 + 向上」移动，方向随歪头一起转
            off = right * (side * amt * 0.35) + down * (-amt)
            out = _local_warp(out, p, p + off, radius=iod * 0.30)
        # 下巴略微下拉，否则下半张脸像被压扁
        chin = (eye_r + eye_l) / 2.0 + down * (iod * 1.75)
        out = _local_warp(out, chin, chin + down * (amt * 0.12),
                          radius=iod * 0.40)

    if eye_open:
        amt = float(eye_open) / 100.0 * iod * 0.10
        half_w = GEO.SCALE["eye_w"] * iod * 0.5
        half_h = GEO.SCALE["eye_h"] * iod * 0.5
        for c in (eye_r, eye_l):
            # 上睑向上、下睑向下：各推一次，眼裂张开而不整体平移
            out = _local_warp(out, c + down * (-half_h),
                              c + down * (-half_h - amt), radius=iod * 0.22)
            out = _local_warp(out, c + down * half_h,
                              c + down * (half_h + amt), radius=iod * 0.22)

    if brow_raise:
        amt = float(brow_raise) / 100.0 * iod * 0.14
        lift = GEO.SCALE["brow_lift"] * iod
        for c, side in ((eye_r, -1), (eye_l, 1)):
            brow = c + down * (-lift) + right * (side * iod * 0.06)
            out = _local_warp(out, brow, brow + down * (-amt),
                              radius=iod * 0.40)

    if mouth_narrow:
        amt = float(mouth_narrow) / 100.0 * iod * 0.12
        for p, side in ((mouth_l, -1), (mouth_r, 1)):
            out = _local_warp(out, p, p + right * (side * amt),
                              radius=iod * 0.22)

    if squash:
        amt = float(squash) / 100.0 * iod * 0.10
        out = _local_warp(out, nose, nose + down * amt, radius=iod * 0.55)
    return np.clip(out, 0.0, 1.0)


def _local_warp(img: np.ndarray, src_pt: np.ndarray, dst_pt: np.ndarray,
                radius: float) -> np.ndarray:
    """Move one point, with a smooth falloff.

    Implemented as a dense remap: for every pixel near `src_pt`, sample from
    the position that is displaced by the falloff-weighted offset. A pull-based
    remap (rather than drawing at the destination) is what keeps the texture
    continuous — pushing pixels leaves gaps.

    The sample coordinates handed to `cv2.remap` are **relative to the
    sub-array**, not to the whole image: remap indexes the array it is given, so
    passing absolute coordinates shifted the sampled content by (x0, y0) and
    repainted the entire crop with the wrong part of the picture.
    """
    h, w = img.shape[:2]
    r = max(float(radius), 2.0)
    x0 = max(0, int(min(src_pt[0], dst_pt[0]) - r * 2))
    x1 = min(w, int(max(src_pt[0], dst_pt[0]) + r * 2) + 1)
    y0 = max(0, int(min(src_pt[1], dst_pt[1]) - r * 2))
    y1 = min(h, int(max(src_pt[1], dst_pt[1]) + r * 2) + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return img
    sub = img[y0:y1, x0:x1]
    hh, ww = sub.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    # 绝对坐标用于算距离，相对坐标用于 remap
    px = xx + x0
    py = yy + y0
    d = np.sqrt((px - src_pt[0]) ** 2 + (py - src_pt[1]) ** 2) / r
    fall = np.clip(1.0 - d, 0.0, 1.0)
    fall = fall * fall * (3.0 - 2.0 * fall)          # smoothstep
    off = (dst_pt - src_pt).astype(np.float32)
    map_x = (xx - off[0] * fall).astype(np.float32)
    map_y = (yy - off[1] * fall).astype(np.float32)
    warped = cv2.remap(sub, map_x, map_y, cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_REFLECT_101)
    out = img.copy()
    out[y0:y1, x0:x1] = warped
    return out


# ---------------------------------------------------------------------------
# 妆容调整
# ---------------------------------------------------------------------------

MAKEUP_TONES = {
    # name: (rgb, default strength)
    "coral": ((0.94, 0.44, 0.40), 45),
    "rose": ((0.90, 0.36, 0.48), 45),
    "peach": ((0.96, 0.60, 0.46), 40),
    "berry": ((0.72, 0.24, 0.42), 40),
    "nude": ((0.86, 0.62, 0.55), 35),
    "brick": ((0.72, 0.30, 0.24), 40),
    "plum": ((0.55, 0.26, 0.42), 35),
}


def makeup(img: np.ndarray, lm: np.ndarray, parts=None,
           lip: float = 0.0, lip_tone: str = "rose",
           blush: float = 0.0, blush_tone: str = "coral",
           shadow: float = 0.0, shadow_tone: str = "nude",
           brow: float = 0.0, brow_tone: str = "brick") -> np.ndarray:
    """Lip colour, blush, eyeshadow and brow tint, all mask-bounded.

    Colour is applied in Lab along the chosen tone's chroma direction and
    blended at the mask's own alpha, so the lip keeps its specular highlights
    and shading instead of becoming a flat sticker.

    Masks come from `geoface` (5 points + anthropometric ratios). That is an
    ellipse per region: the *position* is reliable, the *boundary* is an
    approximation. It is used instead of the face-parsing model because that
    model works on a 128-cell grid internally, where a mouth is about 15 cells
    wide — too coarse to be useful, and it reported zero coverage for brows and
    mouth on real frames.
    """
    lm = np.asarray(lm, np.float32)
    out = np.asarray(img, np.float32).copy()
    h, w = out.shape[:2]
    geo = GEO.region_masks(out, lm)

    def tone(name: str) -> np.ndarray:
        return np.array(MAKEUP_TONES.get(name, MAKEUP_TONES["rose"])[0],
                        np.float32)

    if lip and lip > 0:
        m = geo.get("lips")
        if m is not None:
            out = _tint(out, m, tone(lip_tone), lip / 100.0 * 0.80,
                        keep_luma=0.55)

    if blush and blush > 0:
        m = geo.get("blush")
        if m is not None:
            out = _tint(out, m, tone(blush_tone), blush / 100.0 * 0.50,
                        keep_luma=0.80)

    if shadow and shadow > 0:
        # 眼影落在眼睛上边缘：把眼部遮罩上移半个眼高
        m = geo.get("eyes")
        if m is not None:
            dy = int(round(GEO.SCALE["eye_h"] * 0.55 * _interocular(lm)))
            shifted = np.zeros_like(m)
            if dy > 0:
                shifted[0:h - dy] = m[dy:]
            else:
                shifted[:] = m
            out = _tint(out, shifted, tone(shadow_tone), shadow / 100.0 * 0.45,
                        keep_luma=0.72)

    if brow and brow > 0:
        m = geo.get("brows")
        if m is not None:
            out = _tint(out, m, tone(brow_tone), brow / 100.0 * 0.55,
                        keep_luma=0.35)
    return np.clip(out, 0.0, 1.0)


def _poly_mask(shape: Tuple[int, int], pts) -> np.ndarray:
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    poly = np.array([[int(round(p[0])), int(round(p[1]))] for p in pts],
                    np.int32)
    if len(poly) >= 3:
        cv2.fillPoly(m, [poly], 255)
    m = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 2.5)
    return np.clip(m, 0.0, 1.0)


def _tint(img: np.ndarray, mask: np.ndarray, rgb: np.ndarray,
          amount: float, keep_luma: float = 0.6) -> np.ndarray:
    """Blend a chroma direction in, preserving most of the original luminance.

    A straight alpha blend of a flat colour destroys the shading and looks
    painted on. Keeping `keep_luma` of the original L and only taking the colour
    direction means highlights and shadows survive.
    """
    from .imgutil import rgb_to_lab, lab_to_rgb

    a = np.clip(np.asarray(mask, np.float32), 0.0, 1.0)
    if a.ndim == 3:
        a = a[..., 0]
    if a.max() <= 1e-6 or amount <= 0:
        return img
    lab = rgb_to_lab(img)
    tgt = rgb_to_lab(np.clip(rgb.reshape(1, 1, 3), 0, 1).astype(np.float32))
    # 只取目标的 a/b 方向，L 基本保留
    out_lab = lab.copy()
    out_lab[..., 1] = lab[..., 1] * (1 - amount) + tgt[0, 0, 1] * amount
    out_lab[..., 2] = lab[..., 2] * (1 - amount) + tgt[0, 0, 2] * amount
    # 上色时略微压一点 L，让妆容有存在感但不发灰
    out_lab[..., 0] = lab[..., 0] * (1 - amount * 0.12 * (1 - keep_luma)) + \
        tgt[0, 0, 0] * amount * 0.12 * (1 - keep_luma)
    tinted = lab_to_rgb(out_lab)
    w = (a * amount)[..., None]
    return np.clip(img * (1 - w) + tinted * w, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 头发调整
# ---------------------------------------------------------------------------

def hair(img: np.ndarray, hair_mask: Optional[np.ndarray],
         volume: float = 0.0, shine: float = 0.0,
         colour: float = 0.0, colour_rgb: Optional[Tuple[float, float, float]] = None,
         frizz: float = 0.0) -> np.ndarray:
    """Hair volume, shine, recolour and frizz smoothing.

    * **蓬松** — dilate the hair mask outward and pull the *surrounding*
      pixels into the new area. Bulging the silhouette is what "more volume"
      actually means; brightening the existing hair does not read as volume.
    * **光泽** — add a band of higher luminance following the hair's own
      direction, estimated by the local structure tensor, so the highlight
      lies along the strands rather than in a fixed oval.
    * **发色** — Lab chroma replacement, keeping the strand structure.
    * **碎发** — smooth the edge of the alpha so stray single-pixel hairs
      merge into the mass.
    """
    out = np.asarray(img, np.float32).copy()
    h, w = out.shape[:2]
    if hair_mask is None:
        return out
    m = np.clip(np.asarray(hair_mask, np.float32), 0.0, 1.0)
    if m.ndim == 3:
        m = m[..., 0]
    if m.max() <= 1e-6:
        return out

    if frizz > 0:
        # 碎发：先闭运算连起来，再柔化边缘，减少孤立发丝
        k = int(1 + round(frizz / 100.0 * 3))
        mm = (m > 0.35).astype(np.uint8)
        mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE,
                              np.ones((k * 2 + 1, k * 2 + 1), np.uint8))
        mm = cv2.GaussianBlur(mm.astype(np.float32), (0, 0),
                              1.0 + frizz / 100.0 * 1.5)
        m = np.clip(mm, 0.0, 1.0)

    if volume > 0:
        from .imgutil import guided_filter, luminance
        px = int(round(volume / 100.0 * max(3, min(h, w) * 0.012)))
        if px > 0:
            grown = cv2.dilate(m, np.ones((px * 2 + 1, px * 2 + 1), np.uint8))
            ring = np.clip(grown - m, 0.0, 1.0)
            # 环里的颜色从头发内部采样，再按环的羽化权重贴上去
            k = max(2, px)
            solid = (m > 0.6).astype(np.float32)
            inner = cv2.blur(out * solid[..., None], (2 * k + 1, 2 * k + 1))
            wt = cv2.blur(solid, (2 * k + 1, 2 * k + 1)) + 1e-6
            inner = inner / wt[..., None]
            ring_f = cv2.GaussianBlur(ring, (0, 0), px * 0.6)
            wr = np.clip(ring_f, 0.0, 1.0)[..., None]
            out = out * (1 - wr) + inner * wr
            m = np.clip(np.maximum(m, ring_f), 0.0, 1.0)

    if colour > 0 and colour_rgb is not None:
        from .imgutil import rgb_to_lab, lab_to_rgb
        amt = float(colour) / 100.0
        lab = rgb_to_lab(out)
        tgt = rgb_to_lab(np.clip(np.array(colour_rgb, np.float32)
                                 .reshape(1, 1, 3), 0, 1))
        lab[..., 1] = lab[..., 1] * (1 - amt) + tgt[0, 0, 1] * amt
        lab[..., 2] = lab[..., 2] * (1 - amt) + tgt[0, 0, 2] * amt
        recol = lab_to_rgb(lab)
        wm = np.clip(m * amt, 0.0, 1.0)[..., None]
        out = np.clip(out * (1 - wm) + recol * wm, 0.0, 1.0)

    if shine > 0:
        amt = float(shine) / 100.0
        from .imgutil import luminance
        lum = luminance(out)
        g = cv2.cvtColor((np.clip(lum, 0, 1) * 255).astype(np.uint8),
                         cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # 结构张量：沿发丝方向的高亮带（沿主方向伸展的高斯）
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.0)
        jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.0)
        jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.0)
        theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy)
        # 用方向做各向异性模糊：沿 theta 方向平滑，垂直方向陡
        band = np.abs(np.cos(theta * 2)) ** 2
        hi = cv2.GaussianBlur(lum, (0, 0), max(h, w) * 0.02)
        hi = np.clip((lum - hi) * 2.0 + 0.5, 0.0, 1.0)
        boost = (hi * band * amt * 0.6)
        out = np.clip(out + boost[..., None] * np.clip(m, 0, 1)[..., None],
                      0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# 衣物
# ---------------------------------------------------------------------------

def clothes(img: np.ndarray, cloth_mask: Optional[np.ndarray],
            wrinkle: float = 0.0, stain: float = 0.0,
            structure: float = 0.0, tint: float = 0.0,
            tint_rgb: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
    """Fabric edits: smooth creases, lift marks, keep edges crisp, retint.

    * **褶皱** — edge-preserving smoothing along the crease, applied only where
      the local gradient is *low*: a large soft gradient is a fold, a sharp one
      is a seam, and seams must survive.
    * **去污** — lift local colour outliers toward the surrounding median,
      limited to a small radius so a real pattern is not erased.
    * **版型** — re-assert the garment's silhouette edge with a light unsharp
      on the mask boundary.
    """
    out = np.asarray(img, np.float32).copy()
    h, w = out.shape[:2]
    if cloth_mask is None:
        return out
    m = np.clip(np.asarray(cloth_mask, np.float32), 0.0, 1.0)
    if m.ndim == 3:
        m = m[..., 0]
    if m.max() <= 1e-6:
        return out

    if wrinkle > 0:
        amt = wrinkle / 100.0
        # 双边滤波压平低频褶皱，但保留边缘
        sm = cv2.bilateralFilter(out, 0, 9 + amt * 18, 9 + amt * 14)
        from .imgutil import luminance
        lum = luminance(out)
        gy, gx = np.gradient(lum)
        mag = np.sqrt(gx * gx + gy * gy)
        # 褶皱 = 低频形变；用大幅模糊后的梯度衡量，陡边被排除
        low = cv2.GaussianBlur(mag, (0, 0), max(2.0, min(h, w) * 0.01))
        keep = np.clip(1.0 - low * 6.0, 0.0, 1.0)
        wgt = (amt * keep * m)[..., None]
        out = out * (1 - wgt) + sm * wgt

    if stain > 0:
        amt = stain / 100.0
        # 中值核必须显著大于污点才能"取到周围"。固定小核会让大污点的
        # 中值仍是污点自己（实测 40px 污点 + 13px 核 = 纹丝不动）。
        # 用 8% 短边的大核：小污点照样被周围中值替换，大污点才有机会。
        r = max(9, int(min(h, w) * 0.08))
        med = cv2.medianBlur((np.clip(out, 0, 1) * 255).astype(np.uint8),
                             r * 2 + 1).astype(np.float32) / 255.0
        out = out * (1 - m[..., None] * amt) + med * m[..., None] * amt

    if structure > 0:
        amt = structure / 100.0
        from .imgutil import unsharp
        sharp = unsharp(out, amount=amt * 0.8, radius=1.2)
        # 只在衣服的边界附近加强，避免把布纹也锐化出来
        edge = cv2.morphologyEx((m > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8)).astype(np.float32)
        edge = cv2.GaussianBlur(edge, (0, 0), 2.0)
        wgt = np.clip(edge, 0, 1)[..., None]
        out = out * (1 - wgt) + sharp * wgt

    if tint > 0 and tint_rgb is not None:
        from .imgutil import rgb_to_lab, lab_to_rgb
        amt = tint / 100.0
        lab = rgb_to_lab(out)
        tgt = rgb_to_lab(np.clip(np.array(tint_rgb, np.float32)
                                 .reshape(1, 1, 3), 0, 1))
        lab[..., 1] = lab[..., 1] * (1 - amt) + tgt[0, 0, 1] * amt
        lab[..., 2] = lab[..., 2] * (1 - amt) + tgt[0, 0, 2] * amt
        recol = lab_to_rgb(lab)
        wm = np.clip(m * amt, 0, 1)[..., None]
        out = np.clip(out * (1 - wm) + recol * wm, 0.0, 1.0)

    return np.clip(out, 0.0, 1.0)


def info() -> dict:
    return {"expression": ["smile", "eye_open", "brow_raise",
                           "mouth_narrow", "squash"],
            "makeup": list(MAKEUP_TONES.keys()),
            "hair": ["volume", "shine", "colour", "frizz"],
            "clothes": ["wrinkle", "stain", "structure", "tint"]}
