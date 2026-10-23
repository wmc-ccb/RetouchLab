"""Skin retouching engine.

Implements the classic frequency-separation workflow plus targeted blemish
removal, mirroring the parameter vocabulary recovered from the analysed
application ("spotless_acne", "spotless_spot", "spotless_texture",
"freqSplit_face_high/low", "removeEyeBag_keepSilkworm", "teethWhitening_*").

All operations are original implementations of published techniques.
"""
from __future__ import annotations

import numpy as np
import cv2

from .imgutil import (
    gaussian, gaussian2d, guided_filter, blend, feather, luminance, smoothstep,
    srgb_to_linear, linear_to_srgb, rgb_to_lab, lab_to_rgb,
)
from .face import Face, skin_tone_mask, face_region_mask, detect_skin_global


# --------------------------------------------------------------------------
# frequency separation
# --------------------------------------------------------------------------

def split_frequencies(img: np.ndarray, radius: float) -> tuple:
    """Split into a smooth low-frequency layer and a detail high-frequency layer.

    Returns (low, high) where `low + high == img`.
    """
    low = gaussian(img, radius)
    high = img - low
    return low, high


def apply_frequency_separation(img: np.ndarray, mask: np.ndarray,
                               low_strength: float = 0.0,
                               high_strength: float = 0.0,
                               radius: float = 6.0) -> np.ndarray:
    """Flatten the low band and/or boost the high band inside `mask`.

    This is the "低频平整 / 高频增强" pair: the low band carries uneven
    tone and colour, the high band carries pores and fine texture.
    """
    if low_strength == 0.0 and high_strength == 0.0:
        return img.copy()

    low, high = split_frequencies(img, radius)
    out = img.copy()

    if low_strength != 0.0:
        # pull the low band toward its own local mean -> evens out blotchiness
        flattened = gaussian(low, radius * 2.5)
        target = low + (flattened - low) * (low_strength / 100.0)
        out = out + (target - low) * mask[..., None]

    if high_strength != 0.0:
        out = out + high * (high_strength / 100.0) * mask[..., None]

    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# blemish detection and removal  ("spotless_acne", "spotless_spot")
# --------------------------------------------------------------------------

def detect_blemishes(img: np.ndarray, mask: np.ndarray,
                     sensitivity: float = 50.0,
                     min_radius: int = 1, max_radius: int = 9) -> np.ndarray:
    """Return a soft mask over pixels that look like small dark/red spots.

    Both cues are *local* contrasts, which is what makes the sensitivity
    control behave monotonically:

    * luminance black-hat — a small dark blob on a lighter field;
    * redness difference-of-gaussians — locally redder than its surroundings.

    A global redness measure would let a whole face pass the threshold at
    high sensitivity, merge into one large component and then be discarded
    by the size filter, so detection would fall as sensitivity rose.

    `sensitivity` raises detection monotonically: a higher value lowers both
    thresholds, and the size ceiling grows with it so that neighbouring spots
    merging into a larger blob are not silently dropped.
    """
    lum = luminance(img)

    # --- cue 1: black-hat, inherently local -----------------------------
    # The structuring element must be wider than the largest blemish, or a
    # spot that fills the kernel produces no response at all: MORPH_CLOSE
    # then cannot find a lighter ring to close against.
    ksize = int(max_radius * 2 + 5)
    if ksize % 2 == 0:
        ksize += 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(lum, cv2.MORPH_CLOSE, k)
    bh = np.clip(closed - lum, 0.0, None)

    # --- cue 1b: reject broad soft shading ------------------------------
    # The eye socket, nose side and jaw are large areas that are simply
    # *darker* than the cheek. A black-hat responds to them too, and healing
    # them lifts the whole region, which reads as a white ring around the eye.
    # Real blemishes are small and compact, so require the dark response to be
    # locally peaked: subtract a wider opening of the same response, which
    # keeps small blobs and discards broad gradients.
    wide = int(ksize * 2.5) | 1
    kw = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (wide, wide))
    broad = cv2.morphologyEx(bh, cv2.MORPH_OPEN, kw)
    bh = np.clip(bh - broad, 0.0, None)

    # --- cue 2: local redness (DoG), not global redness ------------------
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    redness = np.clip(r - (g + b) / 2.0, 0.0, None)
    sigma = max(1.5, max_radius * 0.9)
    redness_dog = np.clip(redness - gaussian2d(redness, sigma), 0.0, None)

    # --- normalise against the skin region ------------------------------
    sel = mask > 0.5
    bh_ref = float(np.percentile(bh[sel], 99)) if sel.any() else float(np.percentile(bh, 99))
    red_ref = float(np.percentile(redness_dog[sel], 99)) if sel.any() else float(np.percentile(redness_dog, 99))
    bh_n = bh / max(bh_ref, 1e-4)
    red_n = redness_dog / max(red_ref, 1e-4)

    sens = np.clip(sensitivity / 100.0, 0.01, 1.0)
    t_bh = 0.50 - 0.42 * sens
    t_red = 0.55 - 0.42 * sens

    spot = np.maximum(smoothstep(t_bh, t_bh + 0.30, bh_n),
                      smoothstep(t_red, t_red + 0.35, red_n) * 0.85)
    spot = spot * mask

    # --- size filter ------------------------------------------------------
    binary = (spot > 0.30).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros_like(spot)
    min_area = max(1, min_radius ** 2)
    # The ceiling scales with sensitivity (a lower threshold merges
    # neighbours) but stays bounded: a component far larger than any plausible
    # blemish is a broad feature such as eye make-up or a shadow, and healing
    # it would flatten a whole region.
    max_area = int((max_radius * 4) ** 2 * (1.0 + 2.0 * sens))
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            keep[labels == i] = 1.0
        elif area > max_area:
            # an over-large component is a merged cluster or a broad feature:
            # keep only its confident cores, and only when it is not so large
            # that it must be structure rather than blemishes
            if area <= max_area * 4:
                comp = (labels == i)
                core = comp & (spot > 0.75)
                if core.any():
                    keep[core] = 1.0
    spot = spot * keep
    return feather(spot, 1.5)


def heal_blemishes(img: np.ndarray, spot_mask: np.ndarray,
                   strength: float = 100.0) -> np.ndarray:
    """Replace blemished pixels with a smooth reconstruction of nearby skin.

    The replacement is inpainted from the surrounding clean skin, so tone and
    colour come from real neighbours rather than a global blur.

    Two details matter for this to actually remove a spot:

    * the mask is **dilated** before inpainting, because the soft detector
      response covers the spot core but tapers off over its rim, and an
      undilated mask leaves a dark ring behind;
    * `strength` blends toward the inpainted result rather than masking it,
      so the whole dilated area is replaced, not just the confident core.
    """
    if strength <= 0 or spot_mask is None or spot_mask.max() <= 0:
        return img.copy()

    binary = (spot_mask > 0.18).astype(np.uint8)
    if binary.max() == 0:
        return img.copy()

    # grow the region so the spot's rim is included; the radius scales with
    # the image so behaviour is resolution independent
    h, w = img.shape[:2]
    grow = max(2, int(round(min(h, w) * 0.006)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1,) * 2)
    binary = cv2.dilate(binary, kernel, iterations=1)
    hole = binary * 255

    ys, xs = np.nonzero(hole)
    pad = grow * 6
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad + 1)

    patch = img[y0:y1, x0:x1]
    patch_hole = hole[y0:y1, x0:x1]
    u8 = (np.clip(patch, 0, 1) * 255).astype(np.uint8)
    healed = cv2.inpaint(cv2.cvtColor(u8, cv2.COLOR_RGB2BGR), patch_hole,
                         max(3, grow), cv2.INPAINT_TELEA)
    healed = cv2.cvtColor(healed, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # soften the seam so the patch blends into surrounding texture
    healed = gaussian(healed, max(0.8, grow * 0.4))

    # blend the whole dilated area toward the reconstruction
    m = np.clip(cv2.GaussianBlur(binary.astype(np.float32), (0, 0), grow * 0.5),
                0.0, 1.0)[y0:y1, x0:x1]
    m = np.maximum(m, binary[y0:y1, x0:x1].astype(np.float32))
    m = m * np.clip(strength / 100.0, 0.0, 1.0)

    out = img.copy()
    out[y0:y1, x0:x1] = blend(patch, healed, m)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def spotless(img: np.ndarray, mask: np.ndarray, acne: float = 0.0, spot: float = 0.0,
             texture_keep: float = 0.0, face_scale: float = 0.0,
             soft_shade: np.ndarray = None) -> np.ndarray:
    """Combined blemish pass.

    `acne` targets small red bumps, `spot` targets flatter pigment marks, and
    `texture_keep` scales back how much of the original grain is preserved.

    Strength and sensitivity both rise with the slider: the slider sets how
    aggressively spots are found *and* how completely they are replaced, so
    the result is monotonic in the control.

    `face_scale` (face size in pixels) scales the search radius so that a
    small face is not scanned with a frame-sized structuring element.

    `soft_shade` is an optional 0..1 map of areas that are *supposed* to be
    darker than their surroundings — the eye socket, the sides of the nose,
    the jaw shadow. Blemishes are small and compact, whereas these are broad
    and soft; healing them lifts the whole area and reads as a white ring.
    Supplying the map keeps detection and healing out of those regions.
    """
    if acne <= 0 and spot <= 0:
        return img.copy()

    # blemish radius: ~2.5% of face height, clamped to a sane pixel range
    if face_scale > 0:
        small_r = int(np.clip(round(face_scale * 0.022), 2, 12))
        large_r = int(np.clip(round(face_scale * 0.040), 3, 20))
    else:
        small_r, large_r = 5, 9

    gate = mask if soft_shade is None else mask * (1.0 - np.clip(soft_shade, 0.0, 1.0))
    # The suppression must also apply to the *heal* step, not just detection:
    # a broad soft gradient can still pass the detector and then be inpainted
    # bright, which is what turns the eye socket into a white ring.
    heal_gate = gate

    out = img
    if acne > 0:
        a = np.clip(acne / 100.0, 0.0, 1.0)
        m = detect_blemishes(img, gate, sensitivity=20.0 + 70.0 * a,
                             min_radius=1, max_radius=small_r)
        m = m * heal_gate
        out = heal_blemishes(out, m, strength=min(100.0, 40.0 + 60.0 * a))
    if spot > 0:
        s = np.clip(spot / 100.0, 0.0, 1.0)
        m = detect_blemishes(out, gate, sensitivity=15.0 + 65.0 * s,
                             min_radius=2, max_radius=large_r)
        m = m * heal_gate
        out = heal_blemishes(out, m, strength=min(100.0, 35.0 + 65.0 * s))

    if texture_keep > 0:
        # put a controlled amount of the original high-frequency detail back
        _, high = split_frequencies(img, 3.0)
        keep = (texture_keep / 100.0) * 0.55
        out = np.clip(out + high * keep * mask[..., None], 0.0, 1.0)
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# skin smoothing  (磨皮 / 质感肌)
# --------------------------------------------------------------------------

def smooth_skin(img: np.ndarray, mask: np.ndarray, amount: float = 0.0,
                texture: float = 50.0, radius_scale: float = 1.0,
                face_scale: float = 0.0) -> np.ndarray:
    """Edge-preserving skin smoothing that keeps pore-level texture.

    `amount` sets the overall strength; `texture` controls how much
    high-frequency detail survives (higher = more skin texture retained).

    `face_scale` is the face size in pixels. When supplied, the filter radius
    derives from the face rather than the frame, so a small face in a large
    image is not smoothed with a frame-sized kernel (which destroys the iris
    and the lip edge). Falls back to a frame-relative radius when unknown.
    """
    if amount <= 0:
        return img.copy()

    k = np.clip(amount / 100.0, 0.0, 1.0)
    t = np.clip(texture / 100.0, 0.0, 1.0)

    h, w = img.shape[:2]
    if face_scale > 0:
        # ~1.2% of the face height: enough to even out pores and blotches,
        # small enough to leave the iris, lash line and lip edge intact
        base = max(2.0, face_scale * 0.012) * radius_scale
    else:
        base = max(3.0, min(h, w) * 0.006) * radius_scale
    radius = int(np.clip(base, 2, 31))
    eps = (0.006 + (1.0 - t) * 0.05) ** 2

    guide = luminance(img)
    smoothed = guided_filter(guide, img, radius, eps)
    # a second pass removes residual blotchiness in the low band
    smoothed = guided_filter(luminance(smoothed), smoothed, radius, eps)
    result = blend(img, smoothed, mask * k)

    # restore a fraction of the original detail so skin does not look plastic
    detail_r = max(1.0, base * 0.8)
    _, high = split_frequencies(img, detail_r)
    _, high_s = split_frequencies(result, detail_r)
    detail_gain = (1.0 - k) + t * 0.85
    result = result + (high - high_s) * detail_gain * mask[..., None] * 0.9

    return np.clip(result, 0.0, 1.0).astype(np.float32)


def even_skin_tone(img: np.ndarray, mask: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """Reduce redness and blotchiness without blurring structure.

    Every edit is confined by `mask`, and the Lab round trip is blended back
    through the mask rather than replacing the frame.

    Two bugs lived here and both were invisible in the output, only in the
    pixels nobody looks at:

    * the chroma smoothing was applied to the *whole* image — the mask gated
      only the redness step — so a strongly coloured background was silently
      desaturated (measured: 0.47 max change outside the face box, where the
      operation is supposed to be a no-op);
    * `lab_to_rgb` on the full frame is not a lossless round trip (measured
      0.0039 = one 8-bit level), so even a correctly masked edit perturbed
      every pixel in the image.

    `tests/verify_locality.py` catches both.
    """
    if amount <= 0:
        return img.copy()
    k = np.clip(amount / 100.0, 0.0, 1.0)
    mk = np.clip(mask, 0.0, 1.0)

    lab = rgb_to_lab(img)
    a = lab[..., 1]
    b = lab[..., 2]
    # local mean of the chroma channels
    a_s = gaussian2d(a, 8.0)
    b_s = gaussian2d(b, 8.0)
    # gated by the mask: chroma smoothing outside the skin is not wanted
    lab[..., 1] = a + (a_s - a) * k * mk
    lab[..., 2] = b + (b_s - b) * k * mk
    adjusted = lab_to_rgb(lab)

    # blend rather than replace, so pixels the mask excludes stay exact
    out = blend(img, adjusted, mk)

    # pull overall redness down slightly (already colour-selective)
    r, g, bl = out[..., 0], out[..., 1], out[..., 2]
    red_push = np.clip((r - (g + bl) / 2.0) - 0.02, 0.0, None)
    out = out.copy()
    out[..., 0] = out[..., 0] - red_push * k * 0.55 * mk
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# eye area  ("removeEyeBag_keepSilkworm")
# --------------------------------------------------------------------------

def eye_bag_reduction(img: np.ndarray, face: Face, degree: float = 0.0,
                      keep_silkworm: float = 0.0) -> np.ndarray:
    """Lighten the under-eye shadow while optionally preserving the lower lid roll.

    `keep_silkworm` protects the aegyo-sal highlight, which sits immediately
    below the lash line.
    """
    if degree <= 0 or not face.eyes:
        return img.copy()

    h, w = img.shape[:2]
    k = np.clip(degree / 100.0, 0.0, 1.0)
    protect = np.clip(keep_silkworm / 100.0, 0.0, 1.0)

    eye_r, eye_l = face.eyes[0], face.eyes[1]
    eye_dist = float(np.linalg.norm(np.array(eye_l[:2]) - np.array(eye_r[:2])))
    ew = eye_dist * 0.50
    eh = eye_dist * 0.30

    out = img
    for (ex, ey, ewidth, eheight) in (eye_r, eye_l):
        cx = ex + ewidth / 2.0
        cy = ey + eheight / 2.0 + eye_dist * 0.16      # just below the eye
        m = radial_falloff_local(h, w, cx, cy, ew, eh)

        # protect the silkworm band right under the lash line
        if protect > 0:
            band_c = ey + eheight * 0.95
            band = radial_falloff_local(h, w, cx, band_c, ew * 0.9, eh * 0.5)
            m = m * (1.0 - band * protect)

        lab = rgb_to_lab(out)
        L = lab[..., 0]
        # lift the shadow toward the local surrounding brightness
        local = gaussian2d(L, eye_dist * 0.22)
        lift = np.clip(local - L, 0.0, None) * m * k * 0.85
        lab[..., 0] = np.clip(L + lift, 0.0, 1.0)
        # slightly reduce blue/purple cast in the same area
        bb = lab[..., 2]
        lab[..., 2] = bb + (gaussian2d(bb, eye_dist * 0.22) - bb) * m * k * 0.5
        # blend through this eye's mask: the Lab round trip is not lossless,
        # so rebuilding the frame would perturb pixels outside the eye too
        out = blend(out, lab_to_rgb(lab), m)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def radial_falloff_local(h: int, w: int, cx: float, cy: float,
                         rx: float, ry: float) -> np.ndarray:
    from .imgutil import radial_falloff
    return radial_falloff(h, w, cx, cy, max(rx, 1.0), max(ry, 1.0))


# --------------------------------------------------------------------------
# teeth and eyes  ("teethWhitening_brightness/deYellow", "eyeHighlight")
# --------------------------------------------------------------------------

def _mouth_mask(face: Face, shape: tuple, inner: float = 0.55) -> np.ndarray:
    h, w = shape[:2]
    if not face.mouth:
        return np.zeros((h, w), np.float32)
    mx, my, mw, mh = face.mouth
    m = radial_falloff_local(h, w, mx + mw / 2, my + mh / 2,
                             mw * inner, mh * inner * 0.8)
    return m


def teeth_whiten(img: np.ndarray, face: Face, brightness: float = 0.0,
                 de_yellow: float = 0.0) -> np.ndarray:
    """Brighten and de-yellow teeth inside the mouth region."""
    if (brightness <= 0 and de_yellow <= 0) or not face.mouth:
        return img.copy()

    h, w = img.shape[:2]
    region = _mouth_mask(face, (h, w), inner=0.50)

    # within the mouth, teeth are the bright, low-saturation pixels
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    s = hsv[..., 1]
    teeth = smoothstep(0.35, 0.62, v) * (1.0 - smoothstep(0.25, 0.55, s))
    # teeth sit in the lower hue range but are not skin: exclude strong red
    hue = hsv[..., 0]
    teeth *= (1.0 - smoothstep(0.02, 0.08, np.minimum(hue, 1.0 - hue)))
    m = feather(region * teeth, 2.0)
    if m.max() <= 1e-6:
        return img.copy()

    out = img
    if brightness > 0:
        kb = np.clip(brightness / 100.0, 0, 1)
        # Same reasoning as the Lab path below: srgb_to_linear -> linear_to_srgb
        # is not a lossless round trip, so a full-frame rebuild moves pixels the
        # mask excludes (measured 0.0061 outside the mouth). Blend instead.
        lin = srgb_to_linear(out) * (1.0 + kb * 0.45 * m[..., None])
        out = blend(out, linear_to_srgb(np.clip(lin, 0.0, None)), m)

    if de_yellow > 0:
        ky = np.clip(de_yellow / 100.0, 0, 1)
        lab = rgb_to_lab(out)
        b = lab[..., 2]
        # yellow sits at positive b*; pull it toward neutral
        lab[..., 2] = b - np.clip(b, 0.0, None) * ky * 0.85 * m
        # blend through the mask: the Lab round trip is lossy (0.0039), so
        # rebuilding the whole frame would touch every pixel
        out = blend(out, lab_to_rgb(lab), m)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def eye_highlight(img: np.ndarray, face: Face, amount: float = 0.0) -> np.ndarray:
    """Brighten the specular highlight inside the eye.

    The mask is kept *inside* the eye and gated to already-bright pixels, so
    only the existing catchlight and the brightest part of the sclera lift.
    A wider mask would brighten the lid and skin around the eye, which reads
    as a white ring rather than a catchlight.
    """
    if amount <= 0 or not face.eyes:
        return img.copy()
    h, w = img.shape[:2]
    k = np.clip(amount / 100.0, 0.0, 1.0)

    out = img
    for (ex, ey, ew, eh) in face.eyes:
        # well inside the eye box: 30% of its width and height
        m = radial_falloff_local(h, w, ex + ew / 2, ey + eh / 2, ew * 0.30, eh * 0.28)
        lab = rgb_to_lab(out)
        L = lab[..., 0]
        # only lift the brighter part of the eye so the pupil stays dark
        gate = smoothstep(0.45, 0.80, L)
        lab[..., 0] = np.clip(L + gate * m * k * 0.18, 0.0, 1.0)
        # blend through this eye's mask; see even_skin_tone for why
        out = blend(out, lab_to_rgb(lab), m)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# full-skin pass
# --------------------------------------------------------------------------

def retouch_skin(img: np.ndarray, face: Face = None, params: dict = None,
                 semantic=None) -> np.ndarray:
    """Run the whole skin pipeline for one face (or globally when face is None).

    Mask selection depends on what is available:

    * **with `semantic`** (a `semantic.Regions`): the learned face-skin and
      body-skin classes are used for *both* jobs. This is the best case and it
      dissolves an earlier compromise — the chroma test had to be kept away
      from blemish detection because it attenuated the very colour outliers a
      blemish is (measured: it cut the detector response to ~34% at spot
      centres). A segmentation does not work that way: it reports "skin" for a
      red spot on a cheek because it learned the region, not the colour.
    * **without `semantic`**: blemish work falls back to the geometric face
      region, and smoothing to the chroma skin mask.
    """
    p = params or {}
    out = img

    face_scale = 0.0
    if face is not None:
        face_scale = float(max(face.box[2], face.box[3]))

    near = None
    if face is not None:
        near = face_region_mask(img, face)

    # The semantic gate is only used when the model is actually confident on
    # this image. Its absolute output scale is image-dependent (a real photo
    # reaches ~0.95, a stylised or synthetic one ~0.04), and trusting a weak
    # mask silently disables skin processing instead of reporting a problem.
    sem_ok = False
    if semantic is not None:
        try:
            sem_ok = semantic.confidence() >= 0.25
        except Exception:
            sem_ok = False

    if sem_ok:
        sem_skin = semantic.skin()
        if near is not None:
            # one face at a time: keep semantic skin but limit to this face's
            # neighbourhood so a multi-face image is not re-processed per face
            tone_mask = np.clip(sem_skin * near, 0.0, 1.0)
            # blemish gate: semantic skin, plus a little of the geometric
            # region so a spot right on the jaw line is not cut in half
            region = np.clip(np.maximum(sem_skin * near, near * 0.6), 0.0, 1.0)
        else:
            tone_mask = sem_skin
            region = sem_skin
    else:
        region = near if near is not None else face_region_mask(img, None)
        if face is not None and face.skin_mask is not None:
            tone_mask = skin_tone_mask(img, face)
        else:
            tone_mask = detect_skin_global(img)

    if region.max() <= 1e-6 and tone_mask.max() <= 1e-6:
        return out

    # ---- work in a window around the face -------------------------------
    # The masks are zero away from the subject, so every filter below was
    # doing full-frame work for a result that only changes inside a small
    # region. On a 1600px preview with a 276px face this was ~5.3 s of an
    # ~8.9 s run, with the face occupying under 2% of the pixels. Cropping to
    # a padded window and pasting the result back makes the cost track the
    # face rather than the frame.
    h, w = out.shape[:2]
    if face is not None:
        x, y, bw, bh = face.box
        pad = 0.45
        wx0 = max(0, int(x - bw * (0.5 + pad)))
        wy0 = max(0, int(y - bh * (0.5 + pad)))
        wx1 = min(w, int(x + bw * (0.5 + pad)) + 1)
        wy1 = min(h, int(y + bh * (0.5 + pad)) + 1)
    else:
        wx0 = wy0 = 0
        wx1, wy1 = w, h

    if (wx1 - wx0) < 8 or (wy1 - wy0) < 8:
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    win = out[wy0:wy1, wx0:wx1]
    m_region = region[wy0:wy1, wx0:wx1]
    m_tone = tone_mask[wy0:wy1, wx0:wx1]
    m_shade = None
    if face is not None and face.soft_shade is not None:
        m_shade = face.soft_shade[wy0:wy1, wx0:wx1]

    if m_region.max() <= 1e-6 and m_tone.max() <= 1e-6:
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    # a view of the face whose coordinates are window-local
    win_face = _window_face(face, wx0, wy0) if face is not None else None

    win = spotless(win, m_region,
                   acne=p.get("spotless_acne", 0.0),
                   spot=p.get("spotless_spot", 0.0),
                   texture_keep=p.get("spotless_texture", 0.0),
                   face_scale=face_scale,
                   soft_shade=m_shade)

    win = even_skin_tone(win, m_tone, p.get("even_tone", 0.0))

    win = smooth_skin(win, m_tone,
                      amount=p.get("smooth", 0.0),
                      texture=p.get("texture", 50.0),
                      face_scale=face_scale)

    if p.get("freq_low", 0.0) or p.get("freq_high", 0.0):
        win = apply_frequency_separation(
            win, m_tone,
            low_strength=p.get("freq_low", 0.0),
            high_strength=p.get("freq_high", 0.0),
            radius=max(3.0, min(win.shape[:2]) * 0.004))

    if win_face is not None:
        win = eye_bag_reduction(win, win_face,
                                degree=p.get("removeEyeBag", 0.0),
                                keep_silkworm=p.get("keepSilkworm", 0.0))
        win = eye_highlight(win, win_face, p.get("eyeHighlight", 0.0))
        win = teeth_whiten(win, win_face,
                           brightness=p.get("teethWhitening", 0.0),
                           de_yellow=p.get("teethDeYellow", 0.0))

    # paste back, feathering the seam so the window edge is invisible
    out = out.copy()
    out[wy0:wy1, wx0:wx1] = win
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _window_face(face: Face, dx: float, dy: float) -> Face:
    """A copy of `face` whose coordinates are shifted into a local window.

    Masks are deliberately shared rather than offset: the callers that use
    them index the window, and the masks are already local to the face.
    """
    import copy
    f = copy.copy(face)
    x, y, bw, bh = face.box
    f.box = (x - dx, y - dy, bw, bh)
    f.landmarks = face.landmarks - np.array([dx, dy], np.float32)
    if face.eyes:
        f.eyes = [(ex - dx, ey - dy, ew, eh) for (ex, ey, ew, eh) in face.eyes]
    if face.mouth:
        mx, my, mw, mh = face.mouth
        f.mouth = (mx - dx, my - dy, mw, mh)
    if face.dense is not None:
        d = copy.copy(face.dense)
        d.points = face.dense.points - np.array([dx, dy], np.float32)
        f.dense = d
    return f
