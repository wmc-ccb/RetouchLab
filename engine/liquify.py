"""Liquify / reshaping engine.

Implements local warps driven by control points: the technique behind face
slimming, eye enlargement, limb reshaping and body proportion changes. The
control names mirror the ones recovered from the analysed application
("slimFace", "enlargeEye", "legLengthen_*", "aiBodyReshape_bodySmooth").

Sign conventions
----------------
`cv2.remap` samples the output pixel at (x, y) from the input at
(x + dx, y + dy), so a positive `dx` moves *content* to the left. Every
primitive below therefore negates the desired content motion when it
accumulates the displacement field. The public semantics are the intuitive
ones:

* ``push_displacement``  — content moves along `angle` (radians, image
  coordinates: 0 = right, pi/2 = down).
* ``scale_displacement`` — ``amount > 0`` magnifies around the centre.
* ``axis_displacement``  — ``amount_x > 0`` narrows horizontally,
  ``amount_y > 0`` lengthens vertically.

Warps accumulate into one displacement grid and are applied with a single
remap, so several adjustments compose without repeated resampling.
"""
from __future__ import annotations

import numpy as np
import cv2

from .face import Face
from .imgutil import smoothstep, feather, gaussian
from .pose import L_SHOULDER, R_SHOULDER


# --------------------------------------------------------------------------
# displacement primitives
# --------------------------------------------------------------------------

def _grid(shape: tuple):
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx, yy


def _falloff(d: np.ndarray, radius: float) -> np.ndarray:
    """Smooth 1 -> 0 falloff over `radius`."""
    w = 1.0 - smoothstep(0.0, max(radius, 1e-3), d)
    return (w * w * (3.0 - 2.0 * w)).astype(np.float32)


def push_displacement(dx: np.ndarray, dy: np.ndarray, shape: tuple,
                      cx: float, cy: float, radius: float,
                      strength: float, angle: float = 0.0) -> None:
    """Accumulate a local translation with a smooth falloff (a "push" brush).

    `strength` is in pixels; content moves along `angle` (radians, 0 = +x,
    pi/2 = +y i.e. downward on screen).
    """
    xx, yy = _grid(shape)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    w = _falloff(d, radius)
    dx -= np.cos(angle) * strength * w
    dy -= np.sin(angle) * strength * w


def scale_displacement(dx: np.ndarray, dy: np.ndarray, shape: tuple,
                       cx: float, cy: float, rx: float, ry: float,
                       amount: float) -> None:
    """Local magnification: `amount` > 0 enlarges, < 0 shrinks."""
    xx, yy = _grid(shape)
    nx = (xx - cx) / max(rx, 1e-3)
    ny = (yy - cy) / max(ry, 1e-3)
    d = np.sqrt(nx * nx + ny * ny)
    w = _falloff(d, 1.0)
    dx -= (xx - cx) * amount * w
    dy -= (yy - cy) * amount * w


def axis_displacement(dx: np.ndarray, dy: np.ndarray, shape: tuple,
                      cx: float, cy: float, rx: float, ry: float,
                      amount_x: float = 0.0, amount_y: float = 0.0,
                      falloff: float = 1.0) -> None:
    """Squeeze horizontally or stretch vertically inside an ellipse.

    ``amount_x > 0`` narrows (content moves toward the vertical axis);
    ``amount_y > 0`` lengthens (content moves away from the horizontal axis).

    ``falloff`` sets where the effect reaches zero, in units of the ellipse
    radius. Values near 1 confine the warp to the ellipse; larger values
    spread the transition over a wider ring, which is what makes a face
    slimming read as smooth rather than pinched.
    """
    xx, yy = _grid(shape)
    nx = (xx - cx) / max(rx, 1e-3)
    ny = (yy - cy) / max(ry, 1e-3)
    d = np.sqrt(nx * nx + ny * ny)
    w = _falloff(d, falloff)
    if amount_x:
        dx += (xx - cx) * amount_x * w
    if amount_y:
        dy -= (yy - cy) * amount_y * w


def warp(img: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Apply a displacement field with one bilinear remap."""
    h, w = img.shape[:2]
    xx, yy = _grid((h, w))
    map_x = (xx + dx).astype(np.float32)
    map_y = (yy + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LANCZOS4,
                     borderMode=cv2.BORDER_REFLECT).astype(np.float32)


# --------------------------------------------------------------------------
# face reshaping
# --------------------------------------------------------------------------

def slim_face(img: np.ndarray, face: Face, amount: float = 0.0,
              jaw: float = 0.0, cheekbone: float = 0.0) -> np.ndarray:
    """Narrow the jaw and cheek line.

    Implemented as an inward horizontal squeeze concentrated on the lower
    half of the face, strongest at the jaw.
    """
    if amount == 0 and jaw == 0 and cheekbone == 0:
        return img.copy()

    h, w = img.shape[:2]
    x, y, bw, bh = face.box
    cx, cy = face.center

    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)

    if amount != 0:
        k = amount / 100.0
        axis_displacement(dx, dy, (h, w), cx, cy + bh * 0.10,
                          bw * 0.62, bh * 0.68, amount_x=k * 0.22, falloff=1.8)

    if jaw != 0:
        k = jaw / 100.0
        axis_displacement(dx, dy, (h, w), cx, cy + bh * 0.30,
                          bw * 0.48, bh * 0.34, amount_x=k * 0.26, falloff=1.8)

    if cheekbone != 0:
        k = cheekbone / 100.0
        axis_displacement(dx, dy, (h, w), cx, cy - bh * 0.02,
                          bw * 0.56, bh * 0.22, amount_x=k * 0.16, falloff=1.8)

    return warp(img, dx, dy)


def enlarge_eyes(img: np.ndarray, face: Face, amount: float = 0.0,
                 height: float = 0.0, distance: float = 0.0) -> np.ndarray:
    """Scale the eyes up and optionally change eye height or spacing."""
    if amount == 0 and height == 0 and distance == 0:
        return img.copy()

    h, w = img.shape[:2]
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)

    eye_r, eye_l = face.eyes[0], face.eyes[1]
    eye_dist = float(np.linalg.norm(np.array(eye_l[:2]) - np.array(eye_r[:2])))

    for (ex, ey, ew, eh) in (eye_r, eye_l):
        ecx, ecy = ex + ew / 2, ey + eh / 2
        if amount != 0:
            scale_displacement(dx, dy, (h, w), ecx, ecy,
                               ew * 0.80, eh * 1.05, amount / 100.0 * 0.18)
        if height != 0:
            axis_displacement(dx, dy, (h, w), ecx, ecy,
                              ew * 0.80, eh * 1.05, amount_y=height / 100.0 * 0.14)
        if distance != 0:
            sign = 1.0 if ecx < w / 2 else -1.0
            push_displacement(dx, dy, (h, w), ecx, ecy, eye_dist * 0.60,
                              -sign * distance / 100.0 * eye_dist * 0.10, angle=0.0)

    return warp(img, dx, dy)


def reshape_face(img: np.ndarray, face: Face, params: dict,
                 return_field: bool = False):
    """Apply every facial geometry control in one composed warp.

    Supported keys: slimFace, jaw, cheekbone, enlargeEye, eyeHeight,
    eyeDistance, noseWidth, mouthWidth, chinLength, foreheadHeight.

    `return_field` returns the displacement field instead of the warped image.
    A warp's change is exactly confined to where the field is non-zero — a
    zero-displacement remap is a bit-exact identity (verified) — so the field
    is what a locality test needs to assert against, and returning it here
    avoids the test re-deriving the geometry and drifting out of sync.
    """
    p = params
    keys = ["slimFace", "jaw", "cheekbone", "enlargeEye", "eyeHeight", "eyeDistance",
            "noseWidth", "mouthWidth", "chinLength", "foreheadHeight"]
    if not any(p.get(k) for k in keys):
        h, w = img.shape[:2]
        if return_field:
            z = np.zeros((h, w), np.float32)
            return z.copy(), z.copy()
        return img.copy()

    h, w = img.shape[:2]
    x, y, bw, bh = face.box
    cx, cy = face.center
    lm = face.landmarks

    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)

    # ---- width controls -------------------------------------------------
    if p.get("slimFace"):
        k = p["slimFace"] / 100.0
        axis_displacement(dx, dy, (h, w), cx, cy + bh * 0.10,
                          bw * 0.62, bh * 0.68, amount_x=k * 0.22, falloff=1.8)
    if p.get("jaw"):
        k = p["jaw"] / 100.0
        axis_displacement(dx, dy, (h, w), cx, cy + bh * 0.30,
                          bw * 0.48, bh * 0.34, amount_x=k * 0.26, falloff=1.8)
    if p.get("cheekbone"):
        k = p["cheekbone"] / 100.0
        axis_displacement(dx, dy, (h, w), cx, cy - bh * 0.02,
                          bw * 0.56, bh * 0.22, amount_x=k * 0.16, falloff=1.8)

    # ---- eyes -----------------------------------------------------------
    eye_r, eye_l = face.eyes[0], face.eyes[1]
    eye_dist = float(np.linalg.norm(np.array(eye_l[:2]) - np.array(eye_r[:2])))
    for (ex, ey, ew, eh) in (eye_r, eye_l):
        ecx, ecy = ex + ew / 2, ey + eh / 2
        if p.get("enlargeEye"):
            scale_displacement(dx, dy, (h, w), ecx, ecy, ew * 0.80, eh * 1.05,
                               p["enlargeEye"] / 100.0 * 0.18)
        if p.get("eyeHeight"):
            axis_displacement(dx, dy, (h, w), ecx, ecy, ew * 0.80, eh * 1.05,
                              amount_y=p["eyeHeight"] / 100.0 * 0.14)
    if p.get("eyeDistance"):
        for (ex, ey, ew, eh) in (eye_r, eye_l):
            ecx, ecy = ex + ew / 2, ey + eh / 2
            sign = 1.0 if ecx < w / 2 else -1.0
            push_displacement(dx, dy, (h, w), ecx, ecy, eye_dist * 0.60,
                              -sign * p["eyeDistance"] / 100.0 * eye_dist * 0.10)

    # ---- nose -----------------------------------------------------------
    # With the 106-point model the nostril positions are measured; without it
    # they are extrapolated from the box, which is noticeably less accurate on
    # a face that is turned or smiling.
    if p.get("noseWidth"):
        k = p["noseWidth"] / 100.0
        wings = face.nose_wing_points()
        if wings is not None:
            wl, wr = wings
            ncx = float((wl[0] + wr[0]) / 2.0)
            ncy = float((wl[1] + wr[1]) / 2.0)
            half = float(abs(wr[0] - wl[0]) / 2.0)
            rx = max(half * 1.5, bw * 0.10)
            ry = max(bh * 0.10, 8.0)
        else:
            ncx, ncy = float(lm[2][0]), float(lm[2][1]) + bh * 0.06
            rx, ry = bw * 0.16, bh * 0.16
        axis_displacement(dx, dy, (h, w), ncx, ncy, rx, ry,
                          amount_x=k * 0.14, falloff=1.6)

    # ---- mouth ----------------------------------------------------------
    if p.get("mouthWidth"):
        k = p["mouthWidth"] / 100.0
        lips = face.lip_points()
        if lips is not None and len(lips) >= 2:
            mcx = float(lips[:, 0].mean())
            mcy = float(lips[:, 1].mean())
            half = float((lips[:, 0].max() - lips[:, 0].min()) / 2.0)
            axis_displacement(dx, dy, (h, w), mcx, mcy,
                              max(half * 1.25, 6.0), max(bh * 0.07, 5.0),
                              amount_x=k * 0.13, falloff=1.6)
        elif face.mouth:
            mx, my, mw, mh = face.mouth
            axis_displacement(dx, dy, (h, w), mx + mw / 2, my + mh / 2,
                              mw * 0.60, mh * 0.70, amount_x=k * 0.13)

    # ---- chin: positive lengthens, so content moves down (+y) -----------
    if p.get("chinLength"):
        k = p["chinLength"] / 100.0
        chin = face.dense.chin() if face.has_dense() else None
        if chin is not None:
            ccx, ccy = float(chin[0]), float(chin[1])
        else:
            ccx, ccy = cx, cy + bh * 0.42
        push_displacement(dx, dy, (h, w), ccx, ccy, bw * 0.34,
                          k * bh * 0.06, angle=np.pi / 2)

    # ---- forehead: positive raises the hairline, content moves up (-y) --
    if p.get("foreheadHeight"):
        k = p["foreheadHeight"] / 100.0
        push_displacement(dx, dy, (h, w), cx, cy - bh * 0.42, bw * 0.38,
                          k * bh * 0.05, angle=-np.pi / 2)

    if return_field:
        return dx, dy
    return warp(img, dx, dy)


# --------------------------------------------------------------------------
# body reshaping
# --------------------------------------------------------------------------

def reshape_body(img: np.ndarray, params: dict, faces=None,
                 pose=None, return_field: bool = False):
    """Proportion changes below the face: legs, waist, arms, neck, shoulders.

    Two modes:

    * **Pose-driven** (preferred): when a `Pose` is supplied, every band comes
      from real joint positions — shoulder, hip, knee and ankle — so the warp
      follows a subject who is off-centre, turned, or in a non-standing pose.
    * **Frame-derived fallback**: without a pose, bands are estimated from the
      image frame assuming a centred upright subject. This is a documented
      approximation and will be wrong for anything else.

    `return_field` returns (dx, dy) instead of the image; see `reshape_face`.
    """
    keys = ["legLengthen", "slimLeg", "slimWaist", "slimArm", "neckLength",
            "shoulderWidth", "heightGain", "bodySmooth"]
    if not any(params.get(k) for k in keys):
        h, w = img.shape[:2]
        if return_field:
            z = np.zeros((h, w), np.float32)
            return z.copy(), z.copy()
        return img.copy()

    h, w = img.shape[:2]
    cx = w / 2.0
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)

    use_pose = pose is not None and pose.hip_c is not None

    if use_pose:
        # ---- pose-driven bands ------------------------------------------
        sh = pose.shoulder_c
        hip = pose.hip_c
        ank = pose.ankle_c
        sh_w = max(pose.shoulder_w, 1.0)
        hip_w = max(pose.hip_w, 1.0)

        def band_at(t):
            """Point and width at torso fraction t (0 = shoulders, 1 = hips)."""
            c = sh + (hip - sh) * t
            wd = sh_w + (hip_w - sh_w) * t
            return c, wd

        # legs: hip -> ankle
        leg_c = (hip + ank) / 2 if ank is not None else None
        leg_len = pose.leg_len if pose.leg_len > 1 else pose.torso_h * 1.5
        leg_w = hip_w * 0.62

        if params.get("legLengthen") and leg_c is not None:
            k = params["legLengthen"] / 100.0
            axis_displacement(dx, dy, (h, w), leg_c[0], leg_c[1],
                              leg_w * 1.25, leg_len * 0.55,
                              amount_y=k * 0.16, falloff=1.6)
        if params.get("slimLeg") and leg_c is not None:
            k = params["slimLeg"] / 100.0
            axis_displacement(dx, dy, (h, w), leg_c[0], leg_c[1],
                              leg_w * 1.20, leg_len * 0.60,
                              amount_x=k * 0.14, falloff=1.6)
        if params.get("slimWaist"):
            k = params["slimWaist"] / 100.0
            c, wd = band_at(0.62)
            axis_displacement(dx, dy, (h, w), c[0], c[1],
                              wd * 0.52, pose.torso_h * 0.30,
                              amount_x=k * 0.16, falloff=1.7)
        if params.get("slimArm"):
            k = params["slimArm"] / 100.0
            # push each arm inward, anchored on the shoulder
            for idx in (L_SHOULDER, R_SHOULDER):
                if not pose.has(idx):
                    continue
                p = pose.pt(idx)
                side = 1.0 if p[0] > sh[0] else -1.0
                push_displacement(dx, dy, (h, w), p[0], p[1] + pose.torso_h * 0.30,
                                  sh_w * 0.55, -side * k * sh_w * 0.06, angle=0.0)
        if params.get("neckLength"):
            k = params["neckLength"] / 100.0
            head_h = pose.torso_h * 0.55
            push_displacement(dx, dy, (h, w), sh[0], sh[1] - pose.torso_h * 0.25,
                              sh_w * 0.60, k * head_h * 0.12, angle=-np.pi / 2)
        if params.get("shoulderWidth"):
            k = params["shoulderWidth"] / 100.0
            axis_displacement(dx, dy, (h, w), sh[0], sh[1] + pose.torso_h * 0.12,
                              sh_w * 0.80, pose.torso_h * 0.28,
                              amount_x=-k * 0.14, falloff=1.7)
        if params.get("heightGain"):
            k = params["heightGain"] / 100.0
            axis_displacement(dx, dy, (h, w), cx, h * 0.5, w * 0.5, h * 0.5,
                              amount_y=k * 0.06)
        top = float(sh[1] - pose.torso_h * 0.5)
    else:
        # ---- frame-derived fallback -------------------------------------
        if faces:
            f = faces[0]
            top = f.box[1] + f.box[3]
            head_h = f.box[3]
        else:
            top = h * 0.18
            head_h = h * 0.14

        body_h = max(h - top, 1.0)

        if params.get("legLengthen"):
            k = params["legLengthen"] / 100.0
            cy = top + body_h * 0.78
            axis_displacement(dx, dy, (h, w), cx, cy, w * 0.42, body_h * 0.30,
                              amount_y=k * 0.16)
        if params.get("slimLeg"):
            k = params["slimLeg"] / 100.0
            cy = top + body_h * 0.72
            axis_displacement(dx, dy, (h, w), cx, cy, w * 0.40, body_h * 0.34,
                              amount_x=k * 0.13)

        if params.get("slimWaist"):
            k = params["slimWaist"] / 100.0
            cy = top + body_h * 0.42
            axis_displacement(dx, dy, (h, w), cx, cy, w * 0.30, body_h * 0.16,
                              amount_x=k * 0.15)

        if params.get("slimArm"):
            k = params["slimArm"] / 100.0
            cy = top + body_h * 0.30
            for side in (-1, 1):
                ax = cx + side * w * 0.28
                push_displacement(dx, dy, (h, w), ax, cy, w * 0.16,
                                  -side * k * w * 0.02, angle=0.0)

        if params.get("neckLength"):
            k = params["neckLength"] / 100.0
            push_displacement(dx, dy, (h, w), cx, top + head_h * 0.10, head_h * 1.1,
                              k * head_h * 0.10, angle=-np.pi / 2)

        if params.get("shoulderWidth"):
            k = params["shoulderWidth"] / 100.0
            cy = top + body_h * 0.08
            axis_displacement(dx, dy, (h, w), cx, cy, w * 0.42, body_h * 0.12,
                              amount_x=-k * 0.12)

        if params.get("heightGain"):
            k = params["heightGain"] / 100.0
            axis_displacement(dx, dy, (h, w), cx, h * 0.5, w * 0.5, h * 0.5,
                              amount_y=k * 0.06)

    if return_field:
        return dx, dy

    out = warp(img, dx, dy)

    # ---- body contour smoothing ----------------------------------------
    if params.get("bodySmooth"):
        from .skin import smooth_skin
        m = np.full((h, w), 1.0, np.float32)
        m[: int(max(0.0, top))] = 0.0            # keep the face untouched
        # when a pose is available, restrict smoothing to the torso and legs
        # so hands and background are not blurred
        if use_pose and pose.hip_c is not None:
            keep = np.zeros((h, w), np.float32)
            keep[int(max(0, sh[1])):, :] = 1.0
            m = m * keep
        out = smooth_skin(out, m, amount=params["bodySmooth"], texture=45.0)

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def free_liquify(img: np.ndarray, strokes: list) -> np.ndarray:
    """Apply user-drawn liquify strokes.

    Each stroke is a dict: {x, y, radius, strength, angle} where angle is in
    degrees and measures the direction content is pushed (0 = right,
    90 = down). Composed in one pass.
    """
    if not strokes:
        return img.copy()
    h, w = img.shape[:2]
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    for s in strokes:
        push_displacement(dx, dy, (h, w), float(s["x"]), float(s["y"]),
                          float(s.get("radius", 60)),
                          float(s.get("strength", 10)),
                          angle=np.deg2rad(float(s.get("angle", 0))))
    return warp(img, dx, dy)
