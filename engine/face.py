"""Face detection and landmark helpers.

Wraps OpenCV's YuNet detector (a small ONNX model) and derives the facial
regions the retouch stages need: skin, eyes, mouth/teeth, brows and cheeks.

The reference application models people as clusters with a type tag
(PT_F / PT_M / PT_C / PT_O / PT_O_M). We do not reproduce that proprietary
classification; we expose a simple, explainable age/gender-agnostic region
model instead and document the difference.
"""
from __future__ import annotations

from .paths import models_dir
import os
import threading
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import cv2

from .imgutil import feather, radial_falloff, smoothstep, meshgrid_f32

MODEL_NAME = "face_detection_yunet_2023mar.onnx"


def model_path() -> str:
    return os.path.join(models_dir(), MODEL_NAME)


def _plateau(h: int, w: int, cx: float, cy: float,
             rx: float, ry: float, rim: float = 1.30) -> np.ndarray:
    """Face gate with a full-weight plateau and a rim taper.

    1.0 inside the ellipse defined by (rx, ry), falling smoothly to 0 at
    `rim` times that radius. Compared with a plain radial falloff this keeps
    the whole face at full weight, so operations that multiply by the mask do
    not lose strength toward the forehead, temples and jaw.
    """
    yy, xx = meshgrid_f32(h, w)
    d = np.sqrt(((xx - cx) / max(rx, 1e-3)) ** 2 + ((yy - cy) / max(ry, 1e-3)) ** 2)
    return (1.0 - smoothstep(1.0, max(rim, 1.001), d)).astype(np.float32)


@dataclass
class Face:
    """One detected face plus the landmark-derived regions."""
    box: tuple                      # (x, y, w, h) in pixels
    landmarks: np.ndarray           # (5, 2): right eye, left eye, nose, right mouth, left mouth
    score: float
    # filled by build_regions
    eyes: List[tuple] = field(default_factory=list)
    mouth: Optional[tuple] = None
    skin_mask: Optional[np.ndarray] = None      # skin for smoothing / tone
    region_mask: Optional[np.ndarray] = None    # skin for blemish work (geometry only)
    soft_shade: Optional[np.ndarray] = None     # areas that are naturally darker
    dense: Optional[object] = None              # DenseLandmarks (106 points) when available

    # ------------------------------------------------------------------
    # dense-landmark accessors: prefer the 106-point model when present,
    # because the 5-point set only marks eye centres and mouth corners,
    # so nose wings, brow arch and jaw contour had to be extrapolated.
    # ------------------------------------------------------------------
    def has_dense(self) -> bool:
        return self.dense is not None and len(getattr(self.dense, "points", [])) == 106

    def dense_group(self, name: str) -> Optional[np.ndarray]:
        if not self.has_dense():
            return None
        g = self.dense.group(name)
        return g if len(g) else None

    def nose_wing_points(self) -> Optional[tuple]:
        """Left/right nostril points; falls back to an estimate from the box."""
        if self.has_dense():
            return self.dense.nose_wings()
        x, y, bw, bh = self.box
        nx, ny = self.landmarks[2]
        half = bw * 0.09
        return (np.array([nx - half, ny + bh * 0.10], np.float32),
                np.array([nx + half, ny + bh * 0.10], np.float32))

    def jaw_points(self) -> Optional[np.ndarray]:
        """Jaw contour points; falls back to None (caller uses the box)."""
        return self.dense_group("jaw")

    def brow_points(self, side: str) -> Optional[np.ndarray]:
        return self.dense_group(f"{side}_brow")

    def eye_outline(self, side: str) -> Optional[np.ndarray]:
        return self.dense_group(f"{side}_eye")

    def lip_points(self) -> Optional[np.ndarray]:
        if not self.has_dense():
            return None
        pts = [self.dense_group("outer_lip"), self.dense_group("inner_lip")]
        pts = [p for p in pts if p is not None]
        return np.vstack(pts) if pts else None

    @property
    def center(self) -> tuple:
        x, y, w, h = self.box
        return (x + w / 2.0, y + h / 2.0)

    @property
    def area(self) -> float:
        return float(self.box[2] * self.box[3])


class FaceDetector:
    """YuNet-backed detector; safe to reuse across images (and across threads).

    cv2.FaceDetectorYN 的底层 Net **不是线程安全的**：服务端线程池并发
    处理 /api/render、/api/faces 时，两个线程同时进 setInputSize/detect
    会触发原生层断言（buf.shape() == m.shape()），偶发直接把整个进程
    拖进死循环（GIL 被原生代码占住，全端点无响应，实测 CPU 烧到
    1000s+）。所有调用必须过同一把锁。
    """

    _detect_lock = threading.Lock()

    def __init__(self, score_threshold: float = 0.6, nms_threshold: float = 0.3,
                 top_k: int = 50, dense: bool = True):
        path = model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"face model not found at {path}. "
                "Download face_detection_yunet_2023mar.onnx into the models/ directory."
            )
        self._det = cv2.FaceDetectorYN.create(
            path, "", (320, 320), score_threshold, nms_threshold, top_k)
        # the 106-point model is optional: without it the engine falls back to
        # 5-point landmarks and extrapolated regions
        self._dense = None
        self._want_dense = dense

    def _landmarker(self):
        if not self._want_dense:
            return None
        if self._dense is None:
            try:
                from .landmarks import Landmark106
                self._dense = Landmark106()
            except Exception:
                self._want_dense = False
        return self._dense

    def detect(self, img: np.ndarray, max_faces: int = 8,
               dense: bool = None) -> List[Face]:
        """Detect faces in a float32 RGB image.

        When the 106-point model is available it is run on each face and
        attached as `Face.dense`; failure there is non-fatal and leaves the
        face usable with 5-point landmarks only.
        """
        h, w = img.shape[:2]
        bgr = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        with FaceDetector._detect_lock:
            self._det.setInputSize((w, h))
            _, raw = self._det.detect(bgr)
        faces: List[Face] = []
        if raw is None:
            return faces
        for row in raw[:max_faces]:
            x, y, bw, bh = row[:4]
            pts = row[4:14].reshape(5, 2)
            score = float(row[14])
            faces.append(Face(box=(float(x), float(y), float(bw), float(bh)),
                              landmarks=pts.astype(np.float32), score=score))
        faces.sort(key=lambda f: f.area, reverse=True)

        want = self._want_dense if dense is None else dense
        lm = self._landmarker() if want else None
        if lm is not None:
            for f in faces:
                x, y, bw, bh = f.box
                box = np.array([x, y, x + bw, y + bh], np.float32)
                try:
                    f.dense = lm.predict(img, box)
                except Exception:
                    f.dense = None
        return faces

    # ------------------------------------------------------------------
    # region derivation
    # ------------------------------------------------------------------

    @staticmethod
    def build_regions(face: Face, shape: tuple, feather_px: float = None) -> Face:
        """Derive eye/mouth boxes and the two region masks for one face."""
        h, w = shape[:2]
        x, y, bw, bh = face.box
        cx, cy = face.center
        lm = face.landmarks

        eye_r, eye_l = lm[0], lm[1]
        mouth_r, mouth_l = lm[3], lm[4]

        # eye boxes: the 5-point model gives centres, so size them from inter-eye distance
        eye_dist = float(np.linalg.norm(eye_l - eye_r))
        ew = eye_dist * 0.42
        eh = eye_dist * 0.26
        face.eyes = []
        for c in (eye_r, eye_l):
            face.eyes.append((float(c[0] - ew / 2), float(c[1] - eh / 2), float(ew), float(eh)))

        # mouth: the 5-point model gives the two mouth *corners*, so the lip
        # band sits between them. Height is derived from the inter-eye
        # distance and centred on the corner line, not pushed downward, or the
        # cut-out lands on the chin and leaves the lips unprotected.
        mx = float((mouth_r[0] + mouth_l[0]) / 2.0)
        my = float((mouth_r[1] + mouth_l[1]) / 2.0)
        mw = float(np.linalg.norm(mouth_l - mouth_r)) * 1.25
        mh = eye_dist * 0.40
        face.mouth = (mx - mw / 2, my - mh / 2, mw, mh)

        # ---- local window --------------------------------------------------
        # Every mask below is zero well before the edges of a large preview,
        # so building them full-frame wastes most of the work: on a 1600px
        # image the accumulations below dominated `build_regions` (measured:
        # 1.2 s). Compute inside a window around the face instead and paste
        # the results back into full-size arrays.
        pad = 0.55
        wx0 = max(0, int(x - bw * (0.5 + pad)))
        wy0 = max(0, int(y - bh * (0.5 + pad)))
        wx1 = min(w, int(x + bw * (0.5 + pad)) + 1)
        wy1 = min(h, int(y + bh * (0.5 + pad)) + 1)
        lh, lw = (wy1 - wy0), (wx1 - wx0)
        lcx, lcy = cx - wx0, cy - wy0

        def _local(m):
            """Paste a local-window mask into a full-size zero array."""
            out = np.zeros((h, w), np.float32)
            if m is not None:
                out[wy0:wy1, wx0:wx1] = m
            return out

        # `protect` and `shade` are accumulated with `maximum`, so they can be
        # built directly in local coordinates.
        protect_l = np.zeros((lh, lw), np.float32)

        def _cut(cx_, cy_, rx_, ry_, strength=1.0):
            nonlocal protect_l
            protect_l = np.maximum(
                protect_l,
                radial_falloff(lh, lw, cx_ - wx0, cy_ - wy0, rx_, ry_) * strength)

        for (ex, ey, ewx, ehx) in face.eyes:
            ecx, ecy = ex + ewx / 2, ey + ehx / 2
            # Protect the eyeball itself: lid, lash line, iris, sclera.
            # Broad socket shading is handled in the detector (which rejects
            # large soft gradients), so this cut-out stays close to the eye and
            # leaves the forehead and cheek available for real blemishes.
            _cut(ecx, ecy, ewx * 1.05, ehx * 1.30)
            # brow: a narrow band just above the eye
            _cut(ecx, ecy - ehx * 1.60, ewx * 1.05, ehx * 0.50)
        if face.mouth:
            mx2, my2, mw2, mh2 = face.mouth
            mcx, mcy = mx2 + mw2 / 2, my2 + mh2 / 2
            # The lip band is taller than the corner-to-corner box suggests:
            # smoothing the lips erases them, so cut a region that covers the
            # full upper and lower lip plus a margin.
            _cut(mcx, mcy, mw2 * 0.62, mh2 * 1.55)
            _cut(mcx, mcy - mh2 * 0.55, mw2 * 0.50, mh2 * 0.75)   # philtrum
            _cut(mcx, mcy + mh2 * 0.60, mw2 * 0.50, mh2 * 0.70)   # chin crease
        # nostrils
        nx, ny = lm[2]
        _cut(float(nx), float(ny) + bh * 0.12, bw * 0.14, bh * 0.06)
        # `protect` is built from hard-ish radial falloffs; feathering it keeps
        # the skin gate free of visible edges where a cut-out meets the face.
        protect = _local(feather(protect_l, max(2.0, min(bw, bh) * 0.035)))

        # ---- soft-shade map ----------------------------------------------
        # Regions that are *supposed* to be darker than their surroundings.
        # A blemish detector responds to these broad gradients as if they were
        # spots; healing them lifts the whole area and produces a visible ring.
        # Unlike `protect`, this map is a soft suppression weight rather than a
        # hard exclusion, so genuine spots inside it can still be found.
        shade_l = np.zeros((lh, lw), np.float32)

        def _shade(cx_, cy_, rx_, ry_, weight):
            nonlocal shade_l
            shade_l = np.maximum(
                shade_l,
                radial_falloff(lh, lw, cx_ - wx0, cy_ - wy0, rx_, ry_) * weight)

        for (ex, ey, ewx, ehx) in face.eyes:
            ecx, ecy = ex + ewx / 2, ey + ehx / 2
            # the orbital socket, including the under-eye hollow
            _shade(ecx, ecy + ehx * 0.55, ewx * 1.15, ehx * 1.55, 0.92)
        if face.mouth:
            mx3, my3, mw3, mh3 = face.mouth
            # the crease under the lower lip
            _shade(mx3 + mw3 / 2, my3 + mh3 * 0.95, mw3 * 0.45, mh3 * 0.55, 0.75)
        # sides of the nose
        for sign in (-1, 1):
            _shade(float(nx) + sign * bw * 0.075, float(ny) + bh * 0.10,
                   bw * 0.055, bh * 0.075, 0.70)
        face.soft_shade = _local(shade_l)

        # ---- region for blemish work (geometry only) ---------------------
        # A plain radial falloff wastes most of its area on the transition: at
        # 0.7 of the radius it is already below 0.4, which silently suppresses
        # real blemishes near the forehead and jaw line. Use a plateau that
        # holds full weight across the face and tapers only just outside it.
        #
        # The plateau must stay *inside* the face box. A generous rim would
        # extend the gate over hair and background, and because the blemish
        # detector normalises against this region, that leaks skin operations
        # into the background and shows up as a bright rectangle around the
        # head.
        region_l = _plateau(lh, lw, lcx, lcy - bh * 0.02,
                            bw * 0.36, bh * 0.40, rim=1.42)
        region_l = region_l * (1.0 - protect_l)
        face.region_mask = _local(region_l)

        # ---- region for smoothing and tone work --------------------------
        skin_l = _plateau(lh, lw, lcx, lcy - bh * 0.02,
                          bw * 0.36, bh * 0.40, rim=1.42)
        skin_l = skin_l * (1.0 - protect_l)
        # taper the lower edge into the neck
        lyy, _lxx = meshgrid_f32(lh, lw)
        skin_l *= 1.0 - smoothstep(cy + bh * 0.28, cy + bh * 0.62, lyy + wy0)

        if feather_px:
            skin_l = feather(skin_l, feather_px)
        face.skin_mask = _local(skin_l)
        return face


def skin_tone_mask(img: np.ndarray, face: Face = None) -> np.ndarray:
    """Skin likelihood used for *smoothing and tone* work.

    Combines the geometric face region (which already excludes eyes, brows,
    nostrils and mouth) with a chroma rule in YCrCb.

    Note: blemish *detection* must not use a chroma gate. Blemishes are colour
    outliers by definition, so a chroma test attenuates exactly the pixels
    that need healing. Use `face_region_mask` for that stage.
    """
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    cr = ycc[..., 1]
    cb = ycc[..., 2]
    m = ((cr > 0.52) & (cr < 0.72) & (cb > 0.40) & (cb < 0.52)).astype(np.float32)
    m = feather(m, 2.0)
    if face is not None and face.skin_mask is not None:
        m = m * face.skin_mask
    return m.astype(np.float32)


def face_region_mask(img: np.ndarray, face: Face = None) -> np.ndarray:
    """Geometric skin region for blemish detection and healing.

    Selects *where* to look without judging *what colour* a pixel is, so
    colour-outlier pixels (the blemishes) survive the gate. Eyes, brows,
    nostrils and mouth are cut out, so a dark iris is never treated as a spot.

    The region is deliberately generous: a mask that falls off inside the face
    suppresses detection near the rim, which is a silent failure.
    """
    if face is not None:
        if face.region_mask is not None:
            return face.region_mask.astype(np.float32)
        h, w = img.shape[:2]
        x, y, bw, bh = face.box
        cx, cy = face.center
        return radial_falloff(h, w, cx, cy - bh * 0.02,
                              bw * 0.62, bh * 0.70).astype(np.float32)

    # no face: fall back to a widened chroma mask so spots are not excluded
    h, w = img.shape[:2]
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    cr = ycc[..., 1]
    cb = ycc[..., 2]
    m = ((cr > 0.48) & (cr < 0.76) & (cb > 0.36) & (cb < 0.56)).astype(np.float32)
    return feather(m, 3.0)


def detect_skin_global(img: np.ndarray) -> np.ndarray:
    """Whole-image skin mask when no face is detected."""
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    cr = ycc[..., 1]
    cb = ycc[..., 2]
    m = ((cr > 0.52) & (cr < 0.72) & (cb > 0.40) & (cb < 0.52)).astype(np.float32)
    return feather(m, 3.0)
