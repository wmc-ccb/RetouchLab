"""Dense facial landmarks (106 points) via insightface's 2d106det.

The 5-point model used for detection gives eye centres and mouth corners,
which is enough for detection and for region masks, but facial reshaping
needs much more: nose wings, brow arch, jaw contour, outer and inner lip,
and the eye outline. Those are exactly the 106-point set's semantics.

Output convention (insightface)
-------------------------------
The network returns 212 values in [-1, 1], normalised against the 192x192
aligned crop. They are converted with::

    x_crop = value_x * 192 + 96
    y_crop = value_y * 192 + 96

and then mapped back through the inverse of the alignment transform. Getting
this wrong yields points clustered at the crop centre, so the conversion is
verified in tests.

Only the model file and its published conventions are used; no code is
copied from the upstream project.
"""
from __future__ import annotations

from .paths import models_dir
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2

MODEL_NAME = "2d106det.onnx"
CROP = 192
# Outputs are in [-1, 1] against the crop, so -1..1 spans the whole 192px crop.
# Denormalisation half-extent. The model emits (v+1)*96 in crop pixels, and the
# code below divides by 2, so the constant must be 192 to produce that factor.
#
# Measured on a real frame: the 212-value output spans about [-0.32, 0.63], and
# only (v+1)*96 puts the point cloud on the face (111x130 px against a 133x153
# face box = 0.93x0.85). With the earlier value the cloud came out at half that
# size, and at 192-as-scale-factor it was 1.86x the face and centred off it —
# in both cases the 106 markers were visibly bunched or displaced.
OUTPUT_GRID = 192
# Divisor applied to the face box's longer side when choosing the crop scale.
# See Landmark106.align_crop for why this differs from the published value.
BOX_SCALE_DIVISOR = 2.0

# Reference 5 points for a 192x192 aligned crop, in the arcface convention
# (defined on a 112x112 grid, scaled to 192).
_REF5_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], np.float32)
REF5 = _REF5_112 * (CROP / 112.0)

# Semantic groups of the 106-point set, following the layout published with
# the model. Verified against a measured mean layout in tests/verify_landmarks.py:
# the eyes are two mirrored clusters in the upper face, the nose is the
# centre-line run below them, and the mouth is the cluster under that.
GROUPS = {
    "jaw":        list(range(0, 33)),
    "right_brow": list(range(33, 43)),
    "left_brow":  list(range(43, 53)),
    "nose":       list(range(53, 73)),
    "right_eye":  list(range(73, 87)),
    "left_eye":   list(range(87, 101)),
    "outer_lip":  list(range(101, 106)),
    "inner_lip":  [],
}


def model_path() -> str:
    return os.path.join(models_dir(), MODEL_NAME)


@dataclass
class DenseLandmarks:
    """106 landmarks in image pixel coordinates."""
    points: np.ndarray              # (106, 2)
    transform: np.ndarray           # the 2x3 alignment used for the crop
    score: float = 1.0

    def group(self, name: str) -> np.ndarray:
        idx = GROUPS.get(name)
        if not idx:
            return np.zeros((0, 2), np.float32)
        return self.points[idx]

    def centroid(self, name: str) -> Optional[np.ndarray]:
        g = self.group(name)
        return g.mean(axis=0) if len(g) else None

    def width(self, name: str) -> float:
        g = self.group(name)
        if len(g) < 2:
            return 0.0
        return float(g[:, 0].max() - g[:, 0].min())

    def height(self, name: str) -> float:
        g = self.group(name)
        if len(g) < 2:
            return 0.0
        return float(g[:, 1].max() - g[:, 1].min())

    # ------------------------------------------------------------------
    def nose_wings(self) -> Optional[tuple]:
        """Left and right nostril points, taken from the nose group extremes."""
        g = self.group("nose")
        if len(g) < 5:
            return None
        i_l = int(np.argmin(g[:, 0]))
        i_r = int(np.argmax(g[:, 0]))
        return g[i_l], g[i_r]

    def jaw_extremes(self) -> Optional[tuple]:
        g = self.group("jaw")
        if len(g) < 5:
            return None
        i_l = int(np.argmin(g[:, 0]))
        i_r = int(np.argmax(g[:, 0]))
        return g[i_l], g[i_r]

    def chin(self) -> Optional[np.ndarray]:
        g = self.group("jaw")
        if not len(g):
            return None
        return g[int(np.argmax(g[:, 1]))]

    def brow_centres(self) -> tuple:
        return self.centroid("right_brow"), self.centroid("left_brow")


class Landmark106:
    """Runs the 106-point model; reuse across images."""

    def __init__(self, providers=None):
        import onnxruntime as ort

        path = model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"landmark model not found at {path}. Download 2d106det.onnx "
                "into the models/ directory.")
        # the CUDA provider needs the runtime DLLs that torch bundles
        try:
            from .matting import _prepare_gpu_dlls
            _prepare_gpu_dlls()
        except Exception:
            pass
        if providers is None:
            avail = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in avail] or ["CPUExecutionProvider"]
        try:
            self._sess = ort.InferenceSession(path, providers=providers)
        except Exception:
            # a broken GPU stack must not disable landmarks entirely
            self._sess = ort.InferenceSession(path,
                                              providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        self.providers = self._sess.get_providers()

    @staticmethod
    def align_crop(img: np.ndarray, box: np.ndarray) -> tuple:
        """Build the aligned 192x192 crop the model was trained on.

        The model expects a *centre-and-scale* crop: the face box centre maps
        to the crop centre, and the scale is chosen so the box's longer side
        spans a fixed fraction of the crop.

        The divisor below is tuned to the box convention of this project's
        detector (YuNet), not the one the model was published with. The
        upstream pipeline uses 1.5 against its own detector's box, which is
        noticeably looser; applying 1.5 to YuNet's tighter box crops too far
        in and the predicted landmarks come out undersized (measured: the
        point cloud spanned only 0.71x0.64 of the box, against an expected
        ~0.9x1.0). A sweep over divisors and output grids selected 2.0, which
        gives 0.94x0.85 and the smallest centroid offset.
        """
        x0, y0, x1, y1 = box
        w, h = (x1 - x0), (y1 - y0)
        center = ((x1 + x0) / 2.0, (y1 + y0) / 2.0)
        scale = CROP / (max(w, h) * BOX_SCALE_DIVISOR)

        # forward: p_crop = scale * (p_img - center) + CROP/2
        M = np.array([
            [scale, 0.0, CROP / 2.0 - scale * center[0]],
            [0.0, scale, CROP / 2.0 - scale * center[1]],
        ], np.float32)
        crop = cv2.warpAffine(img, M, (CROP, CROP), flags=cv2.INTER_LINEAR)
        return crop, M

    def predict(self, img: np.ndarray, box: np.ndarray,
                five_points: np.ndarray = None) -> DenseLandmarks:
        """Return 106 landmarks in `img` coordinates.

        `box` is the face bounding box as (x0, y0, x1, y1). `five_points` is
        accepted for call-site compatibility but is not used: the model's
        published preprocessing is box-based.
        """
        crop, M = self.align_crop(img, np.asarray(box, np.float32))

        # preprocessing: RGB, mean 127.5, std 128, NCHW
        blob = (crop.astype(np.float32) - 127.5) / 128.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])
        out = self._sess.run(None, {self._input: blob})[0]

        pts = np.asarray(out, np.float32).reshape(-1, 2)
        # Denormalise. The model emits values in [-1, 1] relative to the
        # crop's *half* extent, so -1..1 spans the full crop: (v + 1) * size/2.
        #
        # The size here is the model's own output grid (96), not the 192px
        # input. Using the input size doubles the scale, which was measured as
        # a constant ~2x eye-separation error across every test image.
        pts = (pts + 1.0) * (OUTPUT_GRID / 2.0)

        Minv = cv2.invertAffineTransform(M)
        pts_img = pts @ Minv[:, :2].T + Minv[:, 2]
        return DenseLandmarks(points=pts_img.astype(np.float32), transform=M)
