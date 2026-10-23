"""Pose estimation and body-region derivation.

Wraps MediaPipe's pose landmarker to replace the geometric guesses in
`liquify.reshape_body` with real joint positions. Without this, body
reshaping has to assume a centred, upright subject and derives bands from
the frame; with it, the waist, hips, knees and ankles come from the subject.

The landmarker runs on a downscaled copy because body proportion work does
not need pixel-accurate joints, and the model is much faster on small input.
"""
from __future__ import annotations

from .paths import models_dir
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import cv2

MODEL_NAME = "pose_landmarker_lite.task"

# MediaPipe pose landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32


def model_path() -> str:
    return os.path.join(models_dir(), MODEL_NAME)


@dataclass
class Pose:
    """One detected person's joints, in image pixels."""
    landmarks: np.ndarray          # (33, 3): x, y, visibility
    width: int
    height: int
    score: float = 1.0

    # derived, filled by `derive`
    shoulder_c: Optional[np.ndarray] = None
    hip_c: Optional[np.ndarray] = None
    ankle_c: Optional[np.ndarray] = None
    torso_h: float = 0.0
    leg_len: float = 0.0
    shoulder_w: float = 0.0
    hip_w: float = 0.0
    body_angle: float = 0.0        # torso tilt from vertical, radians

    def pt(self, idx: int) -> np.ndarray:
        p = self.landmarks[idx]
        return np.array([p[0] * self.width, p[1] * self.height], np.float32)

    def vis(self, idx: int) -> float:
        return float(self.landmarks[idx][2])

    def has(self, *idxs: int, min_vis: float = 0.4) -> bool:
        return all(self.vis(i) >= min_vis for i in idxs)

    def derive(self) -> "Pose":
        """Compute the summary geometry the reshape stages need."""
        if self.has(L_SHOULDER, R_SHOULDER):
            self.shoulder_c = (self.pt(L_SHOULDER) + self.pt(R_SHOULDER)) / 2
            self.shoulder_w = float(np.linalg.norm(
                self.pt(L_SHOULDER) - self.pt(R_SHOULDER)))
        if self.has(L_HIP, R_HIP):
            self.hip_c = (self.pt(L_HIP) + self.pt(R_HIP)) / 2
            self.hip_w = float(np.linalg.norm(self.pt(L_HIP) - self.pt(R_HIP)))
        if self.has(L_ANKLE, R_ANKLE):
            self.ankle_c = (self.pt(L_ANKLE) + self.pt(R_ANKLE)) / 2

        if self.shoulder_c is not None and self.hip_c is not None:
            v = self.hip_c - self.shoulder_c
            self.torso_h = float(np.linalg.norm(v))
            # angle of the torso from the image's vertical axis
            self.body_angle = float(np.arctan2(v[0], max(v[1], 1e-6)))
        if self.hip_c is not None and self.ankle_c is not None:
            self.leg_len = float(np.linalg.norm(self.ankle_c - self.hip_c))
        return self

    # ------------------------------------------------------------------
    def waist_band(self, t: float) -> tuple:
        """Return (centre, half_width, half_height) for a band along the torso.

        `t` is 0 at the shoulders and 1 at the hips.
        """
        if self.shoulder_c is None or self.hip_c is None:
            return None
        c = self.shoulder_c + (self.hip_c - self.shoulder_c) * float(t)
        w = self.shoulder_w + (self.hip_w - self.shoulder_w) * float(t)
        return c, w * 0.62, self.torso_h * 0.22

    def leg_band(self, t: float) -> tuple:
        """Return (centre, half_width, half_height) along the legs, 0=hip 1=ankle."""
        if self.hip_c is None or self.ankle_c is None:
            return None
        c = self.hip_c + (self.ankle_c - self.hip_c) * float(t)
        return c, max(self.hip_w, 1.0) * 0.95, self.leg_len * 0.22


class PoseEstimator:
    """MediaPipe-backed estimator; reuse across images."""

    def __init__(self, max_poses: int = 2, min_confidence: float = 0.35,
                 work_long_edge: int = 768):
        path = model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"pose model not found at {path}. Download "
                "pose_landmarker_lite.task from the MediaPipe model page "
                "into the models/ directory.")
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._vision = vision
        self._mp = __import__("mediapipe")
        self._work_long_edge = work_long_edge
        base = mp_python.BaseOptions(model_asset_path=path)
        opts = vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=max_poses,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._lm = vision.PoseLandmarker.create_from_options(opts)

    def detect(self, img: np.ndarray) -> List[Pose]:
        """Detect poses in a float32 RGB image."""
        h, w = img.shape[:2]
        scale = min(1.0, self._work_long_edge / max(h, w))
        if scale < 1.0:
            small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = img

        u8 = (np.clip(small, 0, 1) * 255).astype(np.uint8)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=u8)
        res = self._lm.detect(mp_img)

        out: List[Pose] = []
        for lm in res.pose_landmarks:
            arr = np.array([[p.x, p.y, p.visibility] for p in lm], np.float32)
            pose = Pose(landmarks=arr, width=w, height=h).derive()
            out.append(pose)
        out.sort(key=lambda p: p.torso_h, reverse=True)
        return out


def pick_primary(poses: List[Pose]) -> Optional[Pose]:
    """Choose the subject the body controls should act on: the largest torso."""
    if not poses:
        return None
    return max(poses, key=lambda p: (p.torso_h, p.leg_len))
