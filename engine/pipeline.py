"""End-to-end retouching pipeline.

`RetouchPipeline.run()` applies the stages in the order a portrait retoucher
would: geometry -> skin -> colour -> optics -> grain -> output. Every stage is
skippable, and each parameter has a documented default so an empty parameter
set is a no-op.

Parameter groups
----------------
geometry : slimFace, jaw, cheekbone, enlargeEye, eyeHeight, eyeDistance,
           noseWidth, mouthWidth, chinLength, foreheadHeight,
           legLengthen, slimLeg, slimWaist, slimArm, neckLength,
           shoulderWidth, heightGain, bodySmooth
skin     : spotless_acne, spotless_spot, spotless_texture, even_tone,
           smooth, texture, freq_low, freq_high, removeEyeBag,
           keepSilkworm, eyeHighlight, teethWhitening, teethDeYellow
tone     : exposure, contrast, brightness, highlights, shadows, whites,
           blacks, texture_tone, clarity, dehaze, vibrance, saturation
colour   : temp, tint, wb_mode, hsl_<band>_<hue|sat|lum>,
           cam_r_hue, cam_r_sat, cam_g_hue, cam_g_sat, cam_b_hue, cam_b_sat
detail   : sharpening, sharp_radius, sharp_detail, nr_luminance, nr_color,
           nr_detail, nr_smoothness, ai_denoise, ai_denoise_strength
optics   : distortion, vignetting, vignetting_midpoint
grain    : grain_amount, grain_size, grain_roughness, grain_shadows,
           grain_midtones, grain_highlights
matting  : matte_enabled, bg_blur, bg_darken, bg_desat, bg_mode, bg_color_r/g/b,
           matte_feather, matte_size
output   : long_edge, protect_background
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import cv2

from . import tone as T
from . import skin as S
from . import liquify as L
from .face import FaceDetector, Face
from .imgutil import fit_long_edge, load_image, save_image, encode_jpeg

# every parameter the pipeline understands, with its neutral value
DEFAULTS: Dict[str, float] = {
    # geometry
    "slimFace": 0.0, "jaw": 0.0, "cheekbone": 0.0,
    "enlargeEye": 0.0, "eyeHeight": 0.0, "eyeDistance": 0.0,
    "noseWidth": 0.0, "mouthWidth": 0.0, "chinLength": 0.0, "foreheadHeight": 0.0,
    "legLengthen": 0.0, "slimLeg": 0.0, "slimWaist": 0.0, "slimArm": 0.0,
    "neckLength": 0.0, "shoulderWidth": 0.0, "heightGain": 0.0, "bodySmooth": 0.0,
    # skin
    "spotless_acne": 0.0, "spotless_spot": 0.0, "spotless_texture": 0.0,
    "even_tone": 0.0, "smooth": 0.0, "texture": 50.0,
    "freq_low": 0.0, "freq_high": 0.0,
    "removeEyeBag": 0.0, "keepSilkworm": 0.0, "eyeHighlight": 0.0,
    "teethWhitening": 0.0, "teethDeYellow": 0.0,
    # tone
    "exposure": 0.0, "contrast": 0.0, "brightness": 0.0,
    "highlights": 0.0, "shadows": 0.0, "whites": 0.0, "blacks": 0.0,
    "texture_tone": 0.0, "clarity": 0.0, "dehaze": 0.0,
    "vibrance": 0.0, "saturation": 0.0,
    # colour
    "temp": 0.0, "tint": 0.0,
    "cam_r_hue": 0.0, "cam_r_sat": 0.0, "cam_g_hue": 0.0, "cam_g_sat": 0.0,
    "cam_b_hue": 0.0, "cam_b_sat": 0.0,
    # detail
    "sharpening": 0.0, "sharp_radius": 1.0, "sharp_detail": 25.0,
    "nr_luminance": 0.0, "nr_color": 0.0, "nr_detail": 50.0, "nr_smoothness": 50.0,
    "ai_denoise": 0.0, "ai_denoise_strength": 50.0,
    # optics
    "distortion": 0.0, "vignetting": 0.0, "vignetting_midpoint": 50.0, "ca_amount": 0.0,
    # grain
    "grain_amount": 0.0, "grain_size": 25.0, "grain_roughness": 50.0,
    "grain_shadows": 0.0, "grain_midtones": 0.0, "grain_highlights": 0.0,
    # ---- three-way colour grading (颜色分级) ------------------------------
    # The reference application grades in three overlapping zones. Each zone
    # carries a hue rotation (degrees), a saturation offset and a luminance
    # offset; `grade_balance` moves the zone crossover and `grade_blending`
    # controls how far they overlap.
    "grade_global_hue": 0.0, "grade_global_sat": 0.0, "grade_global_lum": 0.0,
    "grade_mid_hue": 0.0, "grade_mid_sat": 0.0, "grade_mid_lum": 0.0,
    "grade_high_hue": 0.0, "grade_high_sat": 0.0, "grade_high_lum": 0.0,
    "grade_balance": 0.0, "grade_blending": 0.5,
    # ---- matting / background (AI) ---------------------------------------
    "matte_enabled": 0.0,          # 1 enables the MODNet portrait matte
    "bg_mode": 0.0,                # 0 = keep original backdrop, 1 = solid colour
    "bg_blur": 0.0,                # background blur radius in px
    "bg_darken": 0.0,              # 0..1
    "bg_desat": 0.0,               # 0..1
    "bg_color_r": 1.0,             # used when bg_mode = 1
    "bg_color_g": 1.0,
    "bg_color_b": 1.0,
    "matte_feather": 1.0,          # extra softening of the matte edge, px
    "matte_size": 768.0,           # inference long edge for the matte
    "matte_refine": 1.0,           # 1 = snap the matte to image edges
    # ---- 曲线与色阶（字典参数，见 normalize 的特殊处理）--------------------
    # `levels` = {master|r|g|b: [in_black, in_white, gamma, out_black, out_white]}
    # `curves` = {rgb|r|g|b: [[x, y], ...]}
    # 缺省即"不处理"，所以它们是 None 而不是 0。
    # output
    "long_edge": 0.0,
    # 0..1: how strongly global tone work is held off the background.
    # 0 disables the subject mask (plain global grading); 1 is the default.
    "protect_background": 1.0,
}

HSL_BANDS = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]
for _b in HSL_BANDS:
    DEFAULTS[f"hsl_{_b}_hue"] = 0.0
    DEFAULTS[f"hsl_{_b}_sat"] = 0.0
    DEFAULTS[f"hsl_{_b}_lum"] = 0.0


class RetouchPipeline:
    """Stateful pipeline; reuses the face detector across runs."""

    def __init__(self, detect_faces: bool = True, detect_pose: bool = True,
                 use_matte_for_tone: bool = True, use_semantic: bool = True):
        self.detect_faces = detect_faces
        self.detect_pose = detect_pose
        # Use the portrait matte as the subject boundary for background
        # protection when the model is available. Costs one matte inference per
        # distinct image; disabled automatically if the model is missing.
        self.use_matte_for_tone = use_matte_for_tone
        # Use the semantic segmentation (selfie_multiclass) for the skin gate
        # and the subject boundary. One inference per image; it replaces the
        # chroma skin heuristic and the face-box subject approximation.
        self.use_semantic = use_semantic
        self._detector: Optional[FaceDetector] = None
        self._pose: Optional[object] = None
        self._matte_cache: Dict[int, np.ndarray] = {}
        self._sem_cache: dict = {}

    def _cached_semantic(self, img: np.ndarray):
        """Semantic regions for `img`, memoised by array identity.

        Mirrors `_cached_matte`: the identity key means a caller that keeps
        handing over the same array (which the pipeline does within one run)
        pays for the inference once, and the cache is bounded so a long-lived
        server does not accumulate every image it has served.
        """
        if not self.use_semantic:
            return None
        key = id(img)
        hit = self._sem_cache.get(key)
        if hit is not None and hit.category.shape[:2] == img.shape[:2]:
            return hit
        try:
            from . import semantic as SEM

            if not SEM.available():
                self.use_semantic = False
                return None
            reg = SEM.segment(img)
            if len(self._sem_cache) > 3:
                self._sem_cache.clear()
            self._sem_cache[key] = reg
            return reg
        except Exception:
            self.use_semantic = False
            return None

    def _cached_matte(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Matte for `img`, memoised by array identity.

        The subject mask is rebuilt for every tone operation, so without this
        the matte would be re-inferred a dozen times per run.
        """
        if not self.use_matte_for_tone:
            return None
        key = id(img)
        hit = self._matte_cache.get(key)
        if hit is not None and hit.shape[:2] == img.shape[:2]:
            return hit
        try:
            from . import matting as MT

            if not MT.available():
                self.use_matte_for_tone = False
                return None
            m = MT.alpha(img, size=640)
            # bound the cache: a long-lived server would otherwise hold every
            # image it has ever seen
            if len(self._matte_cache) > 4:
                self._matte_cache.clear()
            self._matte_cache[key] = m
            return m
        except Exception:
            self.use_matte_for_tone = False
            return None

    # ------------------------------------------------------------------
    def detector(self) -> Optional[FaceDetector]:
        if not self.detect_faces:
            return None
        if self._detector is None:
            try:
                self._detector = FaceDetector()
            except FileNotFoundError:
                self.detect_faces = False
        return self._detector

    def pose_estimator(self):
        """Lazy MediaPipe pose estimator; returns None when unavailable."""
        if not self.detect_pose:
            return None
        if self._pose is None:
            try:
                from .pose import PoseEstimator
                self._pose = PoseEstimator()
            except Exception:
                # mediapipe missing or the model file is absent: body controls
                # fall back to frame-derived bands, which is documented
                self.detect_pose = False
        return self._pose

    # ---- 只读探活（供 /api/health 用，不触发模型加载）-------------------
    def detector_loaded(self) -> bool:
        """人脸检测器是否已就绪。

        与 `detector()` 的区别：不构造检测器。健康检查若调用
        `detector()` 会真的把 ONNX 读进内存（首次 2~4 秒），
        探活就会被模型加载拖慢。这里只回答"能不能用"。
        """
        if not self.detect_faces:
            return False
        if self._detector is not None:
            return True
        # 还没加载：看模型文件在不在，在就报可用
        try:
            from .face import model_path as _face_model_path
            import os as _os
            return _os.path.exists(_face_model_path())
        except Exception:
            return False

    def pose_loaded(self) -> bool:
        """姿态模型是否可用（同样不触发加载）。"""
        if not self.detect_pose:
            return False
        if self._pose is not None:
            return True
        try:
            from .pose import model_path as _pose_model_path
            import os as _os
            return _os.path.exists(_pose_model_path())
        except Exception:
            return False

    # ------------------------------------------------------------------
    @staticmethod
    def normalize(params: dict) -> Dict[str, Any]:
        """Merge caller parameters over the neutral defaults.

        Most parameters are scalars and get a numeric default. Three are
        structures and are passed through untouched:

        * `local_adjustments` — a list of mask entries;
        * `levels` — a dict of per-channel level tuples;
        * `curves` — a dict of per-channel control-point lists.

        They only exist in the incoming params, so an absent key correctly means
        "no such edit" rather than a neutral number.
        """
        out = {k: v for k, v in DEFAULTS.items()}
        out["local_adjustments"] = []
        out["levels"] = None
        out["curves"] = None
        out["wb_gains"] = None
        for k, v in (params or {}).items():
            if v is None:
                continue
            if k in ("local_adjustments", "levels", "curves", "wb_gains"):
                out[k] = v
                continue
            out[k] = float(v) if not isinstance(v, str) else v
        return out

    @staticmethod
    def subject_mask(shape: tuple, faces, pose=None,
                     background_weight: float = 0.35,
                     matte: np.ndarray = None,
                     semantic: np.ndarray = None) -> np.ndarray:
        """Weight map that keeps global tone work off the background.

        The map is 1 inside the subject and `background_weight` outside, so
        grading stays coherent across edges instead of switching abruptly.
        Real retouchers solve this with a subject mask; this is the same idea
        built from the detections already on hand.

        Precedence, best first:

        1. `semantic` — the segmentation's non-background classes. Follows hair
           and shoulders *and* keeps accessories, which a portrait matte often
           discards; it also exists on a body shot where no face is found.
        2. `matte` — the portrait alpha.
        3. the face box / pose fallback — a padded ellipse per detected face.
        """
        from .face import _plateau

        h, w = shape[:2]

        if semantic is not None and semantic.shape[:2] == (h, w):
            # remap the 0..1 range into [bg, 1] so the background is damped
            # rather than cut out
            m = background_weight + np.clip(semantic, 0.0, 1.0) * (1.0 - background_weight)
            return np.clip(m, 0.0, 1.0).astype(np.float32)

        if matte is not None and matte.shape[:2] == (h, w):
            # remap the matte's 0..1 range into [bg, 1] so the background is
            # damped rather than cut out
            m = background_weight + np.clip(matte, 0.0, 1.0) * (1.0 - background_weight)
            return np.clip(m, 0.0, 1.0).astype(np.float32)

        m = np.full((h, w), float(background_weight), np.float32)

        for f in (faces or []):
            x, y, bw, bh = f.box
            cx, cy = f.center
            # generous: head, hair and shoulders
            m = np.maximum(m, _plateau(h, w, cx, cy + bh * 0.15,
                                       bw * 0.62, bh * 0.78, rim=1.45))

        if pose is not None and getattr(pose, "hip_c", None) is not None \
                and getattr(pose, "ankle_c", None) is not None:
            top = pose.shoulder_c
            bot = pose.ankle_c
            span = max(pose.torso_h + pose.leg_len, 1.0)
            body = np.zeros((h, w), np.float32)
            for t in (0.0, 0.25, 0.5, 0.75, 1.0):
                c = top + (bot - top) * t
                half = (pose.shoulder_w * 1.1) * (1.0 - 0.35 * t) + 20.0
                body = np.maximum(body, _plateau(h, w, c[0], c[1],
                                                 half, span * 0.18, rim=1.4))
            m = np.maximum(m, body)

        return np.clip(m, 0.0, 1.0)

    @staticmethod
    def _weighted(original, adjusted, mask, weight: float = 1.0):
        """Blend an adjustment back, preserving the untouched areas."""
        if mask is None:
            return adjusted
        a = (mask * weight)[..., None]
        return (original * (1.0 - a) + adjusted * a).astype(np.float32)

    # ------------------------------------------------------------------
    def run(self, img: np.ndarray, params: dict = None,
            detect: bool = True, progress=None) -> tuple:
        """Retouch `img` (float32 RGB in [0,1]).

        Returns (result, info) where info carries the detected faces and the
        stage timings, so a caller can report what actually ran.
        """
        import time
        p = self.normalize(params)
        info: Dict[str, object] = {"stages": [], "faces": [], "applied": []}

        def stage(name, fn):
            t0 = time.perf_counter()
            out = fn()
            info["stages"].append({"stage": name,
                                   "ms": round((time.perf_counter() - t0) * 1000.0, 1)})
            return out

        # ---- optional working-resolution reduction -----------------------
        work = img
        le = p.get("long_edge") or 0.0
        if le and le > 0:
            work = fit_long_edge(work, int(le))

        # ---- face detection ---------------------------------------------
        # Detection runs on a bounded copy: a 5000px photo needs no more
        # precision than a 1600px one, and running the 106-point model at full
        # resolution dominated the whole pipeline (measured: 2.4 s of a 10 s
        # run on a 1600px preview). Boxes scale back to the working image.
        faces: List[Face] = []
        det = self.detector() if detect else None
        detect_scale = 1.0
        if det is not None:
            def _detect():
                nonlocal detect_scale
                hh, ww = work.shape[:2]
                detect_scale = min(1.0, 1600.0 / max(max(hh, ww), 1))
                if detect_scale < 1.0:
                    small = cv2.resize(
                        work, (max(1, int(ww * detect_scale)),
                               max(1, int(hh * detect_scale))),
                        interpolation=cv2.INTER_AREA)
                else:
                    small = work
                fs = det.detect(small)
                if detect_scale < 1.0:
                    inv = 1.0 / detect_scale
                    for fc in fs:
                        bx, by, bwd, bht = fc.box
                        fc.box = (bx * inv, by * inv, bwd * inv, bht * inv)
                        fc.landmarks = fc.landmarks * inv
                        if fc.dense is not None:
                            fc.dense.points = fc.dense.points * inv
                for fc in fs:
                    FaceDetector.build_regions(fc, work.shape)
                return fs
            faces = stage("detect", _detect)
            info["faces"] = [
                {"box": [round(v, 1) for v in f.box], "score": round(f.score, 3),
                 "dense": f.has_dense()}
                for f in faces
            ]
            info["detect_scale"] = round(detect_scale, 3)

        # ---- geometry ----------------------------------------------------
        geo_keys = ["slimFace", "jaw", "cheekbone", "enlargeEye", "eyeHeight",
                    "eyeDistance", "noseWidth", "mouthWidth", "chinLength",
                    "foreheadHeight"]
        if faces and any(p.get(k) for k in geo_keys):
            face = faces[0]
            work = stage("geometry.face", lambda: L.reshape_face(work, face, p))
            info["applied"].append("geometry.face")

        body_keys = ["legLengthen", "slimLeg", "slimWaist", "slimArm",
                     "neckLength", "shoulderWidth", "heightGain", "bodySmooth"]
        body_pose = None
        if any(p.get(k) for k in body_keys):
            # Pose estimation is only worth its cost when a body control is
            # actually engaged. When it succeeds, the warp follows real joints;
            # otherwise reshape_body falls back to frame-derived bands.
            body_pose = None
            est = self.pose_estimator() if detect else None
            if est is not None:
                def _pose():
                    from .pose import pick_primary
                    return pick_primary(est.detect(work))
                try:
                    body_pose = stage("pose", _pose)
                except Exception as exc:
                    info["pose_error"] = str(exc)[:200]
                    body_pose = None
                if body_pose is not None:
                    info["pose"] = {
                        "torso_h": round(body_pose.torso_h, 1),
                        "leg_len": round(body_pose.leg_len, 1),
                        "shoulder_w": round(body_pose.shoulder_w, 1),
                        "hip_w": round(body_pose.hip_w, 1),
                        "body_angle_deg": round(np.degrees(body_pose.body_angle), 1),
                    }
            work = stage("geometry.body",
                         lambda: L.reshape_body(work, p, faces, pose=body_pose))
            info["applied"].append("geometry.body"
                                   + (".pose" if body_pose is not None else ".frame"))

        # ---- semantic regions (cached, used by skin + subject masking) -----
        # selfie_multiclass gives per-class masks (hair / face-skin / body-skin
        # / clothes / accessories / background). One inference per image serves
        # both the skin gate and the background protection below, so it is
        # computed once and reused. Absent model ⇒ both fall back to their
        # previous heuristics.
        sem = None
        if detect and self.use_semantic:
            sem = self._cached_semantic(work)
            if sem is not None:
                info["semantic"] = sem.coverage()

        # ---- skin --------------------------------------------------------
        skin_keys = ["spotless_acne", "spotless_spot", "spotless_texture",
                     "even_tone", "smooth", "freq_low", "freq_high",
                     "removeEyeBag", "eyeHighlight", "teethWhitening",
                     "teethDeYellow"]
        if any(p.get(k) for k in skin_keys):
            if faces:
                # every detected face gets the skin pass
                def _skin_faces():
                    out = work
                    for f in faces:
                        FaceDetector.build_regions(f, out.shape)
                        out = S.retouch_skin(out, f, p, semantic=sem)
                    return out
                work = stage("skin", _skin_faces)
            else:
                work = stage("skin",
                             lambda: S.retouch_skin(work, None, p, semantic=sem))
            info["applied"].append("skin")
            if sem is not None:
                info["skin_gate"] = "semantic"

        # ---- subject weighting -------------------------------------------
        # A portrait retouch should leave the background essentially alone. A
        # global `shadows` lift brightens and desaturates a dark backdrop along
        # with the subject (measured on a real portrait: background level
        # +0.080, saturation -0.026, which reads as a washed-out photo). The
        # mask below keeps global grading off the background while leaving
        # retouching inside the subject at full strength.
        #
        # The protection is per-operation because the controls differ a lot in
        # how much they touch a dark backdrop. `shadows` is the worst offender:
        # on a subject whose face is already bright, the lift lands almost
        # entirely on the background (measured ratio 0.2x, i.e. the background
        # moved five times more than the subject), so it gets the strongest
        # protection. `highlights` and `blacks` barely reach a mid-tone
        # background and are left closer to global for a coherent grade.
        BG_PROTECTION = {
            "shadows": 1.45,
            "blacks": 1.30,
            "exposure": 1.15,
            "brightness": 1.15,
            "dehaze": 1.20,
            "vibrance": 1.10,
            "saturation": 1.10,
            "contrast": 0.75,
            "highlights": 0.75,
            "whites": 0.75,
            "clarity": 0.90,
            "texture_tone": 0.90,
        }
        protect = float(p.get("protect_background", 1.0))

        # The matte is inferred once and held for the whole run. Leaving this
        # to a per-array cache does not work: every tone operation produces a
        # new array, so the key never hits and the network is re-run for each
        # operator (measured: beauty went from 5.4 s to 17.7 s).
        run_matte = None
        sem_subject = None
        # The semantic subject is the preferred boundary: unlike the matte it
        # includes accessories (glasses, jewellery), which a portrait matte
        # often treats as background, and unlike the face box it follows hair
        # and shoulders. The matte is the fallback.
        if sem is not None:
            sem_subject = np.clip(sem.subject(), 0.0, 1.0)
        elif faces and protect > 0.0:
            run_matte = self._cached_matte(work)

        def _mask_for(op: str):
            # The subject mask is useful even without a face: a semantic
            # subject exists for a body shot or a turned-away subject, where
            # face detection finds nothing but the background still must not
            # be graded.
            if sem_subject is None and run_matte is None and not faces:
                return None
            if protect <= 0.0:
                return None
            # `protect` is a 0..1 user control; the per-operation multiplier
            # scales how far the background is pushed down, and is allowed to
            # go past the user setting rather than being clamped by it, or the
            # extra protection for `shadows` and `blacks` would have no effect.
            depth = min(1.0, protect) * BG_PROTECTION.get(op, 1.0)
            bg_w = max(0.02, 1.0 - depth * 0.65)
            return self.subject_mask(work.shape, faces, body_pose,
                                     background_weight=bg_w,
                                     matte=run_matte, semantic=sem_subject)

        # one mask at the default strength, reused where no override applies
        subj = None
        if protect > 0.0 and (faces or sem_subject is not None
                              or run_matte is not None):
            bg_w = 1.0 - protect * 0.65
            # Prefer the semantic subject, then the matte, then the face box:
            # each is a better boundary than the one after it, and tone work
            # leaking onto the backdrop beside the head is the visible symptom
            # of using a worse one.
            subj = self.subject_mask(work.shape, faces, body_pose,
                                     background_weight=bg_w,
                                     matte=run_matte,
                                     semantic=sem_subject)
            info["background_weight"] = round(bg_w, 3)
            if sem_subject is not None:
                info["subject_source"] = "semantic"
            elif run_matte is not None:
                info["subject_source"] = "matte"

        # cache the per-operation masks so a run builds each one once
        _mask_cache: Dict[str, np.ndarray] = {}

        def _mask_cached(op: str):
            if op not in _mask_cache:
                m = _mask_for(op)
                _mask_cache[op] = subj if m is None else m
            return _mask_cache[op]

        # ---- white balance ------------------------------------------------
        # Linear-space channel gains from the auto correction, applied before
        # any tone work so everything downstream sees a neutral frame. Physical
        # white balance belongs in linear light; applying the same numbers to
        # the gamma-encoded image would remove only about half the cast.
        wb_gains = p.get("wb_gains")
        if wb_gains:
            def _wbg():
                from .autofix import apply_gains
                return apply_gains(work, wb_gains)
            work = stage("white_balance.gains", _wbg)
            info["applied"].append("white_balance.gains")

        wb_mode = p.get("wb_mode", "as_shot")
        if p.get("temp") or p.get("tint") or wb_mode == "auto":
            def _wb():
                out = T.white_balance(work, p["temp"], p["tint"], wb_mode)
                return self._weighted(work, out, _mask_cached("white_balance"))
            work = stage("white_balance", _wb)
            info["applied"].append("white_balance")

        # ---- tone ---------------------------------------------------------
        tone_ops = [
            ("exposure", T.exposure), ("contrast", T.contrast),
            ("brightness", T.brightness), ("highlights", T.highlights),
            ("shadows", T.shadows), ("whites", T.whites), ("blacks", T.blacks),
        ]
        active = [(k, fn) for k, fn in tone_ops if p.get(k)]
        if active:
            def _tone():
                out = work
                for k, fn in active:
                    adj = fn(out, p[k])
                    m = _mask_cached(k)
                    out = self._weighted(out, adj, m) if m is not None else adj
                return out
            work = stage("tone", _tone)
            info["applied"].extend(k for k, _ in active)

        presence = [
            ("texture_tone", T.texture), ("clarity", T.clarity),
            ("dehaze", T.dehaze), ("vibrance", T.vibrance),
            ("saturation", T.saturation),
        ]
        active = [(k, fn) for k, fn in presence if p.get(k)]
        if active:
            def _presence():
                out = work
                for k, fn in active:
                    adj = fn(out, p[k])
                    m = _mask_cached(k)
                    out = self._weighted(out, adj, m) if m is not None else adj
                return out
            work = stage("presence", _presence)
            info["applied"].extend(k for k, _ in active)

        # ---- HSL bands + camera primaries ---------------------------------
        hsl_active = []
        for b in HSL_BANDS:
            h_, s_, l_ = p.get(f"hsl_{b}_hue", 0), p.get(f"hsl_{b}_sat", 0), p.get(f"hsl_{b}_lum", 0)
            if h_ or s_ or l_:
                hsl_active.append((b, h_, s_, l_))
        cam_active = any(p.get(k) for k in
                         ["cam_r_hue", "cam_r_sat", "cam_g_hue", "cam_g_sat",
                          "cam_b_hue", "cam_b_sat"])
        if hsl_active or cam_active:
            def _colour():
                out = work
                for b, h_, s_, l_ in hsl_active:
                    out = T.hsl_band(out, b, h_, s_, l_)
                if cam_active:
                    out = T.camera_primaries(
                        out, p["cam_r_hue"], p["cam_r_sat"],
                        p["cam_g_hue"], p["cam_g_sat"],
                        p["cam_b_hue"], p["cam_b_sat"])
                return out
            work = stage("colour_grade", _colour)
            info["applied"].extend(f"hsl_{b}" for b, *_ in hsl_active)
            if cam_active:
                info["applied"].append("camera_primaries")

        # ---- 三区颜色分级 --------------------------------------------------
        # 参考软件把调色拆成 全局/中间调/高光 三个区域，各带色相、饱和度、
        # 明度，再加区域交叉点（balance）与重叠宽度（blending）。这与单一
        # 全局调色是两种不同的控制面：冷阴影 + 暖高光的"电影感"只能用分区
        # 表达，一个全局色相旋转做不到。
        #
        # 激活判断只看九个调整值，不看 blending。blending 默认 0.5，是"区域
        # 如何重叠"的配置而不是效果本身——把它也算进来，空参数就会跑一次
        # 恒等运算（实测：applied 里出现 split_toning，破坏了"空参数必须
        # 零变化"的约定）。balance 同理，它只挪 crossover，不单独产生效果。
        grade_keys = ["grade_global_hue", "grade_global_sat", "grade_global_lum",
                      "grade_mid_hue", "grade_mid_sat", "grade_mid_lum",
                      "grade_high_hue", "grade_high_sat", "grade_high_lum"]
        if any(p.get(k) for k in grade_keys):
            def _grade():
                from .grade import color_grade
                return color_grade(
                    work,
                    global_hue=p.get("grade_global_hue", 0.0),
                    global_sat=p.get("grade_global_sat", 0.0),
                    global_lum=p.get("grade_global_lum", 0.0),
                    mid_hue=p.get("grade_mid_hue", 0.0),
                    mid_sat=p.get("grade_mid_sat", 0.0),
                    mid_lum=p.get("grade_mid_lum", 0.0),
                    high_hue=p.get("grade_high_hue", 0.0),
                    high_sat=p.get("grade_high_sat", 0.0),
                    high_lum=p.get("grade_high_lum", 0.0),
                    balance=p.get("grade_balance", 0.0),
                    blending=p.get("grade_blending", 0.5))
            work = stage("split_toning", _grade)
            info["applied"].append("split_toning")

        # ---- 曲线与色阶 ------------------------------------------------------
        # 放在颜色分级之后：曲线是最后的形状修正，专业流程里调色完再拉曲线定调。
        levels_spec = p.get("levels")
        curves_spec = p.get("curves")
        if levels_spec or curves_spec:
            def _curve_stage():
                from . import curves as CV
                out = work
                if curves_spec:
                    out = CV.curves(out,
                                    rgb=curves_spec.get("rgb"),
                                    r=curves_spec.get("r"),
                                    g=curves_spec.get("g"),
                                    b=curves_spec.get("b"))
                if levels_spec:
                    out = CV.levels(out,
                                    master=levels_spec.get("master"),
                                    r=levels_spec.get("r"),
                                    g=levels_spec.get("g"),
                                    b=levels_spec.get("b"))
                return out
            work = stage("curves_levels", _curve_stage)
            if curves_spec:
                info["applied"].append("curves")
            if levels_spec:
                info["applied"].append("levels")

        # ---- 局部调整 ------------------------------------------------------
        # 渐变/径向局部工具：每个条目自带几何与参数，可叠加。全局控制被子集
        # 复用，所以局部与全局的结果保持一致。放在调色之后、细节之前，让
        # 局部曝光作用于已定调的画面。
        local_entries = p.get("local_adjustments")
        if local_entries:
            def _local():
                from .local import apply_local
                return apply_local(work, local_entries,
                                   strength_scale=1.0)
            work = stage("local_adjustments", _local)
            info["applied"].append("local_adjustments")
            info["local_count"] = len(local_entries)

        # ---- detail -------------------------------------------------------
        if p.get("nr_luminance") or p.get("nr_color"):
            work = stage("noise_reduction",
                         lambda: T.noise_reduction(work, p["nr_luminance"], p["nr_color"],
                                                   p["nr_detail"], p["nr_smoothness"]))
            info["applied"].append("noise_reduction")
        if p.get("ai_denoise"):
            work = stage("ai_denoise",
                         lambda: T.ai_denoise(work, True, p["ai_denoise_strength"]))
            info["applied"].append("ai_denoise")
        if p.get("sharpening"):
            work = stage("sharpening",
                         lambda: T.sharpening(work, p["sharpening"],
                                              p["sharp_radius"], p["sharp_detail"]))
            info["applied"].append("sharpening")

        # ---- optics -------------------------------------------------------
        if p.get("distortion") or p.get("vignetting") or p.get("ca_amount"):
            def _lens():
                out = work
                if p.get("distortion") or p.get("vignetting"):
                    out = T.lens_correction(out, p["distortion"],
                                            p["vignetting"],
                                            p["vignetting_midpoint"])
                if p.get("ca_amount"):
                    out = T.chromatic_aberration(out, p["ca_amount"])
                return out
            work = stage("lens_correction", _lens)
            info["applied"].append("lens_correction")

        # ---- grain --------------------------------------------------------
        if p.get("grain_amount"):
            work = stage("grain", lambda: T.grain(
                work, p["grain_amount"], p["grain_size"], p["grain_roughness"],
                p["grain_shadows"], p["grain_midtones"], p["grain_highlights"]))
            info["applied"].append("grain")

        # ---- matting / background (AI) ------------------------------------
        # Runs last so the backdrop is built from the fully retouched pixels.
        if p.get("matte_enabled") and detect:
            def _matte():
                from . import matting as MT

                if not MT.available():
                    info["matte_error"] = "matting model not installed"
                    return work
                # reuse the matte already computed for background protection
                a = run_matte
                if a is None or a.shape[:2] != work.shape[:2]:
                    a = MT.alpha(work, size=int(p.get("matte_size", 768) or 768),
                                 feather_px=float(p.get("matte_feather", 0.0)))
                if p.get("matte_refine"):
                    a = MT.refine_edges(a, work)
                info["matte_coverage"] = round(float((a > 0.5).mean()), 4)

                solid = bool(p.get("bg_mode"))
                bg_color = ((float(p.get("bg_color_r", 1.0)),
                             float(p.get("bg_color_g", 1.0)),
                             float(p.get("bg_color_b", 1.0)))
                            if solid else None)
                out = MT.compose(
                    work, a,
                    background=None,
                    bg_color=bg_color,
                    blur_bg=float(p.get("bg_blur", 0.0)),
                    darken_bg=float(p.get("bg_darken", 0.0)),
                    desat_bg=float(p.get("bg_desat", 0.0)))
                return out

            work = stage("matting", _matte)
            info["applied"].append("matting")

        return np.clip(work, 0.0, 1.0).astype(np.float32), info


# --------------------------------------------------------------------------
# presets: named starting points, the "工作流/模板" concept
# --------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, float]] = {
    "none": {},

    "natural": {
        "spotless_acne": 35, "spotless_spot": 20, "spotless_texture": 45,
        "even_tone": 25, "smooth": 28, "texture": 70,
        "removeEyeBag": 20, "keepSilkworm": 60,
        "exposure": 4, "contrast": 6, "shadows": 10, "vibrance": 10,
        "sharpening": 22,
    },

    "studio": {
        "slimFace": 18, "jaw": 12, "enlargeEye": 12,
        "spotless_acne": 55, "spotless_spot": 40, "spotless_texture": 30,
        "even_tone": 45, "smooth": 55, "texture": 40,
        "removeEyeBag": 40, "keepSilkworm": 40, "eyeHighlight": 25,
        "teethWhitening": 25, "teethDeYellow": 30,
        "exposure": 6, "contrast": 12, "highlights": -15, "shadows": 20,
        "clarity": 8, "vibrance": 15,
        "sharpening": 30, "vignetting": 12,
    },

    "beauty": {
        "slimFace": 30, "jaw": 22, "cheekbone": 12, "enlargeEye": 22,
        "eyeHeight": 10, "noseWidth": 15, "chinLength": 10,
        "spotless_acne": 70, "spotless_spot": 55, "spotless_texture": 20,
        "even_tone": 60, "smooth": 70, "texture": 30,
        "removeEyeBag": 55, "keepSilkworm": 30, "eyeHighlight": 35,
        "teethWhitening": 40, "teethDeYellow": 45,
        "exposure": 8, "contrast": 10, "shadows": 22, "vibrance": 18,
        "temp": 6, "tint": 3,
        "sharpening": 26,
    },

    "id_photo": {
        "slimFace": 20, "jaw": 15, "eyeDistance": -6,
        "spotless_acne": 50, "spotless_spot": 45, "even_tone": 55,
        "smooth": 40, "texture": 50,
        "removeEyeBag": 35, "keepSilkworm": 20,
        "teethWhitening": 20, "teethDeYellow": 25,
        "exposure": 10, "contrast": 14, "highlights": -10, "shadows": 25,
        "vibrance": 8, "saturation": 4,
        "distortion": 15, "vignetting": 0,
    },

    "film": {
        "even_tone": 15, "smooth": 15, "texture": 75,
        "contrast": 8, "shadows": 15, "highlights": -12,
        "vibrance": -8, "saturation": -12,
        "hsl_orange_hue": 4, "hsl_orange_sat": 10,
        "hsl_blue_sat": -15, "hsl_green_sat": -20,
        "grain_amount": 45, "grain_size": 30, "grain_roughness": 55,
        "grain_shadows": 60, "grain_midtones": 80, "grain_highlights": 40,
        "vignetting": 25,
    },

    "mono": {
        "saturation": -100,
        "contrast": 15, "shadows": 12, "highlights": -10,
        "clarity": 12, "texture_tone": 10,
        "smooth": 18, "texture": 65, "even_tone": 12,
        "grain_amount": 30, "grain_size": 25,
        "sharpening": 25, "vignetting": 18,
    },
}


def apply_preset(name: str) -> Dict[str, float]:
    if name not in PRESETS:
        raise ValueError(f"unknown preset: {name} (have: {', '.join(PRESETS)})")
    return dict(PRESETS[name])
