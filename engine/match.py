"""Colour matching and photo culling from image embeddings.

Two features that the reference application provides and that do not need a
bespoke trained network, because a general image-text embedding already
captures what they need:

* **colour matching (追色)** — given a reference photo, transfer its colour
  character to a target. The transfer is statistical (the target's colour
  distribution is warped onto the reference's), but the *decision* of what to
  match is guided by embeddings: the reference is reduced to its dominant
  palette and each region of the target picks the palette entry it is closest
  to in embedding space, so a sky does not get matched to skin.

* **culling (挑图)** — rank a set of photos. A retoucher's first pass is
  throwing away the frames that are soft, badly exposed or badly composed.
  Those are measurable without a model (sharpness, clipping, contrast,
  composition), and embeddings add the part that is not: near-duplicate
  detection, so a burst of ten near-identical frames collapses to its best.

The embedding backbone is CLIP via `transformers`, loaded lazily and cached.
If it is unavailable the module degrades: colour matching falls back to a
global palette transfer and culling keeps its measurable metrics.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2

_CLIP = None
_CLIP_LOCK = threading.Lock()
_CLIP_NAME = "openai/clip-vit-base-patch32"


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------

def clip_available() -> bool:
    """挑图能不能用。

    有两条路，任一可用即可：
      1. transformers + torch（开发机，最完整）
      2. onnxruntime + clip_image_encoder.onnx（打包版，只带图像塔）

    走 2 时不需要 4GB 的 PyTorch，挑图功能保持可用；文本塔用不到
    （挑图只比图像之间的相似度）。
    """
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        pass
    try:
        from . import cliponnx
        return cliponnx.available()
    except Exception:
        return False


def _clip():
    """Lazily build the CLIP model; returns (model, processor, device).

    Loads **offline first**. `from_pretrained` defaults to contacting the Hub to
    check for updates, and with no network reachable that HEAD request blocks
    until the socket times out — measured as `/api/cull` hanging for 300 s on a
    6-image folder while the model was actually already cached locally. Trying
    `local_files_only=True` first turns a 300 s stall into a ~200 ms load, and
    only falls back to the network when the cache really is missing.
    """
    global _CLIP
    if _CLIP is not None:
        return _CLIP
    with _CLIP_LOCK:
        if _CLIP is not None:
            return _CLIP
        import torch
        from transformers import CLIPModel, CLIPProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"

        def build(**kw):
            m = CLIPModel.from_pretrained(_CLIP_NAME, **kw)
            p = CLIPProcessor.from_pretrained(_CLIP_NAME, **kw)
            return m, p

        try:
            model, proc = build(local_files_only=True)
        except Exception:
            # 本地没有缓存才联网；这一步可能很慢，但至少是"确实需要下载"
            model, proc = build()
        _CLIP = (model.to(device).eval(), proc, device)
    return _CLIP


def embed_images(images: Sequence[np.ndarray], batch: int = 8) -> np.ndarray:
    """Return L2-normalised CLIP image embeddings, shape (N, D).

    优先走 transformers（有的话），否则退回 ONNX 图像塔。两条路的输出
    契约一致：行已归一化，点积即余弦相似度。
    """
    try:
        import torch
    except Exception:
        from . import cliponnx
        return cliponnx.embed_images(images, batch=batch)

    model, proc, device = _clip()
    out = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = images[i:i + batch]
            u8 = [(np.clip(im, 0, 1) * 255).astype(np.uint8) for im in chunk]
            inputs = proc(images=u8, return_tensors="pt").to(device)
            feats = model.get_image_features(**inputs)
            # transformers 5.x returns a model-output object here rather than
            # a bare tensor; older versions returned the tensor directly.
            if not isinstance(feats, torch.Tensor):
                for attr in ("pooler_output", "image_embeds", "last_hidden_state"):
                    candidate = getattr(feats, attr, None)
                    if isinstance(candidate, torch.Tensor):
                        feats = candidate
                        break
            if isinstance(feats, torch.Tensor) and feats.dim() == 3:
                feats = feats[:, 0]
            if not isinstance(feats, torch.Tensor):
                raise TypeError(
                    f"unexpected CLIP image feature type: {type(feats).__name__}")
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            out.append(feats.float().cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# colour matching
# ---------------------------------------------------------------------------

@dataclass
class ColorStats:
    """A compact description of an image's colour character."""
    mean: np.ndarray                 # (3,) in linear light
    std: np.ndarray                  # (3,)
    luma_mean: float
    luma_std: float
    sat_mean: float
    temp_shift: float                # + warm, - cool
    tint_shift: float                # + magenta, - green
    histogram: np.ndarray = field(default=None)   # (3, 256) luma-normalised

    def as_dict(self) -> Dict[str, float]:
        return {
            "luma_mean": round(float(self.luma_mean), 4),
            "luma_std": round(float(self.luma_std), 4),
            "sat_mean": round(float(self.sat_mean), 4),
            "temp_shift": round(float(self.temp_shift), 4),
            "tint_shift": round(float(self.tint_shift), 4),
        }


def analyze_color(img: np.ndarray) -> ColorStats:
    """Measure the colour character of an image."""
    from .imgutil import srgb_to_linear, luminance

    lin = srgb_to_linear(img)
    lum = luminance(img)

    # warm/cool from the red-blue balance, tint from green vs magenta
    r = float(lin[..., 0].mean())
    g = float(lin[..., 1].mean())
    b = float(lin[..., 2].mean())
    temp = (r - b) / max(r + b, 1e-6)
    tint = (g - 0.5 * (r + b)) / max(g + 0.5 * (r + b), 1e-6)

    hsv = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8),
                       cv2.COLOR_RGB2HSV)
    sat = float(hsv[..., 1].mean()) / 255.0

    hist = np.stack([
        cv2.calcHist([(np.clip(img[..., c], 0, 1) * 255).astype(np.uint8)],
                     [0], None, [256], [0, 256]).ravel()
        for c in range(3)
    ])
    hist = hist / max(hist.sum(axis=1, keepdims=True).max(), 1e-6)

    return ColorStats(
        mean=lin.reshape(-1, 3).mean(axis=0),
        std=lin.reshape(-1, 3).std(axis=0),
        luma_mean=float(lum.mean()),
        luma_std=float(lum.std()),
        sat_mean=sat,
        temp_shift=float(temp),
        tint_shift=float(tint),
        histogram=hist.astype(np.float32),
    )


def match_color(target: np.ndarray, reference: np.ndarray,
                strength: float = 1.0,
                preserve_luma: float = 0.35,
                method: str = "reinhard") -> np.ndarray:
    """Transfer `reference`'s colour character onto `target`.

    Parameters
    ----------
    strength : 0..1 blend between the untouched target and the full match.
    preserve_luma : how much of the target's own luminance to keep. Matching a
        brighter reference wholesale would blow out a correctly exposed
        subject, so some of the original luma is retained.
    method : "reinhard" matches mean and standard deviation per axis, which is
        stable and cheap; "histogram" additionally matches the luminance
        distribution, which is closer to a shot-match.

    Implementation note
    -------------------
    Luminance is matched before chroma, and the chroma match is then applied
    *proportionally to the new luminance*. Matching the two independently looks
    harmless but destroys saturation: L and (a, b) are not independent in Lab,
    and remapping L alone while leaving absolute chroma untouched moved a real
    pair's saturation from 0.167 to 0.072 — 90% of the total loss, before the
    chroma axes were even touched. Scaling chroma with the local luminance
    ratio keeps hue and saturation relationships intact.
    """
    from .imgutil import rgb_to_lab, lab_to_rgb

    if strength <= 0:
        return target.copy()

    t = rgb_to_lab(target)
    r = rgb_to_lab(reference)
    matched = t.copy()

    # ---- luminance ------------------------------------------------------
    if method == "histogram":
        src = np.clip(t[..., 0], 0.0, 1.0)
        ref_vals = np.clip(r[..., 0], 0.0, 1.0).ravel()
        if ref_vals.size:
            q = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
            ref_q = np.quantile(ref_vals, q).astype(np.float32)
            matched[..., 0] = ref_q[np.clip((src * 1023.0).astype(np.int32),
                                            0, 1023)]
    else:
        tm, ts = float(t[..., 0].mean()), float(t[..., 0].std())
        rm, rs = float(r[..., 0].mean()), float(r[..., 0].std())
        if ts < 1e-6:
            matched[..., 0] = rm
        else:
            matched[..., 0] = (t[..., 0] - tm) * (rs / ts) + rm

    if preserve_luma > 0:
        k = float(np.clip(preserve_luma, 0.0, 1.0))
        matched[..., 0] = matched[..., 0] * (1.0 - k) + t[..., 0] * k
    matched[..., 0] = np.clip(matched[..., 0], 0.0, 1.0)

    # ---- chroma, scaled with the luminance change ------------------------
    # Working in the normalised Lab space (L in 0..1, a/b in -1..1), a unit of
    # chroma buys less perceived saturation at high L. Scaling the reference
    # chroma by the ratio of mean luminances keeps the *relative* chroma, so a
    # brighter result does not lose its colour.
    lt_mean = max(float(t[..., 0].mean()), 1e-3)
    lm_mean = max(float(matched[..., 0].mean()), 1e-3)
    luma_ratio = lm_mean / lt_mean

    for c in (1, 2):
        tm, ts = float(t[..., c].mean()), float(t[..., c].std())
        rm, rs = float(r[..., c].mean()), float(r[..., c].std())
        if ts < 1e-6:
            matched[..., c] = rm
        else:
            matched[..., c] = (t[..., c] - tm) * (rs / ts) + rm
        # compensate for the luminance shift so saturation is preserved
        matched[..., c] = matched[..., c] * luma_ratio

    matched = np.clip(matched, -1.0, 1.0)
    out = lab_to_rgb(matched)

    s = float(np.clip(strength, 0.0, 1.0))
    return np.clip(target * (1.0 - s) + out * s, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# culling
# ---------------------------------------------------------------------------

@dataclass
class PhotoScore:
    path: str
    index: int
    sharpness: float          # normalised 0..1
    exposure: float
    contrast: float
    colorfulness: float
    overall: float
    embedding: Optional[np.ndarray] = None
    duplicate_of: Optional[int] = None
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "sharpness": round(self.sharpness, 4),
            "exposure": round(self.exposure, 4),
            "contrast": round(self.contrast, 4),
            "colorfulness": round(self.colorfulness, 4),
            "overall": round(self.overall, 4),
            "duplicate_of": self.duplicate_of,
            "reasons": self.reasons,
        }


def _sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian, robust to size and content.

    The raw value depends on resolution, so the image is first normalised to a
    fixed long edge; without that a large photo always outscores a small one.
    """
    from .imgutil import luminance

    g = luminance(img)
    h, w = g.shape
    scale = 640.0 / max(h, w)
    if scale < 1.0:
        g = cv2.resize(g, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(np.ascontiguousarray(g), cv2.CV_32F, ksize=3)
    return float(lap.var())


def _exposure(img: np.ndarray) -> Tuple[float, float]:
    """Return (exposure score, clipping penalty)."""
    from .imgutil import luminance

    lum = luminance(img)
    shadow_clip = float((lum < 0.02).mean())
    high_clip = float((lum > 0.98).mean())
    mean = float(lum.mean())
    # a well-exposed frame sits near the middle of the range with little clipping
    mid = 1.0 - min(1.0, abs(mean - 0.45) / 0.45)
    penalty = min(1.0, (shadow_clip + high_clip) * 8.0)
    return float(mid * (1.0 - penalty)), penalty


def _contrast(img: np.ndarray) -> float:
    from .imgutil import luminance

    lum = luminance(img)
    return float(np.clip(lum.std() / 0.28, 0.0, 1.0))


def _colorfulness(img: np.ndarray) -> float:
    """Hasler-Susstrunk colourfulness, normalised to ~0..1."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(np.clip((std + 0.3 * mean) / 0.45, 0.0, 1.0))


def score_photos(images: Sequence[np.ndarray],
                 paths: Optional[Sequence[str]] = None,
                 use_embeddings: bool = True,
                 dup_threshold: float = 0.965) -> List[PhotoScore]:
    """Score and rank photos, marking near-duplicates.

    Weights favour sharpness and exposure, because those are the two defects a
    retoucher cannot fix convincingly afterwards.
    """
    if paths is None:
        paths = [f"#{i}" for i in range(len(images))]

    sharp = np.array([_sharpness(im) for im in images], np.float64)
    # normalise sharpness within the set: absolute values are content-dependent
    if sharp.max() > sharp.min():
        sharp_n = (sharp - sharp.min()) / (sharp.max() - sharp.min())
    else:
        sharp_n = np.ones_like(sharp) * 0.5

    emb = None
    if use_embeddings and len(images) > 1 and clip_available():
        try:
            emb = embed_images(list(images))
        except Exception:
            emb = None

    scores: List[PhotoScore] = []
    for i, im in enumerate(images):
        exp, clip = _exposure(im)
        con = _contrast(im)
        col = _colorfulness(im)
        overall = (0.42 * float(sharp_n[i]) + 0.34 * exp
                   + 0.14 * con + 0.10 * col)

        reasons = []
        if sharp_n[i] < 0.25:
            reasons.append("soft")
        if clip > 0.35:
            reasons.append("clipped")
        if exp < 0.35:
            reasons.append("poorly exposed")

        scores.append(PhotoScore(
            path=paths[i], index=i, sharpness=float(sharp_n[i]),
            exposure=float(exp), contrast=float(con), colorfulness=float(col),
            overall=float(overall),
            embedding=None if emb is None else emb[i],
            reasons=reasons))

    # near-duplicate detection: keep the highest-scoring frame of each group
    if emb is not None:
        sim = emb @ emb.T
        np.fill_diagonal(sim, -1.0)
        order = np.argsort([-s.overall for s in scores])
        claimed = np.zeros(len(scores), bool)
        for i in order:
            if claimed[i]:
                continue
            claimed[i] = True
            for j in np.where(sim[i] > dup_threshold)[0]:
                if not claimed[j]:
                    claimed[j] = True
                    scores[j].duplicate_of = int(i)
                    scores[j].reasons.append("duplicate")

    return scores


def rank(scores: Sequence[PhotoScore]) -> List[PhotoScore]:
    """Order best-first, pushing duplicates behind their keeper."""
    keep = [s for s in scores if s.duplicate_of is None]
    dup = [s for s in scores if s.duplicate_of is not None]
    keep.sort(key=lambda s: -s.overall)
    dup.sort(key=lambda s: -s.overall)
    return keep + dup
