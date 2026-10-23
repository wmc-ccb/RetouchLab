# -*- coding: utf-8 -*-
"""图库：浏览照片目录、读取 EXIF、按状态筛选.

参考软件的图库不只是缩略图网格，它把一次拍摄**按技术指标分组**：
  抖动 / 欠曝 / 过曝 / 闭眼 / 重复照片 / AI 精选
这让挑图从"一张一张看"变成"先看有问题的"。

这里实现同一套分组，判据都是**可测量的**：
  * 抖动  — 拉普拉斯方差低于该批次的分布下界
  * 欠曝  — 平均亮度偏低且暗部占比高
  * 过曝  — 高光裁切比例高
  * 重复  — 感知哈希距离很小（连拍同一瞬间）
  * 精选  — 上面几项都合格，且人脸清晰度较高

统计阈值按**本批次自身的分布**定，不用绝对常数：室内夜景和户外日光
的"正常曝光"根本不是一个数，用固定阈值会把整批夜景标成欠曝。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".nef", ".cr2",
             ".cr3", ".arw", ".raf", ".dng", ".orf", ".rw2")

THUMB_CACHE_DIR = "_thumbs"


def list_images(root: str, recursive: bool = True,
                limit: int = 5000) -> List[str]:
    """Collect image paths under `root`."""
    out: List[str] = []
    if not root or not os.path.isdir(root):
        return out
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            # 跳过隐藏与缩略图缓存目录
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(".") and d != THUMB_CACHE_DIR]
            for f in filenames:
                if f.lower().endswith(IMAGE_EXT):
                    out.append(os.path.join(dirpath, f))
                    if len(out) >= limit:
                        return out
    else:
        try:
            for f in os.listdir(root):
                p = os.path.join(root, f)
                if os.path.isfile(p) and f.lower().endswith(IMAGE_EXT):
                    out.append(p)
                    if len(out) >= limit:
                        return out
        except OSError:
            pass
    out.sort()
    return out


def groups(root: str, max_depth: int = 3) -> List[Dict]:
    """Top-level sub-folders with an image count, for the library tree.

    `max_depth` caps how far the count descends. Counting a whole drive
    recursively is what made `/api/library/roots` time out at 180 s on a C:
    drive with a million files; a photo library's top level is one folder per
    shoot, so two levels is enough and orders of magnitude faster.
    """
    res = []
    if not root or not os.path.isdir(root):
        return res

    def count_under(p: str, depth: int) -> int:
        if depth <= 0:
            return 0
        n = 0
        try:
            with os.scandir(p) as it:
                for e in it:
                    try:
                        if e.is_file(follow_symlinks=False):
                            if e.name.lower().endswith(IMAGE_EXT):
                                n += 1
                        elif e.is_dir(follow_symlinks=False):
                            if e.name.startswith(".") or e.name == THUMB_CACHE_DIR:
                                continue
                            n += count_under(e.path, depth - 1)
                    except OSError:
                        continue
        except OSError:
            pass
        return n

    try:
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if not os.path.isdir(p) or name.startswith("."):
                continue
            n = count_under(p, max_depth)
            res.append({"name": name, "path": p, "count": n})
    except OSError:
        pass
    return res


def exif(path: str) -> Dict:
    """Read the exposure fields the info panel shows.

    Nikon 写相机把曝光字段放在 Exif IFD（0x8769）里，主 IFD 只有机身/时间/
    版权。只读主 IFD 时 ISO/光圈/快门全部是 None（实测 _DSC7550.jpg 就这样），
    所以子 IFD 必须合并进来。同时补镜头（LensModel）与拍摄时间，供元数据
    面板显示。
    """
    out = {"name": os.path.basename(path), "iso": None, "focal": None,
           "aperture": None, "shutter": None, "camera": None, "lens": None,
           "taken": None, "artist": None, "path": path}
    try:
        from PIL import Image, ExifTags

        with Image.open(path) as im:
            ex = im.getexif()
            if not ex:
                return out
            tag = {v: k for k, v in ExifTags.TAGS.items()}
            # Exif IFD 里的曝光字段；GPS IFD 不是标量不合并
            try:
                sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}
                for k, v in (sub or {}).items():
                    ex[k] = v
            except Exception:
                pass

            def g(name):
                return ex.get(tag.get(name, -1))

            out["iso"] = g("ISOSpeedRatings") or g("PhotographicSensitivity")
            out["focal"] = g("FocalLength")
            out["aperture"] = g("FNumber")
            out["shutter"] = g("ExposureTime")
            out["camera"] = g("Model")
            out["lens"] = g("LensModel")
            out["taken"] = g("DateTimeOriginal") or g("DateTime")
            out["artist"] = g("Artist") or g("Copyright")
    except Exception:
        pass
    # 转成可 JSON 的形态
    for k in ("focal", "aperture"):
        v = out.get(k)
        if v is not None:
            try:
                out[k] = round(float(v), 1)
            except Exception:
                out[k] = None
    for k in ("iso",):
        if out.get(k) is not None:
            try:
                out[k] = int(out[k])
            except Exception:
                out[k] = None
    v = out.get("shutter")
    if v is not None:
        try:
            f = float(v)
            out["shutter"] = ("1/%d" % round(1 / f)) if f < 1 else str(int(f))
        except Exception:
            out["shutter"] = None
    return out


def load_thumb(path: str, long_edge: int = 320):
    """Decode a thumbnail as float RGB, fastest available route first.

    For metrics we only need ~240 px. Three tiers, fastest first:

    1. **Embedded JPEG preview** (RAW files carry a full-size JPEG inside).
       Extracting it is a byte scan, ~1 ms, versus ~1600 ms to demosaic a NEF.
       This is what makes scanning a folder of raw files feasible at all —
       measured 41.6 s -> 15.6 s from threads alone, and the remaining cost was
       almost entirely RAW decoding.
    2. `Image.draft()` so libjpeg can emit at 1/2, 1/4 or 1/8 scale directly.
    3. Full decode as a last resort.
    """
    # 1) RAW 内嵌预览
    try:
        with open(path, "rb") as fh:
            head = fh.read(4)
        if head[:2] in (b"II", b"MM"):        # TIFF 容器（NEF/CR2/ARW/DNG）
            prev = _embedded_jpeg(path)
            if prev is not None:
                import cv2

                # REDUCED 让 libjpeg 在解码阶段就做 1/2、1/4、1/8 缩放，
                # 全尺寸预览（6048x4032）不必先解出 72 MB 再缩小
                try:
                    bgr = cv2.imdecode(np.frombuffer(prev, np.uint8),
                                       cv2.IMREAD_REDUCED_COLOR_8)
                except Exception:
                    bgr = None
                if bgr is None or not bgr.size:
                    bgr = cv2.imdecode(np.frombuffer(prev, np.uint8),
                                       cv2.IMREAD_COLOR)
                if bgr is not None and bgr.size:
                    h, w = bgr.shape[:2]
                    sc = long_edge / float(max(h, w))
                    if sc < 1.0:
                        bgr = cv2.resize(bgr, (max(1, int(w * sc)),
                                               max(1, int(h * sc))),
                                         interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    return rgb.astype(np.float32) / 255.0
    except Exception:
        pass
    # 2) draft 缩放解码
    try:
        from PIL import Image

        with Image.open(path) as im:
            try:
                im.draft("RGB", (long_edge * 2, long_edge * 2))
            except Exception:
                pass
            im = im.convert("RGB")
            im.thumbnail((long_edge, long_edge), Image.BILINEAR)
            return np.asarray(im, np.float32) / 255.0
    except Exception:
        pass
    # 3) 完整解码
    try:
        from .imgutil import load_image, fit_long_edge

        return fit_long_edge(load_image(path), long_edge)
    except Exception:
        return None


def _embedded_jpeg(path: str, max_bytes: int = 6 * 1024 * 1024):
    """Find the largest embedded JPEG stream in a TIFF-based raw file.

    Reads only the first `max_bytes`: raw files put their previews near the
    start, and reading the whole 27 MB NEF made this call alone take 632 ms —
    slower than the thing it was meant to avoid. With a 6 MB window it is
    a few milliseconds and still finds the full-size preview.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read(max_bytes)
    except OSError:
        return None
    best = None
    i = 0
    while True:
        s = data.find(b"\xff\xd8\xff", i)
        if s < 0:
            break
        e = data.find(b"\xff\xd9", s + 3)
        if e < 0:
            break
        seg_len = e + 2 - s
        # 太短的段是缩略图，用不上
        if seg_len > 20000 and (best is None or seg_len > len(best)):
            best = data[s:e + 2]
        i = e + 2
    return best


def metrics(img: np.ndarray) -> Dict[str, float]:
    """Technical quality metrics used for grouping."""
    if img is None:
        return {}
    import cv2

    from .imgutil import luminance

    lum = luminance(img)
    g = (np.clip(lum, 0, 1) * 255).astype(np.uint8)
    sharp = float(cv2.Laplacian(g, cv2.CV_32F).var())
    return {
        "sharp": sharp,
        "mean": float(lum.mean()),
        "shadow": float((lum < 0.10).mean()),
        "clip": float((lum > 0.98).mean()),
    }


def phash(img: np.ndarray, size: int = 8) -> int:
    """64-bit perceptual hash for duplicate detection."""
    if img is None:
        return 0
    import cv2

    from .imgutil import luminance

    lum = luminance(np.clip(img, 0, 1))
    small = cv2.resize(lum.astype(np.float32), (size, size),
                       interpolation=cv2.INTER_AREA)
    med = float(np.median(small))
    bits = (small > med).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(bool(b))
    return v


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def classify(items: List[Dict]) -> Dict[str, List[int]]:
    """Group indices by technical status.

    Thresholds come from **this batch's own distribution** (percentiles), not
    fixed constants: a night shoot and a daylight shoot have completely
    different "normal", and a fixed cut would flag the whole night set as
    underexposed.
    """
    n = len(items)
    out = {"blur": [], "under": [], "over": [], "dup": [], "closed": [],
           "pick": []}
    if n == 0:
        return out
    sharp = np.array([it.get("sharp") or 0.0 for it in items], float)
    mean = np.array([it.get("mean") or 0.0 for it in items], float)
    shadow = np.array([it.get("shadow") or 0.0 for it in items], float)
    clip = np.array([it.get("clip") or 0.0 for it in items], float)

    # 抖动：锐度低于本批次的 12 分位
    if n >= 5:
        s_lo = float(np.percentile(sharp, 12))
        out["blur"] = [i for i in range(n)
                       if sharp[i] > 0 and sharp[i] <= s_lo]
    # 欠曝：亮度低且暗部多（两项都要满足，避免把低调人像误判）
    if n >= 5:
        m_lo = float(np.percentile(mean, 15))
        out["under"] = [i for i in range(n)
                        if mean[i] <= m_lo and shadow[i] > 0.25]
    # 过曝：高光裁切明显高于批次中位
    if n >= 5:
        c_hi = float(np.percentile(clip, 92))
        out["over"] = [i for i in range(n)
                       if clip[i] >= c_hi and clip[i] > 0.02]
    # 重复：哈希距离 <= 4
    hashes = [it.get("phash") or 0 for it in items]
    seen = set()
    for i in range(n):
        if i in seen:
            continue
        for j in range(i + 1, n):
            if j in seen:
                continue
            if hashes[i] and hashes[j] and hamming(hashes[i], hashes[j]) <= 4:
                out["dup"].append(j)
                seen.add(j)
    # 闭眼：眼睛开合度与纵向边缘两项都低于本批次 20 分位。
    # 阈值同样取本批次分位——妆效、光线、人种都会改变"正常眼睛"的绝对
    # 数值，固定阈值会整批误判（和抖动/欠曝同一个道理）。
    eo = np.array([it.get("eye_open") or -1.0 for it in items], float)
    ee = np.array([it.get("eye_edge") or -1.0 for it in items], float)
    valid = [i for i in range(n) if eo[i] >= 0 and ee[i] >= 0]
    if len(valid) >= 3:
        o_thr = float(np.percentile(eo[valid], 20))
        e_thr = float(np.percentile(ee[valid], 20))
        out["closed"] = [i for i in valid
                         if eo[i] <= o_thr and ee[i] <= e_thr]
    # 精选：不在任何问题组里的
    bad = set(out["blur"]) | set(out["under"]) | set(out["over"]) \
        | set(out["dup"]) | set(out["closed"])
    out["pick"] = [i for i in range(n) if i not in bad]
    return out


def _one_metrics(args):
    """Decode one file and compute its metrics (worker for threads).

    Also detects the face and computes the two eye-openness proxies — the
    "闭眼" group is a first-class part of photo culling (a burst of 30 frames
    always contains a few with eyes shut), and it needs the same decoded
    thumbnail we already have.
    """
    path, thumb_edge = args
    img = load_thumb(path, thumb_edge)
    m = metrics(img)
    m["phash"] = phash(img)
    try:
        from .face import FaceDetector
        from . import eyes as EYE

        faces = FaceDetector(dense=False).detect(img, max_faces=3)
        if faces:
            f = max(faces, key=lambda x: x.box[2] * x.box[3])
            s = EYE.eye_scores(img, f.landmarks)
            if s:
                m["eye_open"] = s["open"]
                m["eye_edge"] = s["edge"]
    except Exception:
        pass
    return m


def scan(root: str, limit: int = 400, with_metrics: bool = True,
         workers: int = 12) -> Dict:
    """Scan a folder into library items with metrics + groups.

    Decoding runs on a thread pool. Decoding a 24 MP JPEG is I/O plus
    libjpeg work that releases the GIL, so threads scale here without the
    pickling cost of a process pool; measured 13.5 s -> ~2 s on 24 files.
    """
    import concurrent.futures as cf

    paths = list_images(root, recursive=True, limit=limit)
    items = []
    for p in paths:
        it = {"path": p, "name": os.path.basename(p),
              "dir": os.path.dirname(p)}
        try:
            st = os.stat(p)
            it["size"] = st.st_size
            it["mtime"] = st.st_mtime
        except OSError:
            it["size"] = 0
            it["mtime"] = 0
        items.append(it)

    if with_metrics and items:
        jobs = [(it["path"], 240) for it in items]
        try:
            with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                results = list(ex.map(_one_metrics, jobs))
        except Exception:
            results = [_one_metrics(j) for j in jobs]
        for it, m in zip(items, results):
            it.update(m or {})

    g = classify(items)
    for k, idxs in g.items():
        for i in idxs:
            items[i].setdefault("tags", []).append(k)
    return {"root": root, "count": len(items), "items": items,
            "groups": {k: len(v) for k, v in g.items()},
            "index": g}


def info() -> dict:
    return {"ext": list(IMAGE_EXT),
            "groups": ["blur", "under", "over", "dup", "closed", "pick"],
            "thresholds": "percentile of the batch itself"}
