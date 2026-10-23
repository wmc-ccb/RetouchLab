# -*- coding: utf-8 -*-
"""背景消除：纯色背景的四类问题.

参考软件的「图像美化 → 背景调整」给出四项，都是**纯色背景**（证件照、
影棚）特有的缺陷。每一项对应一个可测量的物理原因，不是滤镜：

1. **祛穿帮** — 背景布没拉满，边缘露出墙、地面、灯架。表现为纯色区域里
   出现与背景色显著不同的色块/线条。检测：在背景区域里找"不是背景色"的
   连通块，用周围背景色填掉。
2. **祛瑕疵** — 背景布本身的污点、褶皱阴影、灰尘。表现为局部亮度凹陷/
   突起，尺度小、对比低。检测：背景区域内的高通响应，超过阈值处用低频
   重建填平。
3. **光影均匀** — 打光不均，背景一侧亮一侧暗（渐变）。表现为背景亮度
   沿某方向单调变化。检测：拟合低频平面/多项式，减去它。
4. **祛色彩断层** — 8 位量化 + 大面积渐变导致的色带（posterization）。
   表现为相邻像素色阶跳变。检测：在低梯度区做抖动 + 轻微降噪。

四者共用同一个前提：**必须先有准确的背景掩膜**。这也是它们能做好而
"AI 换背景"做不好的原因——纯色背景的掩膜可以从颜色直接算出来，不依赖
模型识别。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def background_mask(img: np.ndarray, subject: Optional[np.ndarray] = None,
                    samples: int = 40000) -> np.ndarray:
    """纯色背景掩膜：从四角采样估计背景色，按色差分割。

    四角采样比"整图取中位数"稳：人像通常占中间，四角是背景；而整图中位数
    会被衣服、道具拉偏。若给了 subject 掩膜则直接取反，这是最准的路径。
    """
    if subject is not None:
        s = np.clip(np.asarray(subject, np.float32), 0.0, 1.0)
        if s.ndim == 3:
            s = s[..., 0]
        if s.max() > 0.02:
            return np.clip(1.0 - s, 0.0, 1.0)

    h, w = img.shape[:2]
    k = max(4, min(h, w) // 12)
    corners = np.concatenate([
        img[:k, :k].reshape(-1, 3), img[:k, -k:].reshape(-1, 3),
        img[-k:, :k].reshape(-1, 3), img[-k:, -k:].reshape(-1, 3)])
    # 用中位数而不是均值：角落可能有一点点主体侵入
    bg = np.median(corners, axis=0)

    d = np.linalg.norm(img - bg.reshape(1, 1, 3), axis=2)
    # 阈值按四角自身的离散度自适应：背景越干净阈值越小
    spread = float(np.median(np.linalg.norm(corners - bg, axis=1)))
    thr = max(0.035, spread * 3.0)
    m = np.clip(1.0 - (d - thr * 0.6) / (thr * 0.8), 0.0, 1.0)
    return m.astype(np.float32)


def _bg_stats(img: np.ndarray, bg: np.ndarray) -> Tuple[np.ndarray, float]:
    """背景区域的颜色中位数与典型离散度。"""
    sel = bg > 0.7
    if sel.sum() < 100:
        return np.array([0.5, 0.5, 0.5], np.float32), 0.02
    px = img[sel]
    med = np.median(px, axis=0)
    spread = float(np.median(np.linalg.norm(px - med, axis=1)))
    return med, spread


def fix_showthrough(img: np.ndarray, bg: np.ndarray, strength: float = 1.0,
                    min_area: int = 64) -> Tuple[np.ndarray, dict]:
    """祛穿帮：背景里露出非背景色的块（墙、地面、灯架）用背景色填掉。

    只在**背景区域内部**做：连通的"异色块"若面积够大且不贴着画面边缘，
    就是穿帮。贴边的大块通常是背景布本身的边界，不处理。
    """
    out = np.asarray(img, np.float32).copy()
    b = np.clip(np.asarray(bg, np.float32), 0, 1)
    if b.ndim == 3:
        b = b[..., 0]
    med, spread = _bg_stats(out, b)
    d = np.linalg.norm(out - med.reshape(1, 1, 3), axis=2)
    thr = max(0.06, spread * 4.0)
    # 背景内且明显偏离背景色的像素
    foreign = ((b > 0.6) & (d > thr)).astype(np.uint8)
    if foreign.sum() == 0:
        return out, {"patched": 0, "px": 0}

    n, lab, stats, _ = cv2.connectedComponentsWithStats(foreign, 8)
    h, w = out.shape[:2]
    patched = 0
    px = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x0 = stats[i, cv2.CC_STAT_LEFT]
        y0 = stats[i, cv2.CC_STAT_TOP]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]
        # 贴着画面边缘的大块很可能是背景布边界，不动
        if (x0 <= 1 or y0 <= 1 or x0 + ww >= w - 1 or y0 + hh >= h - 1) \
                and area > 0.05 * w * h:
            continue
        comp = (lab == i)
        # 用该块周围的**背景色**填。采样时排除主体：半径按块尺寸定，但
        # 若把主体颜色也卷进来，填出来的补丁会带肤色，还会因为羽化把主体
        # 边缘一起改掉（实测主体被误改 3.2 万像素）。
        r = max(3, int(max(ww, hh) * 0.6))
        solid = ((b > 0.7) & ~comp).astype(np.float32)
        fill = cv2.blur(out * solid[..., None], (2 * r + 1, 2 * r + 1))
        wt = cv2.blur(solid, (2 * r + 1, 2 * r + 1)) + 1e-6
        fill = fill / wt[..., None]
        # 边缘羽化，避免填出一块硬边
        mask = cv2.GaussianBlur(comp.astype(np.float32), (0, 0),
                                max(1.0, min(ww, hh) * 0.15))
        # 严格限制在背景内：羽化后的 alpha 再乘背景掩膜，主体一点不动
        a = np.clip(mask * float(strength), 0, 1) * (b > 0.5)
        a = a[..., None]
        out = out * (1 - a) + fill * a
        patched += 1
        px += area
    return np.clip(out, 0, 1), {"patched": patched, "px": px}


def fix_blemish(img: np.ndarray, bg: np.ndarray, strength: float = 1.0,
                radius: float = 6.0) -> Tuple[np.ndarray, dict]:
    """祛瑕疵：背景布上的污点/灰尘/小褶皱阴影。

    高通检测 + 低频重建。关键是把"瑕疵"和"背景本身的纹理"分开：只在
    背景区域内、且高频响应超过背景自身噪声水平的像素才修。
    """
    out = np.asarray(img, np.float32).copy()
    b = np.clip(np.asarray(bg, np.float32), 0, 1)
    if b.ndim == 3:
        b = b[..., 0]
    if (b > 0.5).sum() < 100:
        return out, {"fixed": 0}

    # 低频背景（大尺度平滑）
    r = int(max(3, radius * 2))
    low = cv2.blur(out, (2 * r + 1, 2 * r + 1))
    hp = out - low
    # 背景区域的高频强度，用其分位数做阈值
    sel = b > 0.7
    mag = np.linalg.norm(hp, axis=2)
    base = float(np.percentile(mag[sel], 75)) if sel.any() else 0.0
    thr = max(0.012, base * 1.8)
    defect = (sel & (mag > thr)).astype(np.float32)
    if defect.sum() < 8:
        return out, {"fixed": 0}

    mask = cv2.GaussianBlur(defect, (0, 0), max(1.0, radius * 0.5))
    # 羽化会越过背景边界，再乘一次背景掩膜把主体完全排除
    a = np.clip(mask * float(strength), 0, 1) * (b > 0.5)
    a = a[..., None]
    out = out * (1 - a) + low * a
    return np.clip(out, 0, 1), {"fixed": int(defect.sum())}


def fix_uneven_light(img: np.ndarray, bg: np.ndarray,
                     strength: float = 1.0) -> Tuple[np.ndarray, dict]:
    """光影均匀：把背景的亮度渐变拉平。

    拟合一个低阶多项式曲面（背景的照明分布），再把它归一化到中位数。
    只用背景像素拟合，主体不受影响。
    """
    out = np.asarray(img, np.float32).copy()
    b = np.clip(np.asarray(bg, np.float32), 0, 1)
    if b.ndim == 3:
        b = b[..., 0]
    sel = b > 0.7
    if sel.sum() < 200:
        return out, {"ok": False}

    h, w = out.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = (xx / max(w - 1, 1)) * 2 - 1
    yn = (yy / max(h - 1, 1)) * 2 - 1
    # 二次曲面够用：打光不均是低频的，阶数高会拟合进主体
    terms = [np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn]
    A = np.stack([t[sel] for t in terms], axis=1)
    # 用亮度做拟合目标（比三通道分别拟合稳）
    lum = (out * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(axis=2)
    try:
        coef, *_ = np.linalg.lstsq(A, lum[sel], rcond=None)
    except np.linalg.LinAlgError:
        return out, {"ok": False}
    surf = sum(c * t for c, t in zip(coef, terms))
    target = float(np.median(surf[sel]))
    if target <= 1e-6:
        return out, {"ok": False}
    # 增益限制在合理范围，避免把正常暗角也拉平
    gain = np.clip(target / np.maximum(surf, 1e-3), 0.75, 1.35)
    g = 1.0 + (gain - 1.0) * float(strength)
    # 只在背景区域应用，边缘羽化后再乘背景掩膜，主体完全不动
    a = cv2.GaussianBlur(sel.astype(np.float32), (0, 0),
                         max(2.0, min(h, w) * 0.01))
    a = np.clip(a, 0, 1) * sel
    out = out * (1 + (g[..., None] - 1) * a[..., None])
    return np.clip(out, 0, 1), {"ok": True, "range": float(gain.max() - gain.min())}


def fix_banding(img: np.ndarray, bg: np.ndarray, strength: float = 1.0
                ) -> Tuple[np.ndarray, dict]:
    """祛色彩断层：8 位量化的色带。

    做法是**在低梯度区加亚像素抖动**，把量化台阶打散成噪声；人眼对噪声
    不敏感，对台阶敏感。这不是模糊，不损失锐度。
    """
    out = np.asarray(img, np.float32).copy()
    b = np.clip(np.asarray(bg, np.float32), 0, 1)
    if b.ndim == 3:
        b = b[..., 0]
    sel = b > 0.6
    if sel.sum() < 200:
        return out, {"ok": False}

    h, w = out.shape[:2]
    # 梯度小的区域 = 可能有色带
    gx = cv2.Sobel(out, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(out, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.linalg.norm(np.dstack([gx, gy]), axis=2)
    flat = (grad < 0.02) & sel

    # 三角分布抖动（幅度 = 半个量化步长）
    rng = np.random.default_rng(12345)
    amp = 0.5 / 255.0 * float(strength)
    noise = (rng.random((h, w, 1), np.float32) -
             rng.random((h, w, 1), np.float32)) * amp
    a = np.clip(flat.astype(np.float32), 0, 1)[..., None]
    out = out + noise * a
    return np.clip(out, 0, 1), {"ok": True, "flat_px": int(flat.sum())}


def purity(img: np.ndarray, bg: np.ndarray) -> float:
    """背景区域的"纯色程度"（0..1）。

    这几项功能的前提是**纯色背景**。在复杂背景（晚会、街景）上跑，
    "祛瑕疵"会把背景纹理当瑕疵抹掉，"光影均匀"会把真实光影拉平——
    效果上等于给背景糊了一层。

    关键：**平滑的光影渐变不算"不纯"**。证件照背景一侧亮一侧暗很常见，
    那正是"光影均匀"要处理的对象。真正的不纯是**高频的颜色起伏**
    （纹理、杂物、人）。所以这里量的是"去趋势后的残差"：先用低频曲面
    拟合背景的明暗，再看拟合残差有多大。
    """
    b = np.clip(np.asarray(bg, np.float32), 0, 1)
    if b.ndim == 3:
        b = b[..., 0]
    sel = b > 0.7
    if sel.sum() < 200:
        return 0.0
    import cv2 as _cv

    h, w = b.shape[:2]
    # 用一张大尺度模糊估计低频趋势，残差才是纹理
    k = max(15, (min(h, w) // 6) | 1)
    low = _cv.blur(np.asarray(img, np.float32), (k, k))
    resid = np.linalg.norm(img - low, axis=2)
    r = float(np.median(resid[sel]))
    # 纯色背景残差 < 0.012；复杂场景 > 0.06
    score = 1.0 - min(1.0, max(0.0, (r - 0.012) / 0.048))
    # 颜色离散度仍作次要判据，但阈值放宽（允许整体偏色）
    px = img[sel]
    med = np.median(px, axis=0)
    d = np.linalg.norm(px - med, axis=1)
    lo, hi = np.percentile(d, [20, 80])
    band = float(hi - lo)
    tex = 1.0 - min(1.0, max(0.0, band / 0.28))
    return float(max(0.0, min(1.0, 0.7 * score + 0.3 * tex)))


def color_key_alpha(img: np.ndarray, tolerance: float = 0.0
                    ) -> Tuple[np.ndarray, dict]:
    """纯色背景的色键抠图：从四角估计背景色，按色差给 alpha。

    为什么纯色背景优先用它而不是分割模型：MODNet 之类的模型是**人像**
    模型，它假设画面里是真人；对影棚里的人物、道具、静物都不可靠（实测
    合成几何体上 alpha 在身体区域直接掉到 0，换底色时把底色透到身上）。

    纯色背景有确定答案：离背景色远的就是前景。这条路径没有模型不确定性，
    而且边缘是逐像素判定的，比模型输出的分辨率更高。
    """
    from .imgutil import luminance

    src = np.asarray(img, np.float32)
    h, w = src.shape[:2]
    k = max(6, min(h, w) // 10)
    corners = np.concatenate([
        src[:k, :k].reshape(-1, 3), src[:k, -k:].reshape(-1, 3),
        src[-k:, :k].reshape(-1, 3), src[-k:, -k:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    spread = float(np.median(np.linalg.norm(corners - bg, axis=1)))
    if tolerance <= 0:
        tolerance = max(0.05, spread * 4.0)

    d = np.linalg.norm(src - bg.reshape(1, 1, 3), axis=2)
    # 离背景近 -> alpha 0；远 -> 1；过渡带按 tolerance 的 40%
    soft = max(0.02, tolerance * 0.4)
    alpha = np.clip((d - tolerance) / soft, 0.0, 1.0)
    # 主体内部推到 1：主体与背景色差远大于容差时，alpha 会停在 0.98 附近，
    # 残余的 2% 底色会渗到主体上。只把"明显是前景"的部分硬化——判据要用
    # 色差本身，而不是羽化后的 alpha：之前用 alpha>0.98 判前景，而 alpha
    # 在背景侧也能到 0.99（羽化把边界带抬高了），导致背景像素被当成主体。
    fg_core = d > tolerance + soft * 2.0
    # 先只对真前景硬化，再做平滑，最后用同一个 fg_core 找回 1.0
    alpha = np.where(fg_core, 1.0, alpha)
    import cv2 as _cv

    alpha = _cv.GaussianBlur(alpha.astype(np.float32), (0, 0), 1.0)
    # 平滑会在主体内部留下 0.99 左右的残值；用色差判据再拉回 1.0，
    # 边界带（d 在容差附近）仍保持渐变
    alpha = np.where(fg_core, np.maximum(alpha, 0.995), alpha)
    # 背景侧同样硬化到 0：d < 容差一半的地方必然是背景
    alpha = np.where(d < tolerance * 0.5, np.minimum(alpha, 0.005), alpha)
    # 连通性清理：只保留最大的几个前景连通域（去掉背景里零星的噪点）
    binm = (alpha > 0.5).astype(np.uint8)
    n, lab, stats, _ = _cv.connectedComponentsWithStats(binm, 8)
    if n > 1:
        areas = [(int(stats[i, _cv.CC_STAT_AREA]), i) for i in range(1, n)]
        areas.sort(reverse=True)
        keep = set(i for a, i in areas[:3] if a >= 0.0004 * h * w)
        if keep:
            keep_mask = np.isin(lab, list(keep)).astype(np.float32)
            alpha = alpha * _cv.GaussianBlur(keep_mask, (0, 0), 2.0)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32), {
        "bg_colour": [round(float(v), 3) for v in bg],
        "tolerance": round(float(tolerance), 3),
        "spread": round(spread, 4)}


def apply(img: np.ndarray, subject: Optional[np.ndarray] = None,
          showthrough: float = 0.0, blemish: float = 0.0,
          even_light: float = 0.0, banding: float = 0.0,
          enhance: float = 0.0, auto_guard: bool = True
          ) -> Tuple[np.ndarray, dict]:
    """按顺序执行背景处理。

    顺序有依据：先补穿帮（补丁会改变背景色统计），再修瑕疵（补丁边缘可能
    有痕），再拉平光影（需要干净的背景做拟合），最后抖动（抖动要在所有
    几何/颜色操作之后，否则会被后续处理平滑掉）。

    `auto_guard` 会按背景纯度自动降权：复杂背景上重处理等于把背景抹平，
    那不是用户想要的结果。纯背景（证件照）保持全强度。
    """
    out = np.asarray(img, np.float32).copy()
    bg = background_mask(out, subject)
    pur = purity(out, bg)
    # 纯度 < 0.45 时线性降到 0：复杂场景不做破坏性清理
    guard = 1.0
    if auto_guard:
        guard = float(min(1.0, max(0.0, (pur - 0.45) / 0.35)))
    info = {"bg_coverage": round(float((bg > 0.5).mean()), 4),
            "purity": round(pur, 3), "guard": round(guard, 3)}
    if guard <= 0.0 and auto_guard:
        info["skipped"] = ("背景不纯（purity=%.2f），跳过破坏性清理；"
                           "这几项是为纯色背景设计的" % pur)
        return np.clip(out, 0, 1), info
    sc = guard
    if showthrough > 0:
        out, st = fix_showthrough(out, bg, showthrough * sc)
        info["showthrough"] = st
    if blemish > 0:
        out, bl = fix_blemish(out, bg, blemish * sc)
        info["blemish"] = bl
    if even_light > 0:
        out, el = fix_uneven_light(out, bg, even_light * sc)
        info["even_light"] = el
    if enhance > 0:
        # 背景增强：轻微提亮 + 加饱和，让纯色背景更"实"
        med, _ = _bg_stats(out, bg)
        e = float(enhance) / 100.0 * sc
        a = cv2.GaussianBlur(bg, (0, 0), 3.0)[..., None]
        boost = out * (1 + 0.10 * e) + med.reshape(1, 1, 3) * 0.06 * e
        out = np.clip(out * (1 - a * e) + boost * (a * e), 0, 1)
        info["enhance"] = True
    if banding > 0:
        out, bd = fix_banding(out, bg, banding)
        info["banding"] = bd
    return np.clip(out, 0, 1), info


def info() -> dict:
    return {"features": ["showthrough", "blemish", "even_light", "banding",
                         "enhance"],
            "premise": "solid-colour background; mask derived from corner colour"}
