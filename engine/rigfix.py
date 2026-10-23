# -*- coding: utf-8 -*-
"""去道具（灯架/灯）：识别摄影器材并用周围背景补全.

影楼/外拍照里的灯架、柔光箱、补光灯是最常见的"必须后期去掉"的物体。
像素蛋糕把它们归在「智能道具修复」（生成式）；这里用**确定性识别**：

  * **灯架**：细长的近垂直连通块。判据是形状（高宽比 ≥ 3.5、
    填充率高）+ 颜色（金属架是深色低饱和；白色反光板/伞骨也常见，
    所以白色高亮但低饱和的细长块也算）。天空/浅色墙面上最常见。
  * **灯头/柔光箱**：架顶的大块低饱和区域（白/灰/黑罩子），面积大、
    矩形度高（填充率 ≥ 0.7）。
  * **约束**：主体分割排除——灯架常在人身后，掩膜与主体相交部分剔除；
    与现有「AI 祛路人」同一套保护。
  * **补全**：LaMa。细长杆件的上下文（两侧背景）充足，补全效果好。

用户也可以手涂：`mask` 直接给定。
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def _gray(img: np.ndarray) -> np.ndarray:
    from .imgutil import luminance

    return luminance(img)


def detect_rig(img: np.ndarray, subject_mask: Optional[np.ndarray] = None,
               min_len_frac: float = 0.18,
               max_width_frac: float = 0.055
               ) -> Tuple[np.ndarray, dict]:
    """返回 (mask 0..1, info)。细长垂直杆 + 顶部灯头。"""
    h, w = img.shape[:2]
    gray = _gray(img)
    info: dict = {"candidates": 0, "poles": 0, "heads": 0,
                  "applicable": True}

    # ---- 杆件候选：暗（金属架）细长块；亮块（柔光箱面）面积大、不是
    # 杆——它们留给"灯头"阶段，先进杆候选会把箱面与杆连成宽块。
    from .imgutil import rgb_to_hsv

    hsv = rgb_to_hsv(np.clip(img, 0, 1))
    s = hsv[..., 1].astype(np.float32)
    dark = ((gray < 0.32) & (s < 0.45)).astype(np.uint8)
    # 只做**纯竖直**闭运算接回杆上的高光/阴影断点（1×h/60）。
    # 椭圆核哪怕宽 3px 也会逐行桥接三脚架的斜腿，把杆+腿连成一个
    # bbox 105px 宽的稀疏大块，全部漏检（实测两次踩坑）。
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(5, h // 60))))
    cand = dark

    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    min_len = h * min_len_frac
    max_wid = w * max_width_frac
    pole = np.zeros((h, w), np.uint8)
    tops: list = []
    for i in range(1, n):
        x, y, bw, bh, a = (int(stats[i, cv2.CC_STAT_LEFT]),
                           int(stats[i, cv2.CC_STAT_TOP]),
                           int(stats[i, cv2.CC_STAT_WIDTH]),
                           int(stats[i, cv2.CC_STAT_HEIGHT]),
                           int(stats[i, cv2.CC_STAT_AREA]))
        if bh < min_len or a < 12:
            continue
        # bbox 判据对三脚架会失效：三条腿与杆在杆底**几何相连**，整个
        # 支架是一个连通块（bbox 105px 宽是几何事实，不是形态学 bug）。
        # 判据换成**列游程**：杆所在列有一段 ≥min_len 的连续竖直像素，
        # 且这些"杆列"相互紧邻（宽度 ≤ max_wid）。
        colmask = cand[y:y + bh, x:x + bw]
        runs = np.zeros(bw, np.int32)
        for c in range(bw):
            col = colmask[:, c]
            best = cur = 0
            for v in col:
                cur = cur + 1 if v else 0
                best = max(best, cur)
            runs[c] = best
        good = runs >= min_len
        if not good.any():
            continue
        idx = np.where(good)[0]
        if idx[-1] - idx[0] + 1 > max_wid:
            continue
        # 只保留杆列（带 1px 余量），腿的稀疏列不进 mask
        sub = np.zeros_like(colmask)
        sub[:, max(0, idx[0] - 1):idx[-1] + 2] = colmask[:,
            max(0, idx[0] - 1):idx[-1] + 2]
        pole[y:y + bh, x:x + bw] = np.maximum(
            pole[y:y + bh, x:x + bw], sub)
        info["poles"] += 1
        tops.append((x + int(idx[0]), y, int(idx[-1] - idx[0] + 1), bh))
        # 三脚架腿：腿与杆在关节处**几何相连**（不是形态学造出来的），
        # 所以"杆 bbox 下半段"窗口内的暗色像素全部收进 mask——那一段
        # 既有杆也有向外张开的腿，LaMa 都能用两侧背景补。窗口宽度按
        # **杆宽**的倍数（不是 bbox 宽——bbox 含整副三脚架，按它算会把
        # 窗口拉到画面边缘、误伤旁边的人物，实测）。
        fy0 = min(h, y + int(bh * 0.55))
        fy1 = min(h, y + bh + 2)
        pole_w = int(idx[-1] - idx[0] + 1)
        fw = max(24, pole_w * 9)
        fx0 = max(0, x + int(idx[0]) - fw)
        fx1 = min(w, x + int(idx[-1] + 2) + fw)
        legwin = cand[fy0:fy1, fx0:fx1]
        if legwin.any():
            pole[fy0:fy1, fx0:fx1] = np.maximum(
                pole[fy0:fy1, fx0:fx1], legwin)

    # ---- 灯头先于建筑判据：杆顶上方有大块高亮低饱和区（白/灰罩子）且
    # **x 与杆重叠**的杆是灯架，豁免于建筑判据。x 重叠必须查——不查时
    # 相邻窗框的杆都会"借到"旁边柔光箱的亮块（实测 5 个假灯头）。
    head = np.zeros((h, w), np.uint8)
    head_boxes = []
    head_cand = ((gray > 0.80) & (s < 0.25)).astype(np.uint8)
    head_cand = cv2.morphologyEx(
        head_cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (7, 7)))
    hn, hlab, hstats, _ = cv2.connectedComponentsWithStats(head_cand, 8)
    for i in range(1, hn):
        area = int(hstats[i, cv2.CC_STAT_AREA])
        if area < w * h * 0.004 or area > w * h * 0.12:
            continue
        hx0 = hstats[i, cv2.CC_STAT_LEFT]
        hx1 = hx0 + hstats[i, cv2.CC_STAT_WIDTH]
        hy0 = hstats[i, cv2.CC_STAT_TOP]
        fillr = area / float(max(1, hstats[i, cv2.CC_STAT_WIDTH] *
                                 hstats[i, cv2.CC_STAT_HEIGHT]))
        if fillr < 0.55:
            continue
        head_boxes.append((hx0, hx1, hy0))
        head[hlab == i] = 255
    rig_tops = list(tops)
    exempt_x = set()
    if rig_tops:
        # 杆 → x 重叠且 y 窗口内有灯头块
        for (px, py, pw, ph) in rig_tops:
            wy0 = max(0, py - int(h * 0.22))
            wy1 = py + int(ph * 0.15)
            cx0, cx1 = px, px + pw
            for (hx0, hx1, hy0) in head_boxes:
                if hx1 > cx0 - pw * 0.3 and hx0 < cx1 + pw * 0.3 and \
                        wy0 <= hy0 <= wy1:
                    exempt_x.add(round(px))
                    info["heads"] += 1
                    break
        rig_tops = [t for t in rig_tops
                    if round(t[0]) not in exempt_x] or rig_tops

    # ---- 建筑结构判据：等距阵列里的**满高杆**是窗框/栏杆（拒绝）。
    # 带灯头的杆（灯架，exempt_x）豁免——不参与等距判定也不被拒。
    if len(rig_tops) >= 2:
        xs = sorted(t[0] for t in rig_tops if round(t[0]) not in exempt_x)
        gaps = [b - a for a, b in zip(xs, xs[1:]) if b - a > 4]
        if len(gaps) >= 2:
            med = float(np.median(gaps))
            rejected_boxes = []
            for t in rig_tops:
                if round(t[0]) in exempt_x:
                    continue          # 灯架豁免
                tx, ty, tw, th = t
                full_h = th >= h * 0.72
                i = xs.index(tx) if tx in xs else -1
                in_chain = (
                    (0 < i and abs(tx - xs[i - 1] - med) / med < 0.35) or
                    (0 <= i < len(xs) - 1 and
                     abs(xs[i + 1] - tx - med) / med < 0.35))
                if in_chain and full_h:
                    rejected_boxes.append(t)
            if rejected_boxes:
                info["rejected_pattern"] = (
                    "等距满高杆阵列（窗框/栏杆）%d 根" % len(rejected_boxes))
                rig_tops = [t for t in rig_tops if t not in rejected_boxes]
                for (rx, ry, rw, rh) in rejected_boxes:
                    pole[ry:ry + rh, max(0, rx - 2):rx + rw + 2] = 0
        info["poles"] = len(rig_tops)

    # ---- 独立灯箱（无杆）：高亮低饱和大块里**没有配对杆**的那些。
    # LED 平板灯/柔光箱独立挂在天花上时，画面里只有箱没有杆。
    # 判据同灯头（面积/矩形度 ≥0.55），配对过杆的块不算独立灯箱。
    lone_heads = 0
    for (hx0, hx1, hy0) in head_boxes:
        paired = any(
            hx1 > px - pw * 0.3 and hx0 < px + pw * 1.3 and
            max(0, py - int(h * 0.22)) <= hy0 <= py + int(ph * 0.15)
            for (px, py, pw, ph) in rig_tops)
        if not paired:
            lone_heads += 1
    if lone_heads:
        head_cand2 = ((gray > 0.80) & (s < 0.25)).astype(np.uint8)
        head_cand2 = cv2.morphologyEx(
            head_cand2, cv2.MORPH_CLOSE, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (7, 7)))
        hn2, hlab2, hst2, _ = cv2.connectedComponentsWithStats(head_cand2, 8)
        for i in range(1, hn2):
            area = int(hst2[i, cv2.CC_STAT_AREA])
            if area < w * h * 0.004 or area > w * h * 0.12:
                continue
            bx0 = hst2[i, cv2.CC_STAT_LEFT]
            bx1 = bx0 + hst2[i, cv2.CC_STAT_WIDTH]
            by0 = hst2[i, cv2.CC_STAT_TOP]
            fillr = area / float(max(1, hst2[i, cv2.CC_STAT_WIDTH] *
                                     hst2[i, cv2.CC_STAT_HEIGHT]))
            if fillr < 0.55:
                continue
            paired = any(
                bx1 > px - pw * 0.3 and bx0 < px + pw * 1.3 and
                max(0, py - int(h * 0.22)) <= by0 <= py + int(ph * 0.15)
                for (px, py, pw, ph) in rig_tops)
            if not paired:
                head[hlab2 == i] = 255
    info["lone_heads"] = lone_heads

    m = cv2.dilate(np.maximum(pole, head), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (9, 9)))
    m = (m > 0).astype(np.float32)
    # 主体保护
    if subject_mask is not None:
        subj = np.clip(np.asarray(subject_mask, np.float32), 0, 1)
        if subj.shape[:2] != (h, w):
            subj = cv2.resize(subj, (w, h), interpolation=cv2.INTER_LINEAR)
        m = m * (1.0 - (subj > 0.5).astype(np.float32))
    # 软边
    m = cv2.GaussianBlur(m, (0, 0), 2.0)
    info["candidates"] = info["poles"] + info["heads"]
    info["mask_frac"] = round(float((m > 0.5).mean()), 5)
    return np.clip(m, 0, 1), info


def remove_rig(img: np.ndarray, subject_mask: Optional[np.ndarray] = None,
               mask: Optional[np.ndarray] = None,
               strength: float = 1.0) -> Tuple[np.ndarray, dict]:
    """去灯架/灯。`mask` 给定则直接用（手涂），否则自动识别."""
    from . import lama as LM

    h, w = img.shape[:2]
    if mask is None:
        m, dinfo = detect_rig(img, subject_mask)
    else:
        m = np.clip(np.asarray(mask, np.float32), 0, 1)
        if m.ndim == 3:
            m = m[..., 0]
        dinfo = {"applicable": True, "manual": True,
                 "mask_frac": round(float((m > 0.5).mean()), 5)}
    if float((m > 0.5).sum()) < 40:
        return img.copy(), {**dinfo, "skipped": "没有可移除的道具"}

    filled, finfo = LM.fill(img, m)
    a = (m * float(np.clip(strength, 0, 1)))[..., None]
    out = np.clip(img * (1 - a) + filled * a, 0, 1).astype(np.float32)
    return out, {**dinfo, "lama_windows": finfo.get("windows",
                                                    finfo) if isinstance(
        finfo, dict) else None}


def info() -> dict:
    return {"feature": "remove_rig",
            "pipeline": "thin-vertical-pole + head-block → subject "
                        "protection → LaMa",
            "deterministic": True}
