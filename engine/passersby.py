# -*- coding: utf-8 -*-
"""AI 祛路人：把背景里的其他人从画面里抹掉.

参考软件的「AI 祛路人」是一个云端生成功能；这里用**确定性管线**替代：

  1. YuNet 检出所有人脸（多人场景）
  2. 主对象 = 与语义分割主体重叠最大的那张脸；其余是路人
  3. 每个路人以脸为锚，按解剖比例扩成头+身的粗掩膜
  4. 主体掩膜之外的部分交给 LaMa 填充

为什么不做「识别这是路人」的语义判断：selfie_multiclass 只有 6 类、且
是**单主体**模型，它对背景里第二个人输出的是"背景"。YuNet 的人脸框 +
比例外扩虽然粗糙，但**位置是确定的**，配合 LaMa 的填充能力足够把路人
抹掉；而"用模型认路人"在现有模型下做不到，只能靠编。

边界情况（都在测试里钉住）：
  * 只有一张脸 -> 没有路人，返回原图
  * 路人脸框贴着主体 -> 掩膜与主体相交的部分被剔除，不会误删主体
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _subject_mask(img: np.ndarray) -> Optional[np.ndarray]:
    try:
        from . import semantic as SEM

        if not SEM.available():
            return None
        reg = SEM.segment(img)
        return reg.norm("subject") if reg is not None else None
    except Exception:
        return None


def find_passersby(img: np.ndarray
                   ) -> Tuple[List[dict], Optional[np.ndarray],
                              Optional[list]]:
    """Return (passers-by, subject mask, subject face box).

    The subject face box is what lets `remove_passersby` build a guaranteed
    protected zone: the single-subject segmentation (selfie_multiclass) does
    NOT cover other people in a group shot, so a passer-by mask overlapping
    the subject's body can eat the subject (measured: in an 8-person frame
    the coser's head was filled with background). The box + the same
    anatomical expansion as person_mask is the reliable guard.
    """
    from .face import FaceDetector
    from . import geoface as GEO

    h0, w0 = img.shape[:2]
    long0 = max(h0, w0)
    # 大图先降到 ~900px 工作尺度。实测检测率对输入尺寸不是单调的：
    # 同一张 60px 的路人脸，长边 900–1300px 时稳定检出，1400px 以上反而
    # 消失（分数归一被稀释）；preview 通常是 1399px，正好卡在失效边界。
    # 所以只要长边超过 900 就降，让所有人脸落回检出窗口，框映回原尺度。
    det_img, k = img, 1.0
    if long0 > 900:
        k = 900.0 / long0
        det_img = cv2.resize(img, (int(w0 * k), int(h0 * k)),
                             interpolation=cv2.INTER_AREA)
    scales = [1.0]
    if max(det_img.shape[:2]) < 1600:
        # 第二尺度：略放大，补检更小的脸
        s2 = min(1.4, 1600.0 / max(det_img.shape[:2]))
        if s2 > 1.05:
            scales.append(s2)

    det = FaceDetector(dense=False)
    raw = []          # (box_at_working_scale, score)
    for s in scales:
        work = det_img if s == 1.0 else cv2.resize(
            det_img, (int(det_img.shape[1] * s), int(det_img.shape[0] * s)),
            interpolation=cv2.INTER_CUBIC)
        for f in det.detect(work, max_faces=12):
            bx = [float(v) / (s * k) for v in f.box]
            raw.append((bx, float(getattr(f, "score", 0.5) or 0.5)))
    if not raw:
        return [], None

    # 框级去重：IoU > 0.6 视为同一张脸
    raw.sort(key=lambda t: -(t[0][2] * t[0][3]))
    merged: List[List[float]] = []
    for bx, sc in raw:
        dup = False
        for m in merged:
            iou = _iou(bx, m)
            if iou > 0.6:
                dup = True
                break
        if not dup:
            merged.append(bx)
    faces = merged
    subj = _subject_mask(img)
    # face_overlap 用 raw subject 聚合（不用 norm——见 face_overlap 文档）
    subj_raw = None
    try:
        from . import semantic as SEM

        reg = SEM.segment(img)
        subj_raw = np.clip(reg.subject(), 0, 1) if reg is not None else None
    except Exception:
        subj_raw = None
    subj_d = subj   # 打分在工作尺度上，faces 已映回工作尺度

    def score(bx) -> float:
        x, y, w, h = bx
        if subj is None:
            return w * h                      # 没有分割：最大的脸当主体
        m = np.zeros((h0, w0), np.float32)
        m[int(max(0, y)):int(min(h0, y + h)),
          int(max(0, x)):int(min(w0, x + w))] = 1.0
        return float((m * (subj > 0.5)).sum())   # 与主体重叠最大的是主体

    def face_overlap(bx) -> float:
        """脸框内 subject 掩膜的占比（0..1）。

        单主体模型的语义：subject 类"认得"的人才是画面人物。
        overlap 高 = 模型认为他是画面的一部分（**保护**）；
        overlap 低 = 模型根本没把他当画面主体（背景路人，**可删**）。
        实测合影：三个主体级人物 0.86/0.77/0.68，背景小人 0.01~0.04
        ——0.45 的分界把两类完全分开。

        注意必须用 **raw subject 聚合**（reg.subject()），不能用
        norm("subject")：分位归一会把背景弱响应拉高，overlap 失真
        （实测 norm 版把 0.86 的人物算成路人）。
        """
        x, y, bw, bh = [int(v) for v in bx]
        if subj_raw is None:
            return 0.0
        patch = (subj_raw > 0.5)[max(0, y):min(h0, y + bh),
                                 max(0, x):min(w0, x + bw)]
        return float(patch.mean()) if patch.size else 0.0

    order = sorted(faces, key=score, reverse=True)
    subject = order[0]
    rest = []
    for bx in order[1:]:
        if _near(float(subject[0]), float(subject[1]),
                 bx[0], bx[1], max(float(subject[2]), 24.0)):
            continue
        # 太小的（<20px）多半是误检：抹掉一片背景比留着更糟
        if bx[2] < 20 or bx[3] < 20:
            continue
        # **画面人物保护**：subject 模型认得的人（脸框内 subject 占比
        # ≥0.45）不是路人——多人合影里"只有 1 个主体、其余全删"的旧
        # 逻辑会把合影主角删掉（实测 overlap 0.86/0.68 的两张脸被归为
        # 路人）。这里用绝对阈值区分两类。
        if face_overlap(bx) >= 0.45:
            continue
        rest.append({"box": [float(v) for v in bx], "subject": False})
    return rest, subj, [float(v) for v in order[0]]


def _iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    x0 = max(ax0, bx0); y0 = max(ay0, by0)
    x1 = min(ax0 + aw, bx0 + bw); y1 = min(ay0 + ah, by0 + bh)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    return inter / float(aw * ah + bw * bh - inter)


def _near(x0, y0, x1, y1, tol) -> bool:
    return abs(x0 - x1) < tol and abs(y0 - y1) < tol


def person_mask(img: np.ndarray, box, full_body: bool = True) -> np.ndarray:
    """头+身粗掩膜：以人脸框为锚按解剖比例外扩.

    头高 ≈ 2.2×脸高（含头发），肩宽 ≈ 2.6×脸宽，躯干向下延到画框底。
    这不是精确分割——目的是给 LaMa 一个"这里有人"的充分区域，LaMa 会
    参考周围背景把内容补上，所以掩膜略大比略小安全（略小会留下人形残影）。

    `full_body=False`（**远处小人**用）：身体高度按全身比例估
    （站姿全身 ≈ 7.5×脸高），**不直达画底**——背景小人的脸 15px 时
    直达画底的矩形会抹掉整幅背景，实际只需要覆盖脚下那一点。
    """
    h, w = img.shape[:2]
    x, y, bw, bh = [float(v) for v in box]
    m = np.zeros((h, w), np.float32)

    # 头部：脸上方扩 0.8×脸高（头发/头顶），左右各扩 0.35×脸宽
    hx0 = int(max(0, x - bw * 0.35)); hx1 = int(min(w, x + bw * 1.35))
    hy0 = int(max(0, y - bh * 0.85)); hy1 = int(min(h, y + bh * 1.25))
    m[hy0:hy1, hx0:hx1] = 1.0
    # 身体：从下巴往下，宽度按肩宽比例。
    sy0 = hy1 - int(bh * 0.1)
    sw = bw * 1.3
    bx0 = int(max(0, x + bw * 0.5 - sw)); bx1 = int(min(w, x + bw * 0.5 + sw))
    if full_body:
        m[sy0:h, bx0:bx1] = 1.0
    else:
        # 远景小人：站姿全身 ≈ 7.5×脸高，脚下再多留 0.2×脸高
        by1 = int(min(h, y + bh * 7.5 + bh * 0.2))
        m[sy0:by1, bx0:bx1] = 1.0
    # 软边
    m = cv2.GaussianBlur(m, (0, 0), max(3.0, min(bw, bh) * 0.12))
    return np.clip(m, 0, 1)


def _passerby_mask_precise(img: np.ndarray, box, clothes: np.ndarray
                           ) -> np.ndarray:
    """邻人相邻时的精确路人掩膜：头（脸框外扩）+ 本人的衣服区域.

    person_mask 的身体是"肩宽 2.6 倍直达画框底"的粗矩形——相邻人挨着
    时必然盖到旁边的人，减去主体保护区后剩下被裁剪的碎掩膜，LaMa 补完
    就是半透明残影（实测合影）。semantic 的 clothes 类是**多实例**的
    （8 人合影实测覆盖 36%，每人的衣服都在），用它把身体掩膜收紧到
    本人衣服上：脸框正下方一个锥形窗口内的 clothes 像素 = 这个人的身体。
    """
    h, w = img.shape[:2]
    x, y, bw, bh = [float(v) for v in box]
    m = np.zeros((h, w), np.float32)
    # 头：脸上方扩（同 person_mask）
    hx0 = int(max(0, x - bw * 0.35)); hx1 = int(min(w, x + bw * 1.35))
    hy0 = int(max(0, y - bh * 0.85)); hy1 = int(min(h, y + bh * 1.25))
    m[hy0:hy1, hx0:hx1] = 1.0
    # 身体：脸正下方的锥形窗口（向下渐宽），只取 clothes 掩膜
    cy0 = hy1 - int(bh * 0.1)
    span = bw * 1.6
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    prog = np.clip((yy - cy0) / max(h - cy0, 1.0), 0, 1)
    halfw = bw * 0.9 + prog * span
    cone = ((xx >= x + bw / 2 - halfw) &
            (xx <= x + bw / 2 + halfw) & (yy >= cy0)).astype(np.float32)
    body = np.clip(clothes, 0, 1) * cone
    m = np.maximum(m, body)
    # 收紧后仍轻外扩一次，把贴身轮廓包进 mask
    m = cv2.dilate((m > 0.4).astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.0)
    return np.clip(m, 0, 1)


def remove_passersby(img: np.ndarray, strength: float = 1.0,
                     max_people: int = 6, model: str = "lama"
                     ) -> Tuple[np.ndarray, dict]:
    """把背景里的人抹掉，返回 (图, 统计).

    `model`：**lama**（快 ~5s，纯纹理延续，复杂背景常有半透明残影——
    它不懂"人走了之后这里应该是什么"）或 **sdxl**（质量 ~30-60s，扩散
    模型真理解语义，复杂背景明显更干净；需要 ComfyUI 在 8388 运行）。
    sdxl 失败自动回落 lama，info["model"] 记录实际用的那个。
    """
    out = np.asarray(img, np.float32).copy()
    people, subj, subject_box = find_passersby(out)
    info = {"detected": len(people) + 1, "removed": 0,
            "boxes": [p["box"] for p in people], "model": model}
    if not people or strength <= 0:
        return np.clip(out, 1) if False else np.clip(out, 0, 1), info

    h, w = out.shape[:2]
    hole = np.zeros((h, w), np.float32)
    # **强制主体保护区**：主体脸框按 person_mask 同样的解剖比例扩出的
    # 粗掩膜。单主体分割在多人合影里不覆盖其他人的身体（实测 8 人框里
    # subject 只盖住 3/8 张脸的中心），路人掩膜与主体身体相交时，仅靠
    # semantic 剔除会把主体当路人抹掉——实测主体头部被 LaMa 填成背景。
    guard = np.zeros((h, w), np.float32)
    if subject_box:
        guard = np.maximum(guard, person_mask(out, subject_box))
    if subj is not None:
        guard = np.maximum(guard, np.clip(subj, 0, 1))
    # clothes 类（多实例，覆盖每个人的衣服）——相邻路人的精确掩膜用
    clothes = None
    try:
        from . import semantic as SEM

        reg = SEM.segment(out)
        if reg is not None:
            clothes = np.clip(reg.norm("clothes"), 0, 1)
    except Exception:
        clothes = None
    used = 0
    precise_used = 0
    for p in people[:max_people]:
        bx = p["box"]
        # 远处小人（脸 <30px）用**局部身体**掩膜：直达画底的矩形会
        # 抹掉整幅背景，实际只需要覆盖到脚下。
        face_px = float(min(bx[2], bx[3]))
        pm = person_mask(out, bx, full_body=(face_px >= 30.0))
        # 主体必须毫发无损：路人掩膜与保护区相交的部分剔除
        pm_cut = pm * (1.0 - guard)
        # 相邻人（粗掩膜被保护区裁掉 >35%）时，person_mask 的粗矩形
        # 本身就盖到了邻人——换成"头+本人衣服"的精确掩膜，消除残影根因
        if clothes is not None and float((pm - pm_cut).max()) > 0.35:
            pm2 = _passerby_mask_precise(out, bx, clothes)
            pm2 = pm2 * (1.0 - guard)
            # 两个掩膜取并集：精确掩膜盖到的部分 + 粗掩膜的干净部分
            pm = np.maximum(pm_cut, pm2)
            precise_used += 1
        else:
            pm = pm_cut
        hole = np.maximum(hole, pm)
        used += 1
    info["precise_masks"] = precise_used
    if hole.max() < 0.3:
        info["removed"] = 0
        info["note"] = "路人掩膜与主体完全重叠，无可安全消除的区域"
        return np.clip(out, 0, 1), info

    # **范围口径（用户确认）**：祛路人只针对**零散路人**（单个人、
    # 离主体远）。多紧挨路人的大面积删除不在支持范围（LaMa 是雾影、
    # 扩散发明假人，两次实测）——如实说明，建议重拍或裁剪。
    frac = float((hole > 0.5).mean())
    info["hole_frac"] = round(frac, 4)
    info["guard_touch"] = round(
        float((hole * (guard > 0.5)).sum() / max((hole > 0.5).sum(), 1)), 4)
    if frac > 0.15:
        info["removed"] = 0
        info["guidance"] = (
            "待删的路人占画面 %.0f%%——多个紧挨的路人超出自动删除的"
            "支持范围（效果会是半透明残影）。建议裁掉该区域或重拍。"
            "零散单个路人不受影响。" % (frac * 100))
        return np.clip(out, 0, 1), info

    hard = (hole * strength > 0.5).astype(np.uint8)
    if hard.sum() < 32:
        info["removed"] = 0
        return np.clip(out, 0, 1), info

    # ---- sdxl 档：扩散模型生成（质量高，需 ComfyUI）----
    if model == "sdxl":
        try:
            from . import genfill as GF

            res = GF.fill(out, hole,
                          prompt="clean street scene, empty pavement, "
                                 "no people, seamless continuation",
                          negative="person, people, figure, blurry, "
                                   "artifacts, deformed",
                          steps=20, denoise=1.0, model="sdxl")
            filled = np.asarray(res[0], np.float32)
            a = np.clip(hole * float(strength), 0, 1)[..., None]
            out = out * (1 - a) + filled * a
            info["removed"] = used
            info["gen_seconds"] = res[1].get("seconds")
            return np.clip(out, 0, 1), info
        except Exception as e:
            info["sdxl_error"] = str(e)[:100]
            info["model"] = "lama (sdxl 回落)"
            # 落到 lama 继续

    # ---- lama 档：快，纹理延续 ----
    try:
        from . import lama as LM

        if not LM.available():
            raise RuntimeError("LaMa 不可用")
        # fill 返回 (image, info)，不是裸数组——早先直接 np.asarray(filled)
        # 会把 tuple 拆成异构序列报错。
        res = LM.fill(out, hole)
        filled = np.asarray(res[0], np.float32)
        a = np.clip(hole * float(strength), 0, 1)[..., None]
        out = out * (1 - a) + filled * a
        info["removed"] = used
        if isinstance(res[1], dict):
            info["lama"] = {k: res[1][k] for k in list(res[1])[:4]}
    except Exception as e:
        info["error"] = str(e)[:120]
        info["removed"] = 0
    return np.clip(out, 0, 1), info


def info() -> dict:
    return {"feature": "remove_passersby",
            "pipeline": "YuNet faces -> subject overlap -> "
                        "head+body ratio mask -> LaMa",
            "deterministic": True}
