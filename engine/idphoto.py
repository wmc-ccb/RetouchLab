# -*- coding: utf-8 -*-
"""证件照：标准尺寸规格 + 头部占比裁切 + 换底色.

参考软件的「证件照换背景」与「创成式填充 → 证件照换背景」。本地可确定
完成的部分：

  1. **规格裁切**：中国标准（GB/T）证件照是"头部占画面高度的比例 +
     固定输出像素"，不同规格不同（一寸 295x413、二寸 413x579、
     小二寸 413x531、签证 350x450 等）。头顶到画面顶留 8~12% 高，
     眼睛在画面高度 45~52% 处。
  2. **换底色**：纯色背景走色键（bgclean.color_key_alpha，IoU 0.986），
     复杂背景回退抠图模型。

规格表用中国出入境/摄影行业的通用值；头部定位用 YuNet 的人脸框
（脸框到头顶 ≈ 0.45×脸高，到下巴 ≈ 1.25×脸高）。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# name -> (宽px, 高px, 头部占画面高度比例)
# 头高（头顶到下巴）的标准占比来自证件照规范：一寸约 0.62~0.72，
# 二寸约 0.60~0.70；取 0.68 作为通用目标。
SPECS: Dict[str, Tuple[int, int, float]] = {
    "one_inch":   (295, 413, 0.68),    # 一寸 25×35mm
    "two_inch":   (413, 579, 0.66),    # 二寸 35×49mm
    "small_two":  (413, 531, 0.68),    # 小二寸 35×45mm
    "visa":       (350, 450, 0.70),    # 签证 35×45mm
    "passport":   (390, 567, 0.70),    # 护照 33×48mm
}


def _face_box(img: np.ndarray):
    try:
        from .face import FaceDetector

        faces = FaceDetector(dense=False).detect(img, max_faces=3)
        if not faces:
            return None
        # 最大的脸
        return max(faces, key=lambda f: f.box[2] * f.box[3]).box
    except Exception:
        return None


def crop_spec(img: np.ndarray, spec: str = "two_inch"
              ) -> Tuple[np.ndarray, dict]:
    """按规格裁出证件照构图（未换底色）.

    定位逻辑：脸框高 fh -> 估计头高 ≈ 1.55×fh（头顶≈脸上方0.45fh，
    下巴≈脸下方0.25fh + 脸高0.85fh）。目标头高 = 高度×占比 -> 输出
    视野高度 = 头高/占比。头顶留白 = 视野高 × (1-占比) × 0.55（上方
    略小于下方，符合证件照视觉规范）。
    """
    if spec not in SPECS:
        raise ValueError("unknown spec: " + spec)
    w_px, h_px, ratio = SPECS[spec]
    h, w = img.shape[:2]
    box = _face_box(img)
    info = {"spec": spec, "out": [w_px, h_px], "face": None,
            "fallback_center": False}
    if box is None:
        # 没脸：中心裁标准比例
        ar = w_px / h_px
        cw = min(w, int(h * ar))
        ch = int(cw / ar)
        y0 = (h - ch) // 2; x0 = (w - cw) // 2
        out = img[y0:y0 + ch, x0:x0 + cw]
        info["fallback_center"] = True
        out = cv2.resize(out, (w_px, h_px), interpolation=cv2.INTER_AREA)
        return out, info

    x, y, bw, bh = [float(v) for v in box]
    info["face"] = [round(v, 1) for v in (x, y, bw, bh)]
    face_h = bh
    # 头部几何（以 YuNet 框为锚，系数来自人脸测量比例）：
    #   头顶   = 脸上缘 - 0.42×脸高
    #   下巴   = 脸下缘 + 0.30×脸高
    #   头高   ≈ 1.72×脸高
    # 规范要求的"头部占画面高度"就是 头高/view_h = ratio。
    head_top = y - face_h * 0.42
    head_h = face_h * 1.72
    view_h = head_h / ratio
    view_w = view_h * (w_px / h_px)
    # 头顶留白：证件照上方留白略小于下方，取剩余空间的 45%
    top_gap = (view_h - head_h) * 0.45
    cy_top = head_top - top_gap
    cx_mid = x + bw / 2
    x0 = int(round(cx_mid - view_w / 2))
    y0 = int(round(cy_top))
    x1 = int(round(x0 + view_w)); y1 = int(round(y0 + view_h))

    # 越界处理：**先补背景再裁**，而不是缩小视野。
    # 证件照要求头部占比准确（0.6~0.7），缩小视野会让头撑满画面（实测
    # 这张人脸贴着原图上缘，shrink 后头占满整幅）。要补的背景马上会被
    # 换掉，所以用四角估计的背景色填充，视觉上不可见。
    pad_l = max(0, -x0); pad_t = max(0, -y0)
    pad_r = max(0, x1 - w); pad_b = max(0, y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        k = max(6, min(h, w) // 12)
        corners = np.concatenate([
            img[:k, :k].reshape(-1, 3), img[:k, -k:].reshape(-1, 3),
            img[-k:, :k].reshape(-1, 3), img[-k:, -k:].reshape(-1, 3)])
        bgc = np.median(corners, axis=0)
        canvas = np.ones((h + pad_t + pad_b, w + pad_l + pad_r, 3),
                         np.float32) * bgc.reshape(1, 1, 3)
        canvas[pad_t:pad_t + h, pad_l:pad_l + w] = img
        img = canvas
        x0 += pad_l; y0 += pad_t
        x1 += pad_l; y1 += pad_t
        info["padded"] = [pad_l, pad_t, pad_r, pad_b]
    out = img[max(0, y0):min(img.shape[0], y1),
              max(0, x0):min(img.shape[1], x1)]
    out = cv2.resize(out, (w_px, h_px), interpolation=cv2.INTER_AREA)
    return out, info


def make_id_photo(img: np.ndarray, spec: str = "two_inch",
                  colour=(255, 255, 255), keep_lighting: bool = False
                  ) -> Tuple[np.ndarray, dict]:
    """证件照 = 规格裁切 + 换底色.

    换底色在**裁切后的特写**上做：特写里背景占比大、纯色判断更稳，
    也避免全图色键的无关干扰。

    `keep_lighting` 默认**关**。它把新底色乘以原背景的亮度来"保留打光"，
    但原背景里有高对比图案时，乘出来的就是图案本身——实测白色纹路背景
    换白底后纹路全部留着，灰斑的真正来源是它而不是 alpha。证件照要的是
    规范要求的**纯净平色**，明暗保留交给真实影棚打光。
    """
    from . import bgclean as BC
    from .imgutil import luminance

    crop, info = crop_spec(img, spec)
    w_px, h_px, _ = SPECS[spec]
    bgmask = BC.background_mask(crop)
    pur = BC.purity(crop, bgmask)
    info["purity"] = round(pur, 3)
    if pur >= 0.5:
        alpha, kinfo = BC.color_key_alpha(crop)
        info["key"] = "color_key"
        info.update(kinfo)
    else:
        try:
            from . import matting as MT

            if not MT.available():
                raise RuntimeError("matting unavailable")
            alpha = MT.subject_alpha(crop, mode="hybrid")
            info["key"] = "matting"
        except Exception as e:
            info["key"] = "failed"
            info["error"] = str(e)[:100]
            return crop, info
    a = np.clip(alpha, 0, 1)
    # 几何约束：头顶线以上必是背景（覆盖补边区），脸椭圆之外清零。
    ch, cw = crop.shape[:2]
    fbox = None
    try:
        from .face import FaceDetector

        fs = FaceDetector(dense=False).detect(crop, max_faces=3)
        if fs:
            fbox = max(fs, key=lambda f: f.box[2] * f.box[3]).box
    except Exception:
        fbox = None
    if fbox is not None:
        fx, fy, fw, fh = [float(v) for v in fbox]
        # 头顶线：脸上缘 - 0.5×脸高（含发）；这条线以上必是背景
        head_top = fy - fh * 0.5
        if head_top > 0:
            a[0:int(head_top), :] = 0
        # 左右：脸椭圆之外清零（横向灰斑高发区）
        cx_, cy_ = fx + fw / 2, fy + fh / 2
        rx = max(fw * 1.15, cw * 0.46)
        ry = ch * 1.2
        yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
        d = ((xx - cx_) / rx) ** 2 + ((yy - cy_) / ry) ** 2
        a = a * np.clip((1.10 - d) / 0.25, 0.0, 1.0)
    # 二值化 + 色距校验 + 轻羽化。
    # 灰斑的真相：matting 把背景里的高对比纹路（白色图案）误判成前景，
    # 且它们在脸椭圆**内部**——几何约束够不着。但它们的颜色与背景中位数
    # 几乎一致（离背景色 < 0.12），而头发/皮肤离得远。所以对"不是边界带"
    # 的像素做色距校验：离背景色近的一律当背景。
    ch, cw = crop.shape[:2]
    k = max(6, min(ch, cw) // 10)
    corners = np.concatenate([
        crop[:k, :k].reshape(-1, 3), crop[:k, -k:].reshape(-1, 3),
        crop[-k:, :k].reshape(-1, 3), crop[-k:, -k:].reshape(-1, 3)])
    bgc = np.median(corners, axis=0)
    dist_bg = np.linalg.norm(crop - bgc.reshape(1, 1, 3), axis=2)
    near_bg = dist_bg < 0.13
    # 边界带（alpha 0.2~0.8）不动——那是真正的发丝边缘
    edge_band = (alpha > 0.2) & (alpha < 0.8)
    a = np.where(near_bg & ~edge_band, 0.0, a)
    a = (a > 0.5).astype(np.float32)
    # 连通域清理：孤立的亮点/暗点（发顶噪点）不是主体，面积过小的前景块
    # 直接删掉。注意 a 此时还是 2D——早先写 `a[..., 0]` 是取第一列，
    # connectedComponentsWithStats 收到 1D 输入抛错又被 except 吞掉，
    # 清理从来没执行过（实测 39 个 <234px 的噪点全留着）。
    try:
        import cv2 as _cv2

        nlab, lab, stats, _ = _cv2.connectedComponentsWithStats(
            a.astype(np.uint8), 8)
        if fbox is not None and nlab > 1:
            min_area = max(24, int(fbox[2] * fbox[3] * 0.005))
            for i in range(1, nlab):
                if stats[i, _cv2.CC_STAT_AREA] < min_area:
                    a[lab == i] = 0
    except Exception:
        pass
    import cv2 as _cv
    a = _cv.GaussianBlur(a, (0, 0), 1.0)
    a = np.clip(a, 0, 1)[..., None]
    col = np.array(colour, np.float32) / 255.0
    newbg = np.tile(col.reshape(1, 1, 3), (h_px, w_px, 1))
    if keep_lighting:
        lum = luminance(crop)[..., None]
        bgsel = a[..., 0] < 0.5
        if bgsel.sum() > 100:
            base = float(np.median(lum[bgsel]))
            rel = np.clip(lum / max(base, 1e-3), 0.75, 1.25)
            newbg = np.clip(newbg * rel, 0, 1)
    out = np.clip(crop * a + newbg * (1 - a), 0, 1)
    return out, info


def info() -> dict:
    return {"specs": {k: {"px": [v[0], v[1]]} for k, v in SPECS.items()},
            "pipeline": "YuNet -> spec crop -> color-key/matting -> bg swap"}
