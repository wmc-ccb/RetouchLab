# -*- coding: utf-8 -*-
"""导出后处理：批量水印 + 文件命名模板。

批量面板的收尾两件套——交付前打水印、导出名按规则生成。
水印支持文字（PIL，系统字体）与 Logo 图片两种；命名模板支持
{原名}/{序号3}/{日期}/{时间}/{预设} 占位符。
"""
from __future__ import annotations

import os
import re
from typing import Optional

import cv2
import numpy as np

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyhbd.ttc",    # 微软雅黑 Bold
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
)

# 位置代号 → (锚点 x, y)   0/0.5/1 = 左/中/右
_POSITIONS = {
    "tl": (0.0, 0.0), "tc": (0.5, 0.0), "tr": (1.0, 0.0),
    "ml": (0.0, 0.5), "mc": (0.5, 0.5), "mr": (1.0, 0.5),
    "bl": (0.0, 1.0), "bc": (0.5, 1.0), "br": (1.0, 1.0),
}


def _load_font(px: int):
    from PIL import ImageFont
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, px)
            except Exception:
                continue
    return ImageFont.load_default()


def apply_watermark(img: np.ndarray, text: str = "",
                    logo_path: str = "",
                    position: str = "br",
                    opacity: float = 0.65,
                    scale: float = 4.0,
                    margin: float = 3.0) -> np.ndarray:
    """在 float32 RGB 图上叠水印。text 与 logo_path 至少给一个。

    scale: 水印尺寸相对图宽的千分比（4.0 → 0.4% 字高起步，随文字
    长度自适应放宽到可读）；margin: 边距相对图宽百分比。
    """
    if not text and not (logo_path and os.path.exists(logo_path)):
        return img
    from PIL import Image, ImageDraw

    h, w = img.shape[:2]
    base = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    ax, ay = _POSITIONS.get(position, _POSITIONS["br"])
    m = int(w * margin / 100.0)

    if text:
        px = max(14, int(h * scale / 100.0))
        font = _load_font(px)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = int(w * ax - (tw + m) if ax == 1 else
                 (w * ax - tw / 2 if ax == 0.5 else m)) - bbox[0]
        ty = int(h * ay - (th + m) if ay == 1 else
                 (h * ay - th / 2 if ay == 0.5 else m)) - bbox[1]
        alpha = int(255 * float(opacity))
        draw.text((tx, ty), text, font=font, fill=(255, 255, 255, alpha))
        # 细描边保证浅色背景上也可读
        draw.text((tx, ty), text, font=font, fill=(0, 0, 0, alpha // 2),
                  stroke_width=max(1, px // 28), stroke_fill=(0, 0, 0, alpha // 2))
    else:
        logo = Image.open(logo_path).convert("RGBA")
        lw = int(w * scale * 4 / 100.0)
        lh = int(logo.height * lw / max(logo.width, 1))
        logo = logo.resize((lw, lh))
        lx = int(w * ax - (lw + m) if ax == 1 else
                 (w * ax - lw / 2 if ax == 0.5 else m))
        ly = int(h * ay - (lh + m) if ay == 1 else
                 (h * ay - lh / 2 if ay == 0.5 else m))
        a = logo.getchannel("A").point(lambda v: int(v * opacity))
        logo.putalpha(a)
        overlay.alpha_composite(logo, (lx, ly))

    out = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    return np.asarray(out, np.uint8).astype(np.float32) / 255.0


_TPL_TOKEN = re.compile(r"\{(原名|序号3|日期|时间|预设)\}")


def build_filename(pattern: str, src_path: str, seq: int,
                   preset: str = "") -> str:
    """命名模板 → 新文件名（不含扩展名）。

    默认模板 "{原名}_out"；占位符：原名 / 序号3 / 日期 / 时间 / 预设。
    非法字符统一替换成 _，防止模板把路径分隔符带进来。
    """
    base = os.path.splitext(os.path.basename(src_path))[0]
    import time as _t
    table = {
        "原名": base,
        "序号3": "%03d" % seq,
        "日期": _t.strftime("%Y%m%d"),
        "时间": _t.strftime("%H%M%S"),
        "预设": preset or "",
    }
    name = _TPL_TOKEN.sub(lambda m: table[m.group(1)], pattern or "{原名}_out")
    name = re.sub(r'[\\/:*?"<>|]', "_", name).strip() or base
    return name[:120]
