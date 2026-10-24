#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 RetouchLab 应用图标（assets/retouchlab.ico）。

深色圆角底 + 品牌金光圈 + 白色 R。纯 PIL 绘制，无外部素材。
多尺寸（16/24/32/48/64/128/256）打包成 ico。
"""
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "retouchlab.ico")


def draw(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = size * 0.22
    # 深色圆角底
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r,
                        fill=(18, 18, 21, 255))
    # 金色描边光圈（宽度随尺寸缩放）
    w = max(1, size // 16)
    d.rounded_rectangle([w, w, size - 1 - w, size - 1 - w], radius=r * 0.9,
                        outline=(232, 179, 65, 255), width=w)
    # 中心 R
    fs = int(size * 0.58)
    font = None
    for name in ("segoeuib.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/" + name, fs)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    text = "R"
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1]), text,
           font=font, fill=(245, 220, 160, 255))
    return img


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    imgs = [draw(s) for s in (16, 24, 32, 48, 64, 128, 256)]
    imgs[-1].save(OUT, format="ICO",
                  sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    print("written", OUT, os.path.getsize(OUT), "bytes")
