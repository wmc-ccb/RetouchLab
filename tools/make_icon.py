#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 ZhaJinWenQianCake 应用图标（assets/retouchlab.ico）。

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
    # 金色描边光圈
    w = max(1, size // 16)
    d.rounded_rectangle([w, w, size - 1 - w, size - 1 - w], radius=r * 0.9,
                        outline=(232, 179, 65, 255), width=w)
    # 中心: 金色双层蛋糕（两层圆角横条 + 顶上烛光点）
    cx = size / 2
    gold = (232, 179, 65, 255)
    gold_hi = (245, 220, 160, 255)
    bw = size * 0.46
    bh = size * 0.14
    y_base = size * 0.66
    d.rounded_rectangle([cx - bw / 2, y_base - bh, cx + bw / 2, y_base],
                        radius=bh * 0.35, fill=gold)
    bw2 = bw * 0.72
    y2 = y_base - bh - size * 0.02
    d.rounded_rectangle([cx - bw2 / 2, y2 - bh, cx + bw2 / 2, y2],
                        radius=bh * 0.35, fill=gold_hi)
    # 烛光
    cr = max(1, size * 0.035)
    d.ellipse([cx - cr, y2 - bh - size * 0.10 - cr,
               cx + cr, y2 - bh - size * 0.10 + cr], fill=gold)
    return img


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    imgs = [draw(s) for s in (16, 24, 32, 48, 64, 128, 256)]
    imgs[-1].save(OUT, format="ICO",
                  sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
    print("written", OUT, os.path.getsize(OUT), "bytes")
