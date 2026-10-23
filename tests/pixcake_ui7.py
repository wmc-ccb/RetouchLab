# -*- coding: utf-8 -*-
"""Capture the top bar and the bottom filmstrip at 3x zoom for exact layout."""
import os
import subprocess
import sys
import time

from PIL import Image, ImageGrab
import win32gui

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "pixcake")
os.makedirs(OUT, exist_ok=True)


def hwnd():
    out = subprocess.check_output(
        ["powershell", "-NoProfile", "-Command",
         "(Get-Process pixcake | Where-Object {$_.MainWindowHandle -ne 0})"
         ".MainWindowHandle"], text=True)
    for tok in out.replace("\r", " ").replace("\n", " ").split():
        if tok.strip().isdigit():
            return int(tok.strip())
    raise RuntimeError("no pixcake window")


def main():
    h = hwnd()
    l, t, r, b = win32gui.GetWindowRect(h)
    w, ht = r - l, b - t
    print("window %dx%d" % (w, ht))
    win32gui.SetForegroundWindow(h)
    time.sleep(1.2)
    img = ImageGrab.grab(bbox=(l, t, r, b))

    # 顶栏两段：左（logo/工具）与右（账号/导出）
    img.crop((1080, 0, 1500, 44)).resize((1260, 132),
                                         Image.LANCZOS).save(
        os.path.join(OUT, "_tabs3x.png"))
    img.crop((2020, 0, 2558, 44)).resize((1614, 132),
                                         Image.LANCZOS).save(
        os.path.join(OUT, "_topright3x.png"))
    # 底部胶片栏（含上方信息行）
    img.crop((0, ht - 130, 2260, ht)).save(
        os.path.join(OUT, "_film3x.png"))
    # 底栏左段（计数/筛选）放大
    img.crop((0, ht - 64, 700, ht)).resize((2100, 192),
                                           Image.LANCZOS).save(
        os.path.join(OUT, "_filmleft3x.png"))
    # 右侧图标栏完整（帮助/反馈在底部）
    img.crop((w - 44, 0, w, ht)).resize((132, ht * 3),
                                        Image.LANCZOS).save(
        os.path.join(OUT, "_railfull3x.png"))
    print("saved 5 crops")
    return 0


if __name__ == "__main__":
    sys.exit(main())
