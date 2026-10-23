# -*- coding: utf-8 -*-
"""Capture PixCake's right rail at high zoom to read the exact category list."""
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
         "Get-Process pixcake -ErrorAction SilentlyContinue | "
         "Select-Object -ExpandProperty MainWindowHandle"], text=True)
    vals = [int(t.strip()) for t in out.split() if t.strip().isdigit()]
    vals = [v for v in vals if v > 0]
    if not vals:
        raise RuntimeError("no pixcake window")
    return vals[0]


def main():
    h = hwnd()
    win32gui.SetForegroundWindow(h)
    time.sleep(1.2)
    l, t, r, b = win32gui.GetWindowRect(h)
    w, ht = r - l, b - t
    img = ImageGrab.grab(bbox=(l, t, r, b))
    # 右栏：最右侧 44px，全高，放大 4 倍便于读文字
    rail = img.crop((w - 44, 40, w, ht - 10))
    rail = rail.resize((rail.width * 4, rail.height), Image.LANCZOS)
    rail.save(os.path.join(OUT, "_rail4x.png"))
    print("rail saved", rail.size)

    # 顶部工具条（左侧）放大 3 倍
    tb = img.crop((0, 30, 1000, 92))
    tb = tb.resize((tb.width * 2, tb.height * 3), Image.LANCZOS)
    tb.save(os.path.join(OUT, "_lefttoolbar3x.png"))
    print("toolbar saved", tb.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
