# -*- coding: utf-8 -*-
"""Expand 影调 in PixCake and capture the slider ranges."""
import os
import sys
import time

from PIL import Image, ImageGrab
import win32api
import win32con
import win32gui

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "pixcake")
os.makedirs(OUT, exist_ok=True)


def click(x, y, wait=0.5):
    win32api.SetCursorPos((int(x), int(y)))
    time.sleep(0.25)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(0.08)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
    time.sleep(wait)


def main():
    procs = __import__("subprocess").check_output(
        ["powershell", "-NoProfile", "-Command",
         "(Get-Process pixcake -ErrorAction SilentlyContinue | "
         "Where-Object {$_.MainWindowHandle -ne 0} | "
         "Select-Object -First 1).MainWindowHandle"], text=True)
    h = None
    for tok in procs.split():
        if tok.strip().isdigit() and int(tok.strip()) > 0:
            h = int(tok.strip())
    if not h:
        print("no window")
        return 1
    win32gui.SetForegroundWindow(h)
    time.sleep(1.0)
    l, t, r, b = win32gui.GetWindowRect(h)

    # 展开影响面板里的「影调」：它在参数列 x≈2050，y≈846（前一张截图）
    click(l + 2050, t + 846, wait=1.2)
    img = ImageGrab.grab(bbox=(l + 1830, t + 40, r - 40, b - 60))
    img.save(os.path.join(OUT, "_yin_diao.png"))
    print("saved", img.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
