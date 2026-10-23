# -*- coding: utf-8 -*-
"""Capture PixCake's 图库 (library) module — the grid browsing view."""
import os
import subprocess
import sys
import time

from PIL import ImageGrab
import win32api
import win32con
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


def click(x, y):
    win32api.SetCursorPos((int(x), int(y)))
    time.sleep(0.2)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(0.08)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)


def main():
    h = hwnd()
    l, t, r, b = win32gui.GetWindowRect(h)
    w, ht = r - l, b - t
    win32gui.SetForegroundWindow(h)
    time.sleep(1.0)
    print("window %dx%d" % (w, ht))

    # 顶部三个 tab：图库 / 精修 / 创意，横排在窗口上部中央
    # 从截图量：图库 x≈1164, 精修 x≈1250, 创意 x≈1340, y≈17
    for name, x in (("图库", 1164), ("创意", 1340)):
        click(l + x, t + 17)
        time.sleep(2.5)
        img = ImageGrab.grab(bbox=(l, t, r, b))
        p = os.path.join(OUT, "mod_%s.png" % name)
        img.save(p)
        print("  %s -> %s" % (name, os.path.basename(p)))

    # 回到精修
    click(l + 1250, t + 17)
    time.sleep(1.5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
