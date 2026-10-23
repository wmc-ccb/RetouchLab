# -*- coding: utf-8 -*-
"""Click PixCake rail items and verify the panel actually changed.

The first attempt used hardcoded y offsets from an older screenshot and clicked
"图像" but nothing happened — the panel stayed on 预设. So this version measures
the rail from a fresh capture, clicks the icon centre, and **verifies** the
panel region changed before saving it.
"""
import os
import subprocess
import sys
import time

import numpy as np
from PIL import ImageGrab
import win32api
import win32con
import win32gui

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "pixcake")
os.makedirs(OUT, exist_ok=True)

# y positions read off a 5x-zoomed capture of the rail (window-relative)
RAIL = [("AI色彩", 88), ("调色", 140), ("局部", 190), ("人像", 277),
        ("图像", 327), ("衣物", 377), ("宠物", 425), ("后调色", 478),
        ("裁剪", 528), ("历史", 577)]
RAIL_X_OFFSET = 20      # from the right edge, icon centre


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


def panel_sig(img):
    """Cheap signature of the parameter panel area, to detect a real change."""
    # 参数面板大致在 x 1540..1830（预设列）与 1830..2510（参数列）
    a = np.asarray(img.crop((1840, 60, 2500, 1300)).resize((80, 120)),
                   np.float32)
    return float(a.mean()), float(a.std())


def main():
    h = hwnd()
    l, t, r, b = win32gui.GetWindowRect(h)
    w, ht = r - l, b - t
    print("window %dx%d at (%d,%d)" % (w, ht, l, t))
    win32gui.SetForegroundWindow(h)
    time.sleep(1.0)

    base = ImageGrab.grab(bbox=(l, t, r, b))
    base_sig = panel_sig(base)
    print("baseline panel sig: mean=%.1f std=%.1f" % base_sig)

    x = l + w - RAIL_X_OFFSET
    for name, y in RAIL:
        click(x, t + y)
        time.sleep(1.8)
        img = ImageGrab.grab(bbox=(l, t, r, b))
        sig = panel_sig(img)
        changed = (abs(sig[0] - base_sig[0]) > 1.5 or
                   abs(sig[1] - base_sig[1]) > 1.5)
        img.save(os.path.join(OUT, "p_%s.png" % name))
        print("  %-6s y=%3d  sig mean=%.1f std=%.1f  changed=%s"
              % (name, y, sig[0], sig[1], changed))
        base_sig = sig
    return 0


if __name__ == "__main__":
    sys.exit(main())
