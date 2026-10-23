# -*- coding: utf-8 -*-
"""Open Photoshop's Select menu and capture it."""
import os
import sys
import time

from PIL import Image, ImageGrab
import win32api
import win32con
import win32gui

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "ps")
os.makedirs(OUT, exist_ok=True)


def click(x, y, wait=0.35):
    win32api.SetCursorPos((int(x), int(y)))
    time.sleep(0.2)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(0.08)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
    time.sleep(wait)


def main():
    h = None
    for p in range(3):
        procs = __import__("subprocess").check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Process Photoshop -ErrorAction SilentlyContinue | "
             "Where-Object {$_.MainWindowHandle -ne 0} | "
             "Select-Object -First 1).MainWindowHandle"], text=True)
        vals = [int(t.strip()) for t in procs.split()
                if t.strip().isdigit() and int(t.strip()) > 0]
        if vals:
            h = vals[0]
            break
        time.sleep(2)
    if not h:
        print("no PS window")
        return 1
    win32gui.SetForegroundWindow(h)
    time.sleep(1.2)
    l, t, r, b = win32gui.GetWindowRect(h)
    print("window", l, t, r, b)

    shot = ImageGrab.grab(bbox=(l, t, r, b))
    shot.save(os.path.join(OUT, "_before.png"))

    # 菜单栏实测（窗口内坐标）：文件 编辑 图像 图层 文字 选择 滤镜…
    # 选择 在 x≈313（第一次点到 x=250 落在了"文字"）
    click(l + 313, t + 17, wait=0.9)
    menu = ImageGrab.grab(bbox=(l, t, min(r, l + 900), min(b, t + 800)))
    menu.save(os.path.join(OUT, "_select_menu.png"))
    print("saved _select_menu.png", menu.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
