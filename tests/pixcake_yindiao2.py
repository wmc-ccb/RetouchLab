# -*- coding: utf-8 -*-
"""Open PixCake 后调色 → 影调 and capture slider ranges."""
import os
import subprocess
import sys
import time

from PIL import Image, ImageGrab
import win32api
import win32con
import win32gui

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "pixcake")
os.makedirs(OUT, exist_ok=True)

# 右栏 y（窗口内），从 _rail4x.png 读出
RAIL = {"后调色": 478}


def click(x, y, wait=0.6):
    win32api.SetCursorPos((int(x), int(y)))
    time.sleep(0.25)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(0.08)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
    time.sleep(wait)


def main():
    procs = subprocess.check_output(
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
    # SetForegroundWindow 会被前台锁定策略拒绝；TOPMOST 进出绕过它
    try:
        win32gui.SetWindowPos(h, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                              win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        time.sleep(0.4)
        win32gui.SetWindowPos(h, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                              win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    except Exception:
        pass
    time.sleep(0.8)
    l, t, r, b = win32gui.GetWindowRect(h)
    w = r - l

    # 1) 右栏点「后调色」
    click(l + w - 26, t + RAIL["后调色"], wait=1.5)

    # 2) 参数列里找「影调」折叠条并点开。
    #    从上一张全图截图量：影调 在参数列 x≈2050、y≈846（展开后加高，
    #    现在它被收起。保守做法：点它当前 y，然后全列截图确认）
    click(l + 2050, t + 846, wait=1.5)

    img = ImageGrab.grab(bbox=(l + 1830, t + 40, r - 40, b - 60))
    img.save(os.path.join(OUT, "_yindiao2.png"))
    print("saved", img.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
