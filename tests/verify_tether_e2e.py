# -*- coding: utf-8 -*-
"""联机拍摄端到端：模拟相机写文件 → 自动导入 → 套预设 → 自动导出.

不走真相机（Z6 III 的系统接口不开放快门），模拟它的行为：
传输软件把拍好的文件写进监听目录。这条链路与真机完全一致——
FolderSource 只看目录里出现了什么文件。
"""
import io
import json
import os
import shutil
import sys
import time

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
B = "http://127.0.0.1:8765"
FAIL = []


def check(n, ok, d=""):
    print("%-50s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def make_frame(path: str, i: int) -> None:
    """造一张像样的"相机原片"：渐变背景 + 肤色椭圆主体."""
    import cv2

    h, w = 480, 720
    img = np.zeros((h, w, 3), np.float32)
    gx = np.linspace(0.9, 0.75, w, dtype=np.float32)[None, :, None]
    img[:] = np.array([0.60, 0.64, 0.70], np.float32) * gx
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((yy - 240) ** 2 + (xx - 380) ** 2)
    subj = np.clip((120 - r) / 5.0, 0, 1)
    skin = np.array([0.82, 0.66, 0.58], np.float32)
    img = img * (1 - subj[..., None]) + skin * subj[..., None]
    # 每张亮度不同，模拟不同曝光
    img = np.clip(img * (0.9 + 0.08 * i), 0, 1)
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(u8, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])


def main():
    watch = os.path.join(ROOT, "out", "_tether_watch")
    outdir = os.path.join(ROOT, "out", "_tether_out")
    for d in (watch, outdir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    from engine.paths import data_dir as _data_dir  # 状态在 data/ 不在 server/ (paths 迁移)
    clean_st = os.path.join(_data_dir(), "_user_presets.json")
    # 用已有内置预设 natural，避免依赖用户预设

    # 1) 配置并启动监听（启动时目录为空 -> baseline 为空）
    r = requests.post(B + "/api/tether/config", json={
        "kind": "folder", "path": watch, "out_dir": outdir,
        "auto_preset": "natural", "auto_export_dir": outdir,
        "settle_s": 0.6, "auto_import": True, "commit": True}, timeout=60)
    check("启动监听", r.ok, "%s %s" % (r.status_code, r.text[:90]))
    primed = r.json().get("primed", 0) if r.ok else None
    check("空目录 baseline=0", primed == 0, "primed=%s" % primed)

    # 2) 模拟相机连拍 3 张（间隔写入，settle 机制应各收各的）
    names = []
    for i in range(3):
        name = "_DSC%04d.jpg" % (1000 + i)
        make_frame(os.path.join(watch, name), i)
        names.append(name)
        time.sleep(0.9)

    # 3) poll（联机 UI 每 1.5s 轮询一次；这里显式调）
    got = []
    for _ in range(30):
        time.sleep(1)
        r = requests.post(B + "/api/tether/poll", json={}, timeout=300)
        j = r.json() if r.ok else {}
        new = j.get("new") or []
        for e in new:
            got.append(e)
        if len(got) >= 3:
            break
    check("3 张全部导入", len(got) == 3, "%d 张" % len(got))
    check("导入带会话", all(e.get("session") for e in got),
          str([bool(e.get("session")) for e in got]))
    check("导入应用预设", all("natural" in str(e.get("applied", ""))
                             or e.get("preset") for e in got),
          str([str(e.get("applied", e.get("preset")))[:30] for e in got][:2]))

    # 4) 自动导出：outdir 应有处理后的文件
    time.sleep(2)
    produced = [f for f in os.listdir(outdir)
                if f.lower().endswith((".jpg", ".jpeg"))]
    check("自动导出 3 张", len(produced) == 3,
          "%d 张 %s" % (len(produced), produced[:3]))

    # 5) 中途再写 1 张（联机不中断）
    make_frame(os.path.join(watch, "_DSC2000.jpg"), 4)
    got2 = []
    for _ in range(20):
        time.sleep(1)
        r = requests.post(B + "/api/tether/poll", json={}, timeout=300)
        j = r.json() if r.ok else {}
        got2.extend(j.get("new") or [])
        if got2:
            break
    check("追加 1 张继续导入", len(got2) == 1, "%d 张" % len(got2))

    # 6) 状态端点如实反映
    st = requests.get(B + "/api/tether/status", timeout=60).json()
    check("状态 enabled", st.get("enabled") is True)
    inbox = st.get("inbox")
    n_inbox = len(inbox) if isinstance(inbox, list) else (inbox or 0)
    check("inbox 有导入记录", n_inbox >= 1, "inbox=%s" % str(inbox)[:70])

    # 7) 停止
    r = requests.post(B + "/api/tether/stop", timeout=30)
    check("停止监听", r.ok)
    st = requests.get(B + "/api/tether/status", timeout=60).json()
    check("停止后 enabled=false", st.get("enabled") is False)

    # 清理
    shutil.rmtree(watch, ignore_errors=True)
    shutil.rmtree(outdir, ignore_errors=True)

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
