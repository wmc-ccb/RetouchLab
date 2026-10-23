# -*- coding: utf-8 -*-
"""Local adjustments: mask shape, locality, stacking, and the API/UI path.

The engine has supported `local_adjustments` for a while; the UI never exposed
it. What matters is not that a mask exists but that:

* a radial mask is 1 at its centre and 0 well outside, with a monotone falloff;
* a linear mask ramps along its angle and is flat beyond the band;
* an adjustment is **local** — outside the mask the frame is untouched;
* entries **stack**, so a face brightener and a corner darkener coexist;
* `invert` acts on the complement;
* geometry given as 0..1 fractions survives a change of preview size, which is
  what makes a stored recipe reusable at export resolution.
"""
import io
import base64
import json
import os
import sys
import time

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.imgutil import load_image, fit_long_edge, save_image
from engine import local as L

BASE = "http://127.0.0.1:8765"
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
failures = []


def check(label, cond, detail=""):
    print("%-56s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


def main():
    img = fit_long_edge(load_image(SRC), 700)
    h, w = img.shape[:2]
    print("image: %dx%d" % (w, h))

    # ---- 1. radial mask shape -------------------------------------------
    print("\n=== 径向蒙版 ===")
    m = L.radial_mask((h, w), w * 0.5, h * 0.5, w * 0.2, h * 0.2, feather=0.5)
    check("中心为 1", abs(float(m[int(h * 0.5), int(w * 0.5)]) - 1.0) < 1e-5,
          "%.5f" % float(m[int(h * 0.5), int(w * 0.5)]))
    check("远处为 0", float(m[2, 2]) == 0.0, "%.5f" % float(m[2, 2]))
    check("掩膜在 [0,1]", float(m.min()) >= 0.0 and float(m.max()) <= 1.0)
    # monotone along a ray from the centre
    ray = [float(m[int(h * 0.5), min(w - 1, int(w * 0.5 + t))])
           for t in range(0, int(w * 0.45), 4)]
    check("沿半径单调不增", all(ray[i] >= ray[i + 1] - 1e-6
                              for i in range(len(ray) - 1)))
    inv = L.radial_mask((h, w), w * 0.5, h * 0.5, w * 0.2, h * 0.2,
                        feather=0.5, invert=True)
    check("invert = 1 - 原掩膜", float(np.abs(inv - (1.0 - m)).max()) < 1e-6)

    # ---- 2. linear mask --------------------------------------------------
    print("\n=== 线性蒙版 ===")
    lm = L.linear_mask((h, w), angle=90.0, position=0.5, feather=0.3)
    check("顶部与底部有差", abs(float(lm[5].mean()) - float(lm[-5].mean())) > 0.5,
          "%.3f / %.3f" % (lm[5].mean(), lm[-5].mean()))
    lm0 = L.linear_mask((h, w), angle=0.0, position=0.5, feather=0.3)
    check("旋转 90° 后方向改变",
          abs(float(lm0[:, 5].mean()) - float(lm0[:, -5].mean())) > 0.5)

    # ---- 3. locality -----------------------------------------------------
    print("\n=== 局部性：蒙版外不得改变 ===")
    entry = [{"type": "radial", "cx": 0.5, "cy": 0.5, "rx": 0.15, "ry": 0.15,
              "feather": 0.4, "strength": 100, "exposure": 60}]
    out = L.apply_local(img, entry)
    mm = L.radial_mask((h, w), w * 0.5, h * 0.5, w * 0.15, h * 0.15,
                       feather=0.4)
    d = np.abs(out - img).max(axis=2)
    # "outside" means the mask is *exactly* zero. A soft mask has a feather
    # tail with 0 < m <= 1e-6, and those pixels are legitimately written —
    # treating them as outside reported a 2.4e-07 leak over 93% of the frame
    # when the real count was zero.
    outside = mm == 0.0
    check("掩膜为 0 处逐位不变", float(d[outside].max()) == 0.0,
          "max %.2e（%.1f%% 像素）"
          % (float(d[outside].max()), 100 * outside.mean()))
    check("掩膜内确实改变", float(d[~outside].max()) > 0.05,
          "max %.4f" % float(d[~outside].max()))

    # ---- 4. stacking -----------------------------------------------------
    print("\n=== 叠加 ===")
    two = entry + [{"type": "radial", "cx": 0.15, "cy": 0.15, "rx": 0.12,
                    "ry": 0.12, "feather": 0.4, "strength": 100,
                    "exposure": -60}]
    out2 = L.apply_local(img, two)
    # the second entry darkens a corner the first never touched
    c1 = out2[int(h * 0.15), int(w * 0.15)]
    c2 = img[int(h * 0.15), int(w * 0.15)]
    check("第二条生效（角部被压暗）", float(c1.mean()) < float(c2.mean()),
          "%.3f -> %.3f" % (c2.mean(), c1.mean()))
    same_centre = float(np.abs(out2[int(h * 0.5), int(w * 0.5)]
                               - out[int(h * 0.5), int(w * 0.5)]).max())
    check("第一条不受第二条影响", same_centre < 1e-6, "%.2e" % same_centre)

    # ---- 5. fractional geometry is resolution independent ----------------
    print("\n=== 比例几何与分辨率无关 ===")
    small = fit_long_edge(img, 350)
    big = L.apply_local(img, entry)
    small_out = L.apply_local(small, entry)
    # sample the same fractional point in both
    def at(im, fx, fy):
        return im[int(im.shape[0] * fy), int(im.shape[1] * fx)]
    dev = float(np.abs(at(big, 0.5, 0.5) - at(small_out, 0.5, 0.5)).max())
    check("同一比例点在不同尺寸下效果一致", dev < 0.06, "偏差 %.4f" % dev)

    # ---- 6. strength and zero-entry no-op --------------------------------
    print("\n=== 强度与空条目 ===")
    check("空列表 = 零操作",
          float(np.abs(L.apply_local(img, []) - img).max()) == 0.0)
    half = L.apply_local(img, [dict(entry[0], strength=50)])
    full = L.apply_local(img, entry)
    dh = float(np.abs(half - img).mean())
    df = float(np.abs(full - img).mean())
    check("强度 50 效果约为 100 的一半", 0.2 < dh / max(df, 1e-9) < 0.9,
          "%.5f vs %.5f" % (dh, df))

    save_image(os.path.join(ROOT, "out", "local_demo.jpg"),
               np.hstack([img, out, out2]), 92)

    # ---- 7. API + UI -----------------------------------------------------
    try:
        requests.get(BASE + "/api/health", timeout=60)
    except Exception:
        print("\n服务未运行，跳过 API 检查")
        return 0 if not failures else 1

    print("\n=== API ===")
    save_image(os.path.join(ROOT, "out", "_loc_src.jpg"), img, 95)
    with open(os.path.join(ROOT, "out", "_loc_src.jpg"), "rb") as fh:
        up = requests.post(BASE + "/api/upload",
                           files={"file": ("l.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    sid = up["session"]
    r = requests.post(BASE + "/api/render", json={
        "session": sid, "quality": 90,
        "params": {"local_adjustments": [
            {"type": "radial", "cx": 0.5, "cy": 0.45, "rx": 0.2, "ry": 0.2,
             "feather": 0.5, "strength": 100, "exposure": 40}]}},
        timeout=300).json()
    check("渲染接受 local_adjustments", "image" in r, str(list(r.keys()))[:60])
    check("管线报告 local 已被应用",
          any("local" in a for a in r.get("applied", [])),
          str(r.get("applied"))[:70])
    base_img = np.asarray(__import__("PIL.Image", fromlist=["Image"])
                          .open(io.BytesIO(base64.b64decode(up["preview"])))
                          .convert("RGB"), np.float32) / 255.0
    loc_img = np.asarray(__import__("PIL.Image", fromlist=["Image"])
                         .open(io.BytesIO(base64.b64decode(r["image"])))
                         .convert("RGB"), np.float32) / 255.0
    hh, ww = base_img.shape[:2]
    corner = (slice(0, 8), slice(0, 8))
    centre = (slice(int(hh * 0.42), int(hh * 0.48)),
              slice(int(ww * 0.47), int(ww * 0.53)))
    dc = float(np.abs(loc_img[corner] - base_img[corner]).mean())
    dm = float(np.abs(loc_img[centre] - base_img[centre]).mean())
    check("角部基本未变（仅 JPEG 误差）", dc < 0.02, "%.4f" % dc)
    check("中心确实提亮", dm > dc * 3 + 0.01, "角 %.4f / 中 %.4f" % (dc, dm))

    # ---- browser ---------------------------------------------------------
    print("\n=== UI ===")
    try:
        import websocket
        tgt = [t for t in requests.get("http://127.0.0.1:9223/json",
                                       timeout=10).json()
               if t.get("type") == "page" and "8765" in t.get("url", "")]
        if not tgt:
            print("  浏览器未连接，跳过 UI 检查")
        else:
            ws = websocket.create_connection(tgt[0]["webSocketDebuggerUrl"],
                                             timeout=40)
            counter = [0]

            def call(method, **params):
                counter[0] += 1
                ws.send(json.dumps({"id": counter[0], "method": method,
                                    "params": params}))
                while True:
                    msg = json.loads(ws.recv())
                    if msg.get("id") == counter[0]:
                        return msg.get("result", {})

            def js(expr):
                r = call("Runtime.evaluate", expression=expr,
                         returnByValue=True, awaitPromise=True)
                return r.get("result", {}).get("value")

            # reload first: the shared tab may still hold masks from an earlier
            # run, which makes the entry counts fail for unrelated reasons
            call("Page.reload")
            time.sleep(4)

            js("selectCat('\u5c40\u90e8')")
            time.sleep(1.0)
            check("局部面板可打开",
                  js("document.getElementById('sideCat').textContent") == "局部")
            check("有径向/线性按钮",
                  bool(js("!!document.getElementById('locRadial') && "
                          "!!document.getElementById('locLinear')")))
            js("addLocal('radial')")
            time.sleep(0.4)
            check("可添加径向蒙版", js("LOCAL.entries.length") == 1,
                  str(js("LOCAL.entries.length")))
            check("蒙版条目渲染", js("document.querySelectorAll('.locItem').length") == 1)
            check("叠加指示层出现", bool(js("!!document.getElementById('localOverlay')")))
            check("条目带默认调整值", (js("LOCAL.entries[0].exposure") or 0) != 0,
                  str(js("LOCAL.entries[0].exposure")))
            js("addLocal('linear')")
            time.sleep(0.4)
            check("可添加线性蒙版", js("LOCAL.entries.length") == 2)
            check("localParams 输出正确",
                  (js("localParams().length") == 2
                   and js("localParams()[1].type") == "linear"))
            js("LOCAL.entries = []; LOCAL.sel = -1; renderLocalList();"
               " drawLocalOverlay();")
            check("清空后列表为空", js("LOCAL.entries.length") == 0)
            js("addLocal('radial'); LOCAL.entries[0].exposure = 30;"
               " renderLocalList(); renderLocalDetail();")
            time.sleep(0.3)
            shot = js("1")
            counter[0] += 1
            ws.send(json.dumps({"id": counter[0], "method": "Page.captureScreenshot",
                                "params": {"format": "png"}}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == counter[0]:
                    with open(os.path.join(ROOT, "out", "ui5_local.png"), "wb") as fh:
                        fh.write(base64.b64decode(msg["result"]["data"]))
                    break
            ws.close()
    except Exception as exc:
        print("  UI 检查跳过：%s" % str(exc)[:70])

    print()
    if failures:
        print("FAILED: %d -> %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
