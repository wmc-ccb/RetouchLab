# -*- coding: utf-8 -*-
"""Mask panel: semantic sub-masks, locality, and the histogram band."""
import base64
import io
import json
import os
import sys
import time

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BASE = "http://127.0.0.1:8765"
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
OUT = os.path.join(ROOT, "out")
failures = []


def check(label, cond, detail=""):
    print("%-52s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


def dec(b64):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64)))
                      .convert("RGB"), np.float32) / 255.0


def main():
    from engine.imgutil import load_image, fit_long_edge

    # ---- engine level ----------------------------------------------------
    print("=== 引擎：蒙版内调整 ===")
    from engine import local as L
    from engine import semantic as SEM

    img = fit_long_edge(load_image(SRC), 500)
    h, w = img.shape[:2]
    reg = SEM.segment(img)
    check("语义分割可用", reg is not None)

    m = reg.norm("face-skin")
    out, touched = L.apply_through_mask(img, m, {"exposure": 40}, feather=1.5,
                                        return_mask=True)
    d = np.abs(out - img).max(axis=2)
    # "outside" is the **effective** mask after feathering, which is what the
    # call is allowed to write. Using `m == 0` counted the feather ring as a
    # leak: the feather legitimately reaches past every pixel where the input
    # mask is non-zero, so that comparison could never hold.
    outside = ~touched
    check("生效掩膜之外逐位不变", float(d[outside].max()) == 0.0,
          "max %.2e（%.1f%% 像素）"
          % (float(d[outside].max()), 100 * outside.mean()))
    check("生效区比原掩膜大（羽化）",
          int(touched.sum()) > int((m > 0).sum()),
          "%d -> %d" % (int((m > 0).sum()), int(touched.sum())))
    check("掩膜内确实改变", float(d[touched].max()) > 0.03,
          "max %.4f" % float(d[touched].max()))
    check("空控制项 = 零操作",
          float(np.abs(L.apply_through_mask(img, m, {}) - img).max()) == 0.0)
    check("空蒙版 = 零操作",
          float(np.abs(L.apply_through_mask(
              img, np.zeros((h, w), np.float32), {"exposure": 40}) - img)
              .max()) == 0.0)
    inv, tinv = L.apply_through_mask(img, m, {"exposure": 40}, invert=True,
                                     return_mask=True)
    din = np.abs(inv - img).max(axis=2)
    # Sample the mask's **deep interior**, eroded away from the boundary. With
    # invert the mask value there is (1 - ~1) = ~0, but the feather blends
    # across the edge, so a core taken right up to the boundary legitimately
    # picks up 5% of the adjustment and read as a 0.145 leak.
    import cv2 as _cv2
    core = _cv2.erode((m > 0.95).astype(np.uint8),
                      np.ones((15, 15), np.uint8)).astype(bool)
    if core.any():
        check("反转时原掩膜深内部基本不动",
              float(din[core].max()) < 0.02,
              "core max %.4f（%d px）" % (float(din[core].max()),
                                         int(core.sum())))
    else:
        print("%-52s %s" % ("反转时原掩膜深内部基本不动", "跳过（核心区太小）"))
    check("反转时掩膜外确实改变",
          float(din[m < 0.05].max()) > 0.01,
          "max %.4f" % float(din[m < 0.05].max()))

    # ---- API -------------------------------------------------------------
    try:
        requests.get(BASE + "/api/health", timeout=60)
    except Exception:
        print("\n服务未运行，跳过 API")
        return 0 if not failures else 1

    print("\n=== API ===")
    with open(SRC, "rb") as fh:
        up = requests.post(BASE + "/api/upload",
                           files={"file": ("mk.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    sid = up["session"]
    base = dec(up["preview"])

    cat = requests.get(BASE + "/api/masks/" + sid, timeout=300).json()
    check("蒙版目录返回类别", len(cat["classes"]) == 6, str(cat["classes"]))
    check("目录带每类覆盖", bool(cat["coverage"]),
          str(cat["coverage"])[:70])
    check("背景覆盖最大", cat["coverage"]["background"] ==
          max(cat["coverage"].values()), "%.3f" % cat["coverage"]["background"])

    hist = requests.get(BASE + "/api/histogram/" + sid, timeout=60).json()
    check("直方图含 RGB 与亮度", all(k in hist for k in
                                    ("r", "g", "b", "lum", "bins")))
    check("直方图桶数一致",
          len(hist["r"]) == len(hist["g"]) == len(hist["b"]) == hist["bins"])
    check("直方图总和等于像素数",
          sum(hist["lum"]) == hist["total"],
          "%d vs %d" % (sum(hist["lum"]), hist["total"]))
    check("报告死黑/死白", "clipped_low" in hist and "clipped_high" in hist,
          "低 %d / 高 %d" % (hist["clipped_low"], hist["clipped_high"]))

    for keys, ctl in ((['face-skin'], {'exposure': 30}),
                      (['background'], {'saturation': -60}),
                      (['subject', 'hair'], {'contrast': 25}),
                      (['skin'], {'temp': 20})):
        r = requests.post(BASE + "/api/masked", json={
            "session": sid, "masks": keys, "controls": ctl,
            "commit": False}, timeout=300)
        if r.status_code != 200:
            check("蒙版 %s" % "+".join(keys), False, r.text[:80])
            continue
        j = r.json()
        out = dec(j["image"])
        dd = np.abs(out - base).mean(axis=2)
        check("蒙版 %s 生效" % "+".join(keys), dd.mean() > 0.001,
              "覆盖 %.3f 平均变化 %.4f" % (j["info"]["coverage"], dd.mean()))
        # a background-only adjustment must not move the subject's face
        if keys == ['background']:
            from engine import semantic as S2
            reg2 = S2.segment(base)
            face = reg2["face-skin"] > 0.5
            if face.any():
                check("只改背景时脸部基本不动",
                      float(dd[face].mean()) < 0.01,
                      "脸部平均变化 %.4f" % float(dd[face].mean()))

    bad = requests.post(BASE + "/api/masked",
                        json={"session": sid, "masks": []}, timeout=60)
    check("空蒙版列表返回 400", bad.status_code == 400, str(bad.status_code))

    # ---- browser ---------------------------------------------------------
    print("\n=== UI ===")
    try:
        import websocket
        t = [x for x in requests.get("http://127.0.0.1:9223/json", timeout=10)
             .json() if x.get("type") == "page" and "8765" in x.get("url", "")]
        if not t:
            print("  浏览器未连接，跳过")
        else:
            ws = websocket.create_connection(t[0]["webSocketDebuggerUrl"],
                                             timeout=60)
            n = [0]

            def call(m, **p):
                n[0] += 1
                ws.send(json.dumps({"id": n[0], "method": m, "params": p}))
                while True:
                    msg = json.loads(ws.recv())
                    if msg.get("id") == n[0]:
                        return msg.get("result", {})

            def js(e):
                return call("Runtime.evaluate", expression=e,
                            returnByValue=True, awaitPromise=True
                            ).get("result", {}).get("value")

            call("Page.reload")
            time.sleep(4)
            js("""
              (function(){
                S.session = %s; S.meta = %s; S.faces = %s;
                S.base = 'data:image/jpeg;base64,' + %s;
                document.getElementById('dropzone').style.display='none';
                drawBase(); return true; })()
            """ % (json.dumps(sid),
                   json.dumps({"width": up["width"], "height": up["height"],
                               "preview_width": up["preview_width"],
                               "preview_height": up["preview_height"]}),
                   json.dumps(up["faces"]), json.dumps(up["preview"])))
            time.sleep(1.5)
            js("selectCat('\u8499\u7248')")
            time.sleep(2.0)
            check("蒙版面板可打开",
                  js("document.getElementById('sideCat').textContent") == "蒙版")
            check("直方图常驻在参数区顶部",
                  bool(js("!!document.getElementById('globalHist')")))
            nchips = js("document.querySelectorAll('.maskChip').length") or 0
            check("蒙版 chips 已渲染", nchips >= 6, "%d 个" % nchips)
            check("chips 带覆盖百分比",
                  bool(js("(document.querySelector('.maskChip')||{}).textContent"
                          "&& document.querySelector('.maskChip').textContent"
                          ".indexOf('%')>0")))
            check("羽化滑杆存在", bool(js("!!document.getElementById('mkF')")))
            check("调整项已渲染",
                  (js("document.querySelectorAll('[id^=mkr_]').length") or 0) >= 7,
                  str(js("document.querySelectorAll('[id^=mkr_]').length")))
            # selecting a mask must enable the hint and send a preview
            js("document.querySelector('.maskChip').click()")
            time.sleep(1.0)
            check("点击后进入已选状态",
                  (js("MASK.selected.length") or 0) == 1,
                  str(js("MASK.selected")))
            check("提示行反映作用区",
                  "作用区" in (js("document.getElementById('maskHint')"
                                  ".textContent") or ""))
            n[0] += 1
            ws.send(json.dumps({"id": n[0], "method": "Page.captureScreenshot",
                                "params": {"format": "png"}}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == n[0]:
                    with open(os.path.join(OUT, "ui_mask.png"), "wb") as fh:
                        fh.write(base64.b64decode(msg["result"]["data"]))
                    break
            ws.close()
    except Exception as exc:
        print("  UI 检查跳过：%s" % str(exc)[:80])

    print()
    if failures:
        print("FAILED: %d -> %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
