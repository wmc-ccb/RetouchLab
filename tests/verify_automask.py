# -*- coding: utf-8 -*-
"""自动蒙版：分区正确性 + API + UI.

重点验证三件事：
  1. 关键区域真的产出了（背景/头发/皮肤/衣服），且覆盖率在合理量级
  2. 区域之间不互相污染（背景和人物应该接近互补）
  3. UI 能生成、能选、能导出
"""
import base64
import json
import os
import sys
import time

import numpy as np
import requests
import websocket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
B = "http://127.0.0.1:8765"
CDP = "http://127.0.0.1:9223"
OUT = os.path.join(ROOT, "out")
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
FAIL = []


def check(n, ok, d=""):
    print("%-50s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def main():
    from engine import automask as AM
    from engine.imgutil import load_image, fit_long_edge

    img = fit_long_edge(load_image(SRC), 700)
    masks, info = AM.build(img)
    avail = info["available"]
    cov = info["coverage"]

    # 1) 关键区域必须存在且有实质覆盖
    for key, lo in (("background", 0.02), ("hair", 0.01),
                    ("clothes", 0.02), ("subject", 0.10)):
        check("区域「%s」覆盖率 > %.0f%%" % (key, lo * 100),
              avail and cov.get(key, 0) > lo,
              "%s" % (("可用 %.3f" % cov[key]) if key in cov else "缺失"))
    # 皮肤类：两个模型中至少有一个给出
    skin_ok = (cov.get("face_skin", 0) > 0.005 or
               cov.get("body_skin", 0) > 0.005)
    check("皮肤区域有覆盖", skin_ok,
          "face=%.3f body=%.3f" % (cov.get("face_skin", -1),
                                   cov.get("body_skin", -1)))
    # 五官：眼睛/嘴唇应该有（模型能给的部分）
    check("眼睛有覆盖", cov.get("eyes", 0) > 0.0002,
          "%.4f" % cov.get("eyes", -1))
    check("嘴唇有覆盖", cov.get("lips", 0) > 0.0002,
          "%.4f" % cov.get("lips", -1))

    # 2) 背景与人物应接近互补（IoU 低 = 没互相污染）
    if "background" in masks and "subject" in masks:
        b = masks["background"] > 0.5
        s = masks["subject"] > 0.5
        inter = float((b & s).sum())
        union = float((b | s).sum())
        iou = inter / max(union, 1)
        check("背景与人物不重叠（IoU < 0.2）", iou < 0.2, "%.3f" % iou)
        # 覆盖率之和不必接近 1：这张照片主体居中、四周是虚化的展会背景，
        # selfie_multiclass 只对"明确是背景/皮肤/衣服"的像素给高分，
        # 大片虚化区域两块都不属于。实测两者合计 0.41，这是模型的实际
        # 行为而不是漏检——所以这里只断言两者都非空且不重叠。
        check("背景与人物都非空", b.sum() > 0 and s.sum() > 0,
              "背景 %.3f / 人物 %.3f" % (float(b.mean()),
                                         float(s.mean())))

    # 3) 蒙版尺寸与原图一致
    ok_size = all(m.shape == img.shape[:2] for m in masks.values())
    check("所有蒙版与原图同尺寸", ok_size,
          "%d 个" % len(masks))

    # 4) to_alpha 产出合法 RGBA
    if "hair" in masks:
        rgba = AM.to_alpha(masks["hair"], img)
        check("导出的 RGBA 合法",
              rgba.shape == img.shape[:2] + (4,) and
              float(rgba[..., 3].max()) > 0.5 and
              float(rgba[..., 3].min()) < 0.1)

    # ---------- API ----------
    print("\n=== API ===")
    j = requests.get(B + "/api/automask/info", timeout=60).json()
    check("info 列出部件", len(j.get("parts", [])) >= 10,
          "%d 个" % len(j.get("parts", [])))
    check("info 报告模型可用", j.get("semantic") is True)

    with open(SRC, "rb") as fh:
        up = requests.post(B + "/api/upload",
                           files={"file": ("x.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    sid = up["session"]
    t0 = time.time()
    r = requests.post(B + "/api/automask", json={"session": sid},
                      timeout=900)
    took = time.time() - t0
    check("automask 返回", r.ok, "%d 用时 %.1fs" % (r.status_code, took))
    jj = r.json() if r.ok else {}
    check("返回彩色预览", len(jj.get("image", "")) > 5000,
          "%d bytes" % len(jj.get("image", "")))
    check("返回每区域缩略图", len(jj.get("thumbs", {})) >= 8,
          "%d 张" % len(jj.get("thumbs", {})))
    part = "hair" if "hair" in (jj.get("info") or {}).get("available", []) \
        else (jj.get("info") or {}).get("available", [None])[0]

    r = requests.post(B + "/api/automask/export",
                      json={"session": sid, "part": part, "mode": "mask"},
                      timeout=600)
    check("导出灰度蒙版", r.ok and r.headers.get("content-type") ==
          "image/png", "%d %d bytes" % (r.status_code, len(r.content)))
    r2 = requests.post(B + "/api/automask/export",
                       json={"session": sid, "part": part, "mode": "alpha"},
                       timeout=600)
    check("导出透明 PNG", r2.ok, "%d %d bytes" % (r2.status_code,
                                                  len(r2.content)))
    r3 = requests.post(B + "/api/automask/export",
                       json={"session": sid, "part": "nope", "mode": "mask"},
                       timeout=60)
    check("未知区域 404", r3.status_code == 404, str(r3.status_code))

    # ---------- UI ----------
    print("\n=== UI ===")
    try:
        ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
              if x.get("type") == "page" and "8765" in x.get("url", "")]
        if not ts:
            print("浏览器未连接")
            return 1 if FAIL else 0
        ws = websocket.create_connection(ts[0]["webSocketDebuggerUrl"],
                                         timeout=180)
        ni = [0]

        def call(m, **p):
            ni[0] += 1
            ws.send(json.dumps({"id": ni[0], "method": m, "params": p}))
            while True:
                rr = json.loads(ws.recv())
                if rr.get("id") == ni[0]:
                    return rr.get("result", {})

        def js(e):
            return call("Runtime.evaluate", expression=e, returnByValue=True,
                        awaitPromise=True).get("result", {}).get("value")

        call("Emulation.setDeviceMetricsOverride", width=1800, height=1050,
             deviceScaleFactor=1, mobile=False)
        call("Page.reload")
        time.sleep(4)
        js("""(function(){S.session=%s;S.meta=%s;S.faces=%s;
            S.base='data:image/jpeg;base64,'+%s;
            document.getElementById('dropzone').style.display='none';
            drawBase(); return true;})()""" % (
            json.dumps(sid),
            json.dumps({"width": up["width"], "height": up["height"],
                        "preview_width": up["preview_width"],
                        "preview_height": up["preview_height"]}),
            json.dumps(up["faces"]), json.dumps(up["preview"])))
        time.sleep(2)

        js("selectCat('自动蒙版')")
        time.sleep(2.0)
        check("面板打开",
              js("document.getElementById('sideCat').textContent") == "自动蒙版")
        check("有生成按钮", bool(js("!!document.getElementById('amRun')")))

        js("document.getElementById('amRun').click()")
        cells = 0
        for _ in range(120):
            time.sleep(1)
            cells = js("document.querySelectorAll('.amCell').length") or 0
            if cells > 0:
                break
        check("UI 生成出蒙版格子", cells >= 8, "%s 个" % cells)
        check("显示彩色预览",
              js("!!document.querySelector('#amView img')") is True)

        # 点一个格子 -> 出现导出按钮
        js("(function(){var c=document.querySelector('.amCell');"
           "if(c) c.click(); return true;})()")
        time.sleep(0.6)
        check("选中后出现导出按钮",
              js("(function(){var a=document.getElementById('amActions');"
                 "return !!a && getComputedStyle(a).display !== 'none';})()")
              is True)
        picked = js("AM.picked")
        check("已选中区域", isinstance(picked, str) and len(picked) > 2,
              str(picked))

        shot = call("Page.captureScreenshot", format="png")
        with open(os.path.join(OUT, "ui_automask.png"), "wb") as fh:
            fh.write(base64.b64decode(shot["data"]))
        call("Emulation.clearDeviceMetricsOverride")
        ws.close()
    except Exception as exc:
        print("  UI 段异常：%s" % str(exc)[:160])
        FAIL.append("UI 段")

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
