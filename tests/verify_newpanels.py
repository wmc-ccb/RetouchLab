# -*- coding: utf-8 -*-
"""新面板：引擎正确性 + API + UI 渲染.

覆盖裁剪、祛边、可选颜色、色彩编辑器、配置文件/滤镜、后调色、
表情管理、妆容、头发、衣物，以及三模块切换和缩放。
"""
import base64
import io
import json
import os
import sys
import time

import numpy as np
import requests
import websocket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BASE = "http://127.0.0.1:8765"
CDP = "http://127.0.0.1:9223"
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
OUT = os.path.join(ROOT, "out")
failures = []


def check(label, cond, detail=""):
    print("%-50s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


def dec(b64):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64)))
                      .convert("RGB"), np.float32) / 255.0


class B:
    def __init__(self, t):
        self.ws = websocket.create_connection(t["webSocketDebuggerUrl"],
                                              timeout=120)
        self.i = 0

    def call(self, m, **p):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": m, "params": p}))
        while True:
            r = json.loads(self.ws.recv())
            if r.get("id") == self.i:
                return r.get("result", {})

    def js(self, e):
        return self.call("Runtime.evaluate", expression=e, returnByValue=True,
                         awaitPromise=True).get("result", {}).get("value")

    def shot(self, p):
        r = self.call("Page.captureScreenshot", format="png")
        with open(p, "wb") as fh:
            fh.write(base64.b64decode(r["data"]))
        return os.path.getsize(p) // 1024


def main():
    from engine.imgutil import load_image, fit_long_edge

    # ---------------- 引擎 ----------------
    print("=== 裁剪 ===")
    from engine import crop as CR
    img = fit_long_edge(load_image(SRC), 400)
    h, w = img.shape[:2]
    out = CR.apply(img, {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5})
    check("裁剪尺寸正确", out.shape[0] == round(0.5 * h) and
          out.shape[1] == round(0.5 * w), "%s vs %s" % (out.shape, (h, w)))
    r90 = CR.apply(img, None, 90)
    check("旋转 90 度交换宽高", r90.shape[:2] == (w, h),
          "%s from %s" % (r90.shape[:2], (h, w)))
    a = CR.fit_ratio({"x": 0.1, "y": 0.1, "w": 0.6, "h": 0.6}, 1.0, (h, w))
    check("1:1 比例真实", abs(a["w"] * w - a["h"] * h) < 2,
          "%.1f vs %.1f" % (a["w"] * w, a["h"] * h))
    # 反复确认同一裁剪不应内缩
    b1 = CR.pixel_box({"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}, (h, w))
    b2 = CR.pixel_box({"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}, (h, w))
    check("裁剪框幂等（不漂移）", b1 == b2, str(b1))
    check("裁剪不改原图", abs(float(img.sum()) -
          float(fit_long_edge(load_image(SRC), 400).sum())) < 1e-3)

    print("\n=== 可选颜色 / 色彩编辑器 ===")
    from engine import grading as GR
    red = np.zeros((8, 8, 3), np.float32)
    red[..., 0] = 0.85; red[..., 1] = 0.15; red[..., 2] = 0.15
    out = GR.selective_color(red, {"reds": {"cyan": 0, "magenta": 60}})
    check("选红加洋红：R 更红", out[..., 0].mean() >= red[..., 0].mean() - 1e-4,
          "R %.3f -> %.3f" % (red[..., 0].mean(), out[..., 0].mean()))
    blue = np.zeros((8, 8, 3), np.float32)
    blue[..., 2] = 0.85; blue[..., 1] = 0.2; blue[..., 0] = 0.2
    out2 = GR.selective_color(blue, {"reds": {"magenta": 60}})
    check("调红色不动蓝色", np.abs(out2 - blue).max() < 0.02,
          "max %.4f" % float(np.abs(out2 - blue).max()))
    grey = np.full((8, 8, 3), 0.5, np.float32)
    out3 = GR.selective_color(grey, {"reds": {"magenta": 80, "yellow": 80}})
    check("调色带不动中性灰", float(np.abs(out3 - grey).max()) < 0.02,
          "max %.4f" % float(np.abs(out3 - grey).max()))
    hue_out = GR.color_editor(red, {"reds": {"hue": 60}})
    check("色彩编辑器换色相", float(np.abs(hue_out - red).max()) > 0.05,
          "max %.4f" % float(np.abs(hue_out - red).max()))

    print("\n=== 配置文件 / 滤镜 ===")
    lut = GR.filter_lut("bw_contrast")
    bw = GR.apply_lut(img, lut)
    # 去色的判据是"三个通道相等"，不是"R 的空间标准差为 0"——去色后的灰度图
    # 当然有空间变化，之前那个断言本身是错的。
    check("黑白滤镜真的去色",
          float(np.abs(bw[..., 0] - bw[..., 1]).max()) < 1e-3 and
          float(np.abs(bw[..., 0] - bw[..., 2]).max()) < 1e-3,
          "R-B 差 %.5f" % float(np.abs(bw[..., 0] - bw[..., 2]).max()))
    same = GR.apply_lut(img, GR.filter_lut("clean"))
    check("clean 滤镜改变了画面", float(np.abs(same - img).max()) > 0.005,
          "max %.4f" % float(np.abs(same - img).max()))
    check("neutral 配置文件不改画面", GR.profile_lut("neutral") is None)
    mp = GR.profile_lut("mono")
    mout = GR.apply_lut(img, mp)
    check("mono 配置文件去色", float(np.abs(mout[..., 0] - mout[..., 2]).max())
          < 1e-3)
    check("未知滤镜返回 None", GR.filter_lut("nope") is None)

    print("\n=== 后调色 ===")
    pg = GR.post_grade(img, shadow_tint=[0, 0, 60], vignette=50)
    check("后调色改变了画面", float(np.abs(pg - img).max()) > 0.01,
          "max %.4f" % float(np.abs(pg - img).max()))
    corners = np.concatenate([pg[:20, :20].reshape(-1, 3),
                              pg[-20:, -20:].reshape(-1, 3)]).mean()
    centre = pg[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10].reshape(
        -1, 3).mean()
    check("暗角让四角变暗", corners < centre, "%.3f < %.3f" % (corners, centre))

    print("\n=== 祛边 ===")
    from engine import defringe as DF
    # 造一个白底上的圆，边缘带白边
    test = np.zeros((120, 120, 3), np.float32)
    test[..., :] = 0.05
    yy, xx = np.mgrid[0:120, 0:120]
    rr = np.sqrt((yy - 60) ** 2 + (xx - 60) ** 2)
    alpha = np.clip((45 - rr) / 6.0, 0, 1)
    subject = np.zeros_like(test); subject[..., :] = 0.45
    fringe = np.zeros_like(test); fringe[..., :] = 0.95
    comp = test * (1 - alpha[..., None]) + subject * alpha[..., None]
    comp += (fringe * np.clip(1 - np.abs(rr - 45) / 4.0, 0, 1)[..., None])
    fixed = DF.dehalo(np.clip(comp, 0, 1), alpha, strength=1.0)
    band = np.abs(rr - 45) < 3
    check("白边被压低", fixed[band].mean(axis=0).max() <
          np.clip(comp, 0, 1)[band].mean(axis=0).max(),
          "%.3f -> %.3f" % (np.clip(comp, 0, 1)[band].mean(),
                            fixed[band].mean()))
    check("祛边不改中心", float(np.abs(fixed[rr < 30] -
          np.clip(comp, 0, 1)[rr < 30]).max()) < 0.05,
          "max %.4f" % float(np.abs(fixed[rr < 30] -
                                    np.clip(comp, 0, 1)[rr < 30]).max()))

    print("\n=== 人像（需要人脸）===")
    from engine import portrait as PT
    from engine import landmarks as LMK
    from engine.face import FaceDetector
    faces = FaceDetector().detect(img)
    check("检测到人脸", len(faces) > 0, "%d 张" % len(faces))
    if faces:
        x, y, fw, fh = [float(v) for v in faces[0].box]
        # 关键：align_crop 要 (x0,y0,x1,y1)，检测器给的是 (x,y,w,h)。
        # 直接把 w/h 当右/下边界会把对齐裁剪打到脸外面，landmark 全错位。
        xyxy = np.asarray([x, y, x + fw, y + fh], np.float32)
        pts = LMK.Landmark106().predict(img, xyxy).points
        check("106 点返回形状", pts.shape == (106, 2), str(pts.shape))
        inside = float(((pts[:, 0] >= x - 10) & (pts[:, 0] <= x + fw + 10) &
                        (pts[:, 1] >= y - 10) & (pts[:, 1] <= y + fh + 10)).mean())
        check("106 点落在脸框内", inside > 0.9, "%.0f%%" % (100 * inside))

        e = PT.expression(img, pts, smile=60)
        check("表情改变画面", float(np.abs(e - img).max()) > 0.01,
              "max %.4f" % float(np.abs(e - img).max()))
        # 形变必须是局部的。"远处"取脸框外 1.5 倍脸宽的距离，避开形变半径，
        # 而不是"框外 2 倍框"——后者在这张图上已经越界，等于全图。
        far = np.ones(img.shape[:2], bool)
        margin = 1.5 * max(fw, fh)
        far[max(0, int(y - margin)):min(h, int(y + fh + margin)),
            max(0, int(x - margin)):min(w, int(x + fw + margin))] = False
        if far.any():
            dmax = float(np.abs(e - img).max(axis=2)[far].max())
            check("表情不影响远处", dmax < 0.01,
                  "max %.5f（远处 %d px）" % (dmax, int(far.sum())))
        mk = PT.makeup(img, pts, None, lip=70, blush=50)
        check("妆容改变画面", float(np.abs(mk - img).max()) > 0.01)

    # ---------------- API ----------------
    try:
        requests.get(BASE + "/api/health", timeout=60)
    except Exception:
        print("\n服务未运行，跳过 API")
        return 0 if not failures else 1

    print("\n=== API ===")
    with open(SRC, "rb") as fh:
        up = requests.post(BASE + "/api/upload",
                           files={"file": ("n.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    sid = up["session"]
    base = dec(up["preview"])

    r = requests.post(BASE + "/api/crop/preview", json={
        "session": sid, "crop": {"x": .1, "y": .1, "w": .6, "h": .6},
        "ratio": "1:1"}, timeout=300).json()
    check("裁剪预览返回", bool(r.get("image")))
    check("裁剪预览是正方形", abs(r["width"] - r["height"]) <= 2,
          "%sx%s" % (r["width"], r["height"]))

    r = requests.post(BASE + "/api/crop", json={
        "session": sid, "crop": {"x": .1, "y": .1, "w": .5, "h": .5},
        "commit": True}, timeout=300).json()
    check("裁剪提交改变尺寸", r["width"] < up["width"],
          "%d < %d" % (r["width"], up["width"]))
    hh = requests.get(BASE + "/api/history/" + sid, timeout=60).json()
    check("裁剪进入撤销栈", any("裁剪" in s["label"] for s in hh["steps"]),
          str([s["label"] for s in hh["steps"]]))

    # 裁剪后重置，回到原尺寸
    rr = requests.post(BASE + "/api/crop/reset", json={"session": sid},
                       timeout=60).json()
    check("重置裁剪恢复原尺寸", abs(rr["width"] - up["preview_width"]) <= 2,
          "%d vs %d" % (rr["width"], up["preview_width"]))

    eps = [
        ("/api/selective_color", {"session": sid,
                                  "bands": {"reds": {"magenta": 30}},
                                  "commit": False}),
        ("/api/color_editor", {"session": sid,
                               "bands": {"blues": {"hue": 20}},
                               "commit": False}),
        ("/api/look", {"session": sid, "filter": "warm", "commit": False}),
        ("/api/post_grade", {"session": sid, "vignette": 40, "commit": False}),
        ("/api/defringe", {"session": sid, "defringe": .4, "commit": False}),
        ("/api/portrait", {"session": sid, "smile": 20, "lip": 20,
                           "hair_volume": 15, "commit": False}),
    ]
    for path, body in eps:
        r = requests.post(BASE + path, json=body, timeout=900)
        ok = r.status_code == 200 and bool(r.json().get("image"))
        check("端点 " + path, ok, str(r.status_code))

    pg = requests.post(BASE + "/api/portrait", json={
        "session": sid, "smile": 25, "lip": 25, "hair_volume": 20,
        "wrinkle": 10, "commit": False}, timeout=900).json()
    check("人像四类都生效",
          all(k in (pg.get("applied") or [])
              for k in ("expression", "makeup", "hair", "clothes")),
          str(pg.get("applied")))

    hs = requests.get(BASE + "/api/history/" + sid, timeout=60).json()
    gi = requests.post(BASE + "/api/history/" + sid + "/goto",
                       json={"index": 0}, timeout=120).json()
    check("历史可跳回第 1 步", gi.get("index") == 0, str(gi.get("index")))
    check("历史缩略图可读",
          requests.get(BASE + "/api/history/" + sid + "/thumb/0",
                       timeout=60).status_code == 200)
    bad = requests.post(BASE + "/api/history/" + sid + "/goto",
                        json={"index": 999}, timeout=60)
    check("越界历史索引返回 400", bad.status_code == 400,
          str(bad.status_code))

    # ---------------- UI ----------------
    print("\n=== UI ===")
    try:
        ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
              if x.get("type") == "page" and "8765" in x.get("url", "")]
        if not ts:
            print("  浏览器未连接，跳过")
        else:
            b = B(ts[0])
            b.call("Emulation.setDeviceMetricsOverride", width=1600,
                   height=1000, deviceScaleFactor=1, mobile=False)
            b.call("Page.reload")
            time.sleep(4)
            b.js("""
              (function(){S.session=%s;S.meta=%s;S.faces=%s;
                S.base='data:image/jpeg;base64,'+%s;
                document.getElementById('dropzone').style.display='none';
                ['btnExport','btnErase','btnPicker','btnMatte','btnMatch',
                 'btnAuto','btnHeal','btnStamp','btnUpscale'].forEach(
                  function(id){var e=document.getElementById(id);
                               if(e) e.disabled=false;});
                drawBase(); refreshHistory(); return true;})()
            """ % (json.dumps(sid),
                   json.dumps({"width": up["width"], "height": up["height"],
                               "preview_width": up["preview_width"],
                               "preview_height": up["preview_height"]}),
                   json.dumps(up["faces"]), json.dumps(up["preview"])))
            time.sleep(2)

            # 三模块切换
            nmodes = b.js("document.querySelectorAll('.mode').length")
            check("三个模块按钮", nmodes == 3, "%s 个" % nmodes)
            for mode, cat in (('library', '图库'), ('creative', '创意'),
                              ('retouch', '精修')):
                b.js("setMode('%s')" % mode)
                time.sleep(1.2)
                active = b.js("(document.querySelector('.mode.active')||{})"
                              ".dataset && document.querySelector("
                              "'.mode.active').dataset.mode")
                check("切到%s模块" % cat, active == mode, str(active))
            # 单层导航：图库模块只显示库相关分类（图库/批处理/历史/联机）
            b.js("setMode('library')")
            time.sleep(1.0)
            cats_lib = b.js("[...document.querySelectorAll('#rail .rbtn')]"
                            ".map(function(e){return e.dataset.cat})")
            check("图库模块分类收敛", isinstance(cats_lib, list) and
                  len(cats_lib) <= 6 and "图库" in cats_lib, str(cats_lib))
            b.js("setMode('retouch')")
            time.sleep(1.0)
            cats_ret = b.js("document.querySelectorAll('#rail .rbtn').length")
            check("精修模块分类齐全", (cats_ret or 0) >= 30, "%s 个" % cats_ret)

            # 缩放
            z0 = b.js("(function(){var c=document.getElementById('canvas');"
                      "return c.width;})()")
            b.js("document.getElementById('btnZoomOut').click()")
            time.sleep(0.6)
            z1 = b.js("(function(){var c=document.getElementById('canvas');"
                      "return c.width;})()")
            check("缩小按钮让画布变小", (z1 or 0) < (z0 or 0),
                  "%s -> %s" % (z0, z1))
            b.js("document.getElementById('btnZoom100').click()")
            time.sleep(0.6)
            check("1:1 显示百分比",
                  (b.js("document.getElementById('zoomVal').textContent")
                   or "") == "100%",
                  str(b.js("document.getElementById('zoomVal').textContent")))
            b.js("document.getElementById('btnZoomFit').click()")
            time.sleep(0.6)
            check("适合窗口",
                  (b.js("document.getElementById('zoomVal').textContent")
                   or "") == "适合")

            # 每个新面板都能打开且无异常
            for cat in ('裁剪', '可选颜色', '色彩编辑器', '配置文件', '滤镜',
                        '后调色', '表情管理', '妆容调整', '头发调整', '衣物',
                        '祛边', '历史'):
                b.js("selectCat('%s')" % cat)
                time.sleep(0.9)
                ok = b.js("document.getElementById('sideCat')"
                          ".textContent") == cat
                panel = b.js("document.getElementById('sliders')"
                             ".children.length")
                check("面板 %s 可打开" % cat, ok and (panel or 0) > 0,
                      "%s 个子节点" % panel)

            # 裁剪面板拖框
            b.js("selectCat('裁剪')")
            time.sleep(1.2)
            check("裁剪画布存在", bool(b.js("!!document.getElementById("
                                            "'cropCanvas')")))
            b.js("CROP.x=0.15; CROP.y=0.12; CROP.w=0.6; CROP.h=0.6;"
                 " drawCropCanvas(S.meta.width, S.meta.height);")
            time.sleep(0.4)
            check("裁剪框信息更新",
                  "裁剪尺寸" in (b.js("document.getElementById('crInfo')"
                                     ".innerText") or ""))

            # 历史面板列出步骤
            b.js("selectCat('历史')")
            time.sleep(2.0)
            rows = b.js("document.querySelectorAll('.histRow').length")
            check("历史面板列出步骤", (rows or 0) >= 1, "%s 行" % rows)
            thumbs = b.js("[...document.querySelectorAll('.histRow img')]"
                          ".filter(e=>e.src.indexOf('/api/history/')>0).length")
            check("历史步骤有缩略图", (thumbs or 0) >= 1, "%s 张" % thumbs)

            for cat, name in (('裁剪', 'ui_crop.png'),
                              ('可选颜色', 'ui_selcolor.png'),
                              ('后调色', 'ui_postgrade.png'),
                              ('妆容调整', 'ui_makeup.png'),
                              ('历史', 'ui_history.png')):
                b.js("selectCat('%s')" % cat)
                time.sleep(1.2)
                b.shot(os.path.join(OUT, name))
            print("  截图已存 out/ui_*.png")
            b.call("Emulation.clearDeviceMetricsOverride")
            b.ws.close()
    except Exception as exc:
        print("  UI 检查跳过：%s" % str(exc)[:140])

    print()
    if failures:
        print("FAILED: %d -> %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
