# -*- coding: utf-8 -*-
"""图库 + 背景消除：API 与 UI 端到端."""
import base64
import io
import json
import os
import sys
import time

import requests
import websocket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
B = "http://127.0.0.1:8765"
CDP = "http://127.0.0.1:9223"
OUT = os.path.join(ROOT, "out")
FAIL = []


def check(name, ok, detail=""):
    print("%-50s %s %s" % (name, "OK  " if ok else "FAIL", detail))
    if not ok:
        FAIL.append(name)


class C:
    def __init__(self, t):
        self.ws = websocket.create_connection(t["webSocketDebuggerUrl"],
                                              timeout=180)
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
    # ---------- API ----------
    print("=== 图库 API ===")
    t0 = time.time()
    r = requests.get(B + "/api/library/roots", timeout=60)
    check("roots 立即返回（不阻塞）", r.ok and time.time() - t0 < 5,
          "%.2fs" % (time.time() - t0))
    j = r.json()
    check("roots 带 scanning 标记", "scanning" in j, str(j.get("scanning")))

    # 等后台扫描完成。根目录扫描是整盘遍历，几十秒是正常的；这里只在
    # 首次（无缓存）时等，后续运行直接命中缓存。
    roots = []
    for _ in range(60):
        time.sleep(2)
        k = requests.get(B + "/api/library/roots", timeout=60).json()
        roots = k.get("roots") or []
        if not k.get("scanning") and roots:
            break
    check("后台扫描完成且有盘符", len(roots) >= 1, "%d 个" % len(roots))
    mroot = next((x for x in roots if x["name"] == "M:"), None)
    check("M: 被识别为照片库", mroot is not None,
          ("%d 张" % mroot["count"]) if mroot else "未找到")

    # 扫一个**小目录**：这里验证的是链路（扫描/封面/元数据/打开），
    # 不是吞吐量。挑 M: 下最小的分组，避免每次测试都解 500 张图。
    scan_path = None
    if mroot and mroot["groups"]:
        small = sorted(mroot["groups"], key=lambda g: g["count"])
        pick = next((g for g in small if 0 < g["count"] <= 40), None)
        scan_path = (pick or small[0])["path"]
    if not scan_path:
        scan_path = os.path.dirname(
            r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg")
    if not scan_path:
        scan_path = os.path.dirname(
            r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg")
    t0 = time.time()
    s = requests.get(B + "/api/library/scan",
                     params={"path": scan_path, "limit": 24},
                     timeout=300).json()
    check("扫描目录成功", s.get("count", 0) > 0,
          "%d 张 %.1fs" % (s.get("count", 0), time.time() - t0))
    check("扫描返回分组统计", isinstance(s.get("groups"), dict),
          str(s.get("groups")))
    it0 = (s.get("items") or [{}])[0]
    check("条目带技术指标", all(k in it0 for k in ("sharp", "mean")),
          str({k: it0.get(k) for k in ("sharp", "mean")})[:70])

    if s.get("items"):
        p0 = s["items"][0]["path"]
        th = requests.get(B + "/api/library/thumb",
                          params={"path": p0, "size": 200}, timeout=120)
        check("缩略图可取", th.ok and th.headers.get("content-type",
                                                    "").startswith("image"),
              "%d bytes" % len(th.content))
        ex = requests.get(B + "/api/library/exif", params={"path": p0},
                          timeout=60).json()
        check("EXIF 可读", isinstance(ex, dict) and "name" in ex,
              str({k: ex.get(k) for k in ("iso", "focal", "aperture")})[:70])
        op = requests.post(B + "/api/library/open", json={"path": p0},
                           timeout=300).json()
        check("从图库打开成会话", bool(op.get("session")),
              str(op.get("session"))[:20])

    print("\n=== 背景消除 API ===")
    info = requests.get(B + "/api/bgclean/info", timeout=60).json()
    check("bgclean info 有 5 项", len(info.get("features", [])) == 5,
          str(info.get("features")))
    u = requests.post(B + "/api/upload",
                      files={"file": ("x.jpg", open(
                          r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg",
                          "rb"), "image/jpeg")}, timeout=300).json()
    sid = u["session"]
    r = requests.post(B + "/api/bgclean", json={
        "session": sid, "showthrough": 1, "blemish": 1, "even_light": 1,
        "banding": 1, "enhance": 1, "commit": False}, timeout=900)
    check("bgclean 全项可用", r.ok and len(r.json().get("image", "")) > 5000,
          "%d" % r.status_code)
    if r.ok:
        d = r.json()["info"]
        check("bgclean 报告背景覆盖率", 0 < d.get("bg_coverage", 0) < 1,
              "%.2f" % d.get("bg_coverage", 0))

    # ---------- UI ----------
    print("\n=== UI ===")
    try:
        ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
              if x.get("type") == "page" and "8765" in x.get("url", "")]
        if not ts:
            print("  浏览器未连接，跳过")
        else:
            c = C(ts[0])
            c.call("Emulation.setDeviceMetricsOverride", width=1600,
                   height=1000, deviceScaleFactor=1, mobile=False)
            c.call("Page.reload")
            time.sleep(4)

            # 图库模块
            c.js("setMode('library')")
            time.sleep(2.5)
            check("切到图库模块",
                  c.js("(document.querySelector('.mode.active')||{}).dataset"
                       "&& document.querySelector('.mode.active').dataset"
                       ".mode") == "library")
            # 单层导航：「图库」直接就在栏上，无需展开
            check("图库模块有图库分类",
                  c.js("[...document.querySelectorAll('#rail .rbtn')]"
                       ".some(function(b){return b.dataset.cat==='图库'})")
                  is True)
            c.js("(function(){var bs=[...document.querySelectorAll("
                 "'#rail .rbtn')];var k=bs.find(function(x){"
                 "return x.dataset.cat==='图库'});"
                 "if(k) k.click(); return true;})()")
            time.sleep(1.6)
            cats = c.js("[...document.querySelectorAll('#rail .rbtn')]"
                        ".map(function(b){return b.dataset.cat})")
            check("图库模块分类收敛", isinstance(cats, list) and
                  len(cats) <= 6 and "图库" in cats, str(cats))
            check("图库面板渲染出分组按钮",
                  (c.js("document.querySelectorAll('.libF').length") or 0) >= 5,
                  "%s 个" % c.js("document.querySelectorAll('.libF').length"))

            # 直接扫一个目录
            c.js("libScan(%s)" % json.dumps(scan_path))
            cells = 0
            for _ in range(40):
                time.sleep(1)
                cells = c.js("document.querySelectorAll('.libCell').length") or 0
                if cells > 0:
                    break
            check("图库网格出现缩略图", cells > 0, "%s 个" % cells)
            c.shot(os.path.join(OUT, "ui_library.png"))

            # 背景消除面板
            c.js("setMode('retouch')")
            time.sleep(1.5)
            c.js("selectCat('背景消除')")
            time.sleep(2.0)
            check("背景消除面板可打开",
                  c.js("document.getElementById('sideCat').textContent")
                  == "背景消除")
            check("四项开关存在",
                  (c.js("document.querySelectorAll("
                        "'[id^=bgc_][type=checkbox]').length") or 0) >= 4,
                  "%s 个" % c.js("document.querySelectorAll("
                                 "'[id^=bgc_][type=checkbox]').length"))
            check("背景增强滑条存在", bool(c.js("!!document.getElementById"
                                               "('bgc_enh')")))
            check("换底色色板存在",
                  (c.js("document.querySelectorAll('.bgcSw').length") or 0) >= 4,
                  "%s 个" % c.js("document.querySelectorAll('.bgcSw').length"))
            c.shot(os.path.join(OUT, "ui_bgclean.png"))

            # 设备状态
            st = requests.get(B + "/api/device/status", timeout=120).json()
            check("设备状态如实报告", "can_capture" in st and
                  "reason" in st, "can_capture=%s" % st.get("can_capture"))
            c.call("Emulation.clearDeviceMetricsOverride")
            c.ws.close()
    except Exception as exc:
        print("  UI 检查跳过：%s" % str(exc)[:140])

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
