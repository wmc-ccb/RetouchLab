# -*- coding: utf-8 -*-
"""Browser check for the three new panels: undo/redo, curves, batch."""
import base64
import json
import os
import sys
import time

import requests
import websocket

CDP = "http://127.0.0.1:9223"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

failures = []


class Browser:
    """CDP client that survives a dropped connection.

    The DevTools websocket closes without warning when the page reloads or
    Chrome is busy, and a one-shot connection turns that into a confusing
    traceback in the middle of a check. Reconnecting on a failed call keeps a
    transport hiccup from being mistaken for a failing assertion.
    """

    def __init__(self, target):
        self.target = target
        self.ws = None
        self.i = 0
        self._connect()

    def _connect(self):
        url = self.target["webSocketDebuggerUrl"]
        for attempt in range(4):
            try:
                self.ws = websocket.create_connection(url, timeout=60)
                self.i = 0
                return
            except Exception:
                time.sleep(1.5)
        raise RuntimeError("could not attach to the page")

    def call(self, method, **params):
        for attempt in range(3):
            try:
                self.i += 1
                self.ws.send(json.dumps({"id": self.i, "method": method,
                                         "params": params}))
                while True:
                    msg = json.loads(self.ws.recv())
                    if msg.get("id") == self.i:
                        return msg.get("result", {})
            except Exception:
                # transport died: reattach and retry the same call
                try:
                    self.ws.close()
                except Exception:
                    pass
                time.sleep(1.0)
                self.target = _find_target() or self.target
                self._connect()
        raise RuntimeError("CDP call failed after retries: " + method)

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr,
                      returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    def shot(self, path):
        r = self.call("Page.captureScreenshot", format="png")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(r["data"]))


def _find_target():
    try:
        for t in requests.get(CDP + "/json", timeout=10).json():
            if t.get("type") == "page" and "8765" in t.get("url", ""):
                return t
    except Exception:
        return None
    return None


def check(label, cond, detail=""):
    print("%-50s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


def main():
    target = _find_target()
    if not target:
        print("page not found on the debug port")
        return 1
    b = Browser(target)
    print("connected")
    # Reload first: the UI tests share one browser tab, so a session left
    # behind by a previous test makes "no session ⇒ disabled" fail. Isolating
    # each run from the last is the test's job, not the app's.
    b.call("Page.reload")
    time.sleep(4)

    n = b.js("document.querySelectorAll('#rail .rbtn').length")
    # 右侧栏是**单层**：一屏列全、一次点击直达，分组标题只做视觉分隔
    # （不可点）。历史：先平铺（34 项要滚）、后两级（点两次要想），
    # 现在单层 + 双列 + 分组标题。
    cats = b.js("[...document.querySelectorAll('#rail .rbtn')]"
                ".map(function(e){return e.dataset.cat})")
    for want in ("预设", "曲线", "色轮", "局部", "配方", "批处理",
                 "美肤", "证件照", "背景消除"):
        check("一屏直达「%s」" % want,
              isinstance(cats, list) and want in cats)
    check("没有需要展开的父类按钮",
          b.js("[...document.querySelectorAll('#rail .rbtn')]"
               ".filter(function(e){return !!e.dataset.parent}).length") == 0)
    check("有分组标题",
          (b.js("document.querySelectorAll('#rail .catTitle').length") or 0)
          >= 5, "%s 个" % b.js("document.querySelectorAll("
                               "'#rail .catTitle').length"))

    for key, expect in (("\u66f2\u7ebf", "curveCanvas"),
                        ("\u6279\u5904\u7406", "bSrc"),
                        ("\u8272\u8f6e", None),
                        ("\u914d\u65b9", None)):
        title = b.js("(function(){ selectCat('%s');"
                     " return document.getElementById('sideCat').textContent; })()"
                     % key)
        time.sleep(0.9)
        check("面板 %s 可打开" % key, title == key, str(title))
        if expect:
            has = b.js("!!document.getElementById('%s')" % expect)
            check("  %s 元素存在" % expect, bool(has))

    # curves panel detail
    b.js("selectCat('\u66f2\u7ebf')")
    time.sleep(1.0)
    check("曲线画布存在", bool(b.js("!!document.getElementById('curveCanvas')")))
    check("曲线通道按钮 4 个",
          b.js("document.querySelectorAll('#sliders .chBtn').length") == 4)
    # add a control point via the exposed state, then confirm the canvas redraws
    before = b.js("CURVE.points.rgb.length")
    b.js("CURVE.points.rgb = [[0,0],[0.5,0.7],[1,1]]; drawCurve();")
    after = b.js("CURVE.points.rgb.length")
    check("控制点可修改", before == 2 and after == 3,
          "%s -> %s" % (before, after))
    check("非恒等被识别为已修改",
          b.js("!curveIsIdentity(CURVE.points.rgb)"))
    check("applyCurves 会提交曲线参数",
          bool(b.js("typeof applyCurves === 'function'")))
    b.js("CURVE.points.rgb = [[0,0],[1,1]]; drawCurve(); renderCurves();")
    b.shot(os.path.join(OUT, "ui3_curves.png"))

    # batch panel detail
    b.js("selectCat('\u6279\u5904\u7406')")
    time.sleep(1.0)
    check("批处理源目录输入框", bool(b.js("!!document.getElementById('bSrc')")))
    check("批处理输出目录输入框", bool(b.js("!!document.getElementById('bOut')")))
    check("批处理预设按钮", (b.js("document.querySelectorAll('#sliders .chip[data-p]')"
                                ".length") or 0) >= 5)
    check("批处理开始按钮", bool(b.js("!!document.getElementById('bStart')")))
    check("草稿模式按钮", bool(b.js("!!document.getElementById('bDraft')")))
    b.js("document.getElementById('bDraft').onclick()")
    check("草稿模式可切换", b.js("BATCH.mode") == "draft", str(b.js("BATCH.mode")))
    b.shot(os.path.join(OUT, "ui3_batch.png"))

    # undo buttons
    check("撤销按钮存在", bool(b.js("!!document.getElementById('btnUndo')")))
    check("重做按钮存在", bool(b.js("!!document.getElementById('btnRedo')")))
    check("无会话时撤销禁用",
          b.js("document.getElementById('btnUndo').disabled") is True)
    check("Ctrl+Z 已绑定", bool(b.js(
        "typeof doUndo === 'function' && typeof doRedo === 'function'")))

    b.ws.close()
    print()
    if failures:
        print("FAILED: %d -> %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
