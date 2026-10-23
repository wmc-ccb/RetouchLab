# -*- coding: utf-8 -*-
"""全分类遍历：右栏 42 项逐一点击，验证面板渲染、无 JS 异常.

单元/API 测试覆盖了引擎；这条验证**每个面板在真实浏览器里能渲染**——
路由缺失、函数名拼错、DOM id 冲突这类问题只有真点一遍才暴露。
"""
import base64
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

PARENTS = ["AI色彩", "调色", "局部", "人像", "图像", "衣物", "宠物",
           "后调色", "裁剪", "历史", "联机", "自定义"]


def check(n, ok, d=""):
    print("%-52s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def main():
    tree = requests.get(B + "/api/cats", timeout=60).json().get("tree") or {}
    cats = ["预设"] + [k for p in PARENTS for k in tree.get(p, [])]
    print("待遍历 %d 个分类" % len(cats))

    ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
          if x.get("type") == "page" and "8765" in x.get("url", "")]
    if not ts:
        print("浏览器未连接")
        return 1
    ws = websocket.create_connection(ts[0]["webSocketDebuggerUrl"], timeout=180)
    ni = [0]

    def call(m, **p):
        ni[0] += 1
        ws.send(json.dumps({"id": ni[0], "method": m, "params": p}))
        while True:
            r = json.loads(ws.recv())
            if r.get("id") == ni[0]:
                if "exceptionDetails" in r:
                    print("  !! CDP exception:",
                          str(r["exceptionDetails"])[:160])
                return r.get("result", {})

    def js(e):
        return call("Runtime.evaluate", expression=e, returnByValue=True,
                    awaitPromise=True).get("result", {}).get("value")

    # 收集页面 JS 异常
    call("Runtime.enable")

    call("Emulation.setDeviceMetricsOverride", width=1800, height=1050,
         deviceScaleFactor=1, mobile=False)
    call("Page.reload")
    time.sleep(5)

    broken = []
    # async 面板（配方/图库/历史等要 fetch 完才重写 DOM）：读取前等一拍，
    # 否则会读到上一次渲染的残留（实测「配方」读到「联机」的文案）。
    ASYNC_CATS = {"配方", "图库", "历史", "联机", "批处理", "自动蒙版"}
    for cat in cats:
        js("selectCat('%s')" % cat.replace("'", "\\'"))
        if cat in ASYNC_CATS:
            time.sleep(1.2)
        r = js("""
          (function(){
            try {
              var wrap = document.getElementById('sliders');
              var side = document.getElementById('sideCat');
              var txt = wrap ? wrap.innerText.trim() : '';
              return {side: side ? side.textContent : '(null)',
                      len: txt.length,
                      head: txt.slice(0, 24)};
            } catch (e) { return {err: String(e.message).slice(0, 80)}; }
          })()
        """)
        if not r or r.get("err"):
            broken.append((cat, (r or {}).get("err", "no result")))
            print("  ✗ %-10s %s" % (cat, (r or {}).get("err", "no result")))
        elif r.get("side") != cat:
            broken.append((cat, "sideCat=%s" % r.get("side")))
            print("  ✗ %-10s sideCat=%s" % (cat, r.get("side")))
        elif (r.get("len") or 0) < 10:
            broken.append((cat, "空面板"))
            print("  ✗ %-10s 空面板" % cat)
        else:
            print("  ✓ %-10s %d 字  %s" % (cat, r.get("len"), r.get("head")))

    check("全部 %d 个分类面板渲染" % len(cats), not broken,
          "; ".join("%s:%s" % b for b in broken[:5]))

    # 页面级 JS 异常（Runtime.enable 后 exceptionThrown 事件我们不收，
    # 这里用 window.onerror 计数器验证交互期间没有新异常）
    errs = js("window.__err_count || 0")
    check("交互期间无未捕获异常", not errs, "%s 个" % errs)

    shot = None
    js("(function(){try{window.requestRender=function(){};"
       "return 1}catch(e){}})()")
    for _ in range(4):
        try:
            shot = call("Page.captureScreenshot", format="jpeg", quality=70)
            with open(os.path.join(OUT, "ui_walkthrough.png"), "wb") as fh:
                fh.write(base64.b64decode(shot["data"]))
            break
        except Exception:
            time.sleep(1.0)
    call("Emulation.clearDeviceMetricsOverride")
    ws.close()

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
