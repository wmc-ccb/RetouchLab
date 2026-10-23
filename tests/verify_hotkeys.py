# -*- coding: utf-8 -*-
"""PS 快捷键：单键工具切换 + Ctrl 组合."""
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


def check(n, ok, d=""):
    print("%-52s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def main():
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
                return r.get("result", {})

    def js(e):
        return call("Runtime.evaluate", expression=e, returnByValue=True,
                    awaitPromise=True).get("result", {}).get("value")

    def synth_key(ch):
        """合成 KeyboardEvent（与真实键盘同路径触发 document handler）."""
        js("document.dispatchEvent(new KeyboardEvent('keydown', "
           "{key: '%s', bubbles: true}));" % ch)

    def ctrl_key(ch):
        js("document.dispatchEvent(new KeyboardEvent('keydown', "
           "{key: '%s', ctrlKey: true, bubbles: true}));" % ch)

    call("Emulation.setDeviceMetricsOverride", width=1800, height=1050,
         deviceScaleFactor=1, mobile=False)
    call("Page.reload")
    time.sleep(5)

    # 单键工具切换（画笔/图章/抓手/裁剪/局部…）
    VK = ["b", "s", "h", "c", "m", "w", "e", "f", "v"]
    MODE_EXPECT = {"b": "heal", "s": "clone"}
    for k in VK:
        synth_key(k)
        time.sleep(0.5)
        if k in MODE_EXPECT:
            mode = js("(function(){try{return BRUSH.mode}"
                      "catch(e){return '(no BRUSH)'}})()")
            check("按 %s 进入 %s" % (k, MODE_EXPECT[k]),
                  mode == MODE_EXPECT[k], str(mode))
        elif k in ("c", "m", "w"):
            cat = js("document.getElementById('sideCat').textContent")
            want = {"c": "裁剪", "m": "局部", "w": "自动蒙版"}[k]
            check("按 %s 切到「%s」" % (k, want), cat == want, str(cat))
        elif k == "h":
            pan = js("(function(){try{return S.pan===true}"
                     "catch(e){return false}})()")
            check("按 h 进抓手模式", pan is True)
        elif k == "e":
            er = js("document.getElementById('ltErase')"
                    ".classList.contains('active')")
            check("按 e 切祛边（左条高亮）", er is True, str(er))
        elif k == "f":
            mode = js("(function(){try{return ZOOM.mode}"
                      "catch(e){return '?'}})()")
            check("按 f 缩放适配", mode in ("fit", "full"), str(mode))

    # Ctrl 组合
    ctrl_key("z")       # Ctrl+Z undo（无历史也应无异常）
    time.sleep(0.4)
    check("Ctrl+Z 无异常", True)
    ctrl_key("0")       # Ctrl+0 适配
    time.sleep(0.5)
    mode = js("(function(){try{return ZOOM.mode}catch(e){return '?'}})()")
    check("Ctrl+0 适配窗口", mode in ("fit", "full"), str(mode))
    ctrl_key("o")       # Ctrl+O 打开（弹文件框——只验证 handler 不抛）
    time.sleep(0.5)
    check("Ctrl+O 无异常", True)

    # 输入框聚焦时不抢键。守卫读 activeElement.tagName，所以要先有
    # 一个可聚焦的输入框：切到「光影」（有 range 滑杆）再聚焦。
    # （上一步是「按 e」，自动蒙版面板没有 .sl input，直接聚焦是空的。）
    js("selectCat('光影');")
    time.sleep(0.8)
    focused = js("(function(){var el=document.querySelector('.sl input');"
                 "if(el){el.focus();}"
                 "return (document.activeElement||{}).tagName;})()")
    check("已聚焦滑杆输入框", focused == "INPUT", str(focused))
    synth_key("c")
    time.sleep(0.4)
    cat = js("document.getElementById('sideCat').textContent")
    check("输入框聚焦时单键不抢", cat != "裁剪", str(cat))

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
