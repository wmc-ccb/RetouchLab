# -*- coding: utf-8 -*-
"""单层分类导航：一次点击直达 + 分组标题视觉分隔.

历史：先做过平铺（34 项、要滚）、又做过两级（父类→子项、点两次要想），
现在是一屏列全 + 不可点的分组标题。这个测试锁定"单层"这个决定：
右栏不应再出现需要展开的父类按钮。
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


def check(n, ok, d=""):
    print("%-50s %s %s" % (n, "OK  " if ok else "FAIL", d))
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

    call("Emulation.setDeviceMetricsOverride", width=1800, height=1050,
         deviceScaleFactor=1, mobile=False)
    call("Page.reload")
    time.sleep(4)

    print("=== 单层结构 ===")
    # 不应该再有可展开的父类按钮
    parents = js("[...document.querySelectorAll('#rail .rbtn')]"
                 ".filter(function(b){return !!b.dataset.parent}).length")
    check("无可展开父类按钮", parents == 0, "%s 个" % parents)
    # 分组标题存在且不可点
    titles = js("[...document.querySelectorAll('#rail .catTitle')]"
                ".map(function(e){return e.textContent})")
    check("有分组标题", isinstance(titles, list) and len(titles) >= 5,
          str(titles))
    title_clickable = js("(function(){var e=document.querySelector("
                         "'#rail .catTitle');"
                         "return e ? getComputedStyle(e).cursor : 'none';})()")
    check("标题不是按钮（无 pointer 光标）",
          title_clickable not in ("pointer",), str(title_clickable))

    # 关健功能应**一次点击**可达
    cats = js("[...document.querySelectorAll('#rail .rbtn')]"
              ".map(function(b){return b.dataset.cat})")
    for want in ("预设", "光影", "美肤", "曲线", "裁剪", "背景消除", "证件照",
                 "图库", "历史", "联机"):
        check("一屏直达「%s」" % want, isinstance(cats, list) and want in cats)

    print("\n=== 点击行为 ===")
    # 点「美肤」应立即切到美肤面板（一次点击）
    js("(function(){var bs=[...document.querySelectorAll('#rail .rbtn')];"
       "var b=bs.find(function(x){return x.dataset.cat==='美肤'});"
       "if(b) b.click(); return true;})()")
    time.sleep(1.2)
    check("一次点击切到美肤",
          js("document.getElementById('sideCat').textContent") == "美肤",
          str(js("document.getElementById('sideCat').textContent")))
    check("按钮高亮同步",
          js("(function(){var bs=[...document.querySelectorAll("
             "'#rail .rbtn')];var b=bs.find(function(x){"
             "return x.dataset.cat==='美肤'});"
             "return b?b.classList.contains('active'):false;})()") is True)

    # 宠物组现在有「宠物修整」（Round 22 起），应占位出现
    pet = js("(function(){return [...document.querySelectorAll("
             "'#rail .catTitle')].some(function(e){"
             "return e.textContent==='宠物';});})()")
    check("宠物分组占位（有宠物修整）", pet is True)
    petkid = js("[...document.querySelectorAll('#rail .rbtn')]"
                ".some(function(b){return b.dataset.cat==='宠物修整';})")
    check("宠物修整直达", petkid is True)

    shot = call("Page.captureScreenshot", format="png")
    with open(os.path.join(OUT, "ui_simplecat.png"), "wb") as fh:
        fh.write(base64.b64decode(shot["data"]))
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
