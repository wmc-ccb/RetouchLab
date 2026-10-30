# -*- coding: utf-8 -*-
"""二级分类导航：一级大组 + 飞出子项列表（与参考软件同构）.

历史：平铺（34 项要滚）→ 单层+分组标题（一屏直达但 46 个按钮看花眼）
→ 现在回到两级：一级 rail 只列大组，点组在 rail 旁弹出二级子项列表，
点子项开面板；selectCat 直调（热键/脚本）自动展开对应组的二级列表。
本测试锁定"二级"这个决定与它的关键不变量。
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
    js("localStorage.setItem('rl_onboarded','1'); homeDismiss && homeDismiss();")

    print("=== 一级结构 ===")
    groups = js("[...document.querySelectorAll('#rail .rgroup')]"
                ".map(function(b){return b.dataset.group})")
    check("一级只列大组", isinstance(groups, list) and 10 <= len(groups) <= 14,
          "%s 个: %s" % (len(groups or []), groups))
    flat_kids = js("[...document.querySelectorAll('#rail .rbtn')]"
                   ".filter(function(b){return b.dataset.cat==='美肤'||"
                   "b.dataset.cat==='光影'||b.dataset.cat==='细节'}).length")
    check("一级不再平铺子项", flat_kids == 0, "%s 个" % flat_kids)
    preset1 = js("[...document.querySelectorAll('#rail .rbtn')]"
                 ".some(function(b){return b.dataset.cat==='预设';})")
    check("预设一级直达", preset1 is True)

    print("\n=== 二级行为 ===")
    js("(function(){var b=document.querySelector("
       "'#rail .rgroup[data-group=\"人像\"]'); if(b) b.click();})()")
    time.sleep(0.4)
    fly_on = js("!!document.getElementById('railFly')")
    nsub = js("document.querySelectorAll('#railFly .flyBtn').length")
    check("点组弹出二级列表", fly_on is True and (nsub or 0) >= 3,
          "子项 %s" % nsub)
    js("(function(){var b=[...document.querySelectorAll('#railFly .flyBtn')]"
       ".find(function(x){return x.dataset.cat==='美肤'}); if(b) b.click();})()")
    time.sleep(1.0)
    check("点子项开美肤面板",
          js("document.getElementById('sideCat').textContent") == "美肤")
    check("子项高亮同步", js(
        "(function(){var b=document.querySelector('#railFly .flyBtn.active');"
        "return b?b.dataset.cat:'none';})()") == "美肤")
    check("组按钮高亮同步", js(
        "(function(){var b=document.querySelector('#rail .rgroup.active');"
        "return b?b.dataset.group:'none';})()") == "人像")
    js("(function(){document.querySelector('#rail .rgroup[data-group=\"人像\"]')"
       ".click();})()")
    time.sleep(0.3)
    check("再点同组收起二级", js(
        "!document.getElementById('railFly')") is True)

    js("selectCat('曲线')")
    time.sleep(0.4)
    check("直调 selectCat 自动展开调色组", js(
        "(function(){var f=document.getElementById('railFly');"
        "return f?f.dataset.group:'none';})()") == "调色")
    check("直调后子项高亮", js(
        "(function(){var b=document.querySelector('#railFly .flyBtn.active');"
        "return b?b.dataset.cat:'none';})()") == "曲线")
    check("曲线面板已开", js(
        "document.getElementById('sideCat').textContent") == "曲线")

    js("selectCat('美肤')")
    time.sleep(0.3)
    pet = js("[...document.querySelectorAll('#rail .rgroup')]"
             ".some(function(b){return b.dataset.group==='宠物';})")
    check("宠物组占位（有宠物修整）", pet is True)

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
