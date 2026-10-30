# -*- coding: utf-8 -*-
"""分类结构：二级导航（一级大组 + 飞出子项）+ 服务端分组表.

历史演进（这个测试锁定当前决定）：
  1. 平铺 34 项        -> 一屏放不下，要滚 1.8 屏
  2. 两级折叠          -> 要先想"属于哪个父类"再点两次，反而更累
  3. 单层 + 双列 + 分组标题 -> 46 个按钮看花眼
  4. 二级导航（一级大组 + 飞出子项，selectCat 直调自动展开）<- 现在
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

PARENTS = ["AI色彩", "生成式", "调色", "局部", "人像", "图像", "衣物",
           "宠物", "后调色", "裁剪", "历史", "联机", "自定义"]


def check(n, ok, d=""):
    print("%-50s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def main():
    # ---- API：分组表仍然是唯一映射源 ----
    j = requests.get(B + "/api/cats", timeout=60).json()
    tree = j.get("tree") or {}
    check("13 个分组", list(tree.keys()) == PARENTS, str(list(tree.keys())))
    check("子项总数 45", sum(len(v) for v in tree.values()) == 45,
          "%d" % sum(len(v) for v in tree.values()))

    try:
        ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
              if x.get("type") == "page" and "8765" in x.get("url", "")]
        if not ts:
            print("浏览器未连接")
            return 1
        ws = websocket.create_connection(ts[0]["webSocketDebuggerUrl"],
                                         timeout=180)
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
        js("localStorage.setItem('rl_onboarded','1');"
           "homeDismiss && homeDismiss();")

        print("\n=== 一级结构 ===")
        groups = js("[...document.querySelectorAll('#rail .rgroup')]"
                    ".map(function(b){return b.dataset.group})")
        check("一级大组与分组表一致",
              isinstance(groups, list) and groups == PARENTS, str(groups))
        kids_on_rail = js("[...document.querySelectorAll('#rail .rbtn')]"
                          ".filter(function(b){return b.dataset.cat;"
                          "}).length")
        check("一级只保留预设直达", kids_on_rail == 1, "%s 个" % kids_on_rail)

        print("\n=== 二级展开 ===")
        # 每个大组都能弹出自己的子项，数量与分组表一致
        for parent in ("人像", "调色", "图像"):
            js("(function(){var b=document.querySelector("
               "'#rail .rgroup[data-group=\"%s\"]'); if(b) b.click();"
               "})()" % parent)
            time.sleep(0.35)
            nsub = js("document.querySelectorAll('#railFly .flyBtn').length")
            check("「%s」弹出子项" % parent,
                  nsub == len(tree.get(parent, [])),
                  "%s/%s" % (nsub, len(tree.get(parent, []))))
            js("(function(){var b=document.querySelector("
               "'#rail .rgroup[data-group=\"%s\"]'); if(b) b.click();"
               "})()" % parent)
            time.sleep(0.2)

        print("\n=== 二级点选开面板 ===")
        for key, parent in (("美肤", "人像"), ("曲线", "调色"),
                            ("证件照", "图像"), ("图库", "自定义")):
            js("(function(){var b=document.querySelector("
               "'#rail .rgroup[data-group=\"%s\"]'); if(b) b.click();"
               "})()" % parent)
            time.sleep(0.3)
            js("(function(){var b=[...document.querySelectorAll("
               "'#railFly .flyBtn')].find(function(x){"
               "return x.dataset.cat==='%s'}); if(b) b.click();"
               "return true;})()" % key)
            time.sleep(0.9)
            cur = js("document.getElementById('sideCat').textContent")
            check("二级点选「%s」开面板" % key, cur == key, str(cur))

        # requestRender 的按需重绘会让 Chrome 合成器忙到截图排不上队
        # （verify_levels 实测：patch 成 no-op 后 0.1s 出图）。
        js("(function(){try{window.requestRender=function(){};"
           "return 1}catch(e){}})()")
        shot = None
        for _ in range(4):
            try:
                shot = call("Page.captureScreenshot", format="jpeg",
                            quality=70)
                with open(os.path.join(OUT, "ui_cattree.png"), "wb") as fh:
                    fh.write(base64.b64decode(shot["data"]))
                break
            except Exception:
                time.sleep(1.0)
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
