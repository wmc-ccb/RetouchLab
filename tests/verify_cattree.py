# -*- coding: utf-8 -*-
"""分类结构：单层导航（一次点击直达）+ 服务端分组表.

历史演进（这个测试锁定当前决定）：
  1. 平铺 34 项  -> 一屏放不下，要滚 1.8 屏
  2. 两级折叠    -> 要先想"属于哪个父类"再点两次，反而更累
  3. 单层 + 双列 + 不可点的分组标题  <- 现在
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
    # 自动蒙版 + 色阶 + 换天空 + 补草地 后 39 项，
    # 加「生成式」（AI 布景 / 创成式填充 / 道具修复）3 项 = 45
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

        print("\n=== 单层结构 ===")
        check("无可展开父类",
              js("[...document.querySelectorAll('#rail .rbtn')]"
                 ".filter(function(b){return !!b.dataset.parent}).length") == 0)
        titles = js("[...document.querySelectorAll('#rail .catTitle')]"
                    ".map(function(e){return e.textContent})")
        # 宠物组现在有「宠物修整」，不再是空分组
        check("分组标题含宠物",
              isinstance(titles, list) and "宠物" in titles, str(titles))
        check("分组标题数量合理", isinstance(titles, list) and
              8 <= len(titles) <= 13, "%s 个" % len(titles or []))

        # 所有子项一屏可达
        cats = js("[...document.querySelectorAll('#rail .rbtn')]"
                  ".map(function(b){return b.dataset.cat})")
        total_kids = sum(len(v) for v in tree.values())
        check("所有子项都在栏上", isinstance(cats, list) and
              len([c for c in cats if c != '预设']) == total_kids,
              "%d/%d" % (len(cats or []) - 1, total_kids))

        print("\n=== 一屏放得下 ===")
        geo = js("""
          (function(){
            var rail=document.getElementById('rail');
            var r=rail.getBoundingClientRect();
            return {cols2: rail.classList.contains('cols2'),
                    railH: Math.round(r.height),
                    scrollH: Math.round(rail.scrollHeight),
                    fits: rail.scrollHeight <= r.height + 6};
          })()
        """)
        check("宽屏启用双列", geo and geo.get("cols2") is True)
        check("右侧栏一屏放下", geo and geo.get("fits") is True,
              "内容 %spx / 可见 %spx" % (geo.get("scrollH"),
                                        geo.get("railH")))

        print("\n=== 一次点击直达 ===")
        for key in ("美肤", "曲线", "证件照", "图库"):
            js("(function(){var bs=[...document.querySelectorAll("
               "'#rail .rbtn')];var b=bs.find(function(x){"
               "return x.dataset.cat==='%s'});if(b) b.click();"
               "return true;})()" % key)
            time.sleep(0.9)
            cur = js("document.getElementById('sideCat').textContent")
            check("点「%s」直达" % key, cur == key, str(cur))

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
