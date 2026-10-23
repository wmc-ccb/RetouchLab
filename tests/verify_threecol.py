# -*- coding: utf-8 -*-
"""右侧三列：预设 + 参数 + 历史记录（参考软件同构）."""
import base64
import json
import os
import sys
import time

import requests
import websocket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CDP = "http://127.0.0.1:9223"
OUT = os.path.join(ROOT, "out")
FAIL = []


def check(n, ok, d=""):
    print("%-46s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


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
    ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
          if x.get("type") == "page" and "8765" in x.get("url", "")]
    if not ts:
        print("浏览器未连接")
        return 1
    c = C(ts[0])
    c.call("Emulation.setDeviceMetricsOverride", width=1800, height=1050,
           deviceScaleFactor=1, mobile=False)
    c.call("Page.reload")
    time.sleep(4)

    with open(r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg",
              "rb") as fh:
        up = requests.post("http://127.0.0.1:8765/api/upload",
                           files={"file": ("x.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    c.js("""(function(){S.session=%s;S.meta=%s;S.faces=%s;
        S.base='data:image/jpeg;base64,'+%s;
        document.getElementById('dropzone').style.display='none';
        ['btnExport','btnErase','btnPicker','btnMatte','btnMatch','btnAuto',
         'btnHeal','btnStamp','btnUpscale'].forEach(function(id){
           var e=document.getElementById(id); if(e) e.disabled=false;});
        drawBase(); refreshHistory(); return true;})()""" % (
        json.dumps(up["session"]),
        json.dumps({"width": up["width"], "height": up["height"],
                    "preview_width": up["preview_width"],
                    "preview_height": up["preview_height"]}),
        json.dumps(up["faces"]), json.dumps(up["preview"])))
    time.sleep(2)

    print("=== 三列布局 ===")
    geo = c.js("""
      (function(){
        function r(id){ var e=document.getElementById(id);
          if(!e) return null; var b=e.getBoundingClientRect();
          return {l:Math.round(b.left), r:Math.round(b.right),
                  w:Math.round(b.width)}; }
        return {preset:r('presetCol'), param:r('paramCol'),
                hist:r('histCol'), body:r('sideBody'),
                disp:getComputedStyle(document.getElementById('sideBody')).display};
      })()
    """)
    print("   ", json.dumps(geo, ensure_ascii=False))
    check("sideBody 是横向 flex", geo and geo["disp"] == "flex", geo["disp"])
    check("三列都存在",
          geo and geo["preset"] and geo["param"] and geo["hist"])
    if geo and geo["preset"] and geo["param"] and geo["hist"]:
        check("预设在参数左边", geo["preset"]["r"] <= geo["param"]["l"] + 2)
        check("历史在参数右边", geo["hist"]["l"] >= geo["param"]["r"] - 2)
        check("三列都有宽度",
              all(x["w"] > 20 for x in (geo["preset"], geo["param"],
                                        geo["hist"])),
              "预设%d 参数%d 历史%d" % (geo["preset"]["w"], geo["param"]["w"],
                                       geo["hist"]["w"]))

    print("\n=== 预设列内容 ===")
    # 预设列只在有参数的分类下显示；「光影」就是其中之一。
    # 缩略图每张要调一次 /api/render，必须等填充，否则截图与断言都看到空列。
    c.js("selectCat('光影')")
    cards = 0
    for _ in range(40):
        time.sleep(0.5)
        cards = c.js("document.querySelectorAll('#pcList .pcard').length") or 0
        if cards > 0:
            break
    check("预设列渲染出卡片", cards >= 3, "%s 张" % cards)
    check("预设列可见",
          c.js("getComputedStyle(document.getElementById('presetCol'))"
               ".display") == "flex")
    withthumb = 0
    for _ in range(40):
        time.sleep(0.5)
        withthumb = c.js("[...document.querySelectorAll('#pcList img')]"
                         ".filter(function(i){return i.src && "
                         "i.src.indexOf('data:image')===0}).length") or 0
        if withthumb >= min(cards, 3):
            break
    check("预设缩略图已加载", withthumb >= min(cards, 3),
          "%s/%s 张" % (withthumb, cards))

    print("\n=== 历史内容 ===")
    # 用带 commit 的端点逐步提交。`/api/render` 是纯预览，**没有** commit
    # 字段（每个滑条 tick 都变成一步会污染撤销栈），所以这里用 bgclean。
    sid = up["session"]
    for i, lvl in enumerate((0.35, 0.7)):
        r = requests.post("http://127.0.0.1:8765/api/bgclean", json={
            "session": sid, "blemish": lvl, "even_light": lvl,
            "commit": True}, timeout=600)
        if not r.ok:
            print("   bgclean 失败:", r.status_code, r.text[:120])
            break
        time.sleep(0.5)
    c.js("refreshHistory()")
    time.sleep(2.0)

    n = c.js("document.querySelectorAll('#hcList .hitem').length")
    check("历史列渲染出多步", (n or 0) >= 2, "%s 步" % n)
    cur = c.js("document.querySelectorAll('#hcList .hitem.cur').length")
    check("有当前步高亮", (cur or 0) == 1, "%s 个" % cur)
    shown = c.js("[...document.querySelectorAll('#hcList .hitem .t')]"
                 ".map(function(e){return e.textContent}).slice(0,6)")
    print("   步骤:", shown)

    print("\n=== 折叠 ===")
    c.js("document.getElementById('histToggle').click()")
    time.sleep(0.6)
    w1 = c.js("Math.round(document.getElementById('histCol')"
              ".getBoundingClientRect().width)")
    check("折叠后变窄", (w1 or 99) < 40, "%s px" % w1)
    c.js("document.getElementById('histToggle').click()")
    time.sleep(0.6)
    w2 = c.js("Math.round(document.getElementById('histCol')"
              ".getBoundingClientRect().width)")
    check("展开后恢复", (w2 or 0) > 100, "%s px" % w2)

    print("\n=== 跳转 ===")
    if (n or 0) >= 2:
        # `/api/history` 不返回顶层 index，当前位置看每步的 current 标记
        c.js("gotoHistory(0)")
        time.sleep(2.5)
        st = requests.get("http://127.0.0.1:8765/api/history/" + sid,
                          timeout=60).json()
        steps2 = st.get("steps") or []
        at0 = bool(steps2 and steps2[0].get("current"))
        check("能跳回第 1 步", at0,
              "current=" + str([s.get("current") for s in steps2]))
        # 跳转后 UI 高亮应当跟着走，而不是还停在最后一步
        cur_row = c.js("(document.querySelector('#hcList .hitem.cur')"
                       "||{}).textContent")
        check("历史列高亮同步到原图",
              isinstance(cur_row, str) and "原图" in cur_row, str(cur_row))

    # 截图放在最后：预设缩略图、历史、直方图都要先加载完，
    # 否则截到的是一排空框（第一次就是这样误判"预设列是空的"）
    c.shot(os.path.join(OUT, "ui_threecol.png"))

    print("\n=== 移动端 ===")
    c.call("Emulation.setDeviceMetricsOverride", width=390, height=844,
           deviceScaleFactor=2, mobile=True)
    c.call("Page.reload")
    time.sleep(3.5)
    check("移动端历史列自动收起",
          c.js("document.getElementById('histCol')"
               ".classList.contains('collapsed')") is True)
    c.call("Emulation.clearDeviceMetricsOverride")
    c.ws.close()

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
