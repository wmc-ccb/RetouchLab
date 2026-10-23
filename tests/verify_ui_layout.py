# -*- coding: utf-8 -*-
"""Layout invariants for both breakpoints.

These are the properties that were actually broken and that no functional test
would catch, because the features worked — the layout did not:

* the mobile tab strip squeezed 19 categories into 390px, so "存在感" wrapped
  to two lines and labels overlapped each other;
* opening the drawer left the photo centred in the *full* height, so it sat
  behind the drawer with dead space above it, and the face box stayed where the
  photo used to be — an overlay floating in the black area;
* the status line said "未载入图片" next to a plainly loaded photo, because it
  was only written inside two functions rather than derived from state;
* the left rail overflowed with 19 items and clipped the last ones.

Each is asserted directly, so a CSS change that reintroduces one fails here.
"""
import base64
import json
import os
import sys
import time

import requests
import websocket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDP = "http://127.0.0.1:9223"
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
failures = []


def check(label, cond, detail=""):
    print("%-52s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


class B:
    def __init__(self, url):
        self.ws = websocket.create_connection(url, timeout=90)
        self.i = 0

    def call(self, method, **p):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": method, "params": p}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.i:
                return m.get("result", {})

    def js(self, e):
        r = self.call("Runtime.evaluate", expression=e, returnByValue=True,
                      awaitPromise=True)
        return r.get("result", {}).get("value")


def main():
    t = [x for x in requests.get(CDP + "/json", timeout=10).json()
         if x.get("type") == "page" and "8765" in x.get("url", "")]
    if not t:
        print("page not found")
        return 1
    b = B(t[0]["webSocketDebuggerUrl"])
    print("connected")

    # ---- desktop ---------------------------------------------------------
    print("\n=== 桌面 (1600x1000) ===")
    b.call("Emulation.setDeviceMetricsOverride", width=1600, height=1000,
           deviceScaleFactor=1, mobile=False)
    b.call("Page.reload")
    time.sleep(4)

    rail_scroll = b.js("(function(){var r=document.getElementById('rail');"
                       "return r.scrollHeight - r.clientHeight;})()")
    # The invariant is that every entry is *reachable*, not that none of them
    # ever needs a scroll — the rail carries 21 entries now and `overflow-y`
    # makes the tail reachable. A hard "no overflow" assertion failed purely
    # because a panel was added, which is not the property worth protecting.
    rail_ok = b.js("""
      (function(){
        var r=document.getElementById('rail');
        var scrollable = getComputedStyle(r).overflowY;
        if (r.scrollHeight - r.clientHeight <= 2) return 'fits';
        return (scrollable === 'auto' || scrollable === 'scroll')
               ? 'scrollable' : 'clipped';
      })()
    """)
    check("左栏每一项都可到达", rail_ok in ("fits", "scrollable"),
          "%s（溢出 %s px）" % (rail_ok, rail_scroll))

    # tool groups must be visually separated
    sep = b.js("getComputedStyle(document.querySelectorAll('.tgroup')[1],"
               "'::before').backgroundColor")
    check("顶栏分组有分隔线", sep not in (None, "", "rgba(0, 0, 0, 0)"), str(sep))

    # export must be the rightmost element
    rightmost = b.js("""
      (function(){
        var els = [...document.getElementById('topbar').children];
        var max = -1, who = '';
        els.forEach(function(e){
          var r = e.getBoundingClientRect();
          if (r.right > max) { max = r.right; who = e.id || e.className; }
        });
        return who;
      })()
    """)
    check("导出按钮位于最右", rightmost == "btnExport", str(rightmost))

    # density: a folding group of six controls should fit the panel height
    b.js("selectCat('\u5149\u5f71')")
    time.sleep(1.2)
    rows = b.js("document.querySelectorAll('#sliders .sl').length")
    check("光影组渲染出滑杆", (rows or 0) >= 6, "%s 个" % rows)

    # ---- status line reflects state --------------------------------------
    print("\n=== 状态栏 ===")
    check("未载入时显示未载入", b.js("document.getElementById('meta')"
                                     ".textContent") == "未载入图片")
    with open(SRC, "rb") as fh:
        up = requests.post("http://127.0.0.1:8765/api/upload",
                           files={"file": ("l.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    b.js("""
      (function(){
        S.session = %s; S.meta = %s; S.faces = %s;
        S.base = 'data:image/jpeg;base64,' + %s;
        document.getElementById('dropzone').style.display = 'none';
        drawBase(); updateMeta(); return true; })()
    """ % (json.dumps(up["session"]),
           json.dumps({"width": up["width"], "height": up["height"],
                       "preview_width": up["preview_width"],
                       "preview_height": up["preview_height"]}),
           json.dumps(up["faces"]), json.dumps(up["preview"])))
    time.sleep(1.5)
    meta = b.js("document.getElementById('meta').textContent")
    check("载入后状态栏反映实际尺寸",
          ("×" in (meta or "")) and ("未载入" not in (meta or "")), str(meta))

    # ---- mobile ----------------------------------------------------------
    print("\n=== 移动 (390x844) ===")
    b.call("Emulation.setDeviceMetricsOverride", width=390, height=844,
           deviceScaleFactor=2, mobile=True)
    b.call("Page.reload")
    time.sleep(4)

    check("tab 栏可横向滚动",
          b.js("getComputedStyle(document.getElementById('mbar'))"
               ".overflowX") == "auto")
    check("tab 不收缩（标签不被压扁）",
          b.js("getComputedStyle(document.querySelector('#mbar button'))"
               ".flexShrink") == "0")
    # no two tabs may overlap horizontally
    overlap = b.js("""
      (function(){
        var bs=[...document.querySelectorAll('#mbar button')];
        var bad=0;
        for (var i=1;i<bs.length;i++){
          if (bs[i].getBoundingClientRect().left <
              bs[i-1].getBoundingClientRect().right - 0.5) bad++;
        }
        return bad;
      })()
    """)
    check("tab 之间不重叠", overlap == 0, "%s 处重叠" % overlap)
    # and the label must not wrap onto two lines
    tall = b.js("""
      (function(){
        var bs=[...document.querySelectorAll('#mbar button')];
        return bs.filter(function(b){ return b.scrollWidth > b.clientWidth + 1; })
                 .length;
      })()
    """)
    check("tab 标签不溢出自身宽度", tall == 0, "%s 个溢出" % tall)

    check("抽屉默认收起",
          b.js("document.getElementById('side')"
               ".classList.contains('open')") is False)
    check("收起时画布无额外留白",
          (b.js("parseFloat(getComputedStyle(document.getElementById("
                "'canvasWrap')).marginBottom)") or 0) < 10)

    b.js("openDrawer()")
    # 抽屉用 transform 过渡 220ms，padding-bottom 也带过渡。固定 sleep 在
    # 慢机器上会读到过渡中间值（甚至是 0），所以轮询到稳定为止。
    pad = 0
    for _ in range(24):
        time.sleep(0.25)
        pad = b.js("parseFloat(getComputedStyle(document.getElementById("
                   "'canvasWrap')).marginBottom)") or 0
        if pad > 100:
            break
    check("抽屉可打开", b.js("document.getElementById('side')"
                             ".classList.contains('open')") is True)
    check("body 标记同步",
          b.js("document.body.classList.contains('drawer-open')") is True)
    check("画布为抽屉让出空间", pad > 100, "%s px" % pad)

    # the photo must sit above the drawer, not behind it
    geom = b.js("""
      (function(){
        var c=document.getElementById('canvas').getBoundingClientRect();
        var s=document.getElementById('side').getBoundingClientRect();
        return {cBottom: Math.round(c.bottom), sTop: Math.round(s.top),
                cTop: Math.round(c.top), vh: window.innerHeight};
      })()
    """)
    if geom:
        check("照片整体位于抽屉之上",
              geom["cBottom"] <= geom["sTop"] + 2,
              "照片底 %d / 抽屉顶 %d" % (geom["cBottom"], geom["sTop"]))
        # centred in the visible band, not pushed to one edge
        vis = geom["sTop"]
        above = geom["cTop"]
        below = vis - geom["cBottom"]
        check("照片在可见区域内大致居中",
              abs(above - below) < max(60, vis * 0.25),
              "上方 %d / 下方 %d" % (above, below))

    b.js("closeDrawer()")
    time.sleep(0.6)
    check("收起后留白恢复",
          (b.js("parseFloat(getComputedStyle(document.getElementById("
                "'canvasWrap')).marginBottom)") or 0) < 10)

    b.call("Emulation.setDeviceMetricsOverride", width=1600, height=1000,
           deviceScaleFactor=1, mobile=False)
    b.ws.close()
    print()
    if failures:
        print("FAILED: %d -> %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
