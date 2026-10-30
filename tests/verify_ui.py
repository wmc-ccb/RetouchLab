# -*- coding: utf-8 -*-
"""Verify the rebuilt UI in a real browser over CDP.

Checks, per breakpoint:
* desktop (1600px): icon rail visible, right panel visible, film strip hidden,
  the params panel renders 100+ sliders for a group, formulas panel lists 5;
* mobile (390x844): rail hidden, bottom tab bar visible, drawer opens on
  category selection.
"""
import base64
import json
import os
import sys
import time

import requests
import websocket

CDP_HTTP = "http://127.0.0.1:9223"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "out")


class CDP:
    def __init__(self, url):
        self.ws = websocket.create_connection(url, timeout=30)
        self.i = 0

    def call(self, method, **params):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": method,
                                 "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.i:
                return msg.get("result", {})

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr,
                      returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    def shot(self, path):
        r = self.call("Page.captureScreenshot", format="png")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(r["data"]))


def get_ws():
    for t in requests.get(CDP_HTTP + "/json", timeout=10).json():
        if t.get("type") == "page" and "8765" in t.get("url", ""):
            return t["webSocketDebuggerUrl"]
    raise RuntimeError("page not found")


def main():
    c = CDP(get_ws())
    print("connected")
    failures = []

    def check(label, cond, detail=""):
        print("%-52s %s %s" % (label, "OK" if cond else "FAIL", detail))
        if not cond:
            failures.append(label)

    # ---- desktop ---------------------------------------------------------
    c.call("Emulation.setDeviceMetricsOverride", width=1600, height=1000,
           deviceScaleFactor=1, mobile=False)
    c.call("Page.reload")
    time.sleep(3.5)

    rail_vis = c.js("getComputedStyle(document.getElementById('rail')).display")
    side_vis = c.js("getComputedStyle(document.getElementById('side')).display")
    film_vis = c.js("getComputedStyle(document.getElementById('filmbox')).display")
    tabs = c.js("document.querySelectorAll('#rail .rbtn').length")
    check("desktop: icon rail visible", rail_vis in ("flex", "block"), rail_vis)
    check("desktop: right panel visible", side_vis == "flex", side_vis)
    check("desktop: film strip hidden before upload", film_vis == "none",
          film_vis)
    cols2 = c.js("document.getElementById('rail').classList.contains('cols2')")
    check("desktop: rail uses two-column layout", cols2 is True, str(cols2))
    # 二级导航：一级 rail 只列大组（预设直达 + 13 组 = 14）。
    # 组数由 /api/cats 的分组表决定，不写死。
    tree = requests.get("http://127.0.0.1:8765/api/cats",
                        timeout=60).json().get("tree") or {}
    expected_groups = 1 + len(tree)
    check("desktop: icon rail has all categories", tabs == expected_groups,
          "%d buttons (expect %d = 预设 + %d 组)"
          % (tabs, expected_groups, len(tree)))

    # open a params group and count sliders
    n_sliders = c.js("""
      (function(){
        selectCat('光影');
        return document.querySelectorAll('#sliders input[type=range]').length;
      })()
    """)
    check("desktop: 光影 group renders sliders", (n_sliders or 0) >= 3,
          "%s sliders" % n_sliders)

    # formulas panel
    c.js("selectCat('配方')")
    time.sleep(1.2)
    fnames = c.js("[...document.querySelectorAll('#sliders .formula b')]"
                  ".map(e => e.textContent)")
    check("desktop: formulas panel lists 5", isinstance(fnames, list)
          and len(fnames) == 5, str(fnames)[:80])
    c.shot(os.path.join(OUT, "ui2_desktop_formulas.png"))

    # wheel panel
    c.js("selectCat('色轮')")
    time.sleep(0.8)
    fams = c.js("[...document.querySelectorAll('#sliders .wheelBtn')].length")
    check("desktop: wheel shows 6 families", fams == 6, str(fams))
    c.shot(os.path.join(OUT, "ui2_desktop_wheel.png"))
    c.shot(os.path.join(OUT, "ui2_desktop.png"))

    # ---- mobile ----------------------------------------------------------
    c.call("Emulation.setDeviceMetricsOverride", width=390, height=844,
           deviceScaleFactor=2, mobile=True)
    c.call("Page.reload")
    time.sleep(3.0)

    rail_m = c.js("getComputedStyle(document.getElementById('rail')).display")
    mbar_vis = c.js("getComputedStyle(document.getElementById('mbar')).display")
    side_pos = c.js("getComputedStyle(document.getElementById('side')).position")
    check("mobile: rail hidden", rail_m == "none", rail_m)
    check("mobile: bottom tab bar visible", mbar_vis == "flex", mbar_vis)
    check("mobile: side is a drawer (fixed)", side_pos == "fixed", side_pos)

    c.js("selectCat('光影')")
    time.sleep(0.5)
    drawer = c.js("document.getElementById('side').classList.contains('open')")
    check("mobile: drawer opens on category", bool(drawer))
    c.shot(os.path.join(OUT, "ui2_mobile_drawer.png"))
    c.js("closeDrawer()")
    time.sleep(0.4)
    c.shot(os.path.join(OUT, "ui2_mobile_home.png"))

    print()
    if failures:
        print("FAILED:", failures)
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
