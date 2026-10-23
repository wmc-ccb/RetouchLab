# -*- coding: utf-8 -*-
"""Tethered capture: source abstraction, arrival detection, auto chain.

The camera channel arrives last, so the properties that matter are the ones the
*folder* backend can prove today and that must keep holding when a real camera
is plugged in:

* a file that appears is reported **once** — no repeats across polls, so a
  shoot does not re-import the same frame every 2.5 seconds;
* a file still being written is **held back** — a camera or transfer tool
  writes to a temp name, and handing a half-written JPEG to the decoder is a
  real failure mode;
* the auto chain really runs: arrival → session → preset applied → exported,
  and the session's undo history shows the edit;
* one bad file does not lose the rest;
* a source that cannot trigger a shutter says so instead of pretending;
* the PTP source reports an honest reason (the camera currently enumerates as
  WPD/MTP, which cannot be shutter-controlled).
"""
import base64
import io
import json
import os
import shutil
import sys
import time

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BASE = "http://127.0.0.1:8765"
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
WORK = os.path.join(ROOT, "out", "_tether")
failures = []


def check(label, cond, detail=""):
    print("%-54s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


def main():
    from engine import tether as TT
    from engine.imgutil import load_image, fit_long_edge, save_image

    shutil.rmtree(WORK, ignore_errors=True)
    watch = os.path.join(WORK, "in")
    out_dir = os.path.join(WORK, "out")
    os.makedirs(watch, exist_ok=True)

    # ---- folder source semantics -----------------------------------------
    print("=== 文件夹源 ===")
    src = TT.FolderSource(watch, settle_s=1.0)
    check("空目录无新文件", src.poll() == [])
    img = fit_long_edge(load_image(SRC), 500)

    p1 = os.path.join(watch, "a.jpg")
    save_image(p1, img, 92)
    # a fresh file is held back until it stops changing
    check("刚写入的文件先被压住（防止读到半个文件）", src.poll() == [])
    time.sleep(1.4)
    first = src.poll()
    check("稳定后报告一次", len(first) == 1 and first[0] == p1,
          str([os.path.basename(x) for x in first]))
    check("再次轮询不重复报告", src.poll() == [])

    # overwriting the same name with new content must be reported again:
    # a camera writing shot_0001.jpg twice is normal
    time.sleep(1.2)
    save_image(p1, np.clip(img * 0.8, 0, 1), 92)
    time.sleep(1.4)
    again = src.poll()
    check("同名文件被改写会再次报告", len(again) == 1, str(len(again)))

    # a zero-byte file must never be reported
    empty = os.path.join(watch, "empty.jpg")
    open(empty, "wb").close()
    time.sleep(1.4)
    check("零字节文件不报告", all("empty" not in x for x in src.poll()))

    # non-images ignored
    with open(os.path.join(watch, "note.txt"), "w") as fh:
        fh.write("x")
    time.sleep(1.4)
    check("非图片文件被忽略",
          all(not x.endswith(".txt") for x in src.poll()))

    check("缺少目录时状态说明原因",
          "不存在" in TT.FolderSource(r"C:\nope\nope").status()["reason"])

    # a fresh source reports what is already there; arm() primes instead, so
    # pointing at an existing shoot does not import its back catalogue
    check("新源会报告已有文件（未 prime）",
          len(TT.FolderSource(watch, settle_s=0.0).poll()) >= 1)
    probe = TT.FolderSource(watch, settle_s=0.0)
    primed = probe.prime()
    check("prime 标记已有文件", primed >= 1, "%d 个" % primed)
    check("prime 之后不再报告已有文件", probe.poll() == [])

    # ---- other sources answer honestly -----------------------------------
    print("\n=== 其它后端 ===")
    ptp = TT.PtpSource()
    st = ptp.status()
    check("PTP 后端报告不可用", st["available"] is False)
    check("PTP 说明原因（WPD/MTP 无法控快门）", "PTP" in st["reason"])
    check("PTP 不能拍摄", ptp.capture()["supported"] is False)
    check("PTP 无实时取景", ptp.live_frame() is None)

    dcc = TT.DigiCamControlSource()
    d = dcc.status()
    check("digiCamControl 后端自行检测可执行文件",
          d["available"] == bool(d["exe"]), str(d["available"]))
    if not d["available"]:
        check("未安装时说明原因而不是报错", "未找到" in d["reason"])
        check("未安装时拍摄返回明确失败",
              dcc.capture()["ok"] is False)

    cams = TT.detect_cameras(force=True)
    print("    检测到的成像设备: %d 个" % len(cams))
    for c in cams[:4]:
        print("      %-18s class=%-6s vendor=%-8s can_capture=%s %s"
              % (c["name"][:18], c["class"], c.get("vendor") or "-",
                 c["can_capture"], c["note"]))
    check("系统级总线设备不被当作相机",
          all(c["class"] != "System" for c in cams),
          str([c["name"] for c in cams]))
    check("检测结果被缓存（第二次更快）",
          TT.detect_cameras() is TT.detect_cameras())
    z = [c for c in cams if c.get("vendor") == "Nikon" or "Z6" in c["name"]
         or "Z 6" in c["name"]]
    if z:
        check("Nikon 机身被枚举到", True, z[0]["name"])
        check("WPD 设备被正确标记为不能控快门",
              z[0]["can_capture"] is False, z[0]["note"])
    else:
        print("    （当前没有连接相机 —— 插上后会出现在这里）")

    # ---- API + auto chain ------------------------------------------------
    print("\n=== 状态端点耗时 ===")
    try:
        requests.get(BASE + "/api/health", timeout=60)
    except Exception:
        print("\n服务未运行，跳过 API")
        return 0 if not failures else 1

    t0 = time.time()
    first = requests.get(BASE + "/api/tether/status", timeout=60).json()
    dt_first = time.time() - t0
    t0 = time.time()
    requests.get(BASE + "/api/tether/status", timeout=60)
    dt_second = time.time() - t0
    print("    首次 %.2fs · 再次 %.2fs" % (dt_first, dt_second))
    check("缓存后状态端点变快", dt_second < max(0.35, dt_first * 0.5),
          "%.2f -> %.2f" % (dt_first, dt_second))

    print("\n=== API 与自动链路 ===")
    st = first
    check("状态含后端与相机列表",
          "kind" in st and isinstance(st.get("cameras"), list))

    cfg = requests.post(BASE + "/api/tether/config", json={
        "kind": "folder", "path": watch, "settle_s": 1.0,
        "auto_import": True, "auto_preset": "natural",
        "auto_export_dir": out_dir}, timeout=60).json()
    check("配置成功", cfg.get("ok") is True)
    check("开启时 baseline 已有文件（不导入旧照片）",
          cfg.get("primed", 0) >= 1, "primed=%s" % cfg.get("primed"))

    bad = requests.post(BASE + "/api/tether/config",
                        json={"kind": "nope"}, timeout=30)
    check("非法后端 400", bad.status_code == 400, str(bad.status_code))

    # drop two frames in, one of them broken
    save_image(os.path.join(watch, "shot1.jpg"), img, 92)
    save_image(os.path.join(watch, "shot2.jpg"),
               np.clip(img * 1.1, 0, 1), 92)
    with open(os.path.join(watch, "broken.jpg"), "wb") as fh:
        fh.write(b"not a jpeg")
    time.sleep(1.6)

    got = []
    for _ in range(6):
        j = requests.post(BASE + "/api/tether/poll", timeout=300).json()
        got.extend(j.get("new", []))
        if len(got) >= 3:
            break
        time.sleep(1.0)
    names = [g["name"] for g in got]
    check("只收到开启后新增的 3 张", len(got) == 3, str(names))
    ok = [g for g in got if g["ok"]]
    bad_ones = [g for g in got if not g["ok"]]
    check("损坏文件被单独记为失败", len(bad_ones) == 1,
          str([g.get("error", "")[:40] for g in bad_ones]))
    check("其余 2 张成功", len(ok) == 2, "%d 张" % len(ok))
    check("成功项带 session", all(g.get("session") for g in ok))
    check("自动预设被记录", all(g.get("preset") == "natural" for g in ok))

    outs = os.listdir(out_dir) if os.path.isdir(out_dir) else []
    check("自动导出 2 个文件", len(outs) == 2, str(outs))

    # the auto-applied edit must be visible in that session's history
    sid = ok[0]["session"]
    h = requests.get(BASE + "/api/history/" + sid, timeout=30).json()
    labels = [s["label"] for s in h["steps"]]
    check("自动编辑进入撤销栈", any("联机" in x for x in labels), str(labels))

    # repeat polling must not re-import
    more = requests.post(BASE + "/api/tether/poll", timeout=60).json()
    check("再次轮询不重复导入", more.get("new") == [],
          str(len(more.get("new", []))))

    ls = requests.get(BASE + "/api/tether/live", timeout=60)
    check("无实时取景时返回 204", ls.status_code == 204, str(ls.status_code))

    cl = requests.post(BASE + "/api/tether/clear", timeout=30).json()
    check("清空记录", cl.get("ok") is True)
    requests.post(BASE + "/api/tether/stop", timeout=30)

    # ---- browser ---------------------------------------------------------
    print("\n=== UI ===")
    try:
        import websocket
        t = [x for x in requests.get("http://127.0.0.1:9223/json", timeout=10)
             .json() if x.get("type") == "page" and "8765" in x.get("url", "")]
        if not t:
            print("  浏览器未连接，跳过")
        else:
            ws = websocket.create_connection(t[0]["webSocketDebuggerUrl"],
                                             timeout=60)
            n = [0]

            def call(m, **p):
                n[0] += 1
                ws.send(json.dumps({"id": n[0], "method": m, "params": p}))
                while True:
                    msg = json.loads(ws.recv())
                    if msg.get("id") == n[0]:
                        return msg.get("result", {})

            def js(e):
                return call("Runtime.evaluate", expression=e,
                            returnByValue=True, awaitPromise=True
                            ).get("result", {}).get("value")

            call("Page.reload")
            time.sleep(4)
            js("selectCat('\u8054\u673a')")
            time.sleep(2.5)
            check("联机面板可打开",
                  js("document.getElementById('sideCat').textContent") == "联机")
            check("三种数据源按钮",
                  (js("document.querySelectorAll('[data-src]').length") or 0) == 3)
            check("监视目录输入框", bool(js("!!document.getElementById('tethPath')")))
            check("自动导出输入框", bool(js("!!document.getElementById('tethOut')")))
            check("自动预设选项",
                  (js("document.querySelectorAll('[data-ap]').length") or 0) >= 7)
            check("开始/停止/拍摄按钮",
                  bool(js("!!document.getElementById('tethArm') && "
                          "!!document.getElementById('tethStop') && "
                          "!!document.getElementById('tethShoot')")))
            check("实时取景画布", bool(js("!!document.getElementById('tethLive')")))
            check("状态行显示相机与后端",
                  "数据源" in (js("document.getElementById('tethInfo')"
                                  ".textContent") or ""),
                  (js("document.getElementById('tethInfo').textContent") or "")[:60])
            n[0] += 1
            ws.send(json.dumps({"id": n[0], "method": "Page.captureScreenshot",
                                "params": {"format": "png"}}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == n[0]:
                    with open(os.path.join(ROOT, "out", "ui_tether.png"),
                              "wb") as fh:
                        fh.write(base64.b64decode(msg["result"]["data"]))
                    break
            ws.close()
    except Exception as exc:
        print("  UI 检查跳过：%s" % str(exc)[:80])

    print()
    if failures:
        print("FAILED: %d -> %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
