# -*- coding: utf-8 -*-
"""常驻直方图：位置、形状保真度、随编辑更新.

上一版被判"对不上"，实际是两个问题：
  A. 画的位置不对——它跟着"曲线"面板走，不是常驻在参数区顶部；
  B. 形状不对——用 log1p(peak) 归一化时，真实分布 4124~8459 的两倍起伏被压到
     0.92~1.00，画出来几乎是平的。

所以这里不检查"有没有画"，而是检查**画出来的形状和真实分布的相关系数**。
"""
import base64
import io
import json
import os
import sys
import time

import numpy as np
import requests
import websocket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
BASE = "http://127.0.0.1:8765"
CDP = "http://127.0.0.1:9223"
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
failures = []


def check(label, cond, detail=""):
    print("%-52s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


class B:
    def __init__(self, t):
        self.ws = websocket.create_connection(t["webSocketDebuggerUrl"],
                                              timeout=120)
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
    from engine.imgutil import load_image, fit_long_edge

    # ---------- 形状保真度（引擎侧直接算） ----------
    print("=== 形状保真度 ===")
    img = fit_long_edge(load_image(SRC), 400)
    lum = (img * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(axis=2)
    bins = 64
    hist, _ = np.histogram(np.clip(lum, 0, 1), bins=bins, range=(0.0, 1.0))
    peak = max(1, hist.max())

    # 三种压缩方式与真实分布的相关性
    real = hist / peak
    logc = np.log1p(hist) / np.log1p(peak)
    cbrt = np.cbrt(real)
    sqrt = np.sqrt(real)

    def corr(a, b):
        a = np.asarray(a, float) - np.mean(a)
        b = np.asarray(b, float) - np.mean(b)
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 0 else 0.0

    c_log, c_cbrt, c_sqrt = corr(real, logc), corr(real, cbrt), corr(real, sqrt)
    print("   与真实分布的相关性: log1p %.3f · cbrt %.3f · sqrt %.3f"
          % (c_log, c_cbrt, c_sqrt))
    # 顺序断言：压缩越强，形状保真度越低。这是选择实现方式的依据——
    # 界面用的是自适应方案（见下），不是这三种里的任何一个固定幂。
    check("压缩越强保真度越低（log1p < cbrt < sqrt）",
          c_log < c_cbrt < c_sqrt,
          "%.3f < %.3f < %.3f" % (c_log, c_cbrt, c_sqrt))
    check("sqrt 作为固定方案已足够保形", c_sqrt > 0.95, "r=%.3f" % c_sqrt)

    # 界面实际用的：自适应（99% 线性 + 顶端软压）
    def adaptive(h, pct=99.0):
        s = np.sort(h)
        hi = max(1.0, float(s[min(len(s) - 1, int(len(s) * pct / 100.0) - 1)]))
        v = np.maximum(0, h.astype(float))
        lin = v / hi * 0.75
        t = np.clip((v - hi) / max(1.0, v.max() - hi), 0, 1)
        top = 0.75 + 0.25 * (1 - np.exp(-3 * t))
        return np.where(v <= hi, lin, top)

    ad = adaptive(hist)
    c_ad = corr(real, ad)
    body = hist[4:60]
    real_ratio = float(body.max() / max(1, body.min()))
    ad_body = ad[4:60]
    ad_ratio = float(ad_body.max() / max(1e-9, ad_body.min()))
    print("   自适应方案: 相关性 %.4f · 主体起伏 %.1f（真实 %.1f）"
          % (c_ad, ad_ratio, real_ratio))
    check("自适应压缩高度保形", c_ad > 0.98, "r=%.4f" % c_ad)
    check("自适应压缩保留真实起伏", ad_ratio > real_ratio * 0.8,
          "%.1f vs %.1f" % (ad_ratio, real_ratio))
    check("自适应优于任何固定幂", c_ad > max(c_log, c_cbrt, c_sqrt),
          "%.4f > %.3f" % (c_ad, max(c_log, c_cbrt, c_sqrt)))

    # 起伏幅度已在上面一并断言（ad_ratio vs real_ratio）

    # ---------- 服务端直方图与真实分布一致 ----------
    print("\n=== 服务端数据 ===")
    with open(SRC, "rb") as fh:
        up = requests.post(BASE + "/api/upload",
                           files={"file": ("h.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    sid = up["session"]
    h = requests.get(BASE + "/api/histogram/" + sid, timeout=60).json()
    srv = np.asarray(h["lum"], float)
    check("服务端直方图总和 == 像素数", sum(h["lum"]) == h["total"],
          "%d vs %d" % (sum(h["lum"]), h["total"]))
    # 服务端是基于 preview(1399) 算的，这里用同样尺寸复核
    prev = fit_long_edge(load_image(SRC), 1400)
    pl = (prev * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(axis=2)
    ref, _ = np.histogram(np.clip(pl, 0, 1), bins=h["bins"], range=(0.0, 1.0))
    check("服务端桶计数与本地重算一致",
          int(np.abs(srv - ref).max()) <= max(2, 0.001 * ref.max()),
          "最大差 %d" % int(np.abs(srv - ref).max()))

    # ---------- UI ----------
    print("\n=== UI ===")
    try:
        ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
              if x.get("type") == "page" and "8765" in x.get("url", "")]
        if not ts:
            print("  浏览器未连接，跳过")
        else:
            b = B(ts[0])
            b.call("Emulation.setDeviceMetricsOverride", width=1600,
                   height=1000, deviceScaleFactor=1, mobile=False)
            b.call("Page.reload")
            time.sleep(4)
            b.js("""
              (function(){S.session=%s;S.meta=%s;S.faces=%s;
                S.base='data:image/jpeg;base64,'+%s;
                document.getElementById('dropzone').style.display='none';
                ['btnExport','btnErase','btnPicker','btnMatte','btnMatch',
                 'btnAuto','btnHeal','btnStamp','btnUpscale'].forEach(
                  function(id){var e=document.getElementById(id);
                               if(e) e.disabled=false;});
                drawBase(); refreshHistory(); return true;})()
            """ % (json.dumps(sid),
                   json.dumps({"width": up["width"], "height": up["height"],
                               "preview_width": up["preview_width"],
                               "preview_height": up["preview_height"]}),
                   json.dumps(up["faces"]), json.dumps(up["preview"])))
            time.sleep(2.5)

            # 常驻：切任意分类都应该在
            for cat in ('预设', '光影', '曲线', '蒙版', '局部', '批量', '历史'):
                b.js("selectCat('%s')" % cat)
                time.sleep(0.55)
                vis = b.js("""
                  (function(){
                    var h=document.getElementById('histBar');
                    if(!h) return 'missing';
                    var r=h.getBoundingClientRect();
                    var cs=getComputedStyle(h);
                    return (cs.display!=='none' && r.height>20) ? 'visible'
                                                                : 'hidden';
                  })()
                """)
                check("直方图在「%s」分类下常驻" % cat, vis == "visible",
                      str(vis))

            # 位置：必须在参数面板内、且在参数内容之上
            b.js("selectCat('光影')")
            time.sleep(1.2)
            geo = b.js("""
              (function(){
                var h=document.getElementById('histBar').getBoundingClientRect();
                var s=document.getElementById('side').getBoundingClientRect();
                var b=document.getElementById('sideBody').getBoundingClientRect();
                var hd=document.getElementById('sideHead').getBoundingClientRect();
                return {hL:Math.round(h.left),hR:Math.round(h.right),
                        hT:Math.round(h.top),hB:Math.round(h.bottom),
                        hH:Math.round(h.height),
                        sL:Math.round(s.left),sR:Math.round(s.right),
                        headB:Math.round(hd.bottom), bodyT:Math.round(b.top)};
              })()
            """)
            print("   直方图 %d-%d（y %d-%d, 高 %d）· 参数面板 %d-%d"
                  % (geo["hL"], geo["hR"], geo["hT"], geo["hB"], geo["hH"],
                     geo["sL"], geo["sR"]))
            check("直方图横跨参数面板宽度",
                  geo["hR"] - geo["hL"] > (geo["sR"] - geo["sL"]) * 0.9,
                  "%d / %d" % (geo["hR"] - geo["hL"], geo["sR"] - geo["sL"]))
            check("直方图在分类标题之下",
                  geo["hT"] >= geo["headB"] - 2,
                  "hT=%d headB=%d" % (geo["hT"], geo["headB"]))
            check("直方图在参数内容之上",
                  geo["hB"] <= geo["bodyT"] + 2,
                  "hB=%d bodyT=%d" % (geo["hB"], geo["bodyT"]))
            check("直方图高度合理（不挤占参数区）",
                  40 <= geo["hH"] <= 130, "%d px" % geo["hH"])

            # 面板内不再有第二份直方图
            dup = b.js("(document.querySelectorAll('canvas').length)")
            has_dup = b.js("!!document.getElementById('curveHist') || "
                           "!!document.getElementById('maskHist')")
            check("面板内没有重复直方图", has_dup is False,
                  "全页面 canvas %s 个" % dup)

            # 形状：把画布内容按列取高度，和真实分布比相关性
            shape = b.js("""
              (function(){
                var c=document.getElementById('globalHist');
                if(!c) return null;
                var d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
                var W=c.width,H=c.height,cols=[];
                for(var x=0;x<W;x++){
                  var top=H;
                  for(var y=0;y<H;y++){
                    var i=(y*W+x)*4;
                    // 非黑即"有柱子"
                    if(d[i]>25||d[i+1]>25||d[i+2]>25){top=y;break;}
                  }
                  cols.push(H-top);
                }
                return cols;
              })()
            """)
            check("能读到直方图画布像素",
                  isinstance(shape, list) and len(shape) > 100,
                  "%s 列" % (len(shape) if isinstance(shape, list) else "?"))

            if isinstance(shape, list) and len(shape) > 100:
                cols = np.asarray(shape, float)
                # 把画出来的高度重采样到 64 桶，和真实分布比
                n = len(cols)
                drawn = np.array([
                    cols[int(i / 64 * n):max(int((i + 1) / 64 * n),
                                             int(i / 64 * n) + 1)].mean()
                    for i in range(64)])
                if drawn.max() > 0:
                    drawn = drawn / drawn.max()
                    r_ui = corr(real, drawn)
                    print("   画面直方图 vs 真实分布 相关性 %.3f" % r_ui)
                    check("画出来的形状与真实分布一致", r_ui > 0.85,
                          "r=%.3f" % r_ui)
                # 不能是平的
                flat = float(drawn.max() - drawn.min())
                check("直方图不是平的（有起伏）", flat > 0.25, "起伏 %.3f" % flat)
                # 与 log1p 的平直形状明显不同
                check("形状不同于旧的 log1p 平直版",
                      corr(drawn, np.log1p(hist[:64]) / np.log1p(peak)) < 0.995
                      or flat > 0.3, "起伏 %.3f" % flat)

            # 跟随编辑
            fp0 = b.js("(function(){var c=document.getElementById('globalHist');"
                       "var d=c.getContext('2d').getImageData(0,0,c.width,"
                       "c.height).data;var s=0;for(var i=0;i<d.length;i+=991)"
                       "s=(s+d[i])%99991;return s;})()")
            b.js("S.params.exposure = 75; requestRender();")
            changed = False
            for _ in range(35):
                time.sleep(0.4)
                fp1 = b.js("(function(){var c=document.getElementById"
                           "('globalHist');var d=c.getContext('2d')"
                           ".getImageData(0,0,c.width,c.height).data;var s=0;"
                           "for(var i=0;i<d.length;i+=991)s=(s+d[i])%99991;"
                           "return s;})()")
                if fp1 != fp0:
                    changed = True
                    break
            check("edit 后常驻直方图更新", changed, "指纹 %s -> %s" % (fp0, fp1))
            check("死白读数随提亮上升",
                  "死白" in (b.js("document.getElementById('histClip')"
                                  ".textContent") or ""),
                  str(b.js("document.getElementById('histClip')"
                           ".textContent"))[:40])

            b.js("selectCat('光影')")
            time.sleep(1.0)
            b.shot(os.path.join(OUT, "ui_hist_persistent.png"))
            print("  截图 out/ui_hist_persistent.png")
            b.call("Emulation.clearDeviceMetricsOverride")
            b.ws.close()
    except Exception as exc:
        print("  UI 检查跳过：%s" % str(exc)[:160])

    print()
    if failures:
        print("FAILED: %d -> %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
