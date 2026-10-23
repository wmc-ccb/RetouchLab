# -*- coding: utf-8 -*-
"""Undo/redo and batch: engine-level exactness, then the API state machine.

Two levels, because they fail differently.

* The **engine** test proves the snapshot round trip is lossless. This is the
  part a user notices as "undo degraded my photo", and it is the one place
  where PNG-versus-JPEG storage actually matters.
* The **API** test proves the state machine: the seeded original is reachable
  and cannot be walked past, the redo tail is dropped by a new edit, and a
  refused action is a 409 rather than a silent no-op.

An earlier version compared the undo response against the upload response and
reported a 33-level difference. That was a test bug: the two endpoints encode
at different JPEG qualities (90 vs 88), so the comparison measured compression,
not history. The tell was that redo — where both sides are q90 — matched to
0.000000. The API check below therefore compares like with like.
"""
import io
import base64
import os
import sys

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.imgutil import load_image, fit_long_edge, save_image
from engine.history import History

BASE = "http://127.0.0.1:8765"
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
failures = []


def check(label, cond, detail=""):
    print("%-58s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


def decode(b64):
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64)))
                      .convert("RGB"), np.float32) / 255.0


# ---------------------------------------------------------------------------
# engine: the snapshot round trip must be lossless
# ---------------------------------------------------------------------------

def engine_checks(img):
    print("=== 引擎：快照往返（8 位量化，误差 ≤ 0.5/255）===")
    # The snapshot is 8-bit PNG: lossless as a container, quantised to 8 bits
    # per channel. So the invariant is "restores to within half a display
    # level", not "bit-identical float" — and crucially, no drift across
    # repeated cycles, because each state is stored once rather than re-derived.
    Q = 0.5 / 255.0 + 1e-6            # one quantisation step

    h = History(img, depth=6)
    check("初始状态 = 原图", h.current_label() == "原图")
    check("初始不可撤销", not h.can_undo() and not h.can_redo())

    # a deliberately harsh image: gradients plus hard edges, which is where a
    # lossy snapshot format shows up
    a = np.clip(img * 1.15, 0, 1)
    b = np.clip(img ** 0.9, 0, 1)
    h.push(a, "提亮")
    h.push(b, "压暗")
    check("历史长度 = 3", len(h.states) == 3, "%d" % len(h.states))

    r1, lb1 = h.undo()
    e1 = float(np.abs(r1 - a).max())
    check("撤销一次回到 提亮（≤0.5/255）", lb1 == "提亮" and e1 <= Q,
          "max %.2e" % e1)
    r2, lb2 = h.undo()
    e2 = float(np.abs(r2 - img).max())
    check("撤销两次回到 原图（≤0.5/255）", lb2 == "原图" and e2 <= Q,
          "max %.2e" % e2)
    check("到底后不可再撤销", not h.can_undo())
    check("到底后可重做", h.can_redo())

    # repeated undo/redo must not drift: each state is stored once, not
    # re-derived, so the hundredth cycle returns the same bytes as the first
    for _ in range(5):
        h.redo()
        h.undo()
    r3, _ = h.redo()
    e3 = float(np.abs(r3 - a).max())
    check("反复撤销/重做 5 轮无漂移", e3 <= Q, "max %.2e" % e3)

    # the redo tail is dropped by a new edit
    h.undo()
    h.push(np.clip(img * 0.8, 0, 1), "新编辑")
    check("新编辑丢弃重做链", not h.can_redo())
    check("新编辑后末尾标签正确", h.current_label() == "新编辑")

    # depth bound keeps the original reachable
    h2 = History(img, depth=4)
    for i in range(10):
        h2.push(np.clip(img * (1.0 - i * 0.02), 0, 1), "第%d步" % i)
    check("深度受限但长度不超", len(h2.states) <= 4, "%d" % len(h2.states))
    check("深度受限后仍可撤销", h2.can_undo())
    print("   历史占用: %.2f MB（%d 个快照）"
          % (h2.bytes_used / 1048576, len(h2.states)))


# ---------------------------------------------------------------------------
# api: state machine
# ---------------------------------------------------------------------------

def api_checks():
    print("\n=== API：状态机 ===")
    img = fit_long_edge(load_image(SRC), 700)
    save_image(os.path.join(ROOT, "out", "_hist_src.jpg"), img, 95)
    with open(os.path.join(ROOT, "out", "_hist_src.jpg"), "rb") as fh:
        up = requests.post(BASE + "/api/upload",
                           files={"file": ("h.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    sid = up["session"]

    h = requests.get(BASE + "/api/history/" + sid, timeout=30).json()
    check("上传后无历史", not h["can_undo"] and not h["can_redo"])
    check("空历史撤销返回 409",
          requests.post(BASE + "/api/undo", json={"session": sid},
                        timeout=30).status_code == 409)
    check("空历史重做返回 409",
          requests.post(BASE + "/api/redo", json={"session": sid},
                        timeout=30).status_code == 409)

    # one committed wheel edit
    r1 = requests.post(BASE + "/api/wheel",
                       json={"session": sid, "family": "red", "hue": 20,
                             "sat": 30, "commit": True}, timeout=300).json()
    s1 = decode(r1["image"])
    check("已提交的色轮带 committed 标记", r1.get("committed") is True)

    # a preview-only wheel edit must NOT become a step
    n_before = len(requests.get(BASE + "/api/history/" + sid,
                                timeout=30).json()["steps"])
    requests.post(BASE + "/api/wheel",
                  json={"session": sid, "family": "blue", "hue": 10,
                        "commit": False}, timeout=300)
    n_after = len(requests.get(BASE + "/api/history/" + sid,
                               timeout=30).json()["steps"])
    check("预览（未提交）不产生历史步骤", n_before == n_after,
          "%d -> %d" % (n_before, n_after))

    # a second committed edit
    requests.post(BASE + "/api/greypoint",
                  json={"session": sid, "x": 300, "y": 400, "commit": True},
                  timeout=300)
    h = requests.get(BASE + "/api/history/" + sid, timeout=30).json()
    labels = [s["label"] for s in h["steps"]]
    check("历史含原图与两次编辑", labels[0] == "原图" and len(labels) == 3,
          str(labels))
    check("当前标签指向最后一次编辑", h["label"] == "灰点白平衡", h["label"])

    # undo twice -> the original, then redo -> the first edit
    u = requests.post(BASE + "/api/undo", json={"session": sid},
                      timeout=300).json()
    check("撤销一次回到 色轮 red", u["restored"] == "色轮 red", u["restored"])
    u2 = requests.post(BASE + "/api/undo", json={"session": sid},
                       timeout=300).json()
    check("撤销两次回到 原图", u2["restored"] == "原图", u2["restored"])
    check("到底后 can_undo 为假", u2["can_undo"] is False)
    check("到底后 can_redo 为真", u2["can_redo"] is True)
    check("到底后再撤销 409",
          requests.post(BASE + "/api/undo", json={"session": sid},
                        timeout=30).status_code == 409)

    rd = requests.post(BASE + "/api/redo", json={"session": sid},
                       timeout=300).json()
    check("重做回到 色轮 red", rd["restored"] == "色轮 red", rd["restored"])
    # both sides here are the same endpoint at the same quality, so this is a
    # real comparison
    same = float(np.abs(decode(rd["image"]) - s1).max())
    check("重做结果与当时完全一致", same == 0.0, "max %.2e" % same)

    # a new edit after an undo drops the redo chain
    requests.post(BASE + "/api/undo", json={"session": sid}, timeout=300)
    requests.post(BASE + "/api/greypoint",
                  json={"session": sid, "x": 200, "y": 300, "commit": True},
                  timeout=300)
    h = requests.get(BASE + "/api/history/" + sid, timeout=30).json()
    check("撤销后新编辑丢弃重做链", h["can_redo"] is False)
    return sid


def main():
    img = fit_long_edge(load_image(SRC), 700)
    engine_checks(img)
    try:
        requests.get(BASE + "/api/health", timeout=60)
    except Exception:
        print("\n服务未运行，跳过 API 检查")
        return 0 if not failures else 1
    api_checks()
    print()
    if failures:
        print("FAILED: %d -> %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
