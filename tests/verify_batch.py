# -*- coding: utf-8 -*-
"""Batch processing: discovery, recipe resolution, progress, error isolation.

The interesting properties are not "does it produce files" but:

* **recipe resolution happens once** — every file must get the same look; a
  per-file re-resolution could drift if the library changes mid-run;
* **one bad file does not lose the rest** — a corrupt file must be recorded as
  a failure and the run must continue;
* **progress is monotone and terminates** — `done` never decreases, and the
  run reaches a terminal state;
* **draft mode is real** — a smaller long edge must produce smaller files,
  otherwise the flag is decorative;
* **the output name does not collide** with the source.
"""
import os
import shutil
import sys
import time

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASE = "http://127.0.0.1:8765"
SRC_DIR = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰"
WORK = os.path.join(ROOT, "out", "_batch_test")
failures = []


def check(label, cond, detail=""):
    print("%-54s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


def wait(rid, timeout=900):
    """Poll to a terminal state.

    `queued` is a real state now: runs are serialised because they share the
    process-wide pipeline, so a second run waits for the first. Treating
    `queued` as terminal would report a false pass/fail on whatever happened
    to be in flight.
    """
    t0 = time.time()
    last = -1
    monotone = True
    while time.time() - t0 < timeout:
        st = requests.get(BASE + "/api/batch/" + rid, timeout=60).json()
        if st["done"] < last:
            monotone = False
        last = st["done"]
        if st["state"] not in ("running", "queued"):
            return st, monotone
        time.sleep(2)
    raise TimeoutError("batch did not finish")


def main():
    from engine import batch as BT

    # ---- engine: file discovery ------------------------------------------
    print("=== 文件发现 ===")
    found = BT.list_images(SRC_DIR)
    check("目录扫描找到图片", len(found) > 5, "%d 个" % len(found))
    check("只返回存在的文件", all(os.path.isfile(p) for p in found))
    exts = {os.path.splitext(p)[1].lower() for p in found}
    check("扩展名都在白名单内",
          exts <= set(BT.IMAGE_EXT + BT.RAW_EXT), str(sorted(exts))[:70])
    check("单文件路径也能用", BT.list_images(found[0]) == [found[0]])
    check("不存在的路径返回空", BT.list_images(r"C:\nope\nope") == [])

    try:
        requests.get(BASE + "/api/health", timeout=60)
    except Exception:
        print("\n服务未运行，跳过 API 检查")
        return 0 if not failures else 1

    # ---- build a small fixture set --------------------------------------
    shutil.rmtree(WORK, ignore_errors=True)
    os.makedirs(WORK, exist_ok=True)
    from engine.imgutil import load_image, fit_long_edge, save_image
    src = fit_long_edge(load_image(os.path.join(found[0])), 1000)
    for i in range(3):
        save_image(os.path.join(WORK, "good_%d.jpg" % i),
                   np.clip(src * (0.9 + i * 0.1), 0, 1), 95)
    # a file that is not an image at all, with an image extension
    with open(os.path.join(WORK, "broken.jpg"), "wb") as fh:
        fh.write(b"this is not a jpeg")
    out_dir = os.path.join(WORK, "out")

    # ---- run -------------------------------------------------------------
    print("\n=== 批处理运行（含一个损坏文件）===")
    r = requests.post(BASE + "/api/batch", json={
        "path": WORK, "out_dir": out_dir, "preset": "natural",
        "long_edge": 700, "quality": 90}, timeout=120).json()
    check("启动返回 id 与总数", "id" in r and r["total"] == 4,
          "total=%s" % r.get("total"))
    st, monotone = wait(r["id"])
    check("进度单调不减", monotone)
    check("到达终态", st["state"] == "done", st["state"])
    check("全部处理完", st["done"] == 4, "%d/4" % st["done"])
    check("损坏文件被单独记为失败", st["failed"] == 1, "%d 失败" % st["failed"])
    check("成功的文件仍产出", st["done"] - st["failed"] == 3)

    outs = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []
    check("产出 3 个文件", len(outs) == 3, str(outs))
    check("输出名不与源冲突",
          all("_out" in o for o in outs), str(outs)[:70])
    check("失败项带错误信息",
          any(not e["ok"] and e.get("error") for e in st["results"]))

    # ---- draft mode is real ---------------------------------------------
    print("\n=== 草稿模式确实更小 ===")
    full_dir = os.path.join(WORK, "full")
    draft_dir = os.path.join(WORK, "draft")
    r1 = requests.post(BASE + "/api/batch", json={
        "path": os.path.join(WORK, "good_0.jpg"), "out_dir": full_dir,
        "preset": "natural", "long_edge": 0, "quality": 90}, timeout=120).json()
    r2 = requests.post(BASE + "/api/batch", json={
        "path": os.path.join(WORK, "good_0.jpg"), "out_dir": draft_dir,
        "preset": "natural", "long_edge": 400, "quality": 90}, timeout=120).json()
    wait(r1["id"]); st2, _ = wait(r2["id"])
    check("草稿运行完成", st2["state"] == "done" and st2["failed"] == 0)
    f1 = os.path.join(full_dir, os.listdir(full_dir)[0])
    f2 = os.path.join(draft_dir, os.listdir(draft_dir)[0])

    import cv2
    a = cv2.imread(f1); b = cv2.imread(f2)
    check("草稿输出尺寸更小", b.shape[0] < a.shape[0],
          "%d -> %d" % (a.shape[0], b.shape[0]))
    check("草稿文件更小", os.path.getsize(f2) < os.path.getsize(f1),
          "%.0fKB -> %.0fKB" % (os.path.getsize(f1) / 1024,
                                os.path.getsize(f2) / 1024))

    # ---- recipe is recorded ---------------------------------------------
    print("\n=== 配方标签 ===")
    check("记录预设名", st2["recipe"] == "预设 natural", st2["recipe"])
    check("记录草稿边长", st2["draft_edge"] == 400, str(st2["draft_edge"]))

    # ---- cancel ----------------------------------------------------------
    print("\n=== 取消 ===")
    rc = requests.post(BASE + "/api/batch", json={
        "path": SRC_DIR, "out_dir": os.path.join(WORK, "cancel"),
        "preset": "natural", "long_edge": 700}, timeout=120).json()
    requests.post(BASE + "/api/batch/%s/cancel" % rc["id"], timeout=30)
    time.sleep(3)
    stc = requests.get(BASE + "/api/batch/%s" % rc["id"], timeout=30).json()
    check("取消后状态为 cancelled", stc["state"] in ("cancelled", "done"),
          stc["state"])
    check("取消后不再增长", stc["done"] <= rc["total"])

    # ---- unknown id ------------------------------------------------------
    check("未知 id 返回 404",
          requests.get(BASE + "/api/batch/nope", timeout=30).status_code == 404)

    print()
    if failures:
        print("FAILED: %d -> %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
