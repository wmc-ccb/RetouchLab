# -*- coding: utf-8 -*-
"""Quick smoke test for engine/library.py against the real photo library."""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import library as L

LIB = "M:" + os.sep


def main():
    print("顶层分组:")
    for g in L.groups(LIB):
        print("  %-18s %3d 张" % (g["name"], g["count"]))

    target = os.path.join("M:" + os.sep, "场照精修")
    print("\n扫描: %s" % target)
    t0 = time.time()
    r = L.scan(target, limit=30)
    print("  %d 张，耗时 %.1fs" % (r["count"], time.time() - t0))
    print("  分组:", r["groups"])
    for it in r["items"][:8]:
        print("  %-34s sharp=%8.1f mean=%.3f sh=%.3f clip=%.4f tags=%s"
              % (it["name"][:34], it.get("sharp", 0), it.get("mean", 0),
                 it.get("shadow", 0), it.get("clip", 0), it.get("tags")))
    if r["items"]:
        print("\nEXIF:", L.exif(r["items"][0]["path"]))

    # 更大一批，验证分组的统计阈值
    target2 = os.path.join("M:" + os.sep, "26年7月萤火虫")
    print("\n扫描更大批次: %s" % target2)
    t0 = time.time()
    r2 = L.scan(target2, limit=60)
    print("  %d 张，耗时 %.1fs" % (r2["count"], time.time() - t0))
    print("  分组:", r2["groups"])
    assert r2["count"] > 0, "未扫到图"
    return 0


if __name__ == "__main__":
    sys.exit(main())
