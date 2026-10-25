# -*- coding: utf-8 -*-
"""发布前体检：找硬编码绝对路径、前端写死地址、缺失的打包配置。"""
import os
import re

ROOT = r"C:\Users\Rana\ZhaJinWenQianCake"

print("=== 代码里的绝对路径 ===")
pat = re.compile(r"(['\"])([A-Za-z]:[\\/][^'\"]{3,})\1")
found = []
for d in ("engine", "server"):
    p = os.path.join(ROOT, d)
    if not os.path.isdir(p):
        continue
    for f in sorted(os.listdir(p)):
        if not f.endswith(".py"):
            continue
        fp = os.path.join(p, f)
        for i, line in enumerate(
                open(fp, encoding="utf-8", errors="replace"), 1):
            for m in pat.finditer(line):
                found.append("%s:%d  %s" % (f, i, m.group(2)[:70]))
print("  共 %d 处" % len(found))
for h in found[:25]:
    print("   ", h)

print("\n=== 前端写死地址 ===")
idx = os.path.join(ROOT, "web", "index.html")
if os.path.exists(idx):
    s = open(idx, encoding="utf-8", errors="replace").read()
    for i, line in enumerate(s.split("\n"), 1):
        if re.search(r"127\.0\.0\.1:|localhost:|192\.168\.|frp2\.", line):
            print("  L%d: %s" % (i, line.strip()[:90]))

print("\n=== 模型定位方式 ===")
mod = os.path.join(ROOT, "engine", "models.py")
alt = [f for f in os.listdir(os.path.join(ROOT, "engine"))
       if "model" in f.lower() and f.endswith(".py")]
print("  相关文件:", alt)
for f in alt:
    fp = os.path.join(ROOT, "engine", f)
    txt = open(fp, encoding="utf-8", errors="replace").read()
    for line in txt.split("\n"):
        if "MODEL_DIR" in line and ("=" in line) and "os.path" in line:
            print("  %s: %s" % (f, line.strip()[:90]))
            break
