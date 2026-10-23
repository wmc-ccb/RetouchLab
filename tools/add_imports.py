# -*- coding: utf-8 -*-
"""给缺失 `from .paths import models_dir` 的 engine 模块补上导入。"""
import io
import os
import re

ENG = r"C:\Users\Rana\RetouchLab\engine"

fixed, already = [], []
for f in sorted(os.listdir(ENG)):
    if not f.endswith(".py") or f == "paths.py":
        continue
    p = os.path.join(ENG, f)
    s = io.open(p, encoding="utf-8").read()
    if "models_dir(" not in s:
        continue
    if re.search(r"from \.paths import[^\n]*models_dir", s):
        already.append(f)
        continue
    # 插到第一个 from . import 或 from .xxx import 之前；没有就放在最后一个 import 后
    lines = s.split("\n")
    insert_at = None
    for i, ln in enumerate(lines):
        if re.match(r"^from \.|^from \.\w+ import|^import ", ln):
            insert_at = i
            break
    if insert_at is None:
        # 放在 __future__ 之后
        for i, ln in enumerate(lines):
            if ln.startswith("from __future__"):
                insert_at = i + 1
                break
    if insert_at is None:
        insert_at = 0
    # 避免插到模块 docstring 中间：跳过空行和注释
    while insert_at < len(lines) and (
            not lines[insert_at].strip()
            or lines[insert_at].lstrip().startswith("#")):
        insert_at += 1
    lines.insert(insert_at, "from .paths import models_dir")
    io.open(p, "w", encoding="utf-8", newline="").write("\n".join(lines))
    fixed.append((f, insert_at + 1))

print("补导入:", len(fixed))
for f, ln in fixed:
    print("  %-16s 第 %d 行" % (f, ln))
print("已有导入:", already)
