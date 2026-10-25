# -*- coding: utf-8 -*-
"""删除迁移后不再使用的 `here = os.path.dirname(...)` 死代码。"""
import io
import os
import re

ENG = r"C:\Users\Rana\ZhaJinWenQianCake\engine"
HERE_LINE = re.compile(
    r"^[ \t]*here\s*=\s*os\.path\.dirname\(os\.path\.dirname\("
    r"os\.path\.abspath\(__file__\)\)\)[ \t]*\r?\n", re.M)

changed = []
for f in sorted(os.listdir(ENG)):
    if not f.endswith(".py"):
        continue
    p = os.path.join(ENG, f)
    s = io.open(p, encoding="utf-8").read()
    if "here = os.path.dirname(os.path.dirname(" not in s:
        continue
    # 只删除那些在函数内、且删除后没有别处再用 here 的
    n = len(HERE_LINE.findall(s))
    new = HERE_LINE.sub("", s)
    # 删除后如果还有独立的 here 使用（非注释），就跳过该文件
    leftover = [ln for ln in new.split("\n")
                if re.search(r"\bhere\b", ln) and not ln.strip().startswith("#")]
    if leftover:
        changed.append((f, "跳过（仍有 here 使用: %d 处）" % len(leftover)))
        continue
    io.open(p, "w", encoding="utf-8", newline="").write(new)
    changed.append((f, "删除 %d 行" % n))

for f, msg in changed:
    print("  %-16s %s" % (f, msg))
