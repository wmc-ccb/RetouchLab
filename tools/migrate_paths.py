# -*- coding: utf-8 -*-
"""把 engine 各模块的模型路径改成走 engine.paths（支持打包）。

逐个文件精确替换 `here = os.path.dirname(os.path.dirname(...__file__))`
为 `from .paths import models_dir`，并把 `os.path.join(here, "models", X)`
改成 `os.path.join(models_dir(), X)`。

幂等：已经改过的文件会被跳过。
"""
import io
import os
import re

ROOT = r"C:\Users\Rana\RetouchLab"
ENG = os.path.join(ROOT, "engine")

# 匹配 here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE_RE = re.compile(
    r"^here\s*=\s*os\.path\.dirname\(os\.path\.dirname\("
    r"os\.path\.abspath\(__file__\)\)\)\s*$", re.M)
# 匹配 os.path.join(here, "models", ...)
JOIN_RE = re.compile(r'os\.path\.join\(here,\s*"models",')

changed = []
for f in sorted(os.listdir(ENG)):
    if not f.endswith(".py") or f in ("paths.py",):
        continue
    p = os.path.join(ENG, f)
    s = io.open(p, encoding="utf-8").read()
    if "here = os.path.dirname(os.path.dirname(" not in s:
        continue
    orig = s
    # 1) 替换 here 定义为导入
    s = HERE_RE.sub("from .paths import models_dir", s)
    # 2) 替换 join(here, "models", -> join(models_dir(),
    s = JOIN_RE.sub("os.path.join(models_dir(),", s)
    if s != orig:
        io.open(p, "w", encoding="utf-8", newline="").write(s)
        left = s.count("here")
        changed.append((f, left))

print("改动文件:", len(changed))
for f, left in changed:
    print("  %-18s 残留 here 引用 %d" % (f, left))
