# -*- coding: utf-8 -*-
"""图库「我的分组」：把勾选的照片存成命名选片集，持久化在磁盘.

参考软件的图库右栏有「我的分组」：我的选片 / 未选图片 / 备选照片。
选片是摄影工作流的核心一步——一次拍 2000 张，挑出 300 张精修。

实现：分组 = {名字: [文件路径]}，存 JSON。加入时按路径去重；同名追加，
不覆盖（选片是累积的）。
"""
from __future__ import annotations

import json
import os
import threading
from typing import Dict, List

_LOCK = threading.Lock()
_FILE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "server", "_library_groups.json")


def _load() -> Dict[str, List[str]]:
    try:
        with open(_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save(data: Dict[str, List[str]]) -> None:
    tmp = _FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, _FILE)


def list_groups() -> Dict[str, dict]:
    """所有分组，带计数。"""
    with _LOCK:
        data = _load()
    return {name: {"count": len(paths)} for name, paths in data.items()}


def members(name: str) -> List[str]:
    with _LOCK:
        return list(_load().get(name, []))


def add(name: str, paths: List[str]) -> dict:
    """把文件加入分组（按路径去重，追加不覆盖）。"""
    key = (name or "").strip()
    if not key:
        raise ValueError("分组名不能为空")
    clean = [str(p) for p in paths if p]
    if not clean:
        raise ValueError("没有可加入的文件")
    with _LOCK:
        data = _load()
        cur = data.get(key, [])
        added = [p for p in clean if p not in cur]
        data[key] = cur + added
        _save(data)
        return {"group": key, "added": len(added), "total": len(data[key])}


def remove(name: str, paths: List[str]) -> dict:
    with _LOCK:
        data = _load()
        if name not in data:
            raise KeyError(name)
        drop = set(str(p) for p in paths)
        data[name] = [p for p in data[name] if p not in drop]
        removed = len(drop)
        if not data[name]:
            del data[name]          # 空分组自动消失
        _save(data)
    return {"group": name, "removed": removed}


def delete_group(name: str) -> dict:
    with _LOCK:
        data = _load()
        if name not in data:
            raise KeyError(name)
        n = len(data.pop(name))
        _save(data)
    return {"group": name, "removed": n}


def rename(old: str, new: str) -> dict:
    old, new = old.strip(), new.strip()
    if not new:
        raise ValueError("新名字不能为空")
    with _LOCK:
        data = _load()
        if old not in data:
            raise KeyError(old)
        if new in data and new != old:
            raise ValueError("分组「%s」已存在" % new)
        data[new] = data.pop(old)
        _save(data)
    return {"group": new}
