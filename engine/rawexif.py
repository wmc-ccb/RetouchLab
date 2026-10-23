# -*- coding: utf-8 -*-
"""从 RAW（NEF/TIFF 变体）里读 EXIF 暴露字段.

NEF 就是标准 little-endian TIFF（实测文件头 II*\\0 从偏移 0 开始），
IFD 结构直接按 TIFF 6.0 规范解析即可，不需要 PIL（它对 .NEF 扩展名
直接拒开）也引入不起 exiftool。相机把曝光字段放在 Exif IFD（0x8769），
与 JPEG 同一个布局，library.exif 的合并逻辑在这里同样适用。
"""
from __future__ import annotations

import os
import struct
from typing import Dict, Optional, Tuple

# TIFF 类型 -> 字节数
_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4,
              10: 8, 11: 4, 12: 8}

# 需要的字段
_TAGS = {"Make": 0x010F, "Model": 0x0110, "Artist": 0x013B,
         "DateTime": 0x0132}
_EXIF_TAGS = {"ISO": 0x8827, "FNumber": 0x829D, "ExposureTime": 0x829A,
              "FocalLength": 0x920A, "LensModel": 0xA434,
              "DateTimeOriginal": 0x9003}


def _read_ifd(buf: bytes, off: int, le: bool
              ) -> Tuple[Dict[int, object], int]:
    """解析一个 IFD，返回 {tag: value} 与下一个 IFD 偏移."""
    n = struct.unpack_from("<H" if le else ">H", buf, off)[0]
    entries: Dict[int, object] = {}
    p = off + 2
    for _ in range(n):
        tag, typ, cnt = struct.unpack_from(
            "<HHI" if le else ">HHI", buf, p)
        sz = _TYPE_SIZE.get(typ, 1) * cnt
        if 0 <= sz <= 4:
            raw = buf[p + 8:p + 8 + sz]
        else:
            vo = struct.unpack_from("<I" if le else ">I", buf, p + 8)[0]
            raw = buf[vo:vo + sz]
        try:
            if typ == 2:                      # ASCII
                entries[tag] = raw.split(b"\x00")[0].decode(
                    "ascii", "replace")
            elif typ in (3, 8):               # SHORT
                f = "<H" if le else ">H"
                vals = [struct.unpack_from(f, raw, i * 2)[0]
                        for i in range(cnt)]
                entries[tag] = vals[0] if cnt == 1 else vals
            elif typ in (4, 9):               # LONG
                f = "<I" if le else ">I"
                vals = [struct.unpack_from(f, raw, i * 4)[0]
                        for i in range(cnt)]
                entries[tag] = vals[0] if cnt == 1 else vals
            elif typ in (5, 10):              # RATIONAL
                f = "<II" if le else ">II"
                vals = [struct.unpack_from(f, raw, i * 8) for i in range(cnt)]
                entries[tag] = vals[0] if cnt == 1 else vals
            else:
                entries[tag] = raw
        except Exception:
            entries[tag] = raw
        p += 12
    nxt = struct.unpack_from("<I" if le else ">I", buf, off + 2 + n * 12)[0]
    return entries, nxt


def _sub_ifd_offset(entries: Dict[int, object], tag: int = 0x8769
                    ) -> Optional[int]:
    v = entries.get(tag)
    if isinstance(v, tuple) and len(v) == 2:
        return int(v[0] / v[1]) if v[1] else None
    if isinstance(v, int):
        return v
    return None


def raw_exif(path: str) -> Dict:
    """读 NEF/TIFF 变体的 EXIF 暴露字段（与 library.exif 同款键名）."""
    out: Dict[str, object] = {"name": os.path.basename(path),
                              "iso": None, "focal": None, "aperture": None,
                              "shutter": None, "camera": None, "lens": None,
                              "taken": None, "artist": None, "path": path}
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            head = fh.read(8)
            if head[:4] not in (b"II*\x00", b"MM\x00*"):
                return out
            le = head[:2] == b"II"
            ifd_off = struct.unpack("<I" if le else ">I", head[4:8])[0]
            # 只读文件头附近 256KB 足够覆盖 IFD0/ExifIFD（实测）
            fh.seek(0)
            buf = fh.read(min(size, 262144))
            entries, nxt = _read_ifd(buf, ifd_off, le)
            # 链到下一个 IFD0（有的文件把 MakerNote 拆开）
            hops = 0
            while nxt and hops < 3:
                e2, nxt2 = _read_ifd(buf, nxt, le)
                entries.update(e2)
                nxt, hops = nxt2, hops + 1
            sub_off = _sub_ifd_offset(entries)
            sub: Dict[int, object] = {}
            if sub_off and 0 < sub_off < len(buf):
                sub, _ = _read_ifd(buf, sub_off, le)

        def g(d, name):
            return d.get(_TAGS.get(name, -1)) if name in _TAGS else None

        def ge(name):
            return sub.get(_EXIF_TAGS.get(name, -1))

        out["camera"] = g(entries, "Make") or None
        model = g(entries, "Model")
        if model:
            out["camera"] = ("%s %s" % (out["camera"], model)) \
                if out["camera"] else model
        out["artist"] = g(entries, "Artist")
        out["taken"] = ge("DateTimeOriginal") or g(entries, "DateTime")
        iso = ge("ISO")
        out["iso"] = int(iso) if isinstance(iso, int) else None
        fn = ge("FNumber")
        if isinstance(fn, tuple) and fn[1]:
            out["aperture"] = round(fn[0] / fn[1], 1)
        et = ge("ExposureTime")
        if isinstance(et, tuple) and et[0]:
            out["shutter"] = round(et[0] / et[1], 6)
        fl = ge("FocalLength")
        if isinstance(fl, tuple) and fl[1]:
            out["focal"] = round(fl[0] / fl[1], 1)
        out["lens"] = ge("LensModel")
    except Exception:
        pass
    return out
