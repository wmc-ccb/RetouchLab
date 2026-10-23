# -*- coding: utf-8 -*-
"""相机联机（WPD/MTP）：浏览相机存储卡并导入.

为什么先做 WPD 而不是 PTP：
Z6 III 接上后只枚举出 WPD（Windows Portable Devices）接口，这是 MTP 模式，
**只能传文件，不能控快门**。控快门需要机身切到 PTP，或用 digiCamControl
（安装需要 UAC，本会话无法授权）。

但 WPD 本身就能做一件真正有用的事：**浏览相机里的照片并批量导入**。这条
路径不依赖任何第三方工具，Windows 自带驱动即可，所以先把它做扎实。

快门控制在 `tether.PtpSource` 里留了接口，机身切到 PTP 后接入即可。
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

# Windows 的 WPD 通过 COM 访问；用 ctypes 调 PowerShell 的 Shell.Application
# 比直接写 COM 简单可靠，且不引入 pywin32 之外的依赖。

WPD_DEVICE_PATTERNS = ("Z6", "Z 6", "Nikon", "相机", "Camera", "DCIM")


def _ps(cmd: str, timeout: int = 40) -> str:
    """Run PowerShell and decode without assuming UTF-8.

    A Chinese Windows console emits GBK; `text=True` in subprocess raises
    UnicodeDecodeError in its reader thread and the call returns nothing, which
    is how camera detection silently reported zero devices earlier.
    """
    import subprocess

    for enc in ("utf-8", "gbk", None):
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                               capture_output=True, timeout=timeout,
                               creationflags=getattr(subprocess,
                                                     "CREATE_NO_WINDOW", 0))
        except Exception:
            return ""
        raw = r.stdout or b""
        if enc is None:
            return raw.decode("utf-8", errors="replace")
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return ""


def list_devices() -> List[Dict]:
    """Enumerate portable devices (cameras, phones) by display name.

    Uses the Shell namespace `This PC` and keeps only non-drive items, which is
    where MTP cameras appear. A camera in MTP mode is a *storage* device here,
    not an imaging device, which is exactly why it can list files without being
    able to trip the shutter.
    """
    cmd = (
        "$s = New-Object -ComObject Shell.Application;"
        "$pc = $s.NameSpace(17);"
        "if($pc){"
        " $pc.Items() | ForEach-Object {"
        "   [pscustomobject]@{ Name=$_.Name; Path=$_.Path; "
        "IsFolder=$_.IsFolder } } | ConvertTo-Json -Compress }"
    )
    out = _ps(cmd, timeout=40)
    devices = []
    try:
        import json
        data = json.loads(out) if out.strip() else []
        if isinstance(data, dict):
            data = [data]
        for d in data:
            name = (d.get("Name") or "").strip()
            if not name:
                continue
            # 排除本机磁盘
            if len(name) == 2 and name[1] == ":":     # "C:"
                continue
            if any(p in name for p in WPD_DEVICE_PATTERNS):
                devices.append({"name": name,
                                "path": d.get("Path") or "",
                                "kind": "portable"})
    except Exception:
        pass
    return devices


def list_files(device_name: str, subdir: str = "") -> List[Dict]:
    """List image files on a portable device.

    `subdir` is a path *inside* the device (e.g. "DCIM\\100NZ6_3"). Empty means
    the device root.
    """
    # 通过 Shell 递归一层：相机通常把照片放在 DCIM\xxx
    safe = device_name.replace("'", "''")
    sub = subdir.replace("'", "''")
    cmd = (
        "$s = New-Object -ComObject Shell.Application;"
        "$pc = $s.NameSpace(17);"
        "$dev = $null;"
        "foreach($it in $pc.Items()){ if($it.Name -eq '%s'){ $dev = $it; break } }"
        "if($dev){"
        " $folder = $s.NameSpace($dev.Path);"
        " if($folder){"
        "  $folder.Items() | ForEach-Object {"
        "    [pscustomobject]@{ Name=$_.Name; IsFolder=$_.IsFolder; "
        "Size=$_.Size } } | ConvertTo-Json -Compress } }"
    ) % safe
    if sub:
        # 进入子目录
        cmd = (
            "$s = New-Object -ComObject Shell.Application;"
            "$pc = $s.NameSpace(17);"
            "$dev = $null;"
            "foreach($it in $pc.Items()){ if($it.Name -eq '%s'){ $dev = $it; break } }"
            "if($dev){"
            " $f = $s.NameSpace($dev.Path);"
            " foreach($x in $f.Items()){ if($x.Name -eq '%s'){"
            "   $g = $s.NameSpace($x.Path);"
            "   $g.Items() | ForEach-Object {"
            "     [pscustomobject]@{ Name=$_.Name; IsFolder=$_.IsFolder; "
            "Size=$_.Size } } | ConvertTo-Json -Compress"
            "   break } } }"
        ) % (safe, sub.replace("\\", "\\\\"))
    out = _ps(cmd, timeout=40)
    items = []
    try:
        import json
        data = json.loads(out) if out.strip() else []
        if isinstance(data, dict):
            data = [data]
        ext_ok = (".jpg", ".jpeg", ".nef", ".png", ".tif", ".tiff", ".mov",
                  ".mp4")
        for d in data:
            name = (d.get("Name") or "").strip()
            if not name:
                continue
            if d.get("IsFolder"):
                items.append({"name": name, "folder": True, "size": 0})
                continue
            if name.lower().endswith(ext_ok):
                items.append({"name": name, "folder": False,
                              "size": int(d.get("Size") or 0)})
    except Exception:
        pass
    return items


def import_files(device_name: str, names: List[str], dest: str,
                 subdir: str = "") -> List[Dict]:
    """Copy named files off the device into `dest`.

    Uses the Shell `CopyHere` verb, which is how Explorer itself copies off an
    MTP camera. Files that already exist with the same size are skipped so a
    repeated import does not grow the destination.
    """
    os.makedirs(dest, exist_ok=True)
    results = []
    safe_dev = device_name.replace("'", "''")
    safe_dest = dest.replace("'", "''")
    for name in names:
        safe_name = name.replace("'", "''")
        target = os.path.join(dest, name)
        if os.path.exists(target) and os.path.getsize(target) > 0:
            results.append({"name": name, "copied": False,
                            "path": target, "skipped": "exists"})
            continue
        sub_step = ""
        if subdir:
            sub_step = (" $f = $f.Items() | Where-Object { $_.Name -eq '%s' };"
                        " if($f){ $f = $s.NameSpace($f.Path) }"
                        % subdir.replace("'", "''"))
        cmd = (
            "$s = New-Object -ComObject Shell.Application;"
            "$pc = $s.NameSpace(17);"
            "$dev = $null;"
            "foreach($it in $pc.Items()){ if($it.Name -eq '%s'){ $dev=$it;break } }"
            "if($dev){"
            " $f = $s.NameSpace($dev.Path);%s"
            " if($f){"
            "  $item = $f.Items() | Where-Object { $_.Name -eq '%s' };"
            "  if($item){"
            "   $dst = $s.NameSpace('%s');"
            "   if($dst){ $dst.CopyHere($item, 16) } } } }"
        ) % (safe_dev, sub_step, safe_name, safe_dest)
        _ps(cmd, timeout=120)
        # CopyHere 是异步的，轮询等文件落地
        ok = False
        for _ in range(60):
            if os.path.exists(target) and os.path.getsize(target) > 0:
                ok = True
                break
            time.sleep(0.5)
        results.append({"name": name, "copied": ok, "path": target,
                        "skipped": None if ok else "copy failed"})
    return results


def auto_import(device_name: str, dest: str, subdir: str = "",
                limit: int = 500) -> List[Dict]:
    """Import every image on the device (one level of DCIM subfolder)."""
    names = []
    for it in list_files(device_name, subdir):
        if not it["folder"]:
            names.append(it["name"])
        if len(names) >= limit:
            break
    if not names:
        return []
    return import_files(device_name, names, dest, subdir)


def info() -> dict:
    devs = list_devices()
    return {"devices": devs,
            "mode": "WPD/MTP",
            "can_capture": False,
            "note": "MTP 只能传文件。控快门需要机身切 PTP，或用 digiCamControl。"}


if __name__ == "__main__":
    import json
    print(json.dumps(info(), ensure_ascii=False, indent=1))
