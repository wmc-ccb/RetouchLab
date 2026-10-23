# -*- coding: utf-8 -*-
"""Tethered capture: get a photo from a camera into the pipeline.

The camera channel is the last thing to arrive, so the *source* is an interface
rather than a hard dependency. Everything downstream — auto-import, auto-preset,
auto-export, the inbox list — is written against that interface and works today
with a folder, then works unchanged once a real camera backend is plugged in.

Three sources, in order of how much they assume:

* **folder** — watch a directory for new image files. This is not a
  placeholder: it is how a large share of real tethering is done, because
  Nikon's own transfer tools, a card reader, or a camera writing over Wi-Fi all
  end up as "a file appears in a folder". It also means the whole chain is
  testable without any camera attached, and its tests do not need hardware.
* **digicamcontrol** — drive `CameraControlCmd.exe` for capture and live view.
  Detect-and-degrade: the backend reports `available: false` with the reason if
  the executable is absent, rather than failing at capture time.
* **ptp** — reserved for a direct PTP/SDK channel.

Nothing here blocks on a camera being present. `poll()` is stateless per call
(new files since the last poll), so a UI reload does not lose or replay shots.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import time
from typing import Dict, List, Optional

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".nef", ".cr2", ".cr3",
             ".arw", ".raf", ".dng", ".orf", ".rw2")

DCC_CANDIDATES = (
    r"C:\Program Files (x86)\digiCamControl\CameraControlCmd.exe",
    r"C:\Program Files\digiCamControl\CameraControlCmd.exe",
    r"C:\Program Files (x86)\digiCamControl\digiCamControl.exe",
    r"C:\Program Files\digiCamControl\digiCamControl.exe",
)


def _run_ps(cmd: str, timeout: int = 25):
    """Run PowerShell and decode its output without assuming UTF-8.

    On a Chinese Windows the console codepage is GBK, so `subprocess.run(...,
    text=True)` raises UnicodeDecodeError inside its reader thread and the call
    returns nothing usable — which is how camera detection silently reported
    "0 devices" while the camera was plainly attached. Decoding explicitly, with
    a fallback chain and replacement, keeps the bytes that are valid.
    """
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


def _is_image(name: str) -> bool:
    return name.lower().endswith(IMAGE_EXT)


def _list_dir(path: str) -> List[str]:
    if not path or not os.path.isdir(path):
        return []
    try:
        return [os.path.join(path, f) for f in os.listdir(path)
                if _is_image(f) and os.path.isfile(os.path.join(path, f))]
    except OSError:
        return []


class Source:
    """What every capture source has to provide."""

    kind = "base"

    def prime(self) -> int:
        """Baseline whatever is already present. Default: nothing to do."""
        return 0

    def status(self) -> dict:
        raise NotImplementedError

    def poll(self) -> List[str]:
        """New files since the previous call. Must not repeat a file."""
        raise NotImplementedError

    def capture(self) -> dict:
        """Trigger a shot. Sources that cannot trigger say so explicitly."""
        return {"ok": False, "supported": False,
                "reason": "this source cannot trigger the shutter"}

    def live_frame(self) -> Optional[bytes]:
        """A JPEG preview frame, or None when the source has no live view."""
        return None

    def reset(self) -> None:
        """Forget what has been seen (used when the watch target changes)."""


class FolderSource(Source):
    """Watch a directory and report images that appear in it.

    Files are also matched on **modification time**, not just name: a camera or
    transfer tool frequently writes to a temporary name and then renames, and a
    name-only `seen` set would either miss the rename or report a file that is
    still being written. A file younger than `settle_s` is held back so a
    half-written JPEG is never handed to the decoder.
    """

    kind = "folder"

    def __init__(self, path: str = "", settle_s: float = 1.2,
                 pattern: str = "*"):
        self.path = path
        self.settle_s = float(settle_s)
        self.pattern = pattern
        self._seen: Dict[str, float] = {}

    def reset(self) -> None:
        self._seen.clear()

    def prime(self) -> int:
        """Mark everything already present as seen; returns how many.

        Called when a watch is armed. Without it, pointing the tool at a folder
        that already holds a shoot imports the entire back catalogue on the
        first poll — the session is supposed to start from what arrives *after*
        the shutter is connected, not from what happened to be sitting there.
        """
        n = 0
        for fp in _list_dir(self.path):
            try:
                self._seen[fp] = os.path.getmtime(fp)
                n += 1
            except OSError:
                continue
        return n

    def status(self) -> dict:
        exists = bool(self.path) and os.path.isdir(self.path)
        return {
            "kind": self.kind,
            "available": exists,
            "path": self.path,
            "settle_s": self.settle_s,
            "watched": len(self._seen),
            "reason": None if exists else (
                "目录不存在" if self.path else "还没有设置监视目录"),
        }

    def poll(self) -> List[str]:
        now = time.time()
        out: List[str] = []
        for fp in _list_dir(self.path):
            try:
                mtime = os.path.getmtime(fp)
                size = os.path.getsize(fp)
            except OSError:
                continue
            if size <= 0:
                continue
            # still being written: hold it back until it stops growing
            if now - mtime < self.settle_s:
                continue
            prev = self._seen.get(fp)
            if prev is not None and prev == mtime:
                continue
            self._seen[fp] = mtime
            out.append(fp)
        out.sort(key=lambda p: os.path.getmtime(p))
        return out


class DigiCamControlSource(Source):
    """Drive digiCamControl's CLI for capture and live view.

    `available` is answered by looking for the executable, so the UI can say
    "not installed" instead of failing when the shutter is pressed. The capture
    call also lands the file in `download_dir` and then reports it through
    `poll()`, which is the same path the folder source uses — a capture and a
    manually copied file take identical routes downstream.
    """

    kind = "digicamcontrol"

    def __init__(self, exe: str = "", download_dir: str = "",
                 settle_s: float = 1.0):
        self.exe = exe or self._find()
        self.download_dir = download_dir
        self.settle_s = float(settle_s)
        self._seen: Dict[str, float] = {}
        self._folder = FolderSource(download_dir, settle_s)

    @staticmethod
    def _find() -> str:
        for c in DCC_CANDIDATES:
            if os.path.exists(c):
                return c
        return ""

    def reset(self) -> None:
        self._seen.clear()
        self._folder.reset()

    def prime(self) -> int:
        self._folder.path = self.download_dir
        return self._folder.prime()

    def status(self) -> dict:
        ok = bool(self.exe) and os.path.exists(self.exe)
        return {
            "kind": self.kind,
            "available": ok,
            "exe": self.exe,
            "path": self.download_dir,
            "can_capture": ok,
            "reason": None if ok else (
                "未找到 digiCamControl（CameraControlCmd.exe）。"
                "安装后本项自动可用；或改用文件夹监听。"),
        }

    def poll(self) -> List[str]:
        self._folder.path = self.download_dir
        return self._folder.poll()

    def capture(self) -> dict:
        if not self.exe or not os.path.exists(self.exe):
            return {"ok": False, "supported": True,
                    "reason": "digiCamControl 未安装"}
        if not self.download_dir:
            return {"ok": False, "supported": True,
                    "reason": "还没有设置下载目录"}
        os.makedirs(self.download_dir, exist_ok=True)
        name = time.strftime("shot_%Y%m%d_%H%M%S") + ".jpg"
        dest = os.path.join(self.download_dir, name)
        t0 = time.time()
        try:
            p = subprocess.run(
                [self.exe, "/capture", "/filename", dest],
                capture_output=True, text=True, timeout=120,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except subprocess.TimeoutExpired:
            return {"ok": False, "supported": True, "reason": "拍摄超时"}
        except Exception as exc:
            return {"ok": False, "supported": True, "reason": str(exc)[:160]}
        return {
            "ok": p.returncode == 0,
            "supported": True,
            "path": dest if os.path.exists(dest) else None,
            "seconds": round(time.time() - t0, 2),
            "stdout": (p.stdout or "")[-300:],
            "reason": None if p.returncode == 0 else (
                (p.stderr or "exit %d" % p.returncode)[-200:]),
        }

    def live_frame(self) -> Optional[bytes]:
        """digiCamControl writes live view to a file; read it if it is fresh."""
        if not self.exe or not os.path.exists(self.exe):
            return None
        candidates = glob.glob(os.path.join(
            os.path.dirname(self.exe), "**", "liveview*.jpg"), recursive=True)
        candidates += glob.glob(os.path.join(self.download_dir or ".",
                                             "liveview*.jpg"))
        fresh = [c for c in candidates
                 if os.path.exists(c) and time.time() - os.path.getmtime(c) < 5]
        if not fresh:
            return None
        try:
            with open(max(fresh, key=os.path.getmtime), "rb") as fh:
                return fh.read()
        except OSError:
            return None


class PtpSource(Source):
    """Placeholder for a direct PTP/SDK channel.

    Present so the rest of the system can be written against the interface
    today and the UI reports an honest reason instead of a missing option. The
    camera currently enumerates only as WPD (MTP), which cannot be
    shutter-controlled; the body has to be switched to PTP first.
    """

    kind = "ptp"

    def status(self) -> dict:
        return {"kind": self.kind, "available": False,
                "can_capture": False,
                "reason": "相机未处于 PTP 模式；当前系统只枚举出 WPD(MTP) 接口。"
                          "请在机身把 USB 设为 PTP，或接入 Nikon SDK。"}

    def poll(self) -> List[str]:
        return []


def make_source(kind: str, **kw) -> Source:
    if kind == "digicamcontrol":
        return DigiCamControlSource(**{k: v for k, v in kw.items()
                                       if k in ("exe", "download_dir",
                                                "settle_s")})
    if kind == "ptp":
        return PtpSource()
    return FolderSource(**{k: v for k, v in kw.items()
                           if k in ("path", "settle_s", "pattern")})


_CAM_CACHE = {"at": 0.0, "data": []}
_CAM_TTL = 60.0

# Vendor USB ids worth naming. Nikon is the one in use here; the rest are the
# usual tethered bodies.
VENDOR_VID = {"04B0": "Nikon", "04A9": "Canon", "054C": "Sony",
              "04CB": "Fujifilm", "04DA": "Panasonic", "33A2": "Sigma",
              "2A0D": "Phase One"}

_SKIP_WORDS = ("总线", "Bus", "虚拟", "Virtual", "复合", "Composite", "Hub")


def _cameras_registry() -> List[dict]:
    """Enumerate USB devices from the registry.

    `Get-PnpDevice` measured **12-14 seconds** on this machine, which made the
    status endpoint block for that long and left the tether panel empty past a
    UI timeout — and the cache could not hide it, because a 12 s call with an
    8 s TTL simply expired before the next request arrived. The registry key PnP
    reads is available directly and answers in milliseconds, so it is the
    primary path; the PnP query stays as a fallback.
    """
    out: List[dict] = []
    try:
        import winreg
    except ImportError:
        return out
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                              r"SYSTEM\CurrentControlSet\Enum\USB")
    except OSError:
        return out
    try:
        i = 0
        while True:
            try:
                vid_pid = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            up = vid_pid.upper()
            if "VID_" not in up:
                continue
            vendor = VENDOR_VID.get(up.split("VID_")[1][:4])
            try:
                sub = winreg.OpenKey(root, vid_pid)
            except OSError:
                continue
            j = 0
            while True:
                try:
                    inst = winreg.EnumKey(sub, j)
                except OSError:
                    break
                j += 1
                try:
                    dev = winreg.OpenKey(sub, inst)
                except OSError:
                    continue
                name, cls = "", ""
                for value in ("FriendlyName", "DeviceDesc"):
                    try:
                        name = str(winreg.QueryValueEx(dev, value)[0])
                        if name:
                            break
                    except OSError:
                        continue
                try:
                    cls = str(winreg.QueryValueEx(dev, "Class")[0])
                except OSError:
                    cls = ""
                # DeviceDesc is stored as "@usbhub.inf,%name%;USB 复合设备"
                if ";" in name:
                    name = name.split(";")[-1].strip()
                cls = cls or ("WPD" if vendor else "")
                if not (vendor or cls in ("WPD", "Image", "MEDIA")):
                    continue
                if any(w in name for w in _SKIP_WORDS):
                    continue
                out.append({
                    "name": name or (vendor or "camera"),
                    "class": cls,
                    "vendor": vendor,
                    "instance": "USB\\%s\\%s" % (vid_pid, inst),
                    "status": "OK",
                    "can_capture": cls not in ("WPD",),
                    "note": ("MTP 模式：只能传文件，不能控快门"
                             if cls == "WPD" else ""),
                })
    finally:
        try:
            winreg.CloseKey(root)
        except Exception:
            pass
    return out


def detect_cameras(force: bool = False) -> List[dict]:
    """Enumerate attached imaging devices, with a cache.

    Reported so the UI can show what the system can actually see. A body that
    only exposes WPD is listed with `can_capture: false` and the reason, which
    is the difference between "no camera" and "camera connected but in the
    wrong USB mode" — those need very different actions from the user.
    """
    now = time.time()
    if not force and now - _CAM_CACHE["at"] < _CAM_TTL:
        return _CAM_CACHE["data"]

    out = _cameras_registry()
    if not out and os.name == "nt":
        # nothing under Enum\USB: fall back to the slow PnP walk so a device
        # exposed some other way is not silently missed
        try:
            ps = ("Get-PnpDevice -PresentOnly -Class WPD,Image | "
                  "Select-Object Class,FriendlyName,InstanceId,Status | "
                  "ConvertTo-Json -Compress")
            import json
            data = json.loads(_run_ps(ps, timeout=30) or "[]")
            if isinstance(data, dict):
                data = [data]
            for d in data:
                cls = (d.get("Class") or "").strip()
                name = (d.get("FriendlyName") or "").strip()
                if any(w in name for w in _SKIP_WORDS):
                    continue
                out.append({
                    "name": name or "camera", "class": cls, "vendor": None,
                    "instance": d.get("InstanceId"), "status": d.get("Status"),
                    "can_capture": cls not in ("WPD",),
                    "note": ("MTP 模式：只能传文件，不能控快门"
                             if cls == "WPD" else ""),
                })
        except Exception:
            pass
    _CAM_CACHE["at"] = now
    _CAM_CACHE["data"] = out
    return out


def dcc_installed() -> str:
    for c in DCC_CANDIDATES:
        if os.path.exists(c):
            return c
    return ""
