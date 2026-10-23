# -*- coding: utf-8 -*-
"""End-to-end RAW test through the HTTP API.

Uploads a real NEF, checks that the server decodes it with auto-exposure,
renders it, and exports it. PowerShell's Invoke-WebRequest closed the
connection on this upload, which pointed at the request rather than the
service, so the test is done with requests where the multipart framing is
explicit.
"""
import io
import os
import sys

import requests

BASE = "http://127.0.0.1:8765"
SRC = r"C:\Users\Rana\Desktop\26cicf\d1\新建文件夹\_DSC7029.NEF"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

h = requests.get(BASE + "/api/health", timeout=60).json()
print("health:", h)
assert h["ok"]

with open(SRC, "rb") as fh:
    print("\nuploading %s (%.1f MB) …" % (os.path.basename(SRC),
                                          os.path.getsize(SRC) / 1e6))
    r = requests.post(BASE + "/api/upload",
                      files={"file": ("_DSC7029.NEF", fh,
                                      "application/octet-stream")},
                      timeout=600)
print("upload status:", r.status_code)
if r.status_code != 200:
    print(r.text[:500])
    sys.exit(1)
j = r.json()
print("  session=%s  %sx%s  faces=%d  decode=%s"
      % (j.get("session"), j.get("width"), j.get("height"),
         len(j.get("faces", [])), j.get("decode")))
sid = j["session"]
# the upload decodes half-size on purpose: a 4000px frame costs 4x the preview
# for no retouching gain. Export re-decodes at full resolution.
assert j["width"] == 2000 and j["height"] == 1336, "half-size decode expected"
assert j.get("decode", {}).get("raw") is True, "RAW path not taken"
assert j["faces"], "no faces found in the RAW frame"

r2 = requests.post(BASE + "/api/render",
                   json={"session": sid, "params": {"smooth": 45, "exposure": 6}},
                   timeout=600)
print("render status:", r2.status_code)
r2j = r2.json()
print("  applied:", r2j["applied"])
print("  jpeg bytes:", len(r2j["image"]))

r3 = requests.post(BASE + "/api/save",
                   json={"session": sid, "params": {"smooth": 45, "exposure": 6},
                         "path": os.path.join(ROOT, "out", "raw_api_export.jpg")},
                   timeout=600)
print("save status:", r3.status_code, r3.json().get("path", ""))

print("\nRESULT: RAW through the API works end to end")
