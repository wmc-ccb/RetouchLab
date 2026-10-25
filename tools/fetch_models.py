#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 GitHub Release 下载并校验运行所需的全部模型。

被 `一键安装环境.bat` 调用，也可以单独跑：
    python tools/fetch_models.py           # 下载缺失/校验失败的模型
    python tools/fetch_models.py --check   # 只体检不下载

模型清单 tools/models_manifest.json 记录了每个文件的相对路径、
大小与 sha256；资产托管在本仓库 models-v1 Release（子目录路径
映射为扁平资产名：faceparsing/config.json -> faceparsing_config.json）。

特性：断点续传（.part 临时文件）、失败重试、sha256 校验后才落盘、
已存在且校验通过的文件直接跳过——跑多少次都安全。
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "tools", "models_manifest.json")
MODELS_DIR = os.path.join(ROOT, "models")


def asset_name(rel_path: str) -> str:
    """models/faceparsing/config.json -> faceparsing_config.json（Release 资产名）。"""
    return rel_path.replace("/", "_")


def verify(path: str, sha256: str) -> bool:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return False
    return h.hexdigest() == sha256


def download(url: str, dest: str, retries: int = 5):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    for attempt in range(retries):
        try:
            resume_from = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            req = urllib.request.Request(url)
            if resume_from:
                req.add_header("Range", "bytes=%d-" % resume_from)
            with urllib.request.urlopen(req, timeout=60) as r, \
                    open(tmp, "ab" if resume_from else "wb") as f:
                total = int(r.headers.get("Content-Length") or 0) + resume_from
                got = resume_from
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        sys.stdout.write("\r    %.1f / %.1f MB (%d%%)"
                                         % (got / 1048576, total / 1048576,
                                            got * 100 // total))
                        sys.stdout.flush()
            print()
            os.replace(tmp, dest)
            return
        except Exception as e:
            print("\n    第 %d 次尝试失败: %s" % (attempt + 1, str(e)[:80]))
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError("下载失败(已重试 %d 次): %s" % (retries, url))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只体检不下载")
    args = ap.parse_args()

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)

    missing, bad, ok = [], [], 0
    for item in manifest["files"]:
        dest = os.path.join(MODELS_DIR, item["path"].replace("/", os.sep))
        if os.path.exists(dest) and os.path.getsize(dest) == item["size"] \
                and verify(dest, item["sha256"]):
            ok += 1
            continue
        if os.path.exists(dest):
            bad.append(item["path"])
        else:
            missing.append(item["path"])

    print("模型体检: %d 就绪 / %d 缺失 / %d 损坏"
          % (ok, len(missing), len(bad)))
    need = missing + bad
    if not need:
        print("RESULT: ALL MODELS READY")
        return 0
    if args.check:
        for p in need:
            print("  需要:", p)
        return 1

    for rel in need:
        item = next(x for x in manifest["files"] if x["path"] == rel)
        url = "%s/%s" % (manifest["base_url"], asset_name(rel))
        dest = os.path.join(MODELS_DIR, rel.replace("/", os.sep))
        print("  下载 %s" % rel)
        download(url, dest)
        if not verify(dest, item["sha256"]):
            os.remove(dest)
            raise RuntimeError("sha256 校验失败: %s" % rel)
        print("    校验通过")
    print("RESULT: ALL MODELS READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
