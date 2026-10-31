# -*- coding: utf-8 -*-
"""FLUX.1 Fill 高质量档模型下载器（可选组件，~21GB）。

四个文件来自 HuggingFace 镜像 hf-mirror.com（直连 HF 在本机不通）：
  diffusion_models/flux1-fill-dev.safetensors  fp8 量化版 ~11.8GB
      （源: dim/..._fp8 单文件仓库, 官方 fp16 仓库是 gated 需要账号）
  clip/clip_l.safetensors                      ~246MB
  clip/t5xxl_fp16.safetensors                  ~9.5GB
  vae/ae.safetensors                           ~335MB

用法：python tools/fetch_flux.py        # 下载缺失的文件（断点续传）
      python tools/fetch_flux.py status # 只看状态

下载用系统 curl（-L 跟随重定向到 CDN，-C - 断点续传）。
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MIRROR = "https://hf-mirror.com"
FILES = [
    # (ComfyUI/models 子目录, 落地文件名, 来源 URL)
    ("diffusion_models", "flux1-fill-dev.safetensors",
     f"{MIRROR}/dim/black-forest-labs_FLUX.1-Fill-dev_flux1-fill-dev_fp8.safetensors"
     "/resolve/main/black-forest-labs_FLUX.1-Fill-dev_flux1-fill-dev_fp8.safetensors"),
    ("clip", "clip_l.safetensors",
     f"{MIRROR}/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"),
    ("clip", "t5xxl_fp16.safetensors",
     f"{MIRROR}/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"),
    ("vae", "ae.safetensors",
     f"{MIRROR}/modelzpalace/ae.safetensors/resolve/main/ae.safetensors"),
]


def target(sub: str, name: str) -> str:
    from engine.comfy_mgr import find_comfy_dir
    d = find_comfy_dir()
    if not d:
        raise SystemExit("未找到 ComfyUI 目录（tools/install_generative.py 可装）")
    return os.path.join(d, "ComfyUI", "models", sub, name)


def status():
    from engine.comfy_mgr import find_comfy_dir
    d = find_comfy_dir()
    if not d:
        print("ComfyUI 目录未找到")
        return 1
    for sub, name, url in FILES:
        p = os.path.join(d, "ComfyUI", "models", sub, name)
        if os.path.exists(p):
            print("[就绪] %-46s %8.1f MB" % (name, os.path.getsize(p) / 1048576))
        else:
            part = p + ".part"
            have = os.path.getsize(part) / 1048576 if os.path.exists(part) else 0
            print("[缺失] %-46s %s" % (name,
                  ("已下 %.1f MB(part)" % have) if have else "-"))
    return 0


def download():
    curl = r"C:\Windows\System32\curl.exe" if os.name == "nt" else "curl"
    if not os.path.exists(curl if os.name == "nt" else curl):
        curl = "curl"
    for sub, name, url in FILES:
        dest = target(sub, name)
        if os.path.exists(dest):
            print("[跳过] %s 已存在" % name, flush=True)
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print("[下载] %s" % name, flush=True)
        # -L 跟随 HF→CDN 重定向；-C - 断点续传；--retry 5 自动重试
        r = subprocess.run([curl, "-L", "-C", "-", "--retry", "5",
                            "--retry-delay", "3",
                            "--connect-timeout", "30",
                            "-o", dest + ".part", url])
        if r.returncode != 0:
            print("[失败] %s（curl exit %d）；重跑本命令会续传" % (name, r.returncode))
            return 1
        if os.path.exists(dest + ".part"):
            os.replace(dest + ".part", dest)
        print("[完成] %s (%.1f MB)" % (name, os.path.getsize(dest) / 1048576),
              flush=True)
    print("RESULT: FLUX MODELS READY")
    return 0


if __name__ == "__main__":
    if "--status" in sys.argv or "status" in sys.argv:
        sys.exit(status())
    sys.exit(download())
