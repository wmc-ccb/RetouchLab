# -*- coding: utf-8 -*-
"""Download SDXL inpainting (diffusers fp16) from ModelScope.

Files land in ComfyUI's models/diffusers/sdxl-inpainting/ so DiffusersLoader
can load the whole directory. curl per file, resumable, logged.
"""
import os
import subprocess
import sys
import time

BASE = ("https://modelscope.cn/models/AI-ModelScope/"
        "stable-diffusion-xl-1.0-inpainting-0.1/resolve/master/")
DST = (r"C:\Users\Rana\Desktop\ComfyUI\ComfyUI\models\diffusers"
       r"\sdxl-inpainting")
FILES = [
    ("model_index.json", None),
    ("scheduler/scheduler_config.json", None),
    ("text_encoder/config.json", None),
    ("text_encoder/model.fp16.safetensors", 0.23),
    ("text_encoder_2/config.json", None),
    ("text_encoder_2/model.fp16.safetensors", 1.29),
    ("tokenizer/merges.txt", None),
    ("tokenizer/special_tokens_map.json", None),
    ("tokenizer/tokenizer_config.json", None),
    ("tokenizer/vocab.json", None),
    ("tokenizer_2/merges.txt", None),
    ("tokenizer_2/special_tokens_map.json", None),
    ("tokenizer_2/tokenizer_config.json", None),
    ("tokenizer_2/vocab.json", None),
    ("unet/config.json", None),
    ("unet/diffusion_pytorch_model.fp16.safetensors", 4.78),
    ("vae/config.json", None),
    ("vae/diffusion_pytorch_model.fp16.safetensors", 0.16),
]


def main():
    os.makedirs(DST, exist_ok=True)
    failures = []
    for rel, _gb in FILES:
        dst = os.path.join(DST, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print("  skip (exists):", rel)
            continue
        url = BASE + rel
        print("  GET", rel, flush=True)
        r = subprocess.run(["curl", "-L", "--retry", "5", "--retry-delay",
                            "3", "--connect-timeout", "30", "-s", "-S",
                            "-o", dst, url],
                           capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            failures.append((rel, r.stderr[:120]))
            print("    FAIL:", r.stderr[:120], flush=True)
        elif os.path.exists(dst):
            print("    %.2f GB" % (os.path.getsize(dst) / 1073741824),
                  flush=True)
    print("\n失败 %d 个" % len(failures))
    for f in failures:
        print("  ", f)
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(DST) for f in fs)
    print("总大小: %.2f GB" % (total / 1073741824))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
