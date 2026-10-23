# -*- coding: utf-8 -*-
"""把 CLIP 视觉塔导出成 ONNX，供打包版「挑图」用（不需要 PyTorch 运行时）。

产物：models/clip_image_encoder.onnx  (~350MB)

用法：
    python tools/export_clip_onnx.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT = os.path.join(ROOT, "models", "clip_image_encoder.onnx")
NAME = "openai/clip-vit-base-patch32"


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        print("已存在:", OUT, "%.1f MB" % (os.path.getsize(OUT) / 1024 ** 2))
        return 0

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from transformers import CLIPModel

    print("加载 CLIP 视觉塔:", NAME)
    model = CLIPModel.from_pretrained(NAME, local_files_only=True)
    vis = model.vision_model.eval()
    proj = model.visual_projection.eval()

    class VisionOnly(torch.nn.Module):
        """视觉塔 + visual_projection，输出与 get_image_features 一致。

        只导 pooler_output 是错的：那是视觉塔的 768 维池化特征，而
        `get_image_features` 还会过一层 visual_projection 投影到 512 维
        的多模态空间。少了这层，嵌入落在另一个空间里，两张同图的余弦
        相似度只有 0.26（实测），挑图会给出完全错误的排序。
        """

        def __init__(self, vm, vp):
            super().__init__()
            self.vm = vm
            self.vp = vp

        def forward(self, pixel_values):
            pooled = self.vm(pixel_values=pixel_values).pooler_output
            return self.vp(pooled)

    net = VisionOnly(vis, proj).eval()
    dummy = torch.zeros(1, 3, 224, 224)

    print("导出 ONNX ...")
    with torch.no_grad():
        torch.onnx.export(
            net, dummy, OUT,
            input_names=["pixel_values"],
            output_names=["image_features"],
            dynamic_axes={"pixel_values": {0: "batch"},
                          "image_features": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
        )
    # 外部数据文件合并检查
    side = OUT + ".data"
    if os.path.exists(side):
        print("注意：产生了外部权重文件", side)
    sz = os.path.getsize(OUT) / 1024 ** 2
    print("完成: %s  %.1f MB" % (OUT, sz))
    return 0


if __name__ == "__main__":
    sys.exit(main())
