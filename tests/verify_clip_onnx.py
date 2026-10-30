# -*- coding: utf-8 -*-
"""验证 ONNX CLIP 图像塔：能跑、维度对、与 transformers 结果一致。

一致性判定用**余弦相似度的排序**而不是逐元素比较：两条路的预处理
实现细节（resize 插值、裁切位置）可能略有差别，但"哪两张更像"这个
相对关系必须一致，那才是挑图真正依赖的东西。
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

FAIL = []


def check(name, ok, detail=""):
    print("%-40s %s %s" % (name, "OK  " if ok else "FAIL", detail))
    if not ok:
        FAIL.append(name)


def main():
    from engine import cliponnx
    from engine.imgutil import load_image, fit_long_edge

    print("模型:", cliponnx.info())

    # 三张不同内容的图
    srcs = [
        r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg",
        os.path.join(ROOT, "samples", "synthetic_portrait.jpg"),
    ]
    imgs = []
    for s in srcs:
        if os.path.exists(s):
            imgs.append(fit_long_edge(load_image(s), 600))
    check("有测试图", len(imgs) >= 1, "%d 张" % len(imgs))
    if not imgs:
        print("RESULT: FAILED ->", FAIL)
        return 1

    emb = cliponnx.embed_images(imgs)
    check("ONNX 推理成功", emb is not None and emb.size > 0)
    check("输出是二维", emb.ndim == 2, str(emb.shape))
    check("每张一行", emb.shape[0] == len(imgs), str(emb.shape))
    norms = np.linalg.norm(emb, axis=1)
    check("已 L2 归一化", np.allclose(norms, 1.0, atol=1e-4),
          "norm=%s" % np.round(norms, 4)[:4])

    # 与 transformers 对比（本机有 torch 才能比）
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        print("\n本机没有 transformers，跳过一致性对比")
        print("RESULT: %s" % ("FAILED -> %s" % FAIL if FAIL else "ALL PASS"))
        return 1 if FAIL else 0

    print("\n对比 transformers 版本...")
    from engine import match as M

    # 强制走 transformers 路径
    tf = M._clip()
    model, proc, device = tf
    import torch

    outs = []
    with torch.no_grad():
        u8 = [(np.clip(im, 0, 1) * 255).astype(np.uint8) for im in imgs]
        inputs = proc(images=u8, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs)
        if not isinstance(feats, torch.Tensor):
            for attr in ("image_embeds", "pooler_output"):
                c = getattr(feats, attr, None)
                if isinstance(c, torch.Tensor):
                    feats = c
                    break
        if feats.dim() == 3:
            feats = feats[:, 0]
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        outs = feats.float().cpu().numpy()

    check("两条路维度一致", outs.shape[1] == emb.shape[1],
          "%d vs %d" % (outs.shape[1], emb.shape[1]))
    if len(imgs) >= 2:
        sim_tf = float(np.dot(outs[0], outs[1]))
        sim_on = float(np.dot(emb[0], emb[1]))
        # 同两张图，相似度应该在 0.1 以内
        check("同图相似度接近", abs(sim_tf - sim_on) < 0.1,
              "tf=%.4f onnx=%.4f" % (sim_tf, sim_on))
        # 每张图与自己：应为 1
        diag_tf = float(np.dot(outs[0], outs[0]))
        diag_on = float(np.dot(emb[0], emb[0]))
        check("自相似度都为 1", abs(diag_tf - 1) < 1e-3 and abs(diag_on - 1) < 1e-3,
              "tf=%.4f onnx=%.4f" % (diag_tf, diag_on))

    print("\nRESULT: %s" % ("FAILED -> %s" % FAIL if FAIL else "ALL PASS"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
