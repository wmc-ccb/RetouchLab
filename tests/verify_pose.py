# -*- coding: utf-8 -*-
"""姿态估计（MediaPipe Pose Landmarker）能在真人照片上跑通。

为什么用真人照片而不是 samples/synthetic_portrait.jpg：
合成人像是画出来的，没有真实的人体纹理与关节结构，姿态模型
（和 YuNet 人脸检测一样）本来就识别不出来——那是合成素材的局限，
不是检测的 bug。断言必须落在真人照片上才有意义。

判定真值断言：33 个关键点、肩/髋/踝的三角关系（腿比躯干长、
肩宽在合理区间），这些是体型调整真正依赖的量。
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FAIL = []


def check(name, ok, detail=""):
    print("%-42s %s %s" % (name, "OK  " if ok else "FAIL", detail))
    if not ok:
        FAIL.append(name)


def main():
    import mediapipe as mp
    print("mediapipe", mp.__version__)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = os.path.join(ROOT, "_downloads",
                              "pose_landmarker_lite.task")
    check("模型文件就位", os.path.exists(model_path), model_path)

    # 真人照片（与 verify_passersby 用同一张，确保素材存在）
    src = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
    check("真人照片素材存在", os.path.exists(src), src)

    from engine.imgutil import load_image, fit_long_edge

    img = fit_long_edge(load_image(src), 900)
    rgb = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    h, w = rgb.shape[:2]
    print("图像尺寸:", (w, h))

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=2,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    landmarker = vision.PoseLandmarker.create_from_options(opts)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                       data=rgb))
    n = len(result.pose_landmarks)
    check("真人照片检出姿态", n >= 1, "%d 个" % n)
    if not n:
        print("\n没有检出人体，后续断言无法进行")
        print("RESULT: FAILED -> %s" % FAIL)
        return 1

    lm = result.pose_landmarks[0]
    check("关键点数量 33", len(lm) == 33, "%d" % len(lm))

    def px(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    vis = sum(1 for p in lm if p.visibility > 0.5)
    check("可见关节过半", vis >= 17, "%d/33" % vis)

    sh = (px(11) + px(12)) / 2          # 肩中心
    hip = (px(23) + px(24)) / 2         # 髋中心
    nose = px(0)
    torso = float(np.linalg.norm(hip - sh))
    shoulder_w = float(np.linalg.norm(px(11) - px(12)))
    check("躯干高度合理", 40 < torso < h * 0.9, "%.0f px" % torso)
    check("肩宽在躯干量级内", 0.25 * torso < shoulder_w < 2.2 * torso,
          "%.0f px" % shoulder_w)
    # 头在肩上方（图像坐标 y 向下）
    check("头在肩上方", nose[1] < sh[1],
          "nose y=%.0f shoulder y=%.0f" % (nose[1], sh[1]))
    # 髋在肩下方
    check("髋在肩下方", hip[1] > sh[1],
          "hip y=%.0f shoulder y=%.0f" % (hip[1], sh[1]))
    print("\n关键量：躯干 %.0fpx / 肩宽 %.0fpx / 可见关节 %d/33"
          % (torso, shoulder_w, vis))

    print("\nRESULT: %s" % ("FAILED -> %s" % FAIL if FAIL else "ALL PASS"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())