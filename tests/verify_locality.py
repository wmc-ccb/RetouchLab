# -*- coding: utf-8 -*-
"""AI locality: an operation must not touch pixels outside its own mask.

This is the strongest cheap invariant an image pipeline has. A mask bug rarely
announces itself — the visible symptom is a faint halo, a colour cast on the
backdrop, or detail lost in the hair, and all three read as "the effect is a
bit off" rather than "there is a bug". Asserting that the *untouched* region is
bit-identical turns those into hard failures with a coordinate.

Three deliberate choices, each one learned from a wrong first attempt:

* **Engine level, not HTTP.** Every check calls the pipeline function on a
  float array. JPEG transport adds ~0.006 mean error of its own, which would
  drown a real one-level leak.
* **The target is the operation's own mask, not a geometric proxy.** The first
  version used the face box; the chroma skin mask legitimately extends to the
  neck, so that reported leaks that were not leaks. Where an operation derives
  its mask internally the same mask is reconstructed here.
* **Local vs global by design is written down.** Denoising, white balance and a
  global grade *must* change the whole frame; asserting locality there would be
  wrong, so they are listed as global with the reason.

Two contract shapes are checked, because one shape does not fit both:

* `local` — outside the mask, no change; inside, a real change.
* `swap`  — the subject must be untouched while the background must change
  (background replacement), which is the opposite polarity.

The reverse direction is checked as well: where an operation is supposed to
act, it must actually act, otherwise "no leakage" would be satisfied by doing
nothing at all.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.imgutil import load_image, fit_long_edge, blend
from engine.face import FaceDetector, skin_tone_mask, face_region_mask
from engine import semantic as SEM
from engine import skin as S
from engine import tone as T
from engine import matting as MT

SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"

# a leak is a pixel that moved by more than one 8-bit level
TOL = 1.0 / 255.0

failures = []


def worst_leak(d, target):
    """Coordinates and value of the largest change outside the target."""
    outside = (~target)
    if not outside.any():
        return None
    masked = np.where(outside, d, -1.0)
    y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return int(x), int(y), float(masked[y, x])


def report_local(name, before, after, mask, must_change=True, tol=TOL):
    d = np.abs(after - before).max(axis=2)
    tgt = mask > 0.002
    ins, outs = d[tgt], d[~tgt]
    in_max = float(ins.max()) if ins.size else 0.0
    out_max = float(outs.max()) if outs.size else 0.0
    acted = in_max > tol
    ok = (out_max <= tol) and (acted or not must_change)
    wl = worst_leak(d, tgt)
    print("%-26s in max=%.4f | OUT max=%.4f  %s"
          % (name, in_max, out_max,
             "OK" if ok else ("LEAK" if out_max > tol else "NO-OP")))
    if not ok:
        failures.append(name)
        if wl:
            print("        最差泄漏点 x=%d y=%d 值=%.4f" % wl)
    return ok


def report_swap(name, before, after, subject_mask, tol=TOL):
    """Subject untouched, background changed."""
    d = np.abs(after - before).max(axis=2)
    subj = subject_mask > 0.999
    sub_max = float(d[subj].max()) if subj.any() else 0.0
    bg_max = float(d[~subj].max()) if (~subj).any() else 0.0
    ok = sub_max <= tol and bg_max > tol
    print("%-26s 主体 max=%.4f | 背景 max=%.4f  %s"
          % (name, sub_max, bg_max, "OK" if ok else "FAIL"))
    if not ok:
        failures.append(name)
    return ok


def main():
    img = fit_long_edge(load_image(SRC), 900)
    h, w = img.shape[:2]
    print("image: %dx%d" % (w, h))

    det = FaceDetector()
    faces = det.detect(img)
    if not faces:
        print("no face detected; the locality test needs one")
        return 1
    f = faces[0]
    FaceDetector.build_regions(f, img.shape)
    reg = SEM.segment(img)
    chroma = skin_tone_mask(img, f)
    fs = float(max(f.box[2], f.box[3]))
    print("face box: (%.0f,%.0f) %.0fx%.0f" % f.box)

    print()
    print("=" * 88)
    print("局部操作：掩膜外必须不变（容差 %.4f = 1/255）" % TOL)
    print("=" * 88)

    geo = face_region_mask(img, f)
    out = S.spotless(img, geo, acne=70, spot=55, face_scale=fs,
                     soft_shade=f.soft_shade)
    report_local("祛痘(acne70/spot55)", img, out, geo)

    out = S.smooth_skin(img, chroma, amount=70, texture=30, face_scale=fs)
    report_local("磨皮(70)", img, out, chroma)

    out = S.even_skin_tone(img, chroma, 60)
    report_local("匀肤(60)", img, out, chroma)

    # eye ops build a radial mask around each eye; rebuild the same mask
    from engine.skin import radial_falloff_local
    eye_r, eye_l = f.eyes[0], f.eyes[1]
    eye_dist = float(np.linalg.norm(np.array(eye_l[:2]) - np.array(eye_r[:2])))
    ew, eh = eye_dist * 0.50, eye_dist * 0.30
    eyem = np.zeros((h, w), np.float32)
    for (ex, ey, ewidth, eheight) in (eye_r, eye_l):
        cx = ex + ewidth / 2.0
        cy = ey + eheight / 2.0 + eye_dist * 0.16
        eyem = np.maximum(eyem, radial_falloff_local(h, w, cx, cy, ew, eh))

    out = S.eye_bag_reduction(img, f, 60, 0)
    report_local("祛眼袋(60)", img, out, eyem)

    # the highlight mask is a much smaller ellipse inside the eye
    hlm = np.zeros((h, w), np.float32)
    for (ex, ey, ewidth, eheight) in f.eyes:
        hlm = np.maximum(hlm, radial_falloff_local(
            h, w, ex + ewidth / 2, ey + eheight / 2,
            ewidth * 0.30, eheight * 0.28))
    out = S.eye_highlight(img, f, 60)
    report_local("眼神光(60)", img, out, hlm)

    if f.mouth:
        from engine.skin import _mouth_mask
        mo = _mouth_mask(f, (h, w), inner=0.50)
        out = S.teeth_whiten(img, f, 60, 60)
        report_local("牙齿美白(60)", img, out, mo)

    # ---- whole skin pipeline: change must stay on skin ---------------------
    print()
    print("=" * 88)
    print("整条皮肤管线：皮肤掩膜外必须不变")
    print("=" * 88)
    from engine.pipeline import apply_preset
    out = S.retouch_skin(img, f, apply_preset("beauty"), semantic=reg)
    # the pipeline's gate is the semantic skin limited to the face; rebuild it
    if reg is not None and reg.confidence() >= 0.25:
        near = face_region_mask(img, f)
        gate = np.clip(reg.skin() * near, 0.0, 1.0)
        # blemish work also uses the geometric region at 0.6 weight, and the
        # eye/mouth ops use their own masks: the union is the honest target
        target = np.maximum(gate, geo * 0.6)
        target = np.maximum(target, eyem)
        target = np.maximum(target, hlm)
        if f.mouth:
            target = np.maximum(target, _mouth_mask(f, (h, w), inner=0.50))
        report_local("retouch_skin(beauty)", img, out, target)
    else:
        print("  跳过：语义置信度不足")

    # ---- background replacement: opposite polarity ------------------------
    print()
    print("=" * 88)
    print("换背景：主体必须不变，背景必须变")
    print("=" * 88)
    alpha = MT.subject_alpha(img, mode="hybrid")
    bgc = np.zeros_like(img)
    bgc[..., 0], bgc[..., 1], bgc[..., 2] = 0.10, 0.65, 0.75
    comp = MT.compose(img, alpha, background=bgc)
    report_swap("换背景(hybrid)", img, comp, alpha)

    # ---- liquify: a warp is bit-exact where the field is zero -------------
    print()
    print("=" * 88)
    print("液化/美型：位移场为零处必须【逐位精确】（比 1/255 更强的契约）")
    print("=" * 88)
    from engine import liquify as LQ

    # a full-body frame is needed for the pose-driven body warp
    fb = fit_long_edge(load_image(
        os.path.join(ROOT, "samples", "synthetic_fullbody.jpg")), 900)
    try:
        from engine.pose import PoseEstimator, pick_primary
        body_pose = pick_primary(PoseEstimator().detect(fb))
    except Exception:
        body_pose = None
    pose_ok = body_pose is not None
    print("全身姿态: %s" % ("可用" if pose_ok else "不可用（跳过身形测试）"))

    interior = np.zeros((h, w), bool)
    interior[6:-6, 6:-6] = True

    # baseline: a zero field must be an exact identity, borders included
    z = np.zeros((h, w), np.float32)
    ident = LQ.warp(img, z, z)
    print("零位移 warp 恒等性: 全图 max=%.7f  %s"
          % (float(np.abs(ident - img).max()),
             "OK" if np.array_equal(ident, img) else "FAIL"))
    if not np.array_equal(ident, img):
        failures.append("zero-displacement warp is not identity")

    from engine.pipeline import apply_preset
    geo_params = apply_preset("beauty")
    for label, fn in (
        ("slimFace(60)+jaw(40)", lambda: LQ.reshape_face(
            img, f, dict(geo_params, slimFace=60, jaw=40), return_field=True)),
        ("enlargeEye(60)", lambda: LQ.reshape_face(
            img, f, dict(geo_params, enlargeEye=60), return_field=True)),
    ):
        dx, dy = fn()
        warped = LQ.warp(img, dx, dy)
        mag = np.sqrt(dx * dx + dy * dy)
        zone = (mag == 0.0) & interior
        d = np.abs(warped - img).max(axis=2)
        leak = float(d[zone].max()) if zone.any() else 0.0
        moved = float(d[~zone].max())
        ok = leak == 0.0 and moved > TOL
        print("%-24s 位移=0处 max=%.7f (%.1f%% 像素) | 位移>0处 max=%.4f  %s"
              % (label, leak, 100 * zone.mean(), moved,
                 "OK" if ok else "LEAK"))
        if not ok:
            failures.append(label)

    if pose_ok:
        dx, dy = LQ.reshape_body(fb, {"slimLeg": 60, "slimWaist": 50},
                                 pose=body_pose, return_field=True)
        # reshape_body's smoothing step is global by design, so compare the
        # warp alone (bodySmooth is deliberately not in the params)
        warped = LQ.warp(fb, dx, dy)
        mag = np.sqrt(dx * dx + dy * dy)
        fb_interior = np.zeros(fb.shape[:2], bool)
        fb_interior[6:-6, 6:-6] = True
        zone = (mag == 0.0) & fb_interior
        d = np.abs(warped - fb).max(axis=2)
        leak = float(d[zone].max()) if zone.any() else 0.0
        print("%-24s 位移=0处 max=%.7f (%.1f%% 像素) | 位移>0处 max=%.4f  %s"
              % ("reshape_body(瘦腿)", leak, 100 * zone.mean(),
                 float(d[~zone].max()), "OK" if leak == 0.0 else "LEAK"))
        if leak != 0.0:
            failures.append("reshape_body warp")

    # ---- generative fill: needs ComfyUI ----------------------------------
    from engine import genfill
    if genfill.available():
        print()
        print("=" * 88)
        print("生成填充：mask 外必须不变")
        print("=" * 88)
        small = fit_long_edge(img, 640)
        m = np.zeros(small.shape[:2], np.float32)
        m[int(small.shape[0] * 0.05):int(small.shape[0] * 0.18),
          0:int(small.shape[1] * 0.10)] = 1.0
        try:
            filled, _ = genfill.fill(small, m, prompt="dark room background",
                                     steps=12)
            report_local("生成填充(12步)", small, filled, m)
        except Exception as exc:
            print("  跳过：%s" % str(exc)[:70])
    else:
        print("\n  生成填充跳过：ComfyUI 未运行")

    # ---- global by design: record, do not assert --------------------------
    print()
    print("=" * 88)
    print("设计上全局的操作（记录变化量，不断言局部）")
    print("=" * 88)
    for name, o in (("锐化(30)", T.sharpening(img, 30, 1.0, 50)),
                    ("AI降噪(80)", T.ai_denoise(img, True, 80)),
                    ("白平衡(temp20)", T.white_balance(img, 20, 0)),
                    ("颗粒(50)", T.grain(img, 50, 25, 50, 0, 0, 0)),
                    ("曝光(20)", T.exposure(img, 20))):
        d = np.abs(o - img)
        print("%-26s 全图 max=%.4f mean=%.4f  (全局，符合预期)"
              % (name, float(d.max()), float(d.mean())))

    # ---- background protection: compare masked vs unmasked ----------------
    print()
    print("=" * 88)
    print("背景保护：与不保护相比，背景变化应大幅下降而主体基本不变")
    print("=" * 88)
    from engine.pipeline import RetouchPipeline
    pipe = RetouchPipeline()
    sem_subj = reg.subject() if reg is not None else None
    subj = pipe.subject_mask(img.shape, faces, None, background_weight=0.35,
                             semantic=sem_subj)
    ins, outs = subj > 0.9, subj < 0.4
    adj = T.shadows(img, 50)

    plain = np.abs(adj - img).mean(axis=2)                       # no protection
    guarded = np.abs(blend(img, adj, subj) - img).mean(axis=2)   # with mask
    bg_drop = float(plain[outs].mean()) / max(float(guarded[outs].mean()), 1e-9)
    sub_keep = float(guarded[ins].mean()) / max(float(plain[ins].mean()), 1e-9)
    print("  背景变化: 无保护 %.5f → 有保护 %.5f  (下降 %.1fx)"
          % (plain[outs].mean(), guarded[outs].mean(), bg_drop))
    print("  主体变化: 无保护 %.5f → 有保护 %.5f  (保留 %.0f%%)"
          % (plain[ins].mean(), guarded[ins].mean(), 100 * sub_keep))
    ok = bg_drop > 1.8 and sub_keep > 0.9
    print("  %s" % ("OK" if ok else "FAIL"))
    if not ok:
        failures.append("background protection")

    print()
    if failures:
        print("FAILED: %d 项 -> %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
