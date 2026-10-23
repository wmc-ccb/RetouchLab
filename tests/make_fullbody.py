"""Generate a synthetic full-body figure for pose-estimation testing.

MediaPipe pose needs visible shoulders, hips and limbs, so the head-and-
shoulders fixture used by the retouch tests is not enough. This builds a
simple upright figure with the proportions of a standing person.
"""
import os
import numpy as np
import cv2

H, W = 1200, 700
HEAD_CX, HEAD_CY = W * 0.50, H * 0.11
HEAD_R = 52


def _ellipse(img, c, axes, color, angle=0, thickness=-1):
    cv2.ellipse(img, (int(c[0]), int(c[1])), (int(axes[0]), int(axes[1])),
                angle, 0, 360, color, thickness, cv2.LINE_AA)


def limb(img, p0, p1, r, color):
    """A rounded capsule between two joints."""
    p0 = np.array(p0, np.float32)
    p1 = np.array(p1, np.float32)
    cv2.line(img, tuple(p0.astype(int)), tuple(p1.astype(int)), color,
             int(r * 2), cv2.LINE_AA)
    cv2.circle(img, tuple(p0.astype(int)), int(r), color, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(p1.astype(int)), int(r), color, -1, cv2.LINE_AA)


def build():
    # backdrop
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = 0.34 + 0.10 * (1.0 - yy / H)
    img = np.stack([bg * 0.95, bg * 0.98, bg * 1.05], axis=2).astype(np.float32)

    skin = (0.88, 0.70, 0.62)
    cloth = (0.30, 0.36, 0.52)
    trouser = (0.20, 0.22, 0.30)

    # skeleton proportions for a ~7.5-head standing figure
    head = (HEAD_CX, HEAD_CY)
    neck = (HEAD_CX, HEAD_CY + HEAD_R * 1.55)
    sh_l = (HEAD_CX - HEAD_R * 1.55, neck[1] + 18)
    sh_r = (HEAD_CX + HEAD_R * 1.55, neck[1] + 18)
    hip_c = (HEAD_CX, neck[1] + HEAD_R * 5.2)
    hip_l = (HEAD_CX - HEAD_R * 0.85, hip_c[1])
    hip_r = (HEAD_CX + HEAD_R * 0.85, hip_c[1])
    knee_l = (hip_l[0] - 6, hip_c[1] + HEAD_R * 3.6)
    knee_r = (hip_r[0] + 6, hip_c[1] + HEAD_R * 3.6)
    ank_l = (knee_l[0] - 4, hip_c[1] + HEAD_R * 7.2)
    ank_r = (knee_r[0] + 4, hip_c[1] + HEAD_R * 7.2)
    elb_l = (sh_l[0] - 26, sh_l[1] + HEAD_R * 2.6)
    elb_r = (sh_r[0] + 26, sh_r[1] + HEAD_R * 2.6)
    wr_l = (elb_l[0] - 10, elb_l[1] + HEAD_R * 2.5)
    wr_r = (elb_r[0] + 10, elb_r[1] + HEAD_R * 2.5)

    # legs and arms
    for (a, b) in [(hip_l, knee_l), (knee_l, ank_l), (hip_r, knee_r), (knee_r, ank_r)]:
        limb(img, a, b, HEAD_R * 0.42, trouser)
    for (a, b) in [(sh_l, elb_l), (elb_l, wr_l), (sh_r, elb_r), (elb_r, wr_r)]:
        limb(img, a, b, HEAD_R * 0.26, cloth)
    # hands
    _ellipse(img, wr_l, (HEAD_R * 0.26, HEAD_R * 0.34), skin)
    _ellipse(img, wr_r, (HEAD_R * 0.26, HEAD_R * 0.34), skin)

    # torso
    torso = np.array([sh_l, sh_r, (hip_r[0] + 8, hip_r[1]), (hip_l[0] - 8, hip_l[1])], np.int32)
    cv2.fillPoly(img, [torso], cloth, cv2.LINE_AA)
    # neck
    limb(img, neck, (HEAD_CX, sh_l[1] + 6), HEAD_R * 0.30, skin)

    # head
    _ellipse(img, head, (HEAD_R * 0.86, HEAD_R * 1.10), skin)
    # hair
    cv2.ellipse(img, (int(head[0]), int(head[1] - HEAD_R * 0.18)),
                (int(HEAD_R * 0.92), int(HEAD_R * 0.92)), 0, 180, 360,
                (0.18, 0.13, 0.11), -1, cv2.LINE_AA)
    # eyes
    for sign in (-1, 1):
        ex = int(head[0] + sign * HEAD_R * 0.34)
        ey = int(head[1] - HEAD_R * 0.10)
        cv2.circle(img, (ex, ey), int(HEAD_R * 0.13), (0.94, 0.94, 0.93), -1, cv2.LINE_AA)
        cv2.circle(img, (ex, ey), int(HEAD_R * 0.07), (0.28, 0.20, 0.14), -1, cv2.LINE_AA)
    # mouth
    cv2.ellipse(img, (int(head[0]), int(head[1] + HEAD_R * 0.42)),
                (int(HEAD_R * 0.26), int(HEAD_R * 0.12)), 0, 0, 360,
                (0.62, 0.34, 0.33), -1, cv2.LINE_AA)

    # ground shadow
    shadow = np.zeros((H, W), np.float32)
    _ellipse(shadow, (HEAD_CX, ank_l[1] + 12), (HEAD_R * 3.4, HEAD_R * 0.55), 1.0)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 22)
    img *= (1.0 - 0.35 * shadow)[..., None]

    # gentle film grain so the denoiser has something to work with
    rng = np.random.default_rng(5)
    noise = rng.standard_normal((H, W)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.8)
    noise /= max(float(noise.std()), 1e-6)
    img = img + noise[..., None] * 0.016

    return np.clip(img, 0, 1).astype(np.float32)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.normpath(os.path.join(here, "..", "samples"))
    os.makedirs(outdir, exist_ok=True)
    img = build()
    path = os.path.join(outdir, "synthetic_fullbody.jpg")
    u8 = (img * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(u8, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"wrote {path}  {img.shape}")
    return path


if __name__ == "__main__":
    main()
