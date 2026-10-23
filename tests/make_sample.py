"""Generate a synthetic test portrait.

The fixture is deliberately synthetic: it gives the test suite a deterministic
input with a known face position, skin region and blemish set, so the
face-dependent stages can be exercised without shipping a real photo.

Construction order: background -> hair -> face volume -> features -> blemishes
-> texture. Keeping the background clearly non-skin matters, because the skin
chroma mask and the face detector both rely on that separation.
"""
import os
import numpy as np
import cv2

H, W = 1000, 800
FACE_CX, FACE_CY = W * 0.50, H * 0.40
FACE_RX, FACE_RY = W * 0.155, H * 0.185


def _grid(h, w):
    return np.mgrid[0:h, 0:w].astype(np.float32)


def background(h, w):
    """Cool grey studio backdrop with a soft vignette."""
    yy, xx = _grid(h, w)
    g = 0.30 + 0.10 * (1.0 - yy / h)
    img = np.stack([g * 0.94, g * 0.97, g * 1.06], axis=2)
    # soft pool of light behind the subject
    r = np.sqrt(((xx - w * 0.5) / (w * 0.75)) ** 2 + ((yy - h * 0.35) / (h * 0.7)) ** 2)
    img *= (1.15 - 0.35 * np.clip(r, 0, 1.4))[..., None]
    return img.astype(np.float32)


def hair(img):
    """Dark hair mass behind and above the head."""
    h, w = img.shape[:2]
    out = img.copy()
    cv2.ellipse(out, (int(FACE_CX), int(FACE_CY - FACE_RY * 0.10)),
                (int(FACE_RX * 1.30), int(FACE_RY * 1.24)),
                0, 0, 360, (0.16, 0.12, 0.10), -1, cv2.LINE_AA)
    # side strands falling past the jaw
    for sign in (-1, 1):
        pts = np.array([
            [FACE_CX + sign * FACE_RX * 0.95, FACE_CY - FACE_RY * 0.75],
            [FACE_CX + sign * FACE_RX * 1.35, FACE_CY + FACE_RY * 0.60],
            [FACE_CX + sign * FACE_RX * 1.05, FACE_CY + FACE_RY * 1.45],
            [FACE_CX + sign * FACE_RX * 0.70, FACE_CY + FACE_RY * 0.30],
        ], np.int32)
        cv2.fillPoly(out, [pts], (0.16, 0.12, 0.10), cv2.LINE_AA)
    return out


def neck_and_shoulders(img):
    h, w = img.shape[:2]
    out = img.copy()
    # neck
    cv2.rectangle(out,
                  (int(FACE_CX - FACE_RX * 0.42), int(FACE_CY + FACE_RY * 0.70)),
                  (int(FACE_CX + FACE_RX * 0.42), int(h)),
                  (0.74, 0.57, 0.50), -1, cv2.LINE_AA)
    # shoulders
    cv2.ellipse(out, (int(FACE_CX), int(h * 1.16)),
                (int(w * 0.46), int(h * 0.30)), 0, 180, 360,
                (0.42, 0.45, 0.52), -1, cv2.LINE_AA)
    return out


def face_volume(img):
    """Oval face with directional shading, so it reads as a 3D volume."""
    h, w = img.shape[:2]
    yy, xx = _grid(h, w)
    nx = (xx - FACE_CX) / FACE_RX
    ny = (yy - FACE_CY) / FACE_RY
    r = np.sqrt(nx * nx + ny * ny)

    # smooth mask over the oval
    mask = np.clip(1.0 - (r - 0.86) / 0.16, 0.0, 1.0)
    mask = mask * mask * (3 - 2 * mask)

    # base skin
    skin = np.zeros((h, w, 3), np.float32)
    skin[..., 0] = 0.90
    skin[..., 1] = 0.73
    skin[..., 2] = 0.65

    # key light from the upper left, plus a warm bounce from below
    light = (1.0
             - 0.22 * np.clip(nx + 0.55 * ny, 0, 2)
             + 0.10 * np.clip(-ny, 0, 1)
             - 0.30 * np.clip(r - 0.55, 0, 1) ** 1.5)
    skin *= light[..., None]

    # forehead and cheekbone highlights
    for (cx, cy, rx, ry, amt) in [
        (FACE_CX - FACE_RX * 0.30, FACE_CY - FACE_RY * 0.52, FACE_RX * 0.42, FACE_RY * 0.20, 0.10),
        (FACE_CX + FACE_RX * 0.34, FACE_CY - FACE_RY * 0.12, FACE_RX * 0.34, FACE_RY * 0.22, 0.07),
        (FACE_CX - FACE_RX * 0.36, FACE_CY - FACE_RY * 0.14, FACE_RX * 0.32, FACE_RY * 0.22, 0.07),
    ]:
        d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        skin += (np.clip(1.0 - d, 0, 1) ** 1.6)[..., None] * amt

    # jaw shading toward the neck
    jaw = np.clip((ny - 0.55) / 0.45, 0, 1)
    skin *= (1.0 - 0.28 * jaw)[..., None]

    out = img * (1.0 - mask[..., None]) + skin * mask[..., None]
    return np.clip(out, 0, 1).astype(np.float32)


def features(img):
    """Eyes, brows, nose, mouth with teeth, and under-eye shadow."""
    h, w = img.shape[:2]
    out = img.copy()

    eye_y = FACE_CY - FACE_RY * 0.20
    eye_dx = FACE_RX * 0.46
    ew, eh = FACE_RX * 0.30, FACE_RY * 0.105

    for sign in (-1, 1):
        ex = int(FACE_CX + sign * eye_dx)
        ey = int(eye_y)

        # eye socket shading
        cv2.ellipse(out, (ex, ey), (int(ew * 1.25), int(eh * 1.55)), 0, 0, 360,
                    (0.66, 0.52, 0.48), -1, cv2.LINE_AA)
        # sclera
        cv2.ellipse(out, (ex, ey), (int(ew), int(eh)), 0, 0, 360,
                    (0.94, 0.93, 0.92), -1, cv2.LINE_AA)
        # iris + pupil
        cv2.circle(out, (ex, ey), int(ew * 0.46), (0.31, 0.23, 0.16), -1, cv2.LINE_AA)
        cv2.circle(out, (ex, ey), int(ew * 0.20), (0.05, 0.04, 0.04), -1, cv2.LINE_AA)
        # specular highlight
        cv2.circle(out, (ex - int(ew * 0.16), ey - int(eh * 0.30)),
                   max(2, int(ew * 0.10)), (1.0, 1.0, 1.0), -1, cv2.LINE_AA)
        # upper lash line
        cv2.ellipse(out, (ex, ey), (int(ew * 1.02), int(eh * 1.02)), 0, 180, 360,
                    (0.10, 0.08, 0.08), 3, cv2.LINE_AA)
        # lower lash line, lighter
        cv2.ellipse(out, (ex, ey), (int(ew * 0.98), int(eh * 0.96)), 0, 0, 180,
                    (0.55, 0.44, 0.42), 1, cv2.LINE_AA)
        # brow
        cv2.ellipse(out, (ex, int(ey - FACE_RY * 0.20)),
                    (int(ew * 1.20), int(FACE_RY * 0.035)), 0, 180, 360,
                    (0.22, 0.15, 0.11), -1, cv2.LINE_AA)

        # under-eye shadow (the eye-bag cue)
        cv2.ellipse(out, (ex, int(ey + eh * 1.9)), (int(ew * 1.05), int(eh * 0.85)),
                    0, 0, 360, (0.63, 0.49, 0.47), -1, cv2.LINE_AA)

    # nose: bridge highlight + tip and nostril shadow
    ny0 = FACE_CY + FACE_RY * 0.02
    cv2.ellipse(out, (int(FACE_CX), int(ny0 - FACE_RY * 0.22)),
                (int(FACE_RX * 0.055), int(FACE_RY * 0.22)), 0, 0, 360,
                (0.95, 0.80, 0.72), -1, cv2.LINE_AA)
    cv2.ellipse(out, (int(FACE_CX), int(ny0 + FACE_RY * 0.12)),
                (int(FACE_RX * 0.20), int(FACE_RY * 0.10)), 0, 0, 360,
                (0.70, 0.54, 0.48), -1, cv2.LINE_AA)
    for sign in (-1, 1):
        cv2.circle(out, (int(FACE_CX + sign * FACE_RX * 0.15), int(ny0 + FACE_RY * 0.16)),
                   max(2, int(FACE_RX * 0.030)), (0.40, 0.30, 0.27), -1, cv2.LINE_AA)

    # mouth
    my = int(FACE_CY + FACE_RY * 0.55)
    mw, mh = int(FACE_RX * 0.46), int(FACE_RY * 0.115)
    cv2.ellipse(out, (int(FACE_CX), my), (mw, mh), 0, 0, 360,
                (0.63, 0.34, 0.34), -1, cv2.LINE_AA)
    # teeth, deliberately a little yellow so de-yellow has something to do
    cv2.ellipse(out, (int(FACE_CX), my), (int(mw * 0.80), int(mh * 0.52)), 0, 0, 360,
                (0.86, 0.82, 0.68), -1, cv2.LINE_AA)
    # lip line
    cv2.ellipse(out, (int(FACE_CX), my), (mw, mh), 0, 0, 360,
                (0.45, 0.22, 0.22), 2, cv2.LINE_AA)
    return out


def blemishes(img, seed=11):
    """Acne-like spots and pigment marks, placed inside the face oval."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    out = img.copy()
    spots = []

    # scattered small acne on the cheeks, chin and forehead
    tries = 0
    while len(spots) < 46 and tries < 4000:
        tries += 1
        nx = rng.uniform(-0.72, 0.72)
        ny = rng.uniform(-0.80, 0.78)
        if nx * nx + ny * ny > 0.70:
            continue
        # keep away from eyes and mouth
        if abs(ny + 0.20) < 0.22 and abs(abs(nx) - 0.46) < 0.30:
            continue
        if abs(ny - 0.55) < 0.20:
            continue
        x = int(FACE_CX + nx * FACE_RX)
        y = int(FACE_CY + ny * FACE_RY)
        r = int(rng.uniform(2.5, 6.0))
        sev = float(rng.uniform(0.30, 0.80))
        col = (0.70 - 0.28 * sev, 0.24 - 0.10 * sev, 0.22 - 0.09 * sev)
        cv2.circle(out, (x, y), r, col, -1, cv2.LINE_AA)
        # a slightly darker core, as inflamed spots have
        cv2.circle(out, (x, y), max(1, r // 2),
                   (col[0] * 0.82, col[1] * 0.82, col[2] * 0.82), -1, cv2.LINE_AA)
        spots.append((x, y, r))

    # a few larger pigment marks
    for (nx, ny, r) in [(-0.50, 0.24, 9), (0.55, -0.02, 7), (-0.30, 0.52, 6)]:
        x = int(FACE_CX + nx * FACE_RX)
        y = int(FACE_CY + ny * FACE_RY)
        cv2.circle(out, (x, y), r, (0.62, 0.44, 0.36), -1, cv2.LINE_AA)
        spots.append((x, y, r))

    out = cv2.GaussianBlur(out, (0, 0), 0.7)
    return out, spots


def skin_texture(img, seed=7):
    """Pore-level detail plus low-frequency blotchiness."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]

    fine = rng.standard_normal((h, w)).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (0, 0), 0.9)
    fine /= max(float(fine.std()), 1e-6)

    blotch = rng.standard_normal((h // 10, w // 10)).astype(np.float32)
    blotch = cv2.resize(blotch, (w, h), interpolation=cv2.INTER_CUBIC)
    blotch = cv2.GaussianBlur(blotch, (0, 0), 14)
    blotch /= max(float(blotch.std()), 1e-6)

    out = img + fine[..., None] * 0.020 + blotch[..., None] * 0.028
    return np.clip(out, 0, 1).astype(np.float32)


def build():
    img = background(H, W)
    img = neck_and_shoulders(img)
    img = hair(img)
    img = face_volume(img)
    img = features(img)
    img, spots = blemishes(img)
    img = skin_texture(img)
    return img, spots


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.normpath(os.path.join(here, "..", "samples"))
    os.makedirs(outdir, exist_ok=True)

    img, spots = build()
    path = os.path.join(outdir, "synthetic_portrait.jpg")
    u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(u8, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 96])
    print(f"wrote {path}  shape={img.shape}  blemishes={len(spots)}")
    print(f"expected face box: cx={FACE_CX:.0f} cy={FACE_CY:.0f} "
          f"rx={FACE_RX:.0f} ry={FACE_RY:.0f}")
    return path


if __name__ == "__main__":
    main()
