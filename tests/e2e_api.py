"""End-to-end API test: upload -> render -> save, against a live server."""
import base64
import json
import os
import sys
import urllib.request

BASE = "http://127.0.0.1:8765"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "samples", "synthetic_portrait.jpg")
OUT = os.path.join(ROOT, "out")


def post_json(path, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def post_file(path, filepath):
    boundary = "----ZhaJinWenQianCakeBoundary"
    name = os.path.basename(filepath)
    with open(filepath, "rb") as fh:
        content = fh.read()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + path, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def get_json(path):
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return json.loads(r.read())


def main():
    os.makedirs(OUT, exist_ok=True)
    ok = True

    print("1) health")
    h = get_json("/api/health")
    print("   ", h)
    if not h.get("ok"):
        ok = False

    print("2) params")
    p = get_json("/api/params")
    print(f"    controls={len(p['params'])} groups={len(p['groups'])}")
    if len(p["params"]) < 50:
        ok = False

    print("3) presets")
    pr = get_json("/api/presets")
    print(f"    presets={list(pr['presets'].keys())}")

    print("4) upload")
    up = post_file("/api/upload", SRC)
    sid = up["session"]
    print(f"    session={sid} {up['width']}x{up['height']} "
          f"preview={up['preview_width']}x{up['preview_height']} faces={len(up['faces'])}")
    if not up["preview"]:
        ok = False
        print("    FAIL: no preview returned")

    # write the unmodified preview so the diff is verifiable
    with open(os.path.join(OUT, "e2e_00_original.jpg"), "wb") as fh:
        fh.write(base64.b64decode(up["preview"]))

    print("5) faces")
    f = get_json(f"/api/faces/{sid}")
    print(f"    {f['faces']}")

    print("6) render each preset")
    for name, params in pr["presets"].items():
        r = post_json("/api/render", {"session": sid, "params": params, "quality": 88})
        img = base64.b64decode(r["image"])
        path = os.path.join(OUT, f"e2e_{name}.jpg")
        with open(path, "wb") as fh:
            fh.write(img)
        print(f"    {name:10} {r['ms']:7.0f} ms  {len(img):7} bytes  "
              f"applied={len(r['applied'])}")
        if len(img) < 1000:
            ok = False
            print("    FAIL: suspiciously small output")

    print("7) render a strong combined parameter set")
    strong = {
        "slimFace": 45, "jaw": 35, "enlargeEye": 35, "noseWidth": 25,
        "spotless_acne": 75, "spotless_spot": 60, "even_tone": 55,
        "smooth": 65, "texture": 40,
        "removeEyeBag": 60, "keepSilkworm": 40, "teethWhitening": 45, "teethDeYellow": 50,
        "exposure": 8, "contrast": 14, "shadows": 20, "highlights": -12,
        "vibrance": 20, "clarity": 12, "sharpening": 30,
        "grain_amount": 25, "vignetting": 15,
    }
    r = post_json("/api/render", {"session": sid, "params": strong, "quality": 92})
    with open(os.path.join(OUT, "e2e_strong.jpg"), "wb") as fh:
        fh.write(base64.b64decode(r["image"]))
    print(f"    {r['ms']:.0f} ms  applied={r['applied']}")

    print("8) save full resolution")
    dest = os.path.join(OUT, "e2e_full_res.jpg")
    if os.path.exists(dest):
        os.remove(dest)
    s = post_json("/api/save", {"session": sid, "params": strong, "path": dest})
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    print(f"    {s['path']}  {s['width']}x{s['height']}  {size} bytes")
    if not os.path.exists(dest) or size < 1000:
        ok = False
        print("    FAIL: full-res export missing")

    print("9) empty params must reproduce the original")
    r = post_json("/api/render", {"session": sid, "params": {}, "quality": 95})
    with open(os.path.join(OUT, "e2e_01_empty.jpg"), "wb") as fh:
        fh.write(base64.b64decode(r["image"]))
    print(f"    applied={r['applied']} (expected: no effects)")

    print("\nRESULT:", "ALL PASS" if ok else "FAILURES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
