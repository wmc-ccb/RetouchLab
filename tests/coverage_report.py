"""Map the reference application's recovered feature families onto this project.

`extract_param_families.py` recovered the control surface from the worker
binary. The earlier coverage check was a crude prefix match and reported almost
everything as missing, which is not useful. This does the comparison properly:

* each family is classified by what it does;
* it is matched against a concrete local capability, named explicitly;
* the result distinguishes **covered**, **partial** and **absent**, so the
  report says what still has to be built rather than just counting strings.

The comparison is the point of the exercise: it turns an open-ended "replicate
it" into a finite list of work.
"""
from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Recovered families, with what the binary evidence says they do and whether
# this project implements the same capability.
#
# status: "covered"  = a local implementation exists and is tested
#         "partial"  = a comparable capability exists but is not equivalent
#         "absent"   = nothing local implements it
FAMILIES = [
    # ---- colour and tone -------------------------------------------------
    ("ColorGrading", 32, "colour", "covered",
     "engine/tone.py: hsl_band (8 bands) + camera_primaries"),
    ("ColorGrade", 10, "colour", "covered",
     "engine/grade.py: split toning, global/mid/high x hue/sat/lum + balance + blending"),
    ("HueAdjustment", 9, "colour", "covered", "engine/tone.py: hsl_band, 8 bands"),
    ("WhiteBalance", 17, "colour", "covered",
     "temp/tint + auto + engine/wheel.py grey_point picker (/api/greypoint); bracketing still absent"),
    ("SelectColor", 38, "colour", "covered",
     "engine/wheel.py: CMYK colour-wheel families, /api/wheel, verified per-pixel"),
    ("ColorMatch", 29, "colour", "covered", "engine/match.py: match_color + /api/match"),
    ("LocalColor", 18, "colour", "covered",
     "engine/local.py: radial + linear gradient masks, 7 controls, stackable"),
    ("ColorRecommendation", 10, "colour", "partial",
     "engine/match.py scores and dedupes but does not recommend a look"),
    # ---- face and skin ---------------------------------------------------
    ("EffectFace", 15, "skin", "partial",
     "retouch + highlight covered; plump/lifting/redraw absent"),
    ("EffectRetouch", 9, "skin", "partial",
     "surface (frequency separation) covered; foggy/hessian/moisturizing absent"),
    ("PerfectFace", 13, "skin", "partial", "one-click preset; no per-face adaptive mode"),
    ("FacialReshape", 10, "geometry", "covered",
     "engine/liquify.py: reshape_face + 106 landmarks"),
    ("EffectRemove", 13, "skin", "partial",
     "eye bag covered; eye red blood, double eyelid, hair, gingiva, blue veins absent"),
    ("EyePupil", 8, "skin", "partial", "eye highlight exists; pupil reshaping does not"),
    ("OpenEyes", 9, "skin", "absent", "eye opening / squint correction not implemented"),
    ("Skin", 1, "skin", "covered",
     "engine/skin.py: full pipeline; semantic skin gate with confidence fallback"),
    # ---- body ------------------------------------------------------------
    ("LoseFat", 13, "body", "covered", "engine/liquify.py: reshape_body + pose"),
    ("AddAbs", 11, "body", "absent", "muscle definition enhancement not implemented"),
    ("RemoveBody", 9, "body", "partial",
     "body smoothing exists; body-hair removal does not"),
    ("FillLight", 13, "light", "partial",
     "exposure/shadows approximate it; no dedicated relight with a mask"),
    # ---- background ------------------------------------------------------
    ("MatteRefinement", 16, "matting", "covered",
     "engine/matting.py: hybrid alpha (MODNet hair + semantic accessories), 3 modes, refine_edges; /api/matte?mode="),
    ("EffectBg", 17, "matting", "partial",
     "blur/darken/desat/solid + accessory-preserving cutout; floor reflection and de-moire absent"),
    ("LensBlur", 15, "matting", "partial",
     "uniform background blur only; depth-varying / optical bokeh absent"),
    ("GenerativeFill", 10, "generative", "covered",
     "engine/genfill.py: ComfyUI inpaint + ControlNet, /api/fill, UI eraser"),
    ("FillLawn", 14, "generative", "partial",
     "same fill path with a scene prompt; no dedicated scene model"),
    ("ShoeClean", 12, "generative", "partial",
     "same fill path with an object prompt; no dedicated shoe model"),
    # ---- workflow --------------------------------------------------------
    ("FormulaWorkflow", 27, "workflow", "partial",
     "engine/pipeline.py PRESETS cover named looks; no workflow graph or DB"),
    ("CraftFormula", 15, "workflow", "absent", "no node-graph recipe system"),
    ("IdPhoto", 18, "workflow", "partial",
     "solid-background compositing works and is tested; no ID-photo spec engine"),
    ("SceneEnhance", 9, "workflow", "absent", "no scene-level auto enhance with upload"),
    ("EventTracking", 13, "telemetry", "absent",
     "telemetry is deliberately not implemented"),
    ("SaveImage", 10, "io", "covered", "server/app.py: /api/save, encode_jpeg"),
    ("LoadImage", 10, "io", "covered", "engine/imgutil.py: load_image + fit_long_edge"),
    # ---- capture / raw ---------------------------------------------------
    ("GetRaw", 11, "raw", "covered",
     "engine/rawio.py: decode via LibRaw, 13 demosaicers, WB control, p95 auto-exposure"),
    ("SetRaw", 9, "raw", "partial",
     "wb / gamma / brightness / exp_shift settable; no persistent per-file RAW model"),
    ("CameraLink", 13, "capture", "absent", "no tethered capture"),
    ("CanonPower", 24, "raw", "absent", "camera profiles are data, not code"),
    ("FujifilmFine", 16, "raw", "absent", "camera profiles are data, not code"),
    ("NikonCoolpix", 10, "raw", "absent", "camera profiles are data, not code"),
    # ---- agent / platform ------------------------------------------------
    ("PointerAgent", 16, "agent", "absent", "no conversational agent UI"),
    ("PixCake", 9, "platform", "absent", "application shell, not a feature"),
]


def main():
    by_status: dict = {}
    by_domain: dict = {}
    for name, n, domain, status, note in FAMILIES:
        by_status.setdefault(status, []).append((name, n, note))
        by_domain.setdefault(domain, []).append((name, n, status))

    print("=" * 86)
    print("=" * 86)
    for status, label in (("covered", "COVERED"), ("partial", "PARTIAL"), ("absent", "ABSENT")):
        items = by_status.get(status, [])
        total = sum(n for _, n, _ in items)
        print("\n### %s  (%d families, %d symbols)" % (label, len(items), total))
        for name, n, note in items:
            print("  %-22s %3d  %s" % (name, n, note))

    print()
    print("=" * 86)
    print("by domain")
    print("=" * 86)
    for domain in sorted(by_domain):
        items = by_domain[domain]
        c = sum(1 for _, _, s in items if s == "covered")
        p = sum(1 for _, _, s in items if s == "partial")
        a = sum(1 for _, _, s in items if s == "absent")
        bar = "#" * c + "+" * p + "." * a
        print("  %-12s n=%2d covered=%2d partial=%2d absent=%2d  %s"
              % (domain, len(items), c, p, a, bar))

    n_fam = len(FAMILIES)
    n_cov = sum(1 for f in FAMILIES if f[3] == "covered")
    n_par = sum(1 for f in FAMILIES if f[3] == "partial")
    print()
    print("family coverage: %d/%d fully implemented (%.0f%%), %d/%d partial (%.0f%%)"
          % (n_cov, n_fam, 100.0 * n_cov / n_fam, n_par, n_fam, 100.0 * n_par / n_fam))
    print("weighted (full=1, partial=0.5): %.0f%%"
          % (100.0 * (n_cov + 0.5 * n_par) / n_fam))

    print()
    print("=== largest remaining gaps (by symbol count) ===")
    gaps = sorted([f for f in FAMILIES if f[3] != "covered"], key=lambda f: -f[1])
    for name, n, domain, status, note in gaps[:12]:
        tag = "partial" if status == "partial" else "absent"
        print("  [%s] %-22s %3d  %s" % (tag, name, n, note))

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "out", "coverage_report.txt")
    with io.open(out_path, "w", encoding="utf-8") as fh:
        for name, n, domain, status, note in FAMILIES:
            fh.write(u"%s\t%s\t%s\t%d\t%s\n"
                     % (status, domain, name, n, note))
    print("\nwrote %s" % out_path)


if __name__ == "__main__":
    main()
