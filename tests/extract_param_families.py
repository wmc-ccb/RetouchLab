"""Recover the reference application's parameter names from the worker binary.

The previous pass only looked for SCREAMING_SNAKE names paired with a number,
which surfaced SQLite and OpenCV internals rather than the application's own
controls. The real parameter names use CamelCase (`ColorGradeGlobalHue`,
`SharpenEdgeMasking`), which is a much better filter.

This enumerates every CamelCase string in the image and clusters it by family
prefix, which shows the control surface directly — and, by subtraction against
this project's own parameter list, exactly which controls are still missing.
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.extract_param_map import scan

# CamelCase identifier: two or more capitalised words, optionally with digits
CAMEL = re.compile(r"^[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+){1,6}$")
# also accept a leading lowercase for things like `retouch_type`
CAMEL_ANY = re.compile(r"^[A-Za-z][A-Za-z0-9]{3,60}$")


def family(name: str) -> str:
    """The leading word(s) that identify a parameter family."""
    parts = re.findall(r"[A-Z][a-z0-9]*", name)
    if not parts:
        return name
    # two-word prefixes separate families better than one
    for n in (2, 1):
        if len(parts) >= n:
            return "".join(parts[:n])
    return parts[0]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Program Files\PixCake\pixcake_2\pix-worker.exe"
    hits = scan(path)
    print(f"ascii runs: {len(hits):,}")

    names = Counter()
    for _sec, _off, text in hits:
        for token in re.split(r"[\s,;:=\"'\[\]{}()<>|/\\]+", text):
            if 5 <= len(token) <= 60 and CAMEL.match(token):
                names[token] += 1
    print(f"distinct CamelCase identifiers: {len(names):,}")

    fams: dict = defaultdict(Counter)
    for n, c in names.items():
        fams[family(n)][n] = c

    # families with several members are the interesting ones
    interesting = [(f, m) for f, m in fams.items() if len(m) >= 3]
    interesting.sort(key=lambda kv: -len(kv[1]))
    print(f"\nfamilies with >= 3 members: {len(interesting)}")
    print()
    for f, m in interesting[:45]:
        sample = ", ".join(sorted(m)[:8])
        print(f"  {f:<22} {len(m):>3}  {sample}")

    # ---- what this project already covers --------------------------------
    from engine.pipeline import DEFAULTS
    from server.app import PARAM_SPEC

    mine = set(DEFAULTS)
    print(f"\nthis project's parameters: {len(mine)}")
    print(f"this project's UI controls: {len(PARAM_SPEC)}")

    # map each recovered family to whether we have an equivalent
    covered, missing = [], []
    for f, m in interesting[:45]:
        low = f.lower()
        hit = any(low[:6] in k.lower() for k in mine)
        (covered if hit else missing).append((f, len(m), sorted(m)[:4]))

    print(f"\n=== families with a local counterpart ({len(covered)}) ===")
    for f, n, s in covered:
        print(f"  {f:<22} {n:>3}  {s}")

    print(f"\n=== families with NO local counterpart ({len(missing)}) ===")
    for f, n, s in missing:
        print(f"  {f:<22} {n:>3}  {s}")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "out", "pixworker_params.txt")
    with open(out, "w", encoding="utf-8") as fh:
        for f, m in sorted(fams.items()):
            for n in sorted(m):
                fh.write(f"{f}\t{n}\t{m[n]}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
