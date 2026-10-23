"""Extract the parameter-ID catalogue from pix-worker.exe by string analysis.

Full decompilation of this binary is not realistic: 168 MB of its 236 MB is
executable code (statically linked OpenCV, MNN and CUDA kernels), and Ghidra's
auto-analysis exceeded a 15-minute budget. The question this answers, though,
does not need a decompiler.

The pipeline's control surface is identified by numeric ids in the frontend
(`COMMON:{408:70, FaceSurgery:"V2"}`, `30036:"RAW_PROFILE_TS_COLOR"`). If those
ids and their names are string literals in the worker, they can be recovered
directly, and their *grouping* — which ids sit in the same table — is visible
from their adjacency in memory. That is what pins a parameter to an algorithm
family without recovering a single instruction.

Only readable strings are extracted. Nothing is decrypted, unpacked or
executed; the custom `.iVD*` sections are skipped entirely rather than being
treated as code.
"""
from __future__ import annotations

import os
import re
import struct
import sys
from collections import Counter, defaultdict

MIN_LEN = 5
ASCII = re.compile(rb"[\x20-\x7e]{%d,}" % MIN_LEN)


def pe_sections(path: str):
    with open(path, "rb") as fh:
        head = fh.read(0x1000)
    pe = struct.unpack_from("<I", head, 0x3C)[0]
    nsec = struct.unpack_from("<H", head, pe + 6)[0]
    opt = pe + 24
    sizeopt = struct.unpack_from("<H", head, pe + 20)[0]
    base = opt + sizeopt
    out = []
    for i in range(nsec):
        o = base + i * 40
        name = head[o:o + 8].rstrip(b"\x00").decode("latin1", "replace")
        vs, va, rs, ra = struct.unpack_from("<IIII", head, o + 8)
        ch = struct.unpack_from("<I", head, o + 36)[0]
        out.append((name, vs, va, rs, ra, ch))
    return out


def scan(path: str, skip_virtualized: bool = True):
    """Yield (section, file_offset, text) for every ASCII run in the file."""
    secs = pe_sections(path)
    fsize = os.path.getsize(path)
    hits = []
    with open(path, "rb") as fh:
        for (name, vs, va, rs, ra, ch) in secs:
            if rs == 0 or ra >= fsize:
                continue
            # `.iVD*` hold non-standard code that is very likely virtualised;
            # skip rather than guess at its contents
            if skip_virtualized and name.startswith(".iVD"):
                continue
            fh.seek(ra)
            blob = fh.read(min(rs, fsize - ra))
            for m in ASCII.finditer(blob):
                hits.append((name, ra + m.start(), m.group().decode("latin1")))
    return hits


# The frontend used string keys for the parameter catalogue. These are the
# shapes that identify an id alongside a name.
ID_NAME = re.compile(
    r"(?:^|[\"'\s{,])"
    r"(\d{1,6})"
    r"[\"'\s]*[:=][\"'\s]*"
    r"([A-Za-z_][A-Za-z0-9_ ]{2,60})"
)
NAME_ID = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]{2,60})"
    r"[\"'\s]*[:=][\"'\s]*"
    r"(\d{1,6})(?:[\"'\s,}]|$)"
)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Program Files\PixCake\pixcake_2\pix-worker.exe"
    print(f"scanning {os.path.basename(path)} …")
    hits = scan(path)
    print(f"ascii runs >= {MIN_LEN} chars: {len(hits):,}")

    by_sec = Counter(s for s, _, _ in hits)
    print("\nby section:")
    for s, n in by_sec.most_common():
        print(f"  {s:<10} {n:>10,}")

    # ---- id <-> name pairs ------------------------------------------------
    id2names: dict = defaultdict(Counter)
    name2ids: dict = defaultdict(Counter)
    for _sec, _off, text in hits:
        for m in ID_NAME.finditer(text):
            id2names[int(m.group(1))][m.group(2).strip()] += 1
        for m in NAME_ID.finditer(text):
            name2ids[m.group(1)][int(m.group(2))] += 1

    print(f"\ndistinct numeric ids with a candidate name: {len(id2names):,}")
    print(f"distinct names with a candidate id        : {len(name2ids):,}")

    # ---- parameter-looking names -----------------------------------------
    # The frontend's parameter ids were SCREAMING_SNAKE_CASE
    param_like = Counter()
    for name, ids in name2ids.items():
        if name.isupper() and "_" in name and len(name) > 5:
            param_like[name] = sum(ids.values())
    print(f"\nSCREAMING_SNAKE parameter-like names: {len(param_like):,}")
    for name, n in param_like.most_common(30):
        ids = sorted(name2ids[name])
        print(f"  {name:<42} ids={ids[:6]}{' …' if len(ids) > 6 else ''}")

    # ---- named groups -----------------------------------------------------
    print("\n=== candidate parameter groups ===")
    for token in ("FaceSurgery", "Face", "Skin", "Body", "Liquify", "RAW_PROFILE",
                  "COMMON", "Color", "Tone", "Retouch", "Beauty", "Shape"):
        found = [name for name in name2ids if token.lower() in name.lower()]
        if found:
            print(f"  {token:<12} {len(found):>4} names, e.g. {sorted(found)[:4]}")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "out", "pixworker_strings.txt")
    with open(out, "w", encoding="utf-8") as fh:
        for name, ids in sorted(name2ids.items()):
            fh.write(f"{name}\t{sorted(ids)}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
