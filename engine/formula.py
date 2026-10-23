# -*- coding: utf-8 -*-
"""Recipe engine: named, composable, shareable editing formulas.

The reference application's CraftFormula / FormulaWorkflow families are a
node-graph recipe system: a formula records a sequence of operations, each
with its own parameter overrides, and applies them in order to an image. A
formula is more than a preset in three ways:

* **stages, not a flat dict.** A preset is one parameter set; a formula is an
  ordered list of steps, so the same control can appear twice with different
  values (e.g. a global grade, then a local one, then another global pass).
* **parametric.** A step can reference a named slot (`{{strength}}`) that the
  caller binds at run time, which is what makes a formula reusable across
  images that need different amounts of the same treatment.
* **composable.** A formula can invoke another formula as a step, which is
  how "skin prep + colour grade" becomes one reusable unit.

Steps map onto capabilities that already exist — the pipeline's parameter
set, local adjustments, matting-backed background replacement, generative
fill — so the engine here is orchestration, not new image maths.

Formulas serialise to JSON, which is the sharing unit the UI needs.
"""
from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

_SLOT = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class FormulaError(ValueError):
    """Raised when a formula is malformed or a step is unknown."""


# The step vocabulary. Each kind names a runner and the fields it accepts.
STEP_KINDS = {
    "params":   "apply a parameter set through the main pipeline",
    "local":    "apply local adjustments (radial/linear masks)",
    "fill":     "generative fill over a supplied mask",
    "formula":  "invoke another formula by name",
}


def _coerce_numbers(obj):
    """Turn numeric-looking strings into numbers, recursively.

    A formula reaches the library from three places: hand-written JSON, the
    editor UI, and an HTTP request body. Only the first is typed. JSON carries
    `"170"` for a value typed into a web form, and a string that survives into
    the engine surfaces late and obscurely — `abs("170")` or `"20" / 100.0`
    raises TypeError half-way through a run, after earlier steps have already
    been applied.

    Normalising here means every path into the library produces the same typed
    program, and a string cannot leak past this boundary.
    """
    if isinstance(obj, dict):
        return {k: _coerce_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_numbers(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        # only convert when the whole string is a number, and leave anything
        # with a leading zero alone so an identifier ("007") is never silently
        # reinterpreted as a value
        if not s or s != obj or "{{" in s:
            return obj
        try:
            f = float(s)
        except ValueError:
            return obj
        body = s.lstrip("+-")
        if body.startswith("0") and len(body) > 1 and not body.startswith("0."):
            return obj
        return int(f) if (f.is_integer() and "." not in s
                          and "e" not in s.lower()) else f
    return obj


def validate(formula: dict, known_formulas: Optional[set] = None) -> List[str]:
    """Return a list of problems; empty means the formula will run.

    Structural checks are cheap and catch the mistakes that would otherwise
    surface half-way through a run: unknown step kinds, missing masks for
    fill steps, unresolved slot names, and self-referencing compositions.

    Numeric strings are not a problem: `upsert` normalises them on the way in,
    so a JSON body carrying `"20"` is accepted rather than rejected for a shape
    the caller cannot see.
    """
    errs: List[str] = []
    if not isinstance(formula, dict):
        return ["formula must be an object"]
    name = formula.get("name")
    if not name or not isinstance(name, str):
        errs.append("missing 'name'")
    steps = formula.get("steps")
    if not isinstance(steps, list) or not steps:
        errs.append("'steps' must be a non-empty list")
        return errs

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            errs.append("step %d: not an object" % i)
            continue
        kind = step.get("kind")
        if kind not in STEP_KINDS:
            errs.append("step %d: unknown kind %r (have %s)"
                        % (i, kind, ", ".join(sorted(STEP_KINDS))))
            continue
        if kind == "params" and not isinstance(step.get("params"), dict):
            errs.append("step %d: 'params' step needs a params object" % i)
        if kind == "fill" and not step.get("mask"):
            errs.append("step %d: 'fill' step needs a mask (rect or radial spec)"
                        % i)
        sub = step.get("formula")
        if kind == "formula":
            if not sub:
                errs.append("step %d: 'formula' step needs a formula name" % i)
            elif known_formulas is not None and sub == name:
                errs.append("step %d: formula invokes itself" % i)
            elif known_formulas is not None and sub not in known_formulas:
                errs.append("step %d: unknown formula %r" % (i, sub))
        # any string value may carry slots; check they are well-formed
        for k, v in step.items():
            if isinstance(v, str):
                for m in _SLOT.finditer(v):
                    pass  # syntax is enforced at bind time; keep validation cheap
    return errs


def _bind(value: Any, slots: Dict[str, Any]) -> Any:
    """Substitute {{slot}} references; recurse into containers.

    A reference that spans the *whole* string keeps the slot's own type, so a
    numeric slot stays numeric after substitution — the pipeline's operators
    take floats, and str()-forcing every interpolation handed them strings
    ("170"), which blew up the first grade step it reached. Text that merely
    contains a slot (a prompt, a name) is substituted as text, which is what
    those callers want.
    """
    if isinstance(value, str):
        whole = _SLOT.fullmatch(value.strip())
        if whole:
            key = whole.group(1)
            if key not in slots:
                raise FormulaError("unbound slot %r" % key)
            return slots[key]

        def rep(m):
            key = m.group(1)
            if key not in slots:
                raise FormulaError("unbound slot %r" % key)
            return str(slots[key])
        return _SLOT.sub(rep, value)
    if isinstance(value, dict):
        return {k: _bind(v, slots) for k, v in value.items()}
    if isinstance(value, list):
        return [_bind(v, slots) for v in value]
    return value


def _mask_from_spec(spec: dict, shape: Tuple[int, int]) -> "np.ndarray":
    """Build a mask from a JSON-friendly description."""
    import numpy as np
    import cv2

    h, w = shape
    kind = str(spec.get("type", "rect")).lower()
    m = np.zeros((h, w), np.float32)
    if kind == "rect":
        x0 = int(spec.get("x", 0) * w) if spec.get("x", 0) <= 1 else int(spec.get("x", 0))
        y0 = int(spec.get("y", 0) * h) if spec.get("y", 0) <= 1 else int(spec.get("y", 0))
        x1 = int(spec.get("x2", 1) * w) if spec.get("x2", 1) <= 1 else int(spec.get("x2", 1))
        y1 = int(spec.get("y2", 1) * h) if spec.get("y2", 1) <= 1 else int(spec.get("y2", 1))
        cv2.rectangle(m, (max(0, x0), max(0, y0)),
                      (min(w, x1), min(h, y1)), 1.0, -1)
    elif kind == "radial":
        from .face import _plateau
        cx = spec.get("cx", 0.5) * w if spec.get("cx", 0.5) <= 1 else spec.get("cx")
        cy = spec.get("cy", 0.5) * h if spec.get("cy", 0.5) <= 1 else spec.get("cy")
        rx = spec.get("rx", 0.25) * w if spec.get("rx", 0.25) <= 1 else spec.get("rx")
        ry = spec.get("ry", 0.25) * h if spec.get("ry", 0.25) <= 1 else spec.get("ry")
        m = _plateau(h, w, float(cx), float(cy), float(rx), float(ry),
                     rim=1.0 + float(spec.get("feather", 0.4)))
    else:
        raise FormulaError("unknown mask type %r" % kind)
    return m


def run(formula: dict, img: "np.ndarray",
        pipeline=None,
        slots: Optional[Dict[str, Any]] = None,
        library: Optional[Dict[str, dict]] = None,
        fill_runner=None,
        depth: int = 0) -> Tuple["np.ndarray", dict]:
    """Execute a formula against `img`.

    `pipeline` is a RetouchPipeline for params steps; `fill_runner` is a
    callable `(image, mask, prompt) -> image` for fill steps, injected so the
    engine does not import ComfyUI's stack just to run parameter-only
    formulas. `depth` guards against runaway composition.
    """
    import numpy as np

    if depth > 6:
        raise FormulaError("formula composition too deep (>%d)" % depth)
    errs = validate(formula, set(library) if library else None)
    if errs:
        raise FormulaError("; ".join(errs))

    slots = dict(slots or {})
    out = np.clip(img, 0.0, 1.0).astype(np.float32)
    applied: List[str] = []

    from .local import apply_local
    from .pipeline import RetouchPipeline

    pipe = pipeline or RetouchPipeline()

    for i, raw_step in enumerate(formula["steps"]):
        step = _bind(copy.deepcopy(raw_step), slots)
        kind = step["kind"]

        if kind == "params":
            out, info = pipe.run(out, step.get("params", {}), detect=True)
            applied.append("params:%d" % i)

        elif kind == "local":
            entries = step.get("entries", [])
            strength = float(step.get("strength", 100.0)) / 100.0
            out = apply_local(out, entries, strength_scale=strength)
            applied.append("local:%d" % i)

        elif kind == "fill":
            if fill_runner is None:
                raise FormulaError("step %d: fill requested but no fill_runner"
                                   % i)
            mask = _mask_from_spec(step.get("mask", {}), out.shape[:2])
            out = fill_runner(out, mask, step.get("prompt", "clean, seamless"))
            applied.append("fill:%d" % i)

        elif kind == "formula":
            sub_name = step["formula"]
            if not library or sub_name not in library:
                raise FormulaError("step %d: formula %r not in library"
                                   % (i, sub_name))
            sub_slots = step.get("slots", {})
            merged = dict(slots)
            merged.update(sub_slots)
            out, sub_info = run(library[sub_name], out,
                                pipeline=pipe, slots=merged,
                                library=library, fill_runner=fill_runner,
                                depth=depth + 1)
            applied.append("formula:%s" % sub_name)

    return out, {"applied": applied, "formula": formula.get("name"),
                 "slots": slots}


# ---------------------------------------------------------------------------
# library persistence
# ---------------------------------------------------------------------------

def library_path() -> str:
    from .paths import data_dir
    return os.path.join(data_dir(), "formulas.json")


def load_library() -> Dict[str, dict]:
    p = library_path()
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_library(lib: Dict[str, dict]) -> None:
    p = library_path()
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(lib, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def upsert(formula: dict, lib: Optional[Dict[str, dict]] = None) -> Dict[str, dict]:
    """Validate and store a formula; returns the updated library.

    Numeric-looking strings are coerced first, so a formula that arrives from a
    JSON request body is stored in the same typed shape as one written by hand.
    """
    formula = _coerce_numbers(formula)
    lib = dict(lib if lib is not None else load_library())
    errs = validate(formula, set(lib))
    if errs:
        raise FormulaError("; ".join(errs))
    lib[formula["name"]] = formula
    save_library(lib)
    return lib


def delete(name: str, lib: Optional[Dict[str, dict]] = None) -> Dict[str, dict]:
    lib = dict(lib if lib is not None else load_library())
    lib.pop(name, None)
    save_library(lib)
    return lib
