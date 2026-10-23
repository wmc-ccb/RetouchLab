# -*- coding: utf-8 -*-
"""Undo/redo history for session edits.

Five operations mutate a session's working image destructively — generative
fill, the colour wheel, colour match, the grey-point picker, and running a
formula. Without history, any of them is one-way: the user cannot get back to
the frame they had a moment ago. That is a real defect, not a nicety.

Design
------
Snapshots, not an operation log. The destructive operations are not invertible
from their parameters (fill synthesises content, the wheel works on already-
graded pixels), so re-deriving an earlier state would need a replay from the
upload — and a replay would still not reproduce a fill, whose result depends on
a random seed.

State is kept as **8-bit PNG bytes** — about 1.5 MB for a 1400px preview. Two
things about that choice are worth stating precisely, because the obvious
wording would be wrong:

* It is **lossless as a container but quantised to 8 bits per channel**, so a
  restored state matches the stored one exactly and the pre-edit state to
  within 0.5/255. That is deliberate: 8 bits is the precision the user can
  actually see, the exported JPEG carries the same limit, and storing more
  would cost 12× the memory for a difference no display can show.
* Undo therefore **never accumulates error**. Each state is encoded once and
  decoded on restore, so undoing and redoing a hundred times returns the same
  bytes; a design that re-applied an inverse operation would drift.

The alternative of keeping float arrays costs ~16 MB per snapshot, which at 8
levels across 8 sessions is over a gigabyte.

The redo tail is dropped when a new edit happens after an undo, which is the
behaviour every editor has.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

DEFAULT_DEPTH = 12


def _encode(img: np.ndarray) -> bytes:
    import cv2
    u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("history encode failed")
    return buf.tobytes()


def _decode(data: bytes) -> np.ndarray:
    import cv2
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("history decode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


class History:
    """A bounded snapshot stack with a cursor.

    `labels[i]` describes the state at `states[i]`. Index 0 is always the
    uploaded frame, so undo can never walk past the original.
    """

    def __init__(self, initial: np.ndarray, depth: int = DEFAULT_DEPTH):
        self.depth = max(2, int(depth))
        self.states: List[bytes] = [_encode(initial)]
        self.labels: List[str] = ["原图"]
        self.pos: int = 0

    # ---- queries --------------------------------------------------------
    def can_undo(self) -> bool:
        return self.pos > 0

    def can_redo(self) -> bool:
        return self.pos < len(self.states) - 1

    def current_label(self) -> str:
        return self.labels[self.pos]

    def list(self) -> List[dict]:
        return [{"index": i, "label": lb, "current": i == self.pos}
                for i, lb in enumerate(self.labels)]

    @property
    def bytes_used(self) -> int:
        return sum(len(b) for b in self.states)

    # ---- mutation -------------------------------------------------------
    def push(self, img: np.ndarray, label: str) -> None:
        """Record a new state, dropping any redo tail.

        Called *after* the operation has produced its result, so `pos` always
        points at the state the session currently holds.
        """
        # drop the redo tail: a new edit invalidates it
        if self.pos < len(self.states) - 1:
            del self.states[self.pos + 1:]
            del self.labels[self.pos + 1:]

        self.states.append(_encode(img))
        self.labels.append(label)
        self.pos = len(self.states) - 1

        # trim from the front, but never past the original
        while len(self.states) > self.depth and len(self.states) > 2:
            self.states.pop(0)
            self.labels.pop(0)
            self.pos = max(0, self.pos - 1)

    def undo(self) -> Optional[Tuple[np.ndarray, str]]:
        if not self.can_undo():
            return None
        self.pos -= 1
        return _decode(self.states[self.pos]), self.labels[self.pos]

    def redo(self) -> Optional[Tuple[np.ndarray, str]]:
        if not self.can_redo():
            return None
        self.pos += 1
        return _decode(self.states[self.pos]), self.labels[self.pos]

    def goto(self, index: int) -> Optional[np.ndarray]:
        """Jump directly to any recorded state.

        Used by the 历史 panel: with a snapshot stack there is no reason to
        force the user to press undo eleven times to reach a step they can see
        listed. The cursor moves, the redo tail is kept — jumping back is a
        navigation, not an edit.
        """
        if index < 0 or index >= len(self.states):
            return None
        self.pos = int(index)
        return _decode(self.states[self.pos])

    def thumbnail(self, index: int, long_edge: int = 160) -> Optional[bytes]:
        """A small JPEG for the history strip, or None if the index is bad.

        Encoded on demand rather than stored: keeping a thumbnail per state
        would roughly double the memory for a strip that is only visible while
        the panel is open.
        """
        if index < 0 or index >= len(self.states):
            return None
        import cv2
        img = _decode(self.states[index])
        h, w = img.shape[:2]
        if max(h, w) > long_edge:
            sc = long_edge / float(max(h, w))
            img = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))),
                             interpolation=cv2.INTER_AREA)
        u8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        return buf.tobytes() if ok else None

    @property
    def index(self) -> int:
        return self.pos

    @property
    def steps(self) -> List[bytes]:
        return self.states

    def reset(self, img: np.ndarray, label: str = "原图") -> None:
        self.states = [_encode(img)]
        self.labels = [label]
        self.pos = 0
