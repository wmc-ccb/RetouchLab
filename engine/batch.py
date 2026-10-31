# -*- coding: utf-8 -*-
"""Batch processing: run one recipe over many files.

The work is deliberately boring — load, run the pipeline, save — but three
details matter in practice:

* **A recipe is one of three things.** A named preset (fast, consistent), an
  explicit parameter dict (a saved look), or a formula (a multi-step program).
  All three end up as a params dict, so the loop below is identical for each.
* **A draft mode.** Screening a folder is a different job from delivering it:
  a 1280px draft pass is roughly an order of magnitude faster and is enough to
  decide which frames are worth finishing. `long_edge` selects it, and the
  result records which mode produced it.
* **Per-file isolation.** One unreadable file must not lose the other
  nineteen. Failures are recorded per file and the run continues.

Progress is polled rather than streamed, so a long run survives a page reload.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
RAW_EXT = (".nef", ".cr2", ".cr3", ".arw", ".raf", ".dng", ".orf", ".rw2",
           ".srw", ".pef", ".raw")

# a run is kept in memory so the UI can poll it after a reload; bounded so a
# long-lived server does not accumulate every job it has ever served
_MAX_RUNS = 12


def list_images(path: str, recursive: bool = False) -> List[str]:
    """Collect processable files from a directory or a single file."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        return []
    out: List[str] = []
    if recursive:
        for root, _dirs, files in os.walk(path):
            for f in sorted(files):
                if f.lower().endswith(IMAGE_EXT + RAW_EXT):
                    out.append(os.path.join(root, f))
    else:
        for f in sorted(os.listdir(path)):
            fp = os.path.join(path, f)
            if os.path.isfile(fp) and f.lower().endswith(IMAGE_EXT + RAW_EXT):
                out.append(fp)
    return out


class BatchRun:
    """One background batch job, pollable by id."""

    # 预解码缓存：loader 线程提前解码下一张（RAW 解码是批处理里很重的
    # 一步），工作线程从这里取，实现"解码与推理重叠"。
    # 只缓存 1 张（全尺寸 24MP float32 ≈ 288MB，多存会吃爆内存）。
    _PRELOAD_MAX = 1

    def __init__(self, run_id: str, paths: Sequence[str], out_dir: str,
                 recipe_label: str, draft_edge: int = 0):
        self.id = run_id
        self.paths = list(paths)
        self.out_dir = out_dir
        self.recipe_label = recipe_label
        self.draft_edge = int(draft_edge)
        self.total = len(self.paths)
        self.done = 0
        self.failed = 0
        self.started = time.time()
        self.finished: Optional[float] = None
        self.cancelled = False
        self.current = ""
        self.results: List[dict] = []
        self.error: Optional[str] = None
        self.queued = False
        self._lock = threading.Lock()
        self._preload: "OrderedDict[str, object]" = OrderedDict()
        self._preload_lock = threading.Lock()

    # ---- reporting ------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            elapsed = (self.finished or time.time()) - self.started
            if self.queued and not self.finished and not self.cancelled:
                state = "queued"
            else:
                state = ("cancelled" if self.cancelled else
                         "done" if self.finished else "running")
            return {
                "id": self.id,
                "state": state,
                "total": self.total,
                "done": self.done,
                "failed": self.failed,
                "current": self.current,
                "elapsed": round(elapsed, 2),
                "eta": (round(elapsed / self.done * (self.total - self.done), 1)
                        if self.done else None),
                "recipe": self.recipe_label,
                "draft_edge": self.draft_edge,
                "out_dir": self.out_dir,
                "results": list(self.results[-60:]),
                "error": self.error,
            }

    def cancel(self) -> None:
        self.cancelled = True

    # ---- execution ------------------------------------------------------
    def run(self, params_for, save_fn, preload_fn=None) -> None:
        """`params_for(path) -> dict` and `save_fn(path, img, params) -> str`.

        Kept as callables so this module owns the loop, the progress and the
        error isolation while the server owns how a file is decoded and where an
        output is written.

        `preload_fn(path) -> decoded image`（可选）：后台 loader 线程提前
        解码下一张，与本张的推理重叠。缓存只留 1 张。
        """
        # wait for any run already in progress; `started` is reset so the
        # reported elapsed time is execution time, not queue time
        if _RUN_LOCK.locked():
            self.queued = True
        with _RUN_LOCK:
            self.queued = False
            self.started = time.time()
            self._execute(params_for, save_fn, preload_fn)

    def _execute(self, params_for, save_fn, preload_fn=None) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        loader = None
        if preload_fn is not None and len(self.paths) > 1:
            loader = threading.Thread(target=self._preload_loop,
                                      args=(params_for, preload_fn),
                                      daemon=True)
            loader.start()
        for path in self.paths:
            if self.cancelled:
                break
            self.current = os.path.basename(path)
            entry = {"path": path, "name": self.current, "ok": False}
            try:
                params = params_for(path)
                with self._preload_lock:
                    pre = self._preload.pop(path, None)
                out_path = save_fn(path, params, pre)
                entry.update(ok=True, out=out_path)
                with self._lock:
                    self.done += 1
            except Exception as exc:
                entry.update(error=str(exc)[:200], trace=traceback.format_exc()[-400:])
                with self._lock:
                    self.failed += 1
                    self.done += 1
            self.results.append(entry)
        if loader is not None:
            loader.join(timeout=5)
        self.finished = time.time()
        self.current = ""

    def _preload_loop(self, params_for, preload_fn) -> None:
        """提前解码后续图片（跳过第一张——它已被主线程处理）。"""
        for i, path in enumerate(self.paths[1:], 1):
            if self.cancelled:
                return
            with self._preload_lock:
                if len(self._preload) >= self._PRELOAD_MAX:
                    return
            try:
                decoded = preload_fn(path)
            except Exception:
                continue            # 解码失败留给 save_fn 原路径报错
            with self._preload_lock:
                if self.cancelled or len(self._preload) >= self._PRELOAD_MAX:
                    return
                self._preload[path] = decoded


_RUNS: Dict[str, BatchRun] = {}
_ORDER: List[str] = []

# Only one run executes at a time.
#
# Runs share the process-wide pipeline, which caches lazily-built models and
# per-image matte/mask results, and can flip its own `use_semantic` flag when a
# model is missing. Those are not safe to touch from two threads at once, and
# the failure is not a clean exception: two concurrent runs produced an empty
# output directory while each run on its own succeeded. Serialising is the
# correct fix here rather than a lock inside the pipeline — a local tool has no
# reason to run two batches at once, and queuing keeps the model caches warm
# instead of duplicating the models per run.
_RUN_LOCK = threading.Lock()


def start(paths: Sequence[str], out_dir: str, recipe_label: str,
          draft_edge: int, params_for, save_fn, preload_fn=None) -> BatchRun:
    rid = uuid.uuid4().hex[:12]
    run = BatchRun(rid, paths, out_dir, recipe_label, draft_edge)
    _RUNS[rid] = run
    _ORDER.append(rid)
    while len(_ORDER) > _MAX_RUNS:
        _RUNS.pop(_ORDER.pop(0), None)
    t = threading.Thread(target=run.run, args=(params_for, save_fn, preload_fn),
                         daemon=True, name="batch-" + rid)
    t.start()
    return run


def get(rid: str) -> Optional[BatchRun]:
    return _RUNS.get(rid)


def running() -> bool:
    return any(r.finished is None and not r.cancelled for r in _RUNS.values())
