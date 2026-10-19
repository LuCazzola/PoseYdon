"""Frame and clip filters of stage 1 (the dataset's ``clips:`` group).

UniMate trims and filters its clips around canonicalization: the head trim and
the fps halving run on the raw clip, everything else on the canonical one. A
`ClipOp` says which windows of a clip it keeps, or why it drops the clip;
`apply_clip_ops` composes them, so a kept clip carries ONE ``(start, stop,
step)`` window in the frames of the clip as the source gave it, which is what
lets the round trip map prepared frames back to source frames
(``clip_params[clip]["frames"]["window"]``).

The metrics are UniMate's, computed on global positions (`poseydon.build.prepare`
has already centred, scaled and grounded them for a ``prepared`` op).
"""

from __future__ import annotations

from abc import ABC
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import numpy as np

from poseydon.core.animation import Animation
from poseydon.core.registry import Registry

Window = tuple[int, int, int]   # (start, stop, step), as in ``range``

CLIP_OPS: Registry[ClipOp] = Registry("clip op")


class ClipOp(ABC):
    """One filter. ``when`` is ``"source"`` (before ``prepare:``) or ``"prepared"``."""

    name: ClassVar[str]
    when: ClassVar[str] = "prepared"

    def rig(self, rest: Animation, rig: str) -> str | None:
        """A reason to skip the whole rig, or ``None``."""
        return None

    def windows(self, anim: Animation, clip_id: str) -> list[Window] | str:
        """The windows of ``anim`` kept, or the reason the clip is dropped."""
        return [(0, anim.n_frames, 1)]


def _lines(path: str | None) -> dict[str, list[str]]:
    """``<clip> [value]`` lines (``#`` comments); a missing file is empty, like UniMate's ``auto``."""
    if path is None or not Path(path).is_file():
        return {}
    rows = (line.split("#")[0].split() for line in Path(path).read_text().splitlines())
    return {r[0]: r[1:] for r in rows if r}


def apply_clip_ops(
    ops, anim: Animation, clip_id: str, when: str, base: Window | None = None,
) -> list[tuple[Animation, Window]] | str:
    """``anim`` through the ``when`` ops in order: every kept window, as the
    sub-animation and its window in SOURCE frames, or the reason it is dropped.
    ``base`` is ``anim``'s own window in source frames (default: all of it)."""
    current = [(anim, base or (0, anim.n_frames, 1))]
    for op in (o for o in ops if o.when == when):
        kept, reason = [], None
        for sub, (b0, _, bstep) in current:
            result = op.windows(sub, clip_id)
            if isinstance(result, str):
                reason = f"{op.name}: {result}"
                continue
            for start, stop, step in result:
                piece = replace(
                    sub, rotations=sub.rotations[start:stop:step], translations=sub.translations[start:stop:step],
                    fps=sub.fps / step,   # a halved clip plays at half the rate, not twice the speed
                )
                first, every = b0 + bstep * start, bstep * step
                kept.append((piece, (first, first + every * piece.n_frames, every)))
        if not kept:
            return reason or f"{op.name}: no frames left"
        current = kept
    return current


@CLIP_OPS.register("head_trim")
class HeadTrim(ClipOp):
    """Drop the first N frames named in ``file`` (UniMate's ``clip_trims.txt``: ``<clip> <N>``)."""

    name, when = "head_trim", "source"

    def __init__(self, file: str | None = None) -> None:
        self.trims = {k: int(v[0]) for k, v in _lines(file).items()}

    def windows(self, anim, clip_id):
        n = self.trims.get(clip_id, 0)
        if n >= anim.n_frames:
            return f"head trim of {n} frames removes all {anim.n_frames}"
        return [(n, anim.n_frames, 1)]


@CLIP_OPS.register("halve_fps")
class HalveFps(ClipOp):
    """Take every other frame while the rate is ``threshold`` or more (UniMate halves until below 60 fps)."""

    name, when = "halve_fps", "source"

    def __init__(self, threshold: float = 60) -> None:
        self.threshold = threshold

    def windows(self, anim, clip_id):
        step, fps, n = 1, anim.fps, anim.n_frames
        while fps >= self.threshold and n > 1:
            step, fps, n = step * 2, fps / 2, (n + 1) // 2
        return [(0, anim.n_frames, step)]


@CLIP_OPS.register("trim_static")
class TrimStatic(ClipOp):
    """Strip leading and trailing frames where no joint moves more than ``threshold`` per frame."""

    name = "trim_static"

    def __init__(self, threshold: float = 1e-5) -> None:
        self.threshold = threshold

    def windows(self, anim, clip_id):
        if anim.n_frames < 2:
            return [(0, anim.n_frames, 1)]
        step = np.linalg.norm(np.diff(anim.global_positions(), axis=0), axis=-1).max(axis=-1)
        active = np.flatnonzero(step > self.threshold)
        if not len(active):
            return "every frame is static"
        return [(int(active[0]), min(int(active[-1]) + 2, anim.n_frames), 1)]


@CLIP_OPS.register("min_frames")
class MinFrames(ClipOp):
    name = "min_frames"

    def __init__(self, n: int = 8) -> None:
        self.n = n

    def windows(self, anim, clip_id):
        if anim.n_frames < self.n:
            return f"{anim.n_frames} frames < {self.n}"
        return [(0, anim.n_frames, 1)]


@CLIP_OPS.register("max_length")
class MaxLength(ClipOp):
    """At most ``frames`` frames: the first ``frames`` (``first``), or overlapping windows
    starting every ``stride`` frames (``windows``; UniMate's ``compute_clip_windows``,
    stride = max length minus the diffusion length)."""

    name = "max_length"

    def __init__(self, frames: int = 200, mode: str = "first", stride: int = 110) -> None:
        if mode not in ("first", "windows"):
            raise ValueError(f"max_length mode must be `first` or `windows`, got {mode!r}")
        self.frames, self.mode, self.stride = frames, mode, stride

    def windows(self, anim, clip_id):
        n, m = anim.n_frames, self.frames
        if n <= m:
            return [(0, n, 1)]
        if self.mode == "first":
            return [(0, m, 1)]
        out = [(s, s + m, 1) for s in range(0, n - m + 1, self.stride)]
        tail = n - out[-1][1]
        if tail >= max(1, m - self.stride):
            out.append((out[-1][1], n, 1))
        elif tail and n - m <= (out[-2][1] if len(out) >= 2 else 0):
            out[-1] = (n - m, n, 1)
        return out


@CLIP_OPS.register("low_activity")
class LowActivity(ClipOp):
    """Drop a clip whose most active joints and root both barely move (UniMate's spread metric:
    std over frames of root-relative joint positions, mean of the top ``top_k_ratio`` of joints,
    at least ``top_k_min``). Clips listed in ``keep_file`` are exempt."""

    name = "low_activity"

    def __init__(self, threshold: float = 0.02, top_k_ratio: float = 0.25, top_k_min: int = 2,
                 keep_file: str | None = None) -> None:
        self.threshold, self.ratio, self.k_min = threshold, top_k_ratio, top_k_min
        self.keep = set(_lines(keep_file))

    def windows(self, anim, clip_id):
        pos = anim.global_positions()
        root = float(np.linalg.norm(pos[:, 0].std(axis=0)))
        topk = 0.0
        if pos.shape[1] > 1:
            spread = np.linalg.norm((pos[:, 1:] - pos[:, :1]).std(axis=0), axis=-1)
            k = min(max(self.k_min, int(round(self.ratio * len(spread)))), len(spread))
            topk = float(np.partition(spread, -k)[-k:].mean())
        if topk < self.threshold and root < self.threshold and clip_id not in self.keep:
            return f"topk={topk:.4f}, root={root:.4f} < {self.threshold}"
        return [(0, anim.n_frames, 1)]


@CLIP_OPS.register("jump")
class Jump(ClipOp):
    """Drop a clip with one frame that moves the skeleton ``step`` clip extents AND ``ratio``
    times the median step (concatenated actions, a teleporting root); non-finite input too."""

    name = "jump"

    def __init__(self, step: float = 0.2, ratio: float = 8.0) -> None:
        self.step, self.ratio = step, ratio

    def windows(self, anim, clip_id):
        pos = anim.global_positions()
        ok = [(0, anim.n_frames, 1)]
        if not np.isfinite(pos).all():
            return "non-finite positions"
        if len(pos) < 2 or self.step <= 0:
            return ok
        extent = float(np.ptp(pos.reshape(-1, 3), axis=0).max()) or 1.0
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=2).mean(1) / extent
        med, top = float(np.median(steps)), float(steps.max())
        ratio = top / med if med > 1e-9 else float("inf")
        if top >= self.step and ratio >= self.ratio:
            return f"max_step={top:.3f}, median={med:.4f}, ratio={ratio:.1f}"
        return ok


@CLIP_OPS.register("joint_range")
class JointRange(ClipOp):
    """Skip a rig whose prepared skeleton has fewer than ``min`` or more than ``max`` joints."""

    name = "joint_range"

    def __init__(self, min: int = 8, max: int = 150) -> None:
        self.min, self.max = min, max

    def rig(self, rest, rig):
        if not self.min <= rest.n_joints <= self.max:
            return f"{rest.n_joints} joints outside [{self.min}, {self.max}]"
        return None
