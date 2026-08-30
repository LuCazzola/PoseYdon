"""BVH reading.

End Sites are treated as ordinary joints with an offset, a name and no channels.
The Truebones files carry their names in a nonstandard ``#name:`` comment; where
that is missing a name is derived from the parent.

The MOTION block is parsed in a single vectorized pass rather than line by line,
which is where a naive parser spends nearly all its time.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from poseydon.core.anim import Anim
from poseydon.core.rotations import QUAT_IDENTITY, euler_to_quat

CHANNEL_AXIS = {
    "Xrotation": "X",
    "Yrotation": "Y",
    "Zrotation": "Z",
}
_POSITION_CHANNELS = ("Xposition", "Yposition", "Zposition")

_END_SITE_RE = re.compile(r"End\s+Site\s*(?:#\s*name:\s*(\S+))?", re.IGNORECASE)


class BvhParseError(Exception):
    """Raised when a BVH file cannot be parsed."""


def _parse_hierarchy(text: str):
    names: list[str] = []
    parents: list[int] = []
    offsets: list[list[float]] = []
    channels: list[list[str]] = []
    stack: list[int] = []

    def add(name: str) -> int:
        names.append(name)
        parents.append(stack[-1] if stack else -1)
        offsets.append([0.0, 0.0, 0.0])
        channels.append([])
        return len(names) - 1

    pending: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("ROOT ", "JOINT ")):
            pending = add(line.split(None, 1)[1].strip())
        elif line.upper().startswith("END SITE"):
            match = _END_SITE_RE.match(line)
            name = match.group(1) if match and match.group(1) else None
            if name is None:
                name = f"{names[stack[-1]]}_End" if stack else "End"
            pending = add(name)
        elif line.startswith("{"):
            if pending is None:
                raise BvhParseError("found '{' before any ROOT, JOINT or End Site")
            stack.append(pending)
            pending = None
        elif line.startswith("}"):
            if not stack:
                raise BvhParseError("unbalanced '}' in HIERARCHY")
            stack.pop()
        elif line.startswith("OFFSET"):
            if not stack:
                raise BvhParseError("OFFSET outside of any joint block")
            offsets[stack[-1]] = [float(v) for v in line.split()[1:4]]
        elif line.startswith("CHANNELS"):
            if not stack:
                raise BvhParseError("CHANNELS outside of any joint block")
            parts = line.split()
            count = int(parts[1])
            spec = parts[2 : 2 + count]
            if len(spec) != count:
                raise BvhParseError(f"CHANNELS declares {count} names, found {len(spec)}")
            channels[stack[-1]] = spec

    if stack:
        raise BvhParseError("unbalanced '{' in HIERARCHY")
    if not names:
        raise BvhParseError("no joints found in HIERARCHY")

    return (
        tuple(names),
        np.array(parents, dtype=np.int32),
        np.array(offsets, dtype=np.float64),
        channels,
    )


def _parse_motion(text: str, n_channels: int) -> tuple[np.ndarray, float]:
    lines = text.strip().splitlines()
    n_frames: int | None = None
    frame_time: float | None = None
    start = 0
    for index, raw in enumerate(lines):
        line = raw.strip()
        if line.lower().startswith("frames:"):
            n_frames = int(line.split(":", 1)[1])
        elif line.lower().startswith("frame time:"):
            frame_time = float(line.split(":", 1)[1])
            start = index + 1
            break
    if n_frames is None or frame_time is None:
        raise BvhParseError("MOTION block is missing 'Frames:' or 'Frame Time:'")

    block = " ".join(lines[start:])
    values = np.fromstring(block, sep=" ", dtype=np.float64)
    expected = n_frames * n_channels
    if values.size != expected:
        raise BvhParseError(
            f"MOTION block declares {n_frames} frames of {n_channels} channels "
            f"({expected} values) but contains {values.size}"
        )
    return values.reshape(n_frames, n_channels), frame_time


def _channels_to_local(values, channels, n_joints):
    n_frames = values.shape[0]
    rotations = np.broadcast_to(QUAT_IDENTITY, (n_frames, n_joints, 4)).copy()
    root_pos = np.zeros((n_frames, 3), dtype=np.float64)

    column = 0
    for joint, spec in enumerate(channels):
        if not spec:
            continue
        columns = {name: column + offset for offset, name in enumerate(spec)}
        column += len(spec)

        position_names = [n for n in _POSITION_CHANNELS if n in columns]
        if position_names:
            if joint != 0:
                raise BvhParseError(
                    f"joint {joint} has position channels; only the root may translate"
                )
            for axis_index, name in enumerate(_POSITION_CHANNELS):
                if name in columns:
                    root_pos[:, axis_index] = values[:, columns[name]]

        rotation_names = [n for n in spec if n in CHANNEL_AXIS]
        if rotation_names:
            order = "".join(CHANNEL_AXIS[n] for n in rotation_names)
            angles = np.stack([values[:, columns[n]] for n in rotation_names], axis=-1)
            rotations[:, joint] = euler_to_quat(angles, order)

    return rotations, root_pos


def load_bvh(path: str | Path) -> Anim:
    """Read a BVH file into an :class:`Anim`. End Sites become joints."""
    text = Path(path).read_text()
    head, marker, motion = text.partition("MOTION")
    if not marker:
        raise BvhParseError(f"{path}: no MOTION block found")

    names, parents, offsets, channels = _parse_hierarchy(head)
    n_channels = sum(len(spec) for spec in channels)
    values, frame_time = _parse_motion(motion, n_channels)
    rotations, root_pos = _channels_to_local(values, channels, len(names))

    if frame_time <= 0.0:
        raise BvhParseError(f"{path}: non-positive Frame Time {frame_time}")

    return Anim(
        rotations=rotations,
        root_pos=root_pos,
        offsets=offsets,
        parents=parents,
        names=names,
        fps=1.0 / frame_time,
    )
