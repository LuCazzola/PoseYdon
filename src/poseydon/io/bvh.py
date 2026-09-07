"""BVH reading and writing.

One class, :class:`BVH`, holds a parsed file and converts both ways:
``BVH.read`` -> :class:`~poseydon.core.animation.Animation`, and
``BVH.from_animation`` -> a file.

Reading is LOSSLESS and total. BVH permits position channels on any joint,
not just the root, and exporters use them -- so the parser records whatever
the file declares instead of rejecting it. PoseYdon's rigid-bone assumption
is a property of ``RigidBodyAnimation``, applied by an explicit
:meth:`~poseydon.core.animation.Animation.as_rigid_body` call at the point
a caller actually needs it, not a rule the parser enforces.

End Sites are treated as ordinary joints with an offset, a name and no channels.
The Truebones files carry their names in a nonstandard ``#name:`` comment; where
that is missing a name is derived from the parent.

The MOTION block is parsed in a single vectorized pass rather than line by line,
which is where a naive parser spends nearly all its time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from poseydon.core.animation import Animation
from poseydon.core.rotations import QUAT_IDENTITY, euler_to_quat, quat_to_euler

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


def _channels_to_arrays(values, channels, n_joints):
    """Per-joint rotations and positions, straight from the MOTION columns.

    Every joint gets a (F, 3) position slot -- zero-filled if it declared no
    position channels. No contract enforcement: callers decide what a
    non-root position column means. Shared by ``_channels_to_local`` (which
    rejects non-root translation) and ``poseydon.preproc.sanitize_bvh`` (which
    doesn't).

    Joints are grouped by rotation order and converted in one call per order
    rather than one per joint. A 63-joint skeleton otherwise pays 63 separate
    SciPy round trips, which dominates parsing.
    """
    n_frames = values.shape[0]
    rotations = np.broadcast_to(QUAT_IDENTITY, (n_frames, n_joints, 4)).copy()
    positions = np.zeros((n_frames, n_joints, 3), dtype=np.float64)

    by_order: dict[str, list[tuple[int, list[int]]]] = {}
    column = 0
    for joint, spec in enumerate(channels):
        if not spec:
            continue
        columns = {name: column + offset for offset, name in enumerate(spec)}
        column += len(spec)

        for axis_index, name in enumerate(_POSITION_CHANNELS):
            if name in columns:
                positions[:, joint, axis_index] = values[:, columns[name]]

        rotation_names = [n for n in spec if n in CHANNEL_AXIS]
        if rotation_names:
            order = "".join(CHANNEL_AXIS[n] for n in rotation_names)
            by_order.setdefault(order, []).append(
                (joint, [columns[n] for n in rotation_names])
            )

    for order, entries in by_order.items():
        joints = [joint for joint, _ in entries]
        picks = np.array([cols for _, cols in entries])
        # (J_group, F, 3) -> one conversion for the whole group
        angles = values[:, picks].transpose(1, 0, 2)
        rotations[:, joints] = euler_to_quat(
            angles.reshape(-1, 3), order
        ).reshape(len(joints), n_frames, 4).transpose(1, 0, 2)

    return rotations, positions


@dataclass(frozen=True)
class BVH:
    """A parsed BVH file: exactly what the file declares, nothing dropped.

    ``translations`` carries every joint's per-frame position channels. A
    joint that declares none gets its own ``OFFSET`` repeated instead of
    zeros -- zeros would read as "coincident with my parent" once the pair
    is interpreted as a local transform, and it also makes the rigid case
    fall out of the general one exactly (see
    :func:`poseydon.core.kinematics.forward_kinematics`).
    """

    names: tuple[str, ...]
    parents: np.ndarray
    offsets: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    channels: tuple[tuple[str, ...], ...]
    fps: float

    @classmethod
    def read_names(cls, path: str | Path) -> tuple[str, ...]:
        """Just the joint-name tuple, parsing only the HIERARCHY block.

        The same first step as :meth:`read` -- ``text.partition("MOTION")``
        then ``_parse_hierarchy(head)`` -- stopped short of the MOTION block,
        which is where a full parse spends nearly all its time. For a caller
        that only needs to compare joint sets across a large corpus (choosing
        a rest reference by modal skeleton, say), this is a small honest
        reuse rather than a second parser.
        """
        text = Path(path).read_text()
        head, marker, _motion = text.partition("MOTION")
        if not marker:
            raise BvhParseError(f"{path}: no MOTION block found")
        names, _parents, _offsets, _channels = _parse_hierarchy(head)
        return names

    @classmethod
    def read(cls, path: str | Path) -> BVH:
        """Parse a BVH file. Never rejects a legal file."""
        text = Path(path).read_text()
        head, marker, motion = text.partition("MOTION")
        if not marker:
            raise BvhParseError(f"{path}: no MOTION block found")

        names, parents, offsets, channels = _parse_hierarchy(head)
        n_channels = sum(len(spec) for spec in channels)
        values, frame_time = _parse_motion(motion, n_channels)
        if frame_time <= 0.0:
            raise BvhParseError(f"{path}: non-positive Frame Time {frame_time}")
        rotations, translations = _channels_to_arrays(values, channels, len(names))

        for joint, spec in enumerate(channels):
            if not any(name in spec for name in _POSITION_CHANNELS):
                translations[:, joint] = offsets[joint]

        return cls(
            names=names,
            parents=parents,
            offsets=offsets,
            rotations=rotations,
            translations=translations,
            channels=tuple(tuple(spec) for spec in channels),
            fps=1.0 / frame_time,
        )

    def to_animation(self) -> Animation:
        """The parsed file as an :class:`Animation`, still lossless.

        Call ``.as_rigid_body()`` on the result to impose the rigid-bone
        assumption and choose what happens to per-joint translation.
        """
        return Animation(
            rotations=self.rotations,
            translations=self.translations,
            offsets=self.offsets,
            parents=self.parents,
            names=self.names,
            fps=self.fps,
        )

    @classmethod
    def from_animation(
        cls, animation: Animation, channels: tuple[tuple[str, ...], ...] | None = None
    ) -> BVH:
        """Prepare an animation for writing.

        By default joints are given position channels only where they actually
        translate, so a rigid animation writes the ordinary ``6 channels on the
        root, 3 elsewhere`` layout. Pass ``channels`` to declare a specific
        layout instead: undoing ``EnforceRigid`` has to restore the source
        file's declaration even though the values in those channels are now the
        constant rest offsets, so the returned rig is structurally the one the
        user supplied.
        """
        if channels is None:
            moves = ~np.all(
                np.isclose(animation.translations, animation.offsets[np.newaxis], atol=1e-9),
                axis=(0, 2),
            )
            children = _children_of(animation.parents)
            built = []
            for joint in range(animation.n_joints):
                if not children[joint]:
                    built.append(())
                elif joint == 0 or moves[joint]:
                    built.append((*_POSITION_CHANNELS, "Zrotation", "Xrotation", "Yrotation"))
                else:
                    built.append(("Zrotation", "Xrotation", "Yrotation"))
            channels = tuple(built)
        elif len(channels) != animation.n_joints:
            raise ValueError(
                f"channels declares {len(channels)} joints but the animation has "
                f"{animation.n_joints}"
            )

        return cls(
            names=animation.names,
            parents=animation.parents,
            offsets=animation.offsets,
            rotations=animation.rotations,
            translations=animation.translations,
            channels=tuple(tuple(spec) for spec in channels),
            fps=animation.fps,
        )

    def write(self, path: str | Path, order: str = "ZYX") -> None:
        """Write to a BVH file.

        Leaf joints become ``End Site`` entries carrying their name in the
        same ``#name:`` comment :meth:`read` understands, so a round trip
        preserves joint count and naming.
        """
        children = _children_of(self.parents)

        lines: list[str] = ["HIERARCHY"]
        _write_joint(lines, self, children, 0, 0, order)

        lines.append("MOTION")
        lines.append(f"Frames: {self.rotations.shape[0]}")
        # 10 decimals, not the conventional 6: BVH stores the frame TIME, so a
        # round rate becomes a repeating decimal (30 fps -> 0.0333...) and 6
        # decimals reads back as 30.0003 -- enough to fail an fps check against
        # a manifest. 10 keeps the round trip within 3e-8 fps.
        lines.append(f"Frame Time: {1.0 / self.fps:.10f}")

        jointed = [j for j in range(len(self.names)) if children[j]]
        angles = quat_to_euler(self.rotations[:, jointed], order)
        translating = [j for j in jointed if self._has_position(j)]

        for frame in range(self.rotations.shape[0]):
            row: list[str] = []
            positions = {j: self.translations[frame, j] for j in translating}
            for index, joint in enumerate(jointed):
                if joint in positions:
                    row.extend(f"{value:.6f}" for value in positions[joint])
                row.extend(f"{value:.6f}" for value in angles[frame, index])
            lines.append(" ".join(row))

        Path(path).write_text("\n".join(lines) + "\n")

    def _has_position(self, joint: int) -> bool:
        return any(name in self.channels[joint] for name in _POSITION_CHANNELS)


def _children_of(parents: np.ndarray) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(parents.size)]
    for joint in range(1, parents.size):
        children[int(parents[joint])].append(joint)
    return children


def _write_joint(lines, bvh, children, joint, depth, order) -> None:
    pad = "\t" * depth
    is_end_site = not children[joint]

    if is_end_site:
        lines.append(f"{pad}End Site #name: {bvh.names[joint]}")
    elif joint == 0:
        lines.append(f"{pad}ROOT {bvh.names[joint]}")
    else:
        lines.append(f"{pad}JOINT {bvh.names[joint]}")

    lines.append(f"{pad}{{")
    ox, oy, oz = bvh.offsets[joint]
    lines.append(f"{pad}\tOFFSET {ox:.6f} {oy:.6f} {oz:.6f}")

    if not is_end_site:
        channel_names = " ".join(f"{axis}rotation" for axis in order)
        if bvh._has_position(joint):
            lines.append(
                f"{pad}\tCHANNELS 6 Xposition Yposition Zposition {channel_names}"
            )
        else:
            lines.append(f"{pad}\tCHANNELS 3 {channel_names}")
        for child in children[joint]:
            _write_joint(lines, bvh, children, child, depth + 1, order)

    lines.append(f"{pad}}}")
