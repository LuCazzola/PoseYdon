"""Joint names a text encoder can read.

`SkeletonManifest.strip_joint_prefix` existed, was parsed, was documented, and
was read by nothing -- the same dead-key pattern `tpose` carried before plan A1
removed it. The replacement COMPUTES the prefix from the rig's own names, so a
new rig needs no declaration and cannot declare a wrong one.

RAW names stay the canonical identity: manifests and `resolve()` match on them,
and that is what makes a re-exported rig fail loudly rather than silently
mirroring the character. These names exist only for the text encoder.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from os.path import commonprefix

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SEPARATORS = re.compile(r"[_\-.:]+")
_SIDES = {"r": "right", "l": "left"}


@dataclass(frozen=True)
class Humanize:
    """`Bip01_R_Thigh` -> `right thigh`."""

    lowercase: bool = True
    expand_sides: bool = True

    def __call__(self, names: Sequence[str]) -> tuple[str, ...]:
        names = list(names)
        if not names:
            raise ValueError("cannot humanize: no joint names given")

        prefix = self._shared_prefix(names)
        return tuple(self._one(name[len(prefix) :] or name) for name in names)

    @staticmethod
    def _shared_prefix(names: Sequence[str]) -> str:
        """The separator-terminated prefix EVERY joint carries.

        Truncated at a separator so `Bip01_Pelvis`/`Bip01_Spine` yields
        `Bip01_` rather than `Bip01_`+`S`-style partial-word garbage, and so a
        rig whose names merely happen to share letters loses nothing.

        A single-name list has no sibling to confirm sharing against, so it
        is conservative: only a leading token LONGER THAN ONE CHARACTER (up
        to the first separator) is treated as boilerplate -- a lone leading
        letter such as `R_Thigh`'s `R` is a side marker, not a prefix, and
        must reach `_one` untouched so `expand_sides` can see it.
        """
        if len(names) < 2:
            name = names[0]
            positions = [name.find(c) for c in "_-.:"]
            cut = min((p for p in positions if p >= 0), default=-1)
            return name[: cut + 1] if cut > 1 else ""
        shared = commonprefix(list(names))
        cut = max(shared.rfind(c) for c in "_-.:")
        return shared[: cut + 1] if cut >= 0 else ""

    def _one(self, name: str) -> str:
        parts: list[str] = []
        for chunk in _SEPARATORS.split(name):
            if chunk:
                parts.extend(p for p in _CAMEL.split(chunk) if p)

        if self.expand_sides:
            parts = [_SIDES.get(p.lower(), p) if len(p) == 1 else p for p in parts]

        text = " ".join(parts)
        return text.lower() if self.lowercase else text
