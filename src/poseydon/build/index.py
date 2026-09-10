"""The corpus index.

One row per clip. This replaces filename parsing (the reference re-implements it
in eight places), the per-character filelist tree, the hardcoded taxonomy subset
lists, and the script that materialized retargeting pairs into txt files.

JSONL: append-only, diffable, readable without a dependency, and small enough at
corpus scale that a columnar format would be premature.
"""

from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

SEPARATOR = "__"
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def action_slug(stem: str) -> str:
    """Lowercase, collapse non-alphanumeric runs to `_`, trim the ends."""
    return _NON_ALNUM.sub("_", stem.lower()).strip("_")


def strip_skeleton_prefix(action: str, skeleton: str) -> str:
    """Drop a leading, repeated skeleton name from an action slug.

    Truebones filenames embed the species, sometimes twice
    (``Flamingo_Flamingo_OneLEgBEnt_353``), which would otherwise produce ids
    like ``Flamingo__flamingo_flamingo_onelegbent_353``. Never strips down to an
    empty action.
    """
    prefix = f"{action_slug(skeleton)}_"
    while action.startswith(prefix) and len(action) > len(prefix):
        action = action[len(prefix) :]
    return action


def is_all_takes_bundle(stem: str, rig: str) -> bool:
    """Is this filename stem Truebones' all-takes compilation for ``rig``?

    Every rig ships one file holding all of its takes concatenated --
    ``GoatAll.fbx``, ``scorpionALL.fbx``, ``Flamingo-ALL.fbx``, ``Camel_ALL.fbx``
    -- which is not a distinct action and has no BVH counterpart, so consumers
    of the raw corpus must exclude it. The convention is that a bundle's stem is
    the rig's export PREFIX plus "ALL" and NOTHING else.

    Neither half of "and nothing else" is optional, and getting either wrong
    costs real clips:

    * A suffix test (``stem.lower().endswith("all")``) also drops the 13 takes
      whose stem merely ends in those letters -- ``Camel-Fall``, ``Stego-Fall``,
      and ``DEER-WalkCall``, which shows the mechanism is the suffix rather than
      the word "fall".
    * A substring test (``"ALL" in stem.upper()``) additionally drops every file
      of a rig whose export prefix contains "ALL" -- Alligator, Anaconda, Crow,
      HermitCrab, Lion, SabreToothTiger, Tukan -- leaving those rigs with no
      clips at all.

    The prefix is not always ``rig``: Truebones exports some rigs under an alias
    (``Dragon/WyvernALL.fbx``, ``Jaws/SharkALL.fbx``, ``Raptor2/RaptorALL.fbx``),
    so any single unseparated token is accepted as a prefix, and ``rig`` itself
    is accepted whatever separators its own name contains. What is rejected
    either way is a stem carrying a take name beside the prefix, in either
    position: ``Camel-Fall`` before "all", ``AnacondaALL-Twistrattle`` after it.
    """
    if not stem.lower().endswith("all"):
        return False
    prefix = stem[: -len("all")].rstrip("-_")
    return bool(prefix) and (prefix.isalnum() or action_slug(prefix) == action_slug(rig))


def clip_id(skeleton: str, action: str) -> str:
    """Deterministic clip identity: ``{skeleton}__{action}``."""
    if SEPARATOR in skeleton:
        raise ValueError(
            f"skeleton name `{skeleton}` contains the reserved separator "
            f"`{SEPARATOR}`, which divides skeleton from action in clip ids"
        )
    return f"{skeleton}{SEPARATOR}{action}"


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    skeleton: str
    action: str
    split: str
    n_frames: int
    fps: float
    path: str
    tags: tuple[str, ...] = ()


@dataclass
class CorpusIndex:
    records: list[ClipRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def add(self, record: ClipRecord) -> None:
        if any(existing.clip_id == record.clip_id for existing in self.records):
            raise ValueError(f"duplicate clip_id `{record.clip_id}`")
        self.records.append(record)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for record in self.records:
                payload = asdict(record)
                payload["tags"] = list(record.tags)
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> CorpusIndex:
        index = cls()
        fields = {f.name for f in dataclasses.fields(ClipRecord)}
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            # Ignore keys this version does not know: an index written by a
            # newer build must stay readable, or a colleague's corpus becomes
            # a TypeError about an unexpected keyword argument.
            known = {k: v for k, v in payload.items() if k in fields}
            if "tags" in known:
                known["tags"] = tuple(known["tags"])
            index.add(ClipRecord(**known))
        return index

    def query(
        self,
        *,
        skeleton: str | None = None,
        action: str | None = None,
        split: str | None = None,
        tag: str | None = None,
    ) -> list[ClipRecord]:
        return [
            record
            for record in self.records
            if (skeleton is None or record.skeleton == skeleton)
            and (action is None or record.action == action)
            and (split is None or record.split == split)
            and (tag is None or tag in record.tags)
        ]

    def skeletons(self) -> list[str]:
        return sorted({record.skeleton for record in self.records})
