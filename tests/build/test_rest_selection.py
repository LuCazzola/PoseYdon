"""``find_tpose`` must choose the MODAL skeleton, not whichever file is named a T-pose.

For four real rigs (Ant, Crab, Deer, Jaguar) the file named `tpos`/`idle`
disagrees with almost every clip's joint set, and every disagreeing clip is
then rejected by `RestRelative` downstream. The fix picks the rest reference
from the most common joint-name tuple across the rig's clips, applying the
old naming preference only within that modal set.
"""

from __future__ import annotations

from pathlib import Path

from scripts.process_dataset_truebones import find_tpose

_HEADER_TAIL = "MOTION\nFrames: 1\nFrame Time: 0.0083333\n0.0\n"


def _bvh_text(joint_names: list[str]) -> str:
    lines = ["HIERARCHY", f"ROOT {joint_names[0]}", "{", "OFFSET 0.0 0.0 0.0",
              "CHANNELS 3 Xposition Yposition Zposition"]
    for name in joint_names[1:]:
        lines += [
            f"JOINT {name}",
            "{",
            "OFFSET 1.0 0.0 0.0",
            "CHANNELS 3 Zrotation Xrotation Yrotation",
            "End Site",
            "{",
            "OFFSET 1.0 0.0 0.0",
            "}",
            "}",
        ]
    lines.append("}")
    return "\n".join(lines) + "\n" + _HEADER_TAIL


def _write(tmp_path: Path, name: str, joint_names: list[str]) -> Path:
    path = tmp_path / name
    path.write_text(_bvh_text(joint_names))
    return path


def test_rest_selection_picks_modal_skeleton_not_the_named_tpose_file(tmp_path):
    majority_joints = ["Hips", "Spine", "Head"]
    minority_joints = ["Root", "Chest", "Skull", "Tail"]

    tpose = _write(tmp_path, "__TPOSE.bvh", minority_joints)
    walk = _write(tmp_path, "__Walk.bvh", majority_joints)
    run = _write(tmp_path, "__Run.bvh", majority_joints)
    idle = _write(tmp_path, "__Idle.bvh", majority_joints)

    clip_paths = sorted([tpose, walk, run, idle])
    chosen, warnings = find_tpose(clip_paths)

    assert chosen in (walk, run, idle)
    assert chosen != tpose

    assert any(
        "__TPOSE.bvh" in message and chosen.name in message for message in warnings
    ), f"expected a warning naming both files, got: {warnings}"
