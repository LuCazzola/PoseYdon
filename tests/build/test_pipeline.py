"""The four build passes (parity spec §4).

Run against the real prepared corpus -- these artefacts are what training reads,
and a synthetic fixture would not exercise the rig-shaped edge cases (a promoted
rig, a rig whose rest clip is not named tpose) that motivate the design.
"""

from __future__ import annotations

import numpy as np
import pytest

from poseydon.build.corpus import PerRigDirectory
from poseydon.build.names import Humanize
from poseydon.build.pipeline import (
    BuildConfig,
    build_clips,
    build_index,
    build_skeleton,
    build_stats,
)
from poseydon.core.skeleton import SkeletonManifest
from poseydon.data.normalize import BlockPolicy, Normalizer
from tests.conftest import CORPUS

SCHEMA = ("ric_pos", "rot6d", "local_vel", "foot_contact")
POLICY = (
    BlockPolicy("ric_pos", center=True, scale="joint_block"),
    BlockPolicy("rot6d", center=False, scale="joint_block"),
    BlockPolicy("local_vel", center=True, scale="joint_block"),
    BlockPolicy("foot_contact", center=False, scale="none"),
)


@pytest.fixture
def config(tmp_path):
    """Reads the real corpus, writes into tmp_path -- so a test never mutates
    the corpus other tests read."""
    if not (CORPUS / "clips").is_dir():
        pytest.skip("prepared corpus not present -- run stage 1")
    return BuildConfig(
        root=CORPUS,
        out=tmp_path,
        schema=SCHEMA,
        corpus=PerRigDirectory(),
        names=Humanize(),
        normalize=POLICY,
    )


def test_the_clip_pass_writes_one_npz_per_prepared_clip(config):
    written = build_clips(config, "Goat")
    produced = sorted((config.out / "clips" / "Goat").glob("*.npz"))
    source = sorted((CORPUS / "clips" / "Goat").glob("*.bvh"))
    assert written == len(source)
    assert [p.stem for p in produced] == [p.stem for p in source]


def test_each_clip_npz_carries_its_own_facing_quaternion(config):
    """FaceAxis is clip-scoped; folding the wrong clip's rotation in would be
    invisible until someone tried to un-prepare a generated clip."""
    from poseydon.build.prepare import RigTransform

    build_clips(config, "Goat")
    recorded = RigTransform.load(CORPUS / "rigs" / "Goat" / "prepare.npz")
    for path in sorted((config.out / "clips" / "Goat").glob("*.npz")):
        with np.load(path) as data:
            np.testing.assert_allclose(
                data["facing"],
                recorded.clip_params[path.stem]["face_axis"]["rotation"],
            )


def test_the_skeleton_pass_takes_offsets_from_the_rest_clip(config):
    """Prepared clips of one rig do NOT share an OFFSET block -- each carries
    its own facing correction -- so rig-level offsets must come from the rest
    clip specifically. Crab's is `walk`, which is exactly why matching on a
    filename would be wrong."""
    from poseydon.build.corpus import rest_action
    from poseydon.io.bvh import BVH

    rig = "Crab"
    build_skeleton(config, rig)
    manifest = SkeletonManifest.load(CORPUS / "rigs" / rig / "manifest.yaml")
    expected = BVH.read(
        CORPUS / "clips" / rig / f"{rest_action(manifest)}.bvh"
    ).to_animation()

    with np.load(config.out / "rigs" / rig / "skeleton.npz", allow_pickle=True) as data:
        np.testing.assert_allclose(data["offsets"], expected.offsets, atol=1e-9)
        assert list(data["names"]) == list(expected.names)


def test_build_skeleton_warns_but_still_writes_on_a_mismatched_mesh(config):
    """Camel is one of the 14 `PROMOTE_ROOT` rigs: the FBX path never runs
    `PromoteRoot`, so its `mesh.npz` keeps `Hips` and `C_ctrl`, joints the
    promoted BVH skeleton does not have (51 FBX joints against 49 real BVH
    joints -- measured in task 8). Design spec 2026-09-08 Stage 2 says stage 2
    must not trust positional alignment between a disagreeing mesh and
    skeleton -- but `skeleton.npz`/`stats.npz` never read `mesh.npz` at all
    (mesh folding does not exist yet), so the mismatch must be a WARNING, not
    a refusal that withholds the training artefacts (task 9 fix round 1: a
    build over all 73 rigs found this had silently cost Camel and Goat their
    skeleton.npz/stats.npz, for a disagreement in a file training never
    reads). This is that warning path, and it is what makes
    `test_bvh_and_fbx_agree_on_the_skeleton_structure`'s Camel coverage
    meaningful instead of an xfail with nothing behind it."""
    mesh_path = CORPUS / "rigs" / "Camel" / "mesh.npz"
    if not mesh_path.is_file():
        pytest.skip("Camel mesh.npz not present -- run the fbx container for Camel")

    warning = build_skeleton(config, "Camel")

    assert warning is not None, "a mismatched mesh must be reported, not silently accepted"
    assert "51" in warning and "49" in warning, warning
    assert "PromoteRoot" in warning, warning
    assert (config.out / "rigs" / "Camel" / "skeleton.npz").is_file(), (
        "the mismatch must not withhold skeleton.npz -- mesh.npz is not "
        "folded into it"
    )


def test_build_skeleton_accepts_a_mesh_with_a_matching_joint_count(config):
    """A non-promoted rig's mesh.npz must still fold without tripping the
    guard added for Camel -- the warning is for a genuine mismatch, not for
    every mesh.npz."""
    mesh_path = CORPUS / "rigs" / "Crab" / "mesh.npz"
    if not mesh_path.is_file():
        pytest.skip("Crab mesh.npz not present -- run the fbx container for Crab")

    warning = build_skeleton(config, "Crab")
    assert warning is None, warning
    assert (config.out / "rigs" / "Crab" / "skeleton.npz").is_file()


#: Goat's `mesh.npz` disagrees with its own skeleton.npz -- measured in task
#: 8's fix round 1: the raw FBX armature (`Goat-Die.fbx`, confirmed straight
#: off `bpy.ops.import_scene.fbx`, before any PoseYdon processing) carries a
#: `Null` root bone at (0, 0, 0), parent of `Hips`, that the prepared BVH does
#: not have -- 33 FBX joints against 32 real BVH joints. Goat is NOT one of
#: the 14 `PROMOTE_ROOT` rigs, so this is a second, distinct instance of the
#: same "un-promoted ground locator" shape the FBX path already fails to
#: strip -- a genuine data disagreement, not a miscount in `_n_real_joints`
#: (the four other rigs with a `mesh.npz` -- Flamingo, BrownBear, Crab,
#: Scorpion -- all match their real BVH joint count exactly). Since task 9's
#: fix round 1, this mismatch is a WARNING, not a refusal: `build_skeleton`
#: and `build_stats` write their artefacts regardless, because neither reads
#: `mesh.npz`. The tests below no longer xfail on it -- they assert the
#: artefacts ARE written, and check the warning separately.
_GOAT_MESH_MISMATCH_REASON = (
    "Goat: mesh.npz has 33 joints but the rest skeleton has 32 real joints -- "
    "an FBX-only `Null` root bone above `Hips`, present in the raw FBX armature "
    "itself (not a PoseYdon artefact). Goat is not a PROMOTE_ROOT rig, so this "
    "is a genuine data disagreement distinct from Camel's, not a code miscount "
    "-- see task 8's fix round 1 report and the design spec's Limitations."
)


def test_the_skeleton_pass_stores_both_raw_and_humanized_names(config):
    build_skeleton(config, "Goat")
    with np.load(config.out / "rigs" / "Goat" / "skeleton.npz", allow_pickle=True) as d:
        raw, human = list(d["names"]), list(d["humanized"])
    assert len(raw) == len(human)
    assert raw != human, "humanizing must actually change something"
    assert all(n.islower() or not n.isalpha() for n in human if n)


def test_the_skeleton_pass_warns_on_goats_mismatched_mesh(config):
    """Goat's `mesh.npz`/`skeleton.npz` disagreement (see
    `_GOAT_MESH_MISMATCH_REASON`) must surface as a warning from
    `build_skeleton` even though the artefact is written regardless."""
    mesh_path = CORPUS / "rigs" / "Goat" / "mesh.npz"
    if not mesh_path.is_file():
        pytest.skip("Goat mesh.npz not present -- run the fbx container for Goat")
    warning = build_skeleton(config, "Goat")
    assert warning is not None, _GOAT_MESH_MISMATCH_REASON
    assert "33" in warning and "32" in warning, warning


def test_stats_are_fitted_under_the_declared_policy(config):
    """joint_block pooling must be visible in the stored std, not just claimed."""
    build_clips(config, "Goat")
    build_skeleton(config, "Goat")
    build_stats(config, "Goat")

    stats = Normalizer.load(config.out / "rigs" / "Goat" / "stats.npz")
    block = stats.std[:, stats.spec.slice("rot6d")]
    for joint in range(block.shape[0]):
        np.testing.assert_allclose(block[joint], block[joint, 0], atol=1e-12)

    contact = stats.std[:, stats.spec.slice("foot_contact")]
    np.testing.assert_allclose(contact, 1.0)


def test_stats_record_the_schema_they_were_fitted_under(config):
    build_clips(config, "Goat")
    build_skeleton(config, "Goat")
    build_stats(config, "Goat")
    stats = Normalizer.load(config.out / "rigs" / "Goat" / "stats.npz")
    assert stats.spec.names == SCHEMA


def test_the_index_has_one_row_per_clip_and_joins_rig_tags(config):
    for rig in ("Goat", "Crab"):
        build_clips(config, rig)
    index = build_index(config)

    rows = [r for r in index.records if r.skeleton == "Goat"]
    on_disk = sorted((CORPUS / "clips" / "Goat").glob("*.bvh"))
    assert len(rows) == len(on_disk)

    manifest = SkeletonManifest.load(CORPUS / "rigs" / "Goat" / "manifest.yaml")
    assert rows[0].tags == manifest.tags, "tags live once, in the manifest"


def test_the_index_path_points_at_the_npz_not_the_bvh(config):
    """Training never opens a .bvh."""
    build_clips(config, "Goat")
    index = build_index(config)
    assert all(r.path.endswith(".npz") for r in index.records)


def test_an_authored_split_reaches_the_index_row(config):
    """`build_clips` writes `split: train` into each `<action>.yaml`, and
    `--relabel` exists specifically so a hand-edited `split:` survives a
    rebuild -- but `build_index` used to hardcode `split="train"` regardless
    of what the label file said, so editing it had no effect on the index.
    """
    build_clips(config, "Goat")
    label_files = sorted((config.out / "clips" / "Goat").glob("*.yaml"))
    assert label_files, "build_clips must have written label files"
    edited = label_files[0]
    action = edited.stem
    edited.write_text(f"split: val\naction: {action}\n")

    index = build_index(config)
    row = next(r for r in index.records if r.skeleton == "Goat" and r.action == action)
    assert row.split == "val"

    others = [r for r in index.records if r.skeleton == "Goat" and r.action != action]
    assert all(r.split == "train" for r in others), "an unedited label keeps the default"


def test_one_malformed_label_file_does_not_cost_the_whole_index(config):
    """`build_index` runs ONCE, after `build_all`'s per-rig guard, so anything
    that escapes it discards a completed multi-rig build's index entirely.
    Reading the per-clip label file put a new raise on that path; this pins the
    guard that keeps a hand-edited file with a stray character from costing
    every other row.
    """
    build_clips(config, "Goat")
    labels = sorted((config.out / "clips" / "Goat").glob("*.yaml"))
    labels[0].write_text("- this is a list, not a mapping\n")

    warnings: list[str] = []
    index = build_index(config, warnings)

    assert len(index.records) == len(labels), "every clip still gets a row"
    broken = next(r for r in index.records if r.action == labels[0].stem)
    assert broken.split == "train", "the unreadable label falls back to the default"
    assert any(labels[0].name in w for w in warnings), "and says so in warnings"


def test_a_non_string_split_falls_back_rather_than_reaching_the_index(config):
    """`split:` with no value parses as `None`, not `"train"`. The read path
    selects rows by string equality, so writing `None` would produce a row no
    split can ever pick up -- invisible rather than wrong.
    """
    build_clips(config, "Goat")
    labels = sorted((config.out / "clips" / "Goat").glob("*.yaml"))
    labels[0].write_text(f"action: {labels[0].stem}\nsplit:\n")

    warnings: list[str] = []
    index = build_index(config, warnings)

    row = next(r for r in index.records if r.action == labels[0].stem)
    assert row.split == "train"
    assert any("must be a string" in w for w in warnings)
