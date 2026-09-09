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


def test_build_skeleton_refuses_a_mesh_with_a_mismatched_joint_count(config):
    """Camel is one of the 14 `PROMOTE_ROOT` rigs: the FBX path never runs
    `PromoteRoot`, so its `mesh.npz` keeps `Hips` and `C_ctrl`, joints the
    promoted BVH skeleton does not have (51 FBX joints against 49 real BVH
    joints -- measured in task 8). Design spec 2026-09-08 Stage 2 decided the
    FBX corpus stays un-promoted and `build_skeleton` refuses to fold a
    mismatched mesh rather than trusting positional alignment; this is that
    refusal, and it is what makes `test_bvh_and_fbx_agree_on_the_skeleton_
    structure`'s Camel coverage meaningful instead of an xfail with nothing
    behind it."""
    mesh_path = CORPUS / "rigs" / "Camel" / "mesh.npz"
    if not mesh_path.is_file():
        pytest.skip("Camel mesh.npz not present -- run the fbx container for Camel")

    with pytest.raises(ValueError) as excinfo:
        build_skeleton(config, "Camel")

    message = str(excinfo.value)
    assert "51" in message and "49" in message, message
    assert "PromoteRoot" in message, message


def test_build_skeleton_accepts_a_mesh_with_a_matching_joint_count(config):
    """A non-promoted rig's mesh.npz must still fold without tripping the
    guard added for Camel -- the refusal is for a genuine mismatch, not for
    every mesh.npz."""
    mesh_path = CORPUS / "rigs" / "Crab" / "mesh.npz"
    if not mesh_path.is_file():
        pytest.skip("Crab mesh.npz not present -- run the fbx container for Crab")

    build_skeleton(config, "Crab")  # must not raise
    assert (config.out / "rigs" / "Crab" / "skeleton.npz").is_file()


def test_the_skeleton_pass_stores_both_raw_and_humanized_names(config):
    build_skeleton(config, "Ant")
    with np.load(config.out / "rigs" / "Ant" / "skeleton.npz", allow_pickle=True) as d:
        raw, human = list(d["names"]), list(d["humanized"])
    assert len(raw) == len(human)
    assert raw != human, "humanizing must actually change something"
    assert all(n.islower() or not n.isalpha() for n in human if n)


def test_stats_are_fitted_under_the_declared_policy(config):
    """joint_block pooling must be visible in the stored std, not just claimed."""
    build_clips(config, "Ant")
    build_skeleton(config, "Ant")
    build_stats(config, "Ant")

    stats = Normalizer.load(config.out / "rigs" / "Ant" / "stats.npz")
    block = stats.std[:, stats.spec.slice("rot6d")]
    for joint in range(block.shape[0]):
        np.testing.assert_allclose(block[joint], block[joint, 0], atol=1e-12)

    contact = stats.std[:, stats.spec.slice("foot_contact")]
    np.testing.assert_allclose(contact, 1.0)


def test_stats_record_the_schema_they_were_fitted_under(config):
    build_clips(config, "Ant")
    build_skeleton(config, "Ant")
    build_stats(config, "Ant")
    stats = Normalizer.load(config.out / "rigs" / "Ant" / "stats.npz")
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
