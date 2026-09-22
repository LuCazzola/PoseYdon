import json

from scripts.build_skeleton_visual_debug import build_bundle


def test_bundle_records_typed_absence_and_hash(tmp_path):
    bvh=tmp_path/"x.bvh"; bvh.write_text("fixture")
    build_bundle([{"rig":"BvhOnly","action":"x","original_bvh":bvh}], tmp_path/"out")
    data=json.loads((tmp_path/"out"/"BvhOnly"/"metadata.json").read_text())
    assert data["original_bvh"]["sha256"]
    assert data["prepared_fbx"] is None
    assert data["unavailable"]["prepared_fbx"] == "source_surface_missing"


def test_discover_derives_the_training_animation_and_drops_stale_roundtrips(tmp_path):
    """Everything in a bundle must describe the SAME clip.

    `discover` used to carry over whatever `training.anim.npz` a previous run
    had left in the directory. The sample action moves as the raw tree changes,
    so in 60 of 73 bundles that file belonged to a different clip than the
    `original.bvh` beside it -- and the roundtrip built from it therefore
    reconstructed a different animation than the one it was filed next to,
    which reads as a catastrophically broken export rather than a mismatched
    pair.
    """
    import numpy as np
    import pytest
    from scripts.build_skeleton_visual_debug import discover

    from poseydon.core.animation import Animation, RigidBodyAnimation
    from poseydon.core.rotations import QUAT_IDENTITY
    from poseydon.features.reduce import apply_reduction, build_reduction
    from poseydon.io.bvh import BVH
    from tests.conftest import CORPUS

    raw = CORPUS.parent / "_raw" / "truebones"
    if not (raw / "source").is_dir():
        pytest.skip("Truebones corpus not present")

    contexts = {c["rig"]: c for c in discover(CORPUS, raw, tmp_path)}
    derived = [c for c in contexts.values() if c.get("training_anim")]
    assert derived, "no bundle produced a training animation"

    context = derived[0]
    rig, action = context["rig"], context["action"]
    with np.load(CORPUS / "rigs" / rig / "skeleton.npz", allow_pickle=True) as data:
        names = tuple(str(x) for x in data["names"])
        rest = RigidBodyAnimation.from_root_motion(
            rotations=np.tile(QUAT_IDENTITY, (1, len(names), 1)),
            root_pos=np.zeros((1, 3)), offsets=data["offsets"],
            parents=data["parents"].astype(np.int32), names=names, fps=1.0)
    want = apply_reduction(
        BVH.read(CORPUS / "clips" / rig / f"{action}.bvh")
        .to_animation()
        .as_rigid_body(joint_translation="drop"),
        build_reduction(rest),
    )
    got = Animation.load(context["training_anim"])
    assert tuple(got.names) == tuple(want.names)
    np.testing.assert_allclose(got.rotations, want.rotations, atol=1e-12)

    # And a roundtrip left over from an earlier run is removed, not carried
    # forward beside freshly derived training data.
    stale = tmp_path / rig / "roundtrip.bvh"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("from a previous run")
    again = {c["rig"]: c for c in discover(CORPUS, raw, tmp_path)}
    assert not stale.exists()
    assert "roundtrip_bvh" not in again[rig]
