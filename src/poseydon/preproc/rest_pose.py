"""T-pose geometry recovery and bind-pose removal for raw rig exports.

Raw Biped rigs bake an arbitrary, per-joint "bind" rotation into every clip
-- an exporter axis-convention artifact, not real animation. This module
removes it, and separately recovers the true rest-pose bone geometry a raw
T-pose file's position channels carry (see
docs/superpowers/specs/2026-09-05-preproc-package-design.md section 1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from poseydon.core.animation import Animation, RigidBodyAnimation
from poseydon.core.rotations import quat_apply, quat_inverse, quat_mul
from poseydon.io.bvh import BVH


def make_anim_rest_relative(
    anim: Animation, rest_anim: Animation
) -> Animation:
    """Re-express ``anim`` so zero rotation on every joint reproduces the rest pose.

    Ports the reference's ``compute_rots_from_tpos``, verified term-by-term
    against the real source and the real ``Quaternions.__mul__``/``__neg__``
    (design spec section 1). Matched by joint NAME -- ``rest_anim`` need not
    cover every joint ``anim`` does. A joint missing from ``rest_anim``
    falls back to ``anim``'s own frame-0 rotation as its bind (exact for a
    childless End Site, an approximation elsewhere).

    The returned rotations read as identity, for every joint, exactly when
    a frame equals the rest pose (proof: this function's own rotations are
    ``old_global[j] composed with the inverse of joint j's GLOBAL rest
    rotation``, which is identity when ``old_global == rest_global``).
    Offsets must therefore be plain WORLD-SPACE differences of the rest
    pose's own global positions, not offsets rotated into any joint's
    local frame (as ``rest_anim.offsets`` -- built by
    :func:`establish_rest_pose` to pair correctly with the T-pose file's
    OWN, generally non-identity rotations, for alignment purposes -- is):
    identity rotation applied to a world-space offset reproduces that
    offset unchanged, which is exactly what makes FK using these new
    rotations self-consistent at the rest frame. Reusing
    ``rest_anim.offsets`` directly instead (an earlier version of this
    function did) paired identity rotations with parent-local offsets --
    a coordinate-frame mismatch invisible at the rest frame (both give the
    same trivial result there) but visibly wrong everywhere else,
    confirmed by rendering.
    """
    rest_index = {name: i for i, name in enumerate(rest_anim.names)}
    rest_global_pos = rest_anim.global_positions()

    # `local_bind[j]` is joint j's own LOCAL rest rotation. `bind[j]` is the
    # GLOBAL rest rotation -- needed for the PARENT side of the formula
    # below. Both are built in one pass since parents always precede
    # children.
    local_bind = np.empty((anim.n_joints, 4))
    bind = np.empty((anim.n_joints, 4))
    for joint, name in enumerate(anim.names):
        local_bind[joint] = (
            rest_anim.rotations[0, rest_index[name]] if name in rest_index else anim.rotations[0, joint]
        )
        bind[joint] = (
            local_bind[joint]
            if joint == 0
            else quat_mul(bind[anim.parents[joint]], local_bind[joint])
        )

    new_offsets = anim.offsets.copy()
    for joint, name in enumerate(anim.names):
        if joint == 0 or name not in rest_index:
            continue
        rest_i = rest_index[name]
        rest_parent_i = rest_anim.parents[rest_i]
        new_offsets[joint] = rest_global_pos[0, rest_i] - rest_global_pos[0, rest_parent_i]
    new_rotations = anim.rotations.copy()

    new_rotations[:, 0] = quat_mul(anim.rotations[:, 0], quat_inverse(local_bind[0]))

    for joint in range(1, anim.n_joints):
        parent_bind = bind[anim.parents[joint]]

        # Sandwiched between its own LOCAL bind (undone) and its parent's
        # GLOBAL bind (removed, then reapplied) -- the reference's
        # compute_rots_from_tpos. The world-space offsets computed above
        # are the skeleton-level constant this pairs with, reused for
        # every frame and every clip.
        new_rotations[:, joint] = quat_mul(
            quat_mul(quat_mul(parent_bind, anim.rotations[:, joint]), quat_inverse(local_bind[joint])),
            quat_inverse(parent_bind),
        )

    # Per-joint translation is expressed in the PARENT's local frame, and this
    # function has just turned every local frame by that joint's bind rotation
    # -- so a translation has to be carried into the new frame with it. The
    # rigid case is the special case: a joint whose translation is a constant
    # `offsets[j]` gets `B_parent . offsets[j]`, which is exactly the
    # world-space rest delta stored in `new_offsets` above.
    new_translations = anim.translations.copy()
    for joint in range(1, anim.n_joints):
        new_translations[:, joint] = quat_apply(
            bind[anim.parents[joint]], anim.translations[:, joint]
        )

    # The base type, not RigidBodyAnimation: a clip that genuinely translates
    # stays non-rigid, and pinning it here would silently discard the motion.
    return Animation(
        rotations=new_rotations,
        translations=new_translations,
        offsets=new_offsets,
        parents=anim.parents,
        names=anim.names,
        fps=anim.fps,
    )


def establish_rest_pose(path: str | Path) -> RigidBodyAnimation:
    """
    Recover a raw T-pose file's true bone geometry, keeping its own declared rotations.
    """
    raw = BVH.read(path).to_animation().slice(0, 1)
    global_pos, global_rot = raw.global_transforms()

    final_offsets = np.zeros_like(raw.offsets)
    parent_of = raw.parents[1:]
    final_offsets[1:] = quat_apply(
        quat_inverse(global_rot[0, parent_of]),
        global_pos[0, 1:] - global_pos[0, parent_of],
    )

    return RigidBodyAnimation.from_root_motion(
        rotations=raw.rotations,
        root_pos=global_pos[:, 0],
        offsets=final_offsets,
        parents=raw.parents,
        names=raw.names,
        fps=raw.fps,
    )
