"""The IK solver."""

import math

import pytest
import torch

from poseydon.core.torch_kinematics import forward_kinematics, rot6d_to_matrix
from poseydon.io.bvh import load_bvh
from poseydon.solvers import IK_TERMS, SOLVERS, GradientIK, SolverSkeleton


@pytest.fixture(scope="module")
def rig(truebones_dir):
    anim = load_bvh(truebones_dir / "Goat___HeadButt_395.bvh")
    frames = 10
    skeleton = SolverSkeleton(
        parents=torch.from_numpy(anim.parents.astype("int64")),
        offsets=torch.from_numpy(anim.offsets).float(),
    )
    targets = torch.from_numpy(anim.global_positions()[:frames]).float()
    return skeleton, targets


def solve(rig, terms=None, iterations=150, **kwargs):
    skeleton, targets = rig
    solver = GradientIK(
        terms=terms or [(1.0, IK_TERMS.get("position")())], iterations=iterations, **kwargs
    )
    rotations = solver.solve(targets, skeleton)
    positions = forward_kinematics(rotations, targets[:, 0], skeleton.offsets, skeleton.parents)
    return rotations, positions


def test_registries_are_populated():
    assert SOLVERS.names() == ["gradient_ik"]
    assert IK_TERMS.names() == ["contact_pin", "joint_limits", "position", "smoothness"]


def test_solving_reduces_position_error(rig):
    skeleton, targets = rig
    _, positions = solve(rig)

    identity = torch.tensor([1.0, 0, 0, 0, 1.0, 0]).repeat(*targets.shape[:2], 1)
    rest = forward_kinematics(
        rot6d_to_matrix(identity), targets[:, 0], skeleton.offsets, skeleton.parents
    )

    before = (rest - targets).norm(dim=-1).mean()
    after = (positions - targets).norm(dim=-1).mean()
    assert after < 0.5 * before, f"{before:.4f} -> {after:.4f}"


def test_more_iterations_fit_better(rig):
    _, few = solve(rig, iterations=20)
    _, many = solve(rig, iterations=300)
    _, targets = rig
    assert (many - targets).norm(dim=-1).mean() < (few - targets).norm(dim=-1).mean()


def test_bone_lengths_are_exact_by_construction(rig):
    # Rotation-space parameterization means no term has to defend bone lengths.
    skeleton, _targets = rig
    _, positions = solve(rig, iterations=50)

    bones = (positions[:, 1:] - positions[:, skeleton.parents[1:]]).norm(dim=-1)
    rest = skeleton.offsets[1:].norm(dim=-1).expand(positions.shape[0], -1)
    torch.testing.assert_close(bones, rest, rtol=1e-4, atol=1e-5)


def test_output_is_a_valid_rotation(rig):
    rotations, _ = solve(rig, iterations=20)
    gram = rotations @ rotations.transpose(-1, -2)
    torch.testing.assert_close(
        gram, torch.eye(3).expand_as(gram), rtol=1e-5, atol=1e-5
    )


def test_lbfgs_also_converges(rig):
    _, positions = solve(rig, iterations=40, optimizer="lbfgs", learning_rate=1.0)
    _, targets = rig
    assert torch.isfinite(positions).all()
    assert (positions - targets).norm(dim=-1).mean() < 1.0


def test_smoothness_reduces_frame_to_frame_change(rig):
    plain, _ = solve(rig, iterations=100)
    smooth, _ = solve(
        rig,
        terms=[(1.0, IK_TERMS.get("position")()), (10.0, IK_TERMS.get("smoothness")())],
        iterations=100,
    )
    assert (smooth[1:] - smooth[:-1]).pow(2).mean() < (plain[1:] - plain[:-1]).pow(2).mean()


def test_joint_limits_reduce_swing(rig):
    skeleton, _targets = rig
    limited = {j: 5.0 for j in range(1, skeleton.n_joints)}
    term = IK_TERMS.get("joint_limits")(max_swing_deg=limited)

    plain, _ = solve(rig, iterations=100)
    bounded, _ = solve(
        rig, terms=[(1.0, IK_TERMS.get("position")()), (50.0, term)], iterations=100
    )

    def worst_swing(rotations):
        angles = []
        for joint in range(1, skeleton.n_joints):
            rest = skeleton.offsets[joint]
            if rest.norm() < 1e-8:
                continue
            direction = rest / rest.norm()
            rotated = torch.einsum("fij,j->fi", rotations[:, joint], direction)
            expanded = direction.expand_as(rotated)
            sine = torch.linalg.vector_norm(torch.cross(expanded, rotated, dim=-1), dim=-1)
            angles.append(torch.atan2(sine, (rotated * expanded).sum(-1)).max())
        return max(angles)

    assert worst_swing(plain) > math.radians(45), "unconstrained solve should bend freely"
    # The declared cone is respected, not merely improved upon.
    assert worst_swing(bounded) < math.radians(10)


def test_joint_limit_gradients_are_finite_at_rest():
    # A joint sitting at rest has cosine exactly 1, where acos has an infinite
    # derivative. The atan2 formulation must survive it.
    skeleton = SolverSkeleton(
        parents=torch.tensor([-1, 0]), offsets=torch.tensor([[0.0, 0, 0], [0.0, 1, 0]])
    )
    term = IK_TERMS.get("joint_limits")(max_swing_deg={1: 30.0})
    identity = torch.tensor([1.0, 0, 0, 0, 1.0, 0]).repeat(3, 2, 1).requires_grad_(True)
    rotations = rot6d_to_matrix(identity)
    positions = torch.zeros(3, 2, 3)

    term(positions, rotations, positions, skeleton).backward()
    assert torch.isfinite(identity.grad).all()


def test_joint_limits_are_a_no_op_when_undeclared(rig):
    skeleton, targets = rig
    term = IK_TERMS.get("joint_limits")()
    rotations = torch.eye(3).expand(4, skeleton.n_joints, 3, 3)
    positions = targets[:4]
    assert term(positions, rotations, positions, skeleton).item() == 0.0


def test_contact_pin_penalizes_movement_of_planted_joints(rig):
    skeleton, targets = rig
    contact = torch.zeros(targets.shape[0], skeleton.n_joints)
    contact[:, 5] = 1.0
    term = IK_TERMS.get("contact_pin")(contact)

    still = targets[:1].expand_as(targets)
    rotations = torch.eye(3).expand(targets.shape[0], skeleton.n_joints, 3, 3)

    assert term(still, rotations, targets, skeleton).item() == pytest.approx(0.0)
    assert term(targets, rotations, targets, skeleton).item() > 0.0


def test_rejects_bad_input(rig):
    skeleton, targets = rig
    solver = GradientIK(terms=[(1.0, IK_TERMS.get("position")())], iterations=1)

    with pytest.raises(ValueError, match=r"\(F, J, 3\)"):
        solver.solve(targets[..., :2], skeleton)
    with pytest.raises(ValueError, match="joints"):
        solver.solve(targets[:, :-1], skeleton)


def test_rejects_bad_configuration():
    with pytest.raises(ValueError, match="at least one"):
        GradientIK(terms=[])
    with pytest.raises(ValueError, match="optimizer"):
        GradientIK(terms=[(1.0, IK_TERMS.get("position")())], optimizer="newton")
