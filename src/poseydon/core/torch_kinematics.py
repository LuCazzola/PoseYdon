"""Differentiable forward kinematics, in torch.

The numpy path in :mod:`poseydon.core.kinematics` serves ingest; this one is for
anything that needs gradients through FK, which today means the IK solver.
"""

from __future__ import annotations

import torch


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt two 3-vectors into a rotation matrix.

    Optimizing in this representation rather than in quaternions or Euler angles
    means every point in the parameter space maps to a valid rotation, so the
    solver never has to renormalize or fight gimbal lock.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def forward_kinematics(
    rotations: torch.Tensor,
    root_pos: torch.Tensor,
    offsets: torch.Tensor,
    parents: torch.Tensor,
) -> torch.Tensor:
    """Local rotation matrices to global positions.

    Args:
        rotations: ``(F, J, 3, 3)`` local rotation matrices.
        root_pos:  ``(F, 3)`` root translation.
        offsets:   ``(J, 3)`` rest offsets from each joint's parent.
        parents:   ``(J,)`` parent index, ``-1`` for the root.

    Returns:
        ``(F, J, 3)`` global positions.
    """
    n_frames, n_joints = rotations.shape[:2]
    positions: list[torch.Tensor] = [root_pos]
    globals_: list[torch.Tensor] = [rotations[:, 0]]

    for joint in range(1, n_joints):
        parent = int(parents[joint])
        offset = offsets[joint].to(rotations.dtype).expand(n_frames, 3)
        positions.append(
            positions[parent] + torch.einsum("fij,fj->fi", globals_[parent], offset)
        )
        globals_.append(globals_[parent] @ rotations[:, joint])

    return torch.stack(positions, dim=1)
