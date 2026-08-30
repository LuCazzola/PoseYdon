import numpy as np

from poseydon.core.rotations import (
    QUAT_IDENTITY,
    euler_to_quat,
    matrix_to_quat,
    matrix_to_rot6d,
    quat_apply,
    quat_between,
    quat_mul,
    quat_to_matrix,
    rot6d_to_matrix,
)


def random_quats(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def test_quat_layout_is_scalar_last():
    # Identity has w == 1 in the LAST slot.
    assert QUAT_IDENTITY.tolist() == [0.0, 0.0, 0.0, 1.0]


def test_euler_zyx_90_about_z_maps_x_axis_to_y_axis():
    # BVH channel order "Zrotation Yrotation Xrotation" -> order string "ZYX",
    # angles given in that same order.
    q = euler_to_quat(np.array([90.0, 0.0, 0.0]), "ZYX")
    got = quat_apply(q, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(got, [0.0, 1.0, 0.0], atol=1e-12)


def test_quat_matrix_round_trip():
    q = random_quats(64)
    back = matrix_to_quat(quat_to_matrix(q))
    # q and -q are the same rotation; compare via matrices instead.
    np.testing.assert_allclose(quat_to_matrix(back), quat_to_matrix(q), atol=1e-12)


def test_rot6d_round_trip():
    m = quat_to_matrix(random_quats(64, seed=1))
    np.testing.assert_allclose(rot6d_to_matrix(matrix_to_rot6d(m)), m, atol=1e-12)


def test_rot6d_uses_first_two_rows():
    m = quat_to_matrix(random_quats(4, seed=2))
    np.testing.assert_allclose(matrix_to_rot6d(m), m[..., :2, :].reshape(-1, 6), atol=0)


def test_rot6d_reorthonormalizes_a_perturbed_input():
    m = quat_to_matrix(random_quats(8, seed=3))
    d6 = matrix_to_rot6d(m) + 0.01
    out = rot6d_to_matrix(d6)
    eye = np.einsum("...ij,...kj->...ik", out, out)
    np.testing.assert_allclose(eye, np.broadcast_to(np.eye(3), eye.shape), atol=1e-12)


def test_quat_mul_matches_matrix_product():
    a, b = random_quats(32, seed=4), random_quats(32, seed=5)
    np.testing.assert_allclose(
        quat_to_matrix(quat_mul(a, b)),
        quat_to_matrix(a) @ quat_to_matrix(b),
        atol=1e-12,
    )


def test_quat_between_rotates_a_onto_b():
    rng = np.random.default_rng(6)
    a = rng.normal(size=(32, 3))
    b = rng.normal(size=(32, 3))
    a /= np.linalg.norm(a, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    np.testing.assert_allclose(quat_apply(quat_between(a, b), a), b, atol=1e-10)


def test_quat_between_handles_antiparallel_vectors():
    a = np.array([[1.0, 0.0, 0.0]])
    b = -a
    np.testing.assert_allclose(quat_apply(quat_between(a, b), a), b, atol=1e-10)


def test_quat_between_handles_identical_vectors():
    a = np.array([[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(quat_between(a, a), [QUAT_IDENTITY], atol=1e-12)
