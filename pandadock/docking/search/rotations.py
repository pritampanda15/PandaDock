"""
Rotation helpers for the docking inner loop.

`scipy.spatial.transform.Rotation` allocates a Python object per call, which
dominates the runtime when rotating a few dozen atoms hundreds of thousands of
times. These functions do the same arithmetic directly on numpy arrays.
"""

from typing import Tuple

import numpy as np


def cross_vec_array(k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Cross product of a single vector with an array of vectors: k x v[i].

    `np.cross` spends most of its time in axis-normalisation bookkeeping, which
    dominates when the arrays hold only a few dozen atoms and the call happens
    hundreds of thousands of times. Writing the three components out directly
    avoids that overhead entirely.
    """
    kx, ky, kz = k[0], k[1], k[2]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    out = np.empty_like(v)
    out[:, 0] = ky * vz - kz * vy
    out[:, 1] = kz * vx - kx * vz
    out[:, 2] = kx * vy - ky * vx
    return out


def cross_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cross product of two (N, 3) arrays."""
    ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
    bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
    out = np.empty_like(a)
    out[:, 0] = ay * bz - az * by
    out[:, 1] = az * bx - ax * bz
    out[:, 2] = ax * by - ay * bx
    return out


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two single 3-vectors."""
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def rodrigues_matrix(rotvec: np.ndarray) -> np.ndarray:
    """Rotation matrix for an axis-angle vector, via Rodrigues' formula."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)

    k = rotvec / theta
    kx, ky, kz = k
    K = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    return np.eye(3) + sin_t * K + (1.0 - cos_t) * (K @ K)


def rotate_about_axis(
    points: np.ndarray, axis: np.ndarray, pivot: np.ndarray, angle: float
) -> np.ndarray:
    """
    Rotate `points` by `angle` radians about the line through `pivot` along `axis`.

    `axis` need not be normalised.
    """
    norm = float(np.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2))
    if norm < 1e-12 or angle == 0.0:
        return points

    k = axis / norm
    v = points - pivot
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    # v cos + (k x v) sin + k (k . v)(1 - cos)
    rotated = (
        v * cos_a
        + cross_vec_array(k, v) * sin_a
        + k[None, :] * (v @ k)[:, None] * (1.0 - cos_a)
    )
    return rotated + pivot


def rotvec_gradient(
    rotvec: np.ndarray, rot_matrix: np.ndarray, torque: np.ndarray
) -> np.ndarray:
    """
    Gradient of a scalar with respect to an axis-angle vector.

    Given the torque tau = sum_i (c_i - t) x g_i accumulated over the rotated
    points, returns dE/d(rotvec).

    Uses the closed form for the derivative of the SO(3) exponential map
    (Gallego & Yezzi 2015):

        dR/dr_k = [u_k]_x R,   u_k = (r_k r + r x (I - R) e_k) / |r|^2

    Combining with the triple-product identity g . (u x p) = u . (p x g) reduces
    the whole thing to a single 3x3 matrix applied to the torque, which is exact
    and costs no extra energy evaluations.
    """
    theta_sq = float(rotvec @ rotvec)
    if theta_sq < 1e-16:
        # At the identity the exponential map's derivative is the identity, so
        # the gradient is just the torque.
        return torque

    imr = np.eye(3) - rot_matrix
    # Row k of `columns` is the k-th column of (I - R).
    columns = np.ascontiguousarray(imr.T)
    cross_terms = cross_vec_array(rotvec, columns)
    u = (np.outer(rotvec, rotvec) + cross_terms) / theta_sq
    return u @ torque


def wrap_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """
    Reduce an axis-angle vector to the equivalent rotation with angle <= pi.

    The local optimizer moves the orientation parameters continuously and can
    drift many turns away from the origin. The pose is unaffected, but the
    exponential-map derivative divides by |r|^2, so letting |r| grow costs
    precision in the rotation gradient. Wrapping after each optimization keeps
    the parameterisation well conditioned without changing any energy.
    """
    theta = float(np.linalg.norm(rotvec))
    if theta <= np.pi or theta < 1e-12:
        return rotvec

    axis = rotvec / theta
    wrapped = np.mod(theta, 2.0 * np.pi)
    if wrapped > np.pi:
        wrapped -= 2.0 * np.pi
    return axis * wrapped


def quaternion_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle to quaternion in (x, y, z, w) order, matching scipy's layout."""
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rotvec / theta
    half = theta / 2.0
    return np.concatenate([axis * np.sin(half), [np.cos(half)]])


def random_rotvec(rng: np.random.Generator) -> np.ndarray:
    """
    Sample a rotation uniformly over SO(3), returned as an axis-angle vector.

    Draws a uniform random quaternion (Shoemake's method) and converts, which
    gives a genuinely uniform orientation. Sampling the axis-angle components
    independently would concentrate density near the identity.
    """
    u1, u2, u3 = rng.random(3)
    s1, s2 = np.sqrt(1.0 - u1), np.sqrt(u1)
    quat = np.array(
        [
            s1 * np.sin(2.0 * np.pi * u2),
            s1 * np.cos(2.0 * np.pi * u2),
            s2 * np.sin(2.0 * np.pi * u3),
            s2 * np.cos(2.0 * np.pi * u3),
        ]
    )
    # quat is (x, y, z, w). A quaternion and its negation describe the same
    # rotation; taking the w >= 0 representative keeps the resulting angle in
    # [0, pi] so the axis-angle vector stays canonical and well conditioned.
    if quat[3] < 0.0:
        quat = -quat

    w = np.clip(quat[3], -1.0, 1.0)
    theta = 2.0 * np.arccos(w)
    sin_half = np.sqrt(max(1.0 - w * w, 0.0))
    if sin_half < 1e-12:
        return np.zeros(3)
    return quat[:3] / sin_half * theta


def compose_rotvecs(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Axis-angle vector representing `outer` applied after `inner`."""
    r_outer = rodrigues_matrix(outer)
    r_inner = rodrigues_matrix(inner)
    return matrix_to_rotvec(r_outer @ r_inner)


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix to axis-angle vector."""
    cos_theta = (np.trace(matrix) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = np.arccos(cos_theta)

    if theta < 1e-8:
        return np.zeros(3)

    if theta > np.pi - 1e-6:
        # Near pi the skew part vanishes; recover the axis from R + I, whose
        # columns are all parallel to the rotation axis.
        rpi = (matrix + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(rpi), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] > 1e-8:
            axis = rpi[:, k] / axis[k]
            axis = axis / np.linalg.norm(axis)
        return axis * theta

    skew = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    return skew * (theta / (2.0 * np.sin(theta)))
