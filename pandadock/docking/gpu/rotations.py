"""
Batched SO(3) primitives, matching pandadock.docking.search.rotations.

Each function here is the batched counterpart of a scalar CPU function, and the
tests assert they agree elementwise. The CPU versions branch on magnitude --
`if theta < 1e-12: return identity` and similar -- which does not translate
directly: a batch contains both degenerate and ordinary rotations at once, so
every branch has to be evaluated for every element and selected with `where`.

That creates the trap this module is written around, though not the one it first
appears to be. `rotvec / theta` is NaN when theta is zero. In the forward pass
`torch.where` does select the correct branch and the NaN goes away, so a naive
implementation looks right. It is autograd that breaks: `where` routes a zero
gradient into the unselected branch, and 0 * NaN is NaN, so the identity row
returns NaN gradients and -- once an optimiser sums them -- takes the rest of
the batch with it.

Every division below is therefore guarded by a denominator clamped away from
zero rather than by `where` alone. At an exact identity the derivative really is
singular, so the guarded gradient is merely enormous (order 1/eps) instead of
NaN; that is a number an optimiser can clip, which NaN is not. This matters as
soon as the batched local optimiser starts differentiating through these
functions. See tests/test_gpu_rotations.py::test_guarded_division_keeps_gradients_finite.
"""

from __future__ import annotations

import numpy as np

try:
    import torch

    torch_available = True
except ImportError:  # pragma: no cover - exercised only in a base install
    torch = None
    torch_available = False


# Matches the scalar thresholds in the CPU implementation so that the two agree
# about which inputs are degenerate.
EPS_IDENTITY = 1e-12
EPS_ANGLE = 1e-8
EPS_NEAR_PI = 1e-6


def _safe_div(numerator, denominator, eps=EPS_IDENTITY):
    """Divide with a denominator clamped away from zero, killing NaN at source."""
    return numerator / denominator.clamp_min(eps)


def rodrigues_matrix(rotvec: "torch.Tensor") -> "torch.Tensor":
    """
    Rotation matrices for a batch of axis-angle vectors, (B, 3) -> (B, 3, 3).

    Rotations with |r| below EPS_IDENTITY return the identity, as on the CPU.
    """
    theta = rotvec.norm(dim=-1, keepdim=True)
    k = _safe_div(rotvec, theta)

    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = torch.zeros_like(kx)
    K = torch.stack(
        [
            torch.stack([zero, -kz, ky], dim=-1),
            torch.stack([kz, zero, -kx], dim=-1),
            torch.stack([-ky, kx, zero], dim=-1),
        ],
        dim=-2,
    )

    t = theta.unsqueeze(-1)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand_as(K)
    rotation = eye + torch.sin(t) * K + (1.0 - torch.cos(t)) * (K @ K)

    degenerate = (theta.squeeze(-1) < EPS_IDENTITY).view(-1, 1, 1)
    return torch.where(degenerate, eye, rotation)


def matrix_to_rotvec(matrix: "torch.Tensor") -> "torch.Tensor":
    """
    Axis-angle vectors for a batch of rotation matrices, (B, 3, 3) -> (B, 3).

    Mirrors the CPU function's three cases: near-identity returns zero, near-pi
    recovers the axis from R + I because the skew part vanishes there, and the
    ordinary case reads the axis off the skew part.
    """
    trace = matrix.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    theta = torch.arccos(cos_theta)

    skew = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )
    sin_theta = torch.sin(theta).unsqueeze(-1)
    ordinary = skew * _safe_div(theta.unsqueeze(-1), 2.0 * sin_theta.abs())

    # Near pi: the columns of (R + I)/2 are all parallel to the rotation axis, so
    # take the best-conditioned one rather than dividing by a vanishing sine.
    eye = torch.eye(3, dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
    rpi = (matrix + eye) / 2.0
    diag = rpi.diagonal(dim1=-2, dim2=-1).clamp_min(0.0)
    axis_scale = torch.sqrt(diag)
    k = axis_scale.argmax(dim=-1)

    picked = torch.gather(
        rpi, 2, k.view(-1, 1, 1).expand(-1, 3, 1)
    ).squeeze(-1)
    scale = torch.gather(axis_scale, 1, k.view(-1, 1))
    axis = _safe_div(picked, scale, EPS_ANGLE)
    axis = _safe_div(axis, axis.norm(dim=-1, keepdim=True), EPS_ANGLE)
    near_pi_vec = axis * theta.unsqueeze(-1)

    out = torch.where((theta > np.pi - EPS_NEAR_PI).unsqueeze(-1), near_pi_vec, ordinary)
    return torch.where(
        (theta < EPS_ANGLE).unsqueeze(-1), torch.zeros_like(out), out
    )


def compose_rotvecs(outer: "torch.Tensor", inner: "torch.Tensor") -> "torch.Tensor":
    """Axis-angle vector for `outer` applied after `inner`, batched."""
    return matrix_to_rotvec(rodrigues_matrix(outer) @ rodrigues_matrix(inner))


def wrap_rotvec(rotvec: "torch.Tensor") -> "torch.Tensor":
    """
    Reduce each axis-angle vector to the equivalent rotation with angle <= pi.

    Keeps the parameterisation well conditioned without changing any pose, for
    the reason given on the CPU function: the exponential-map derivative divides
    by |r|^2.
    """
    theta = rotvec.norm(dim=-1, keepdim=True)
    axis = _safe_div(rotvec, theta)

    wrapped = torch.remainder(theta, 2.0 * np.pi)
    wrapped = torch.where(wrapped > np.pi, wrapped - 2.0 * np.pi, wrapped)

    untouched = (theta <= np.pi) | (theta < EPS_IDENTITY)
    return torch.where(untouched, rotvec, axis * wrapped)


def random_rotvec(
    n: int, generator=None, dtype=None, device=None
) -> "torch.Tensor":
    """
    Sample `n` rotations uniformly over SO(3), as axis-angle vectors.

    Shoemake's method, matching the CPU implementation: a uniform quaternion
    converted to axis-angle. Sampling the axis-angle components independently
    would concentrate density near the identity, which is the failure this
    guards against -- see the CPU test asserting the angle density is
    (1 - cos t) / pi.
    """
    u = torch.rand(n, 3, generator=generator, dtype=dtype, device=device)
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    s1, s2 = torch.sqrt(1.0 - u1), torch.sqrt(u1)
    two_pi = 2.0 * np.pi

    quat = torch.stack(
        [
            s1 * torch.sin(two_pi * u2),
            s1 * torch.cos(two_pi * u2),
            s2 * torch.sin(two_pi * u3),
            s2 * torch.cos(two_pi * u3),
        ],
        dim=-1,
    )
    # A quaternion and its negation are the same rotation; take the w >= 0
    # representative so the angle lands in [0, pi] and the vector stays canonical.
    quat = torch.where((quat[:, 3:4] < 0.0), -quat, quat)

    xyz, w = quat[:, :3], quat[:, 3].clamp(-1.0, 1.0)
    theta = 2.0 * torch.arccos(w)
    sin_half = torch.sqrt((1.0 - w * w).clamp_min(0.0))
    axis = _safe_div(xyz, sin_half.unsqueeze(-1))
    return torch.where(
        (sin_half < EPS_IDENTITY).unsqueeze(-1),
        torch.zeros_like(xyz),
        axis * theta.unsqueeze(-1),
    )
