"""
Parity between the batched SO(3) primitives and the scalar CPU versions.

The CPU functions branch on magnitude. Those branches are the whole difficulty
of batching them, so the tests deliberately mix ordinary rotations with the
degenerate cases -- exact zero, angles at and just below pi, many-turn vectors --
in a single batch, which is the situation a per-element `if` cannot express and
where a careless `torch.where` produces NaN.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

pytest.importorskip("torch", reason="the GPU search path needs the [gnn] extra")

import torch  # noqa: E402

from pandadock.docking.gpu import rotations as g  # noqa: E402
from pandadock.docking.gpu.grids import dtype_for_device  # noqa: E402
from pandadock.docking.search import rotations as c  # noqa: E402


def available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


DEVICES = available_devices()


def tol(device):
    return (
        dict(rtol=1e-9, atol=1e-9)
        if dtype_for_device(device) == torch.float64
        else dict(rtol=1e-4, atol=1e-4)
    )


def awkward_rotvecs():
    """
    Inputs chosen to hit every branch of the CPU implementations.

    A batch of only well-conditioned rotations would pass even with the safe
    divisions removed, so the degenerate rows are the point of this fixture.
    """
    rng = np.random.default_rng(0)
    rows = [
        [0.0, 0.0, 0.0],  # exact identity: rotvec / theta is 0/0
        [1e-14, 0.0, 0.0],  # below EPS_IDENTITY
        [np.pi, 0.0, 0.0],  # exactly pi, where the skew part vanishes
        [np.pi - 1e-9, 0.0, 0.0],  # just inside the near-pi branch
        [0.0, np.pi - 1e-7, 0.0],  # near-pi on another axis
        [7.5, -3.2, 1.1],  # many turns, needs wrapping
        [0.0, 0.0, 2.0 * np.pi],  # a full turn: identity by another route
    ]
    rows += list(rng.normal(0.0, 1.5, size=(24, 3)))
    return np.asarray(rows, dtype=np.float64)


@pytest.mark.parametrize("device", DEVICES)
def test_rodrigues_matches_cpu_including_degenerate_rows(device):
    vecs = awkward_rotvecs()
    dtype = dtype_for_device(device)
    batched = g.rodrigues_matrix(torch.as_tensor(vecs, dtype=dtype, device=device))

    assert torch.isfinite(batched).all(), "a degenerate row produced NaN or inf"
    got = batched.cpu().double().numpy()
    for i, v in enumerate(vecs):
        assert np.allclose(got[i], c.rodrigues_matrix(v), **tol(device)), f"row {i}: {v}"


@pytest.mark.parametrize("device", DEVICES)
def test_matrix_to_rotvec_matches_cpu(device):
    vecs = awkward_rotvecs()
    dtype = dtype_for_device(device)
    mats = torch.as_tensor(
        np.stack([c.rodrigues_matrix(v) for v in vecs]), dtype=dtype, device=device
    )
    got = g.matrix_to_rotvec(mats)
    assert torch.isfinite(got).all()
    got = got.cpu().double().numpy()

    for i, v in enumerate(vecs):
        expected = c.matrix_to_rotvec(c.rodrigues_matrix(v))
        # Compare the rotations, not the parameters: at pi, r and -r are the same
        # rotation and either is a correct answer.
        assert np.allclose(
            c.rodrigues_matrix(got[i]), c.rodrigues_matrix(expected), **tol(device)
        ), f"row {i}: {v}"


@pytest.mark.parametrize("device", DEVICES)
def test_compose_matches_cpu(device):
    rng = np.random.default_rng(3)
    a = np.vstack([awkward_rotvecs()[:7], rng.normal(0, 1.5, (12, 3))])
    b = np.vstack([awkward_rotvecs()[:7][::-1], rng.normal(0, 1.5, (12, 3))])
    dtype = dtype_for_device(device)

    got = g.compose_rotvecs(
        torch.as_tensor(a, dtype=dtype, device=device),
        torch.as_tensor(b, dtype=dtype, device=device),
    )
    assert torch.isfinite(got).all()
    got = got.cpu().double().numpy()

    for i in range(len(a)):
        expected = c.compose_rotvecs(a[i], b[i])
        assert np.allclose(
            c.rodrigues_matrix(got[i]), c.rodrigues_matrix(expected), **tol(device)
        ), f"row {i}"


@pytest.mark.parametrize("device", DEVICES)
def test_wrap_matches_cpu_and_preserves_the_rotation(device):
    vecs = awkward_rotvecs()
    dtype = dtype_for_device(device)
    got = g.wrap_rotvec(torch.as_tensor(vecs, dtype=dtype, device=device))
    assert torch.isfinite(got).all()
    got = got.cpu().double().numpy()

    for i, v in enumerate(vecs):
        expected = c.wrap_rotvec(v)
        assert np.allclose(got[i], expected, **tol(device)), f"row {i}: {v}"
        # Wrapping must not change the pose, only the parameterisation.
        assert np.allclose(
            c.rodrigues_matrix(got[i]), c.rodrigues_matrix(v), **tol(device)
        )
        assert np.linalg.norm(got[i]) <= np.pi + 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_random_rotvec_is_uniform_over_so3(device):
    """
    The same distribution check the CPU sampler carries.

    Under the Haar measure the rotation angle has density (1 - cos t) / pi on
    [0, pi], with mean pi/2 + 2/pi. Checking the density rather than the spread
    is what catches a sampler that reaches large angles but still clusters near
    the identity -- the failure mode of drawing axis-angle components
    independently.
    """
    gen = torch.Generator(device="cpu").manual_seed(11)
    vecs = g.random_rotvec(40000, generator=gen, dtype=torch.float64)
    angles = vecs.norm(dim=-1).numpy()

    assert angles.min() >= 0.0
    assert angles.max() <= np.pi + 1e-9
    assert np.isclose(angles.mean(), np.pi / 2 + 2 / np.pi, atol=0.03)

    observed, edges = np.histogram(angles, bins=10, range=(0.0, np.pi), density=True)
    midpoints = (edges[:-1] + edges[1:]) / 2
    expected = (1.0 - np.cos(midpoints)) / np.pi
    assert np.max(np.abs(observed - expected)) < 0.02


def test_degenerate_rows_are_exactly_identity():
    """A zero rotation must come back as the identity, not merely as finite."""
    vecs = torch.tensor(
        [[0.0, 0.0, 0.0], [0.3, -0.7, 1.1], [0.0, 0.0, 0.0]], dtype=torch.float64
    )
    mats = g.rodrigues_matrix(vecs)
    eye = torch.eye(3, dtype=torch.float64)
    assert torch.allclose(mats[0], eye)
    assert torch.allclose(mats[2], eye)
    assert not torch.allclose(mats[1], eye)
    assert torch.isfinite(g.matrix_to_rotvec(mats)).all()


def test_guarded_division_keeps_gradients_finite():
    """
    Why the divisions are clamped rather than merely selected with `where`.

    In the forward pass `torch.where` is enough: it picks the identity branch and
    the 0/0 never shows. Autograd is where it fails -- `where` sends a zero
    gradient down the unselected branch, 0 * NaN is NaN, and a single identity
    rotation then returns NaN gradients that contaminate the whole batch once
    they are summed.

    The clamped form yields an enormous gradient instead, which is correct: at an
    exact identity the axis-angle derivative genuinely is singular. Enormous is
    something an optimiser can clip; NaN is not. This becomes load-bearing when
    the batched local optimiser differentiates through these functions.
    """
    vecs = torch.tensor(
        [[0.0, 0.0, 0.0], [0.3, -0.7, 1.1]], dtype=torch.float64, requires_grad=True
    )

    for fn in (g.rodrigues_matrix, g.wrap_rotvec):
        if vecs.grad is not None:
            vecs.grad = None
        fn(vecs).sum().backward()
        assert torch.isfinite(vecs.grad).all(), (
            f"{fn.__name__} produced NaN gradients on the identity row"
        )

    # And confirm the naive alternative really does fail, so this test is
    # protecting against a mistake someone would plausibly make rather than a
    # hypothetical one.
    naive_input = vecs.detach().clone().requires_grad_(True)
    theta = naive_input.norm(dim=-1, keepdim=True)
    naive = torch.where(theta < 1e-12, torch.zeros_like(naive_input), naive_input / theta)
    naive.sum().backward()
    assert torch.isnan(naive_input.grad).any(), (
        "the unguarded form no longer produces NaN; the guard may be unnecessary"
    )
