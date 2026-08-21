"""
Numerical parity between the CPU and batched GPU grid scoring.

The CPU path is the one validated on 814 complexes in the manuscript. The GPU
path has no independent claim to correctness: it is worth having only if it
agrees, so these tests compare it against the CPU implementation rather than
against hand-written expected values.

Every test runs on whatever device is available. On CPU and CUDA the comparison
is in float64 and the tolerance is tight; MPS has no float64, so there the same
tests run in float32 against a float32 reference and the tolerance reflects it.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

pytest.importorskip("torch", reason="the GPU search path needs the [gnn] extra")
pytest.importorskip("rdkit", reason="RDKit is required to build grids")

import torch  # noqa: E402

from pandadock.docking.gpu import TorchAffinityGrids  # noqa: E402
from pandadock.docking.gpu.grids import dtype_for_device  # noqa: E402


# Devices worth exercising on this machine. CPU is always included so the parity
# maths is covered even where no accelerator exists.
def available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


DEVICES = available_devices()


class FakeGrids:
    """
    A stand-in for AffinityGrids carrying the same fields the scorer reads.

    Building real receptor grids is slow and drags in a receptor fixture; the
    interpolation maths under test does not care where the numbers came from, so
    random maps exercise it just as well and let the tests stay fast.
    """

    def __init__(self, seed=0, n_types=4, shape=(12, 13, 14), spacing=0.375):
        rng = np.random.default_rng(seed)
        # float32 because that is what AffinityGrids.build actually produces.
        # Using float64 here would make these tests pass at a tolerance the real
        # grids cannot meet -- see test_x_gradient_differs_only_by_float32_cancellation.
        self.maps = rng.normal(0.0, 2.0, size=(n_types,) + shape).astype(np.float32)
        self.origin = np.array([-2.0, 1.0, 0.5])
        self.spacing = spacing
        self.out_of_box_penalty = 10.0
        self.shape = np.array(shape, dtype=np.int64)


def cpu_reference(grids, coords, type_ids, need_gradient=True):
    """
    The production CPU maths, lifted verbatim from AffinityGrids.

    Duplicated rather than imported because the real method is a method on a
    class that owns receptor state; keeping the arithmetic here makes any
    divergence between the two implementations visible in this file.
    """
    frac = (coords - grids.origin) / grids.spacing
    hi = grids.shape - 1
    clamped = np.clip(frac, 0.0, hi.astype(np.float64) - 1e-6)
    overflow = frac - clamped

    i0 = np.floor(clamped).astype(np.int64)
    i0 = np.minimum(i0, hi - 1)
    t = clamped - i0
    u, v, w = t[:, 0], t[:, 1], t[:, 2]

    x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    m = grids.maps
    c000 = m[type_ids, x0, y0, z0]
    c100 = m[type_ids, x1, y0, z0]
    c010 = m[type_ids, x0, y1, z0]
    c110 = m[type_ids, x1, y1, z0]
    c001 = m[type_ids, x0, y0, z1]
    c101 = m[type_ids, x1, y0, z1]
    c011 = m[type_ids, x0, y1, z1]
    c111 = m[type_ids, x1, y1, z1]

    one_u, one_v, one_w = 1.0 - u, 1.0 - v, 1.0 - w
    c00 = c000 * one_u + c100 * u
    c10 = c010 * one_u + c110 * u
    c01 = c001 * one_u + c101 * u
    c11 = c011 * one_u + c111 * u
    c0 = c00 * one_v + c10 * v
    c1 = c01 * one_v + c11 * v
    values = c0 * one_w + c1 * w

    penalty = grids.out_of_box_penalty * np.sum(overflow**2)
    energy = float(np.sum(values)) + float(penalty)
    if not need_gradient:
        return energy, None

    d_du = ((c100 - c000) * one_v + (c110 - c010) * v) * one_w + (
        (c101 - c001) * one_v + (c111 - c011) * v
    ) * w
    d_dv = (c10 - c00) * one_w + (c11 - c01) * w
    d_dw = c1 - c0
    grad = np.stack([d_du, d_dv, d_dw], axis=1) / grids.spacing
    grad += 2.0 * grids.out_of_box_penalty * overflow / grids.spacing
    return energy, grad


def tolerances(device):
    """
    Bounds for CPU-vs-GPU agreement.

    On a float64 device the energy agrees to round-off, but the x-gradient does
    not agree to 1e-15, and cannot: the CPU subtracts two raw float32 map values
    while this implementation subtracts them after upcasting. See
    test_x_gradient_differs_only_by_float32_cancellation, which pins down that
    this is the only source of disagreement. MPS runs the whole computation in
    float32 and is looser again.
    """
    if dtype_for_device(device) == torch.float64:
        return dict(rtol=1e-6, atol=1e-6)
    return dict(rtol=2e-4, atol=2e-4)


def sample_coords(grids, n_atoms, n_poses, rng, spread=1.0):
    """Poses spread over the box; `spread` > 1 pushes atoms outside it."""
    extent = (grids.shape - 1) * grids.spacing
    lo = grids.origin - extent * (spread - 1.0) / 2.0
    hi = grids.origin + extent * (1.0 + (spread - 1.0) / 2.0)
    return rng.uniform(lo, hi, size=(n_poses, n_atoms, 3))


@pytest.mark.parametrize("device", DEVICES)
def test_energy_and_gradient_match_cpu(device):
    """The core claim: same inputs, same numbers, for every pose in a batch."""
    grids = FakeGrids()
    rng = np.random.default_rng(1)
    n_atoms, n_poses = 17, 8
    coords = sample_coords(grids, n_atoms, n_poses, rng)
    type_ids = rng.integers(0, grids.maps.shape[0], size=n_atoms)

    gpu = TorchAffinityGrids.from_cpu(grids, device=device)
    energy, grad = gpu.score_and_gradient(
        torch.as_tensor(coords), torch.as_tensor(type_ids)
    )

    tol = tolerances(device)
    for b in range(n_poses):
        ref_e, ref_g = cpu_reference(grids, coords[b], type_ids)
        assert np.allclose(energy[b].cpu().double().numpy(), ref_e, **tol)
        assert np.allclose(grad[b].cpu().double().numpy(), ref_g, **tol)


@pytest.mark.parametrize("device", DEVICES)
def test_out_of_box_atoms_are_penalised_identically(device):
    """
    The boundary penalty is what keeps the optimiser in the pocket.

    Dropping it would still produce plausible-looking energies while letting
    poses drift out of the box, so it is tested separately with coordinates that
    deliberately fall outside the grid.
    """
    grids = FakeGrids(seed=3)
    rng = np.random.default_rng(2)
    coords = sample_coords(grids, 11, 4, rng, spread=3.0)
    type_ids = rng.integers(0, grids.maps.shape[0], size=11)

    # Precondition: this really does place atoms outside the grid, otherwise the
    # test would pass while exercising nothing.
    frac = (coords - grids.origin) / grids.spacing
    assert (frac < 0).any() or (frac > (grids.shape - 1)).any()

    gpu = TorchAffinityGrids.from_cpu(grids, device=device)
    energy, grad = gpu.score_and_gradient(
        torch.as_tensor(coords), torch.as_tensor(type_ids)
    )

    tol = tolerances(device)
    for b in range(coords.shape[0]):
        ref_e, ref_g = cpu_reference(grids, coords[b], type_ids)
        assert np.allclose(energy[b].cpu().double().numpy(), ref_e, **tol)
        assert np.allclose(grad[b].cpu().double().numpy(), ref_g, **tol)


@pytest.mark.parametrize("device", DEVICES)
def test_analytic_gradient_matches_finite_differences(device):
    """
    The gradient is checked against the energy it claims to differentiate.

    Parity with the CPU would be satisfied by two implementations that are wrong
    in the same way; this test is independent of both. Points are kept inside the
    grid so the finite difference does not straddle the penalty's clamp.
    """
    grids = FakeGrids(seed=5)
    rng = np.random.default_rng(4)
    n_atoms = 6
    extent = (grids.shape - 1) * grids.spacing
    coords = grids.origin + rng.uniform(0.3, 0.7, size=(1, n_atoms, 3)) * extent
    type_ids = rng.integers(0, grids.maps.shape[0], size=n_atoms)

    gpu = TorchAffinityGrids.from_cpu(grids, device=device)
    tid = torch.as_tensor(type_ids)
    _, grad = gpu.score_and_gradient(torch.as_tensor(coords), tid)
    grad = grad[0].cpu().double().numpy()

    # A step small enough to be local but large enough to survive float32 on MPS.
    h = 1e-5 if dtype_for_device(device) == torch.float64 else 1e-3
    tol = dict(rtol=1e-5, atol=1e-6) if h == 1e-5 else dict(rtol=2e-2, atol=2e-2)

    for atom in range(n_atoms):
        for axis in range(3):
            plus, minus = coords.copy(), coords.copy()
            plus[0, atom, axis] += h
            minus[0, atom, axis] -= h
            e_plus, _ = gpu.score_and_gradient(
                torch.as_tensor(plus), tid, need_gradient=False
            )
            e_minus, _ = gpu.score_and_gradient(
                torch.as_tensor(minus), tid, need_gradient=False
            )
            fd = (float(e_plus[0]) - float(e_minus[0])) / (2 * h)
            assert np.allclose(grad[atom, axis], fd, **tol), (
                f"atom {atom} axis {axis}: analytic {grad[atom, axis]} vs fd {fd}"
            )


@pytest.mark.parametrize("device", DEVICES)
def test_batching_does_not_change_any_single_pose(device):
    """
    Scoring a pose alone and in a batch must give the same answer.

    A broadcasting mistake in the corner gather would show up here as poses
    contaminating each other, which is otherwise easy to miss because every
    individual energy still looks reasonable.
    """
    grids = FakeGrids(seed=7)
    rng = np.random.default_rng(6)
    coords = sample_coords(grids, 9, 5, rng)
    type_ids = torch.as_tensor(rng.integers(0, grids.maps.shape[0], size=9))

    gpu = TorchAffinityGrids.from_cpu(grids, device=device)
    batched, batched_grad = gpu.score_and_gradient(torch.as_tensor(coords), type_ids)

    tol = tolerances(device)
    for b in range(coords.shape[0]):
        single, single_grad = gpu.score_and_gradient(
            torch.as_tensor(coords[b : b + 1]), type_ids
        )
        assert np.allclose(
            batched[b].cpu().double().numpy(), single[0].cpu().double().numpy(), **tol
        )
        assert np.allclose(
            batched_grad[b].cpu().double().numpy(),
            single_grad[0].cpu().double().numpy(),
            **tol,
        )


def test_x_gradient_differs_only_by_float32_cancellation():
    """
    Explains, and bounds, the one place the two implementations disagree.

    The production maps are float32. The CPU x-derivative starts from
    `c100 - c000`, a subtraction of two raw float32 map values, so it loses
    precision to cancellation before anything is promoted to float64. The y and
    z derivatives subtract intermediates that numpy has already promoted, so
    they are unaffected -- which is why the disagreement appears only on axis 0.

    This implementation uploads the maps as float64 and therefore subtracts in
    float64, making it the *more* accurate of the two. That is a deliberate
    choice, not an oversight: do not "fix" it by downcasting the maps to restore
    bit-identity, and do not tighten the tolerance in `tolerances()` expecting
    the x-gradient to follow.

    The energy is unaffected, since it never takes a difference of corners.
    """
    grids = FakeGrids(seed=11)
    rng = np.random.default_rng(12)
    coords = sample_coords(grids, 24, 6, rng)
    type_ids = rng.integers(0, grids.maps.shape[0], size=24)

    gpu = TorchAffinityGrids.from_cpu(grids, device="cpu")
    energy, grad = gpu.score_and_gradient(
        torch.as_tensor(coords), torch.as_tensor(type_ids)
    )
    grad = grad.cpu().double().numpy()

    ref = np.stack([cpu_reference(grids, coords[b], type_ids)[1] for b in range(6)])
    ref_e = np.array([cpu_reference(grids, coords[b], type_ids)[0] for b in range(6)])

    # The energy takes no difference of corners, so it agrees to round-off.
    assert np.allclose(energy.cpu().double().numpy(), ref_e, rtol=1e-12, atol=1e-12)

    err = np.abs(grad - ref)
    # y and z are exact; only x carries the cancellation.
    assert err[..., 1].max() < 1e-12, "y-gradient should be bit-identical"
    assert err[..., 2].max() < 1e-12, "z-gradient should be bit-identical"

    # And the x disagreement stays at the float32 epsilon scale rather than
    # indicating an actual indexing or interpolation bug.
    scale = max(np.abs(ref[..., 0]).max(), 1e-12)
    assert err[..., 0].max() / scale < 1e-6


def test_mps_refuses_float64_rather_than_silently_downcasting():
    """
    MPS has no float64. The dtype rule must be explicit, not discovered at
    runtime by a caller wondering why its tolerances stopped holding.
    """
    assert dtype_for_device("cpu") == torch.float64
    assert dtype_for_device("mps") == torch.float32
