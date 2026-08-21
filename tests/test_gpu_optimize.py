"""
The batched L-BFGS, and the local relaxation built on it.

Two things need proving. First that the optimiser is a competent L-BFGS at all,
which is checked on functions with known answers and against SciPy's
implementation -- the same one the CPU search uses. Second that the gradient it
is given is the right one, which is checked against the CPU objective's
hand-derived DOF gradient.

Results are not expected to be bit-identical to SciPy. The line search differs
(backtracking Armijo rather than strong Wolfe), so the iterates differ; what
must hold is that both reach minima of the same quality.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

pytest.importorskip("torch", reason="the GPU search path needs the [gnn] extra")
pytest.importorskip("rdkit", reason="RDKit is required to build grids")
pytest.importorskip("Bio", reason="BioPython is required to parse receptors")

import torch  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from pandadock.docking.gpu.grids import dtype_for_device  # noqa: E402
from pandadock.docking.gpu.optimize import LBFGSConfig, batched_lbfgs  # noqa: E402
from pandadock.docking.gpu.rigid_search import (  # noqa: E402
    RigidSearchConfig,
    build_rigid_search,
)
from pandadock.docking.search import (  # noqa: E402
    AffinityGrids,
    DockingObjective,
    TorsionTree,
)


def available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


DEVICES = available_devices()
CENTER = np.array([10.0, 10.0, 10.0])


# --------------------------------------------------------------- the optimiser


def test_quadratic_reaches_the_exact_minimum():
    """With a known closed-form answer there is no ambiguity about correctness."""
    rng = np.random.default_rng(0)
    a = torch.tensor(rng.normal(0, 1, (5, 5)), dtype=torch.float64)
    a = a @ a.T + 5 * torch.eye(5, dtype=torch.float64)
    b = torch.tensor(rng.normal(0, 1, 5), dtype=torch.float64)

    def quadratic(x):
        return 0.5 * ((x @ a) * x).sum(-1) - x @ b, x @ a - b

    x, _, converged = batched_lbfgs(
        torch.zeros(32, 5, dtype=torch.float64), quadratic, LBFGSConfig(max_iter=100)
    )
    expected = torch.linalg.solve(a, b)
    assert torch.allclose(x, expected.expand_as(x), atol=1e-6)
    assert bool(converged.all())


def test_rosenbrock_matches_scipy_lbfgs():
    """
    Compared against the exact optimiser the CPU search uses.

    Rosenbrock's curved valley is the standard case where a weak line search or a
    mishandled curvature history stalls, so agreeing with SciPy here is a
    meaningful check rather than a formality.
    """
    def rosen_batch(x):
        x0, x1 = x[:, 0], x[:, 1]
        f = (1.0 - x0) ** 2 + 100.0 * (x1 - x0**2) ** 2
        g = torch.stack(
            [-2 * (1 - x0) - 400 * x0 * (x1 - x0**2), 200 * (x1 - x0**2)], dim=-1
        )
        return f, g

    def rosen_scipy(x):
        return (
            (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2,
            np.array(
                [-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)]
            ),
        )

    starts = np.random.default_rng(1).uniform(-2, 2, size=(64, 2))
    _, energy, _ = batched_lbfgs(
        torch.tensor(starts, dtype=torch.float64), rosen_batch, LBFGSConfig(max_iter=200)
    )
    scipy_best = np.array(
        [
            minimize(
                rosen_scipy, s, jac=True, method="L-BFGS-B",
                options={"maxiter": 200, "maxcor": 10},
            ).fun
            for s in starts
        ]
    )
    mine = energy.numpy()
    assert mine.max() < 1e-8, "did not reach the minimum from every start"
    # Never materially worse than SciPy on any start.
    assert int((mine > scipy_best + 1e-6).sum()) == 0


def test_a_chain_that_cannot_improve_does_not_corrupt_its_neighbours():
    """
    Chains are independent, including when one of them starts at a minimum.

    A chain already at the optimum accepts no step, and the history update must
    be skipped for it rather than storing a zero curvature pair that would make
    rho infinite and propagate NaN into the shared two-loop recursion.
    """
    def quadratic(x):
        return (x**2).sum(-1), 2 * x

    x0 = torch.zeros(4, 3, dtype=torch.float64)
    x0[1] = torch.tensor([1.0, -2.0, 0.5])  # only this chain has work to do
    x, energy, _ = batched_lbfgs(x0, quadratic, LBFGSConfig(max_iter=50))

    assert torch.isfinite(x).all() and torch.isfinite(energy).all()
    assert torch.allclose(x, torch.zeros_like(x), atol=1e-8)


# ------------------------------------------------------- the docking gradient


@pytest.fixture(scope="module")
def docking(tmp_path_factory):
    from Bio.PDB import PDBParser

    mol = Chem.AddHs(Chem.MolFromSmiles("CCOC(=O)c1ccccc1N"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)

    rng = np.random.default_rng(7)
    lines = []
    for i in range(400):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        pos = CENTER + v * rng.uniform(7.5, 12.0)
        name, elem, res = ("CB", "C", "PHE") if i % 2 else ("O", "O", "SER")
        lines.append(
            f"ATOM  {i + 1:5d}  {name:<3s} {res} A{i % 300 + 1:4d}    "
            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00 20.00          {elem:>2s}"
        )
    lines.append("END")
    path = tmp_path_factory.mktemp("rec") / "receptor.pdb"
    path.write_text("\n".join(lines) + "\n")
    receptor = PDBParser(QUIET=True).get_structure("r", str(path))

    grids = AffinityGrids.build(
        receptor, mol, CENTER, np.array([18.0] * 3), spacing=0.4
    )
    # Rigid: the intramolecular term has no pairs, so the CPU objective reduces
    # to the grid energy and the two paths are directly comparable.
    tree = TorsionTree(mol, rigid=True)
    return grids, tree, DockingObjective(tree, grids)


@pytest.mark.parametrize("device", DEVICES)
def test_dof_gradient_matches_the_cpu_objective(device, docking):
    """
    The hybrid chain rule against the CPU's hand-derived one.

    The CPU maps the Cartesian gradient through the derivative of the SO(3)
    exponential map analytically. Here the Cartesian part is the analytic grid
    gradient and the pose part comes from autograd, so agreement confirms that
    autograd reproduces that derivative and it does not need hand-coding.
    """
    grids, tree, objective = docking
    search = build_rigid_search(grids, tree, device=device)

    rng = np.random.default_rng(0)
    n = 16
    translation = rng.uniform(CENTER - 6, CENTER + 6, size=(n, 3))
    rotvec = rng.normal(0.0, 1.2, size=(n, 3))
    x = torch.tensor(
        np.concatenate([translation, rotvec], axis=1),
        dtype=search.dtype,
        device=device,
    )

    energy, grad = search.energy_and_dof_gradient(x)
    energy = energy.cpu().double().numpy()
    grad = grad.cpu().double().numpy()

    tol = (
        dict(rtol=1e-5, atol=1e-5)
        if dtype_for_device(device) == torch.float64
        else dict(rtol=5e-3, atol=5e-3)
    )
    for b in range(n):
        dof = np.zeros(6)
        dof[:3] = translation[b]
        dof[3:6] = rotvec[b]
        ref_e, ref_g = objective.energy_and_gradient(dof)
        assert np.allclose(energy[b], ref_e, **tol), f"energy, pose {b}"
        assert np.allclose(grad[b], ref_g, **tol), f"gradient, pose {b}"


@pytest.mark.parametrize("device", DEVICES)
def test_local_optimize_is_as_good_as_scipy_from_the_same_start(device, docking):
    """
    The relaxation itself, against the SciPy call the CPU makes.

    Compared per pose from identical starting points, so this measures the
    optimiser rather than the sampling around it.
    """
    grids, tree, objective = docking
    search = build_rigid_search(grids, tree, device=device)

    rng = np.random.default_rng(3)
    n = 24
    translation = rng.uniform(CENTER - 5, CENTER + 5, size=(n, 3))
    rotvec = rng.normal(0.0, 1.0, size=(n, 3))

    _, _, energy = search.local_optimize(
        torch.tensor(translation, dtype=search.dtype, device=device),
        torch.tensor(rotvec, dtype=search.dtype, device=device),
    )
    energy = energy.cpu().double().numpy()

    scipy_energy = []
    for b in range(n):
        dof = np.zeros(6)
        dof[:3] = translation[b]
        dof[3:6] = rotvec[b]
        result = minimize(
            objective.scipy_objective, dof, jac=True, method="L-BFGS-B",
            options={"maxiter": 60, "maxcor": 10},
        )
        scipy_energy.append(float(result.fun))
    scipy_energy = np.array(scipy_energy)

    # Both must actually descend, and neither may be systematically worse.
    assert (energy < 0).any()
    median_gap = float(np.median(energy - scipy_energy))
    assert abs(median_gap) < 0.5, f"median gap to scipy {median_gap:.3f} kcal/mol"


@pytest.mark.parametrize("device", DEVICES)
def test_relaxed_energy_belongs_to_the_relaxed_pose(device, docking):
    """The returned energy must be the energy of the returned pose."""
    grids, tree, _ = docking
    search = build_rigid_search(grids, tree, device=device)

    rng = np.random.default_rng(4)
    translation = torch.tensor(
        rng.uniform(CENTER - 5, CENTER + 5, size=(16, 3)),
        dtype=search.dtype, device=device,
    )
    rotvec = torch.tensor(
        rng.normal(0.0, 1.0, size=(16, 3)), dtype=search.dtype, device=device
    )

    t, r, energy = search.local_optimize(translation, rotvec)
    rescored, _ = search.score(t, r)
    tol = (
        dict(rtol=1e-6, atol=1e-6)
        if dtype_for_device(device) == torch.float64
        else dict(rtol=2e-3, atol=2e-3)
    )
    assert torch.allclose(rescored, energy, **tol)
    # And wrap_rotvec has kept the parameterisation well conditioned.
    assert bool((r.norm(dim=-1) <= np.pi + 1e-6).all())


def test_basin_hopping_beats_plain_monte_carlo(docking):
    """
    The result that motivated doing this stage before torsions.

    Stage two found that raw Monte Carlo could not reach what the CPU reaches,
    however many samples it drew, because its moves did not land in minima. With
    the same chain and step budget, walking between relaxed minima must do
    better than walking over the raw surface.
    """
    grids, tree, _ = docking
    config = RigidSearchConfig(n_chains=64, n_steps=6, seed=1)
    search = build_rigid_search(grids, tree, config=config, device="cpu")
    lo, hi = CENTER - 9, CENTER + 9

    _, _, plain = search.run(lo, hi)
    _, _, hopped = search.run_basin_hopping(
        lo, hi, LBFGSConfig(max_iter=40, max_line_search=12)
    )

    assert float(hopped.min()) < float(plain.min()), (
        f"basin hopping {float(hopped.min()):.3f} did not beat plain MC "
        f"{float(plain.min()):.3f}"
    )
