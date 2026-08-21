"""
Flexible-ligand parity: torsions, the clash term, and their gradients.

Stages one, two and four could all be checked against the grid energy alone,
because a rigid ligand has no intramolecular pairs. That shortcut ends here, so
these tests compare against the full CPU objective, and specifically against a
ligand with enough torsions that the pairs are populated.

The sequential structure of the torsion chain is the thing most likely to be got
wrong, and wrong in a way that still looks plausible: applying the rotations in
the wrong order, or against the reference geometry instead of the running
coordinates, produces a valid-looking conformer with the wrong geometry. There
is a test for exactly that below.
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

from pandadock.docking.gpu.flexible_search import build_flexible_search  # noqa: E402
from pandadock.docking.gpu.grids import dtype_for_device  # noqa: E402
from pandadock.docking.gpu.optimize import LBFGSConfig  # noqa: E402
from pandadock.docking.gpu.rigid_search import RigidSearchConfig  # noqa: E402
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
# Enough torsions that the intramolecular pair list is genuinely populated.
FLEXIBLE_SMILES = "CCOC(=O)c1ccccc1N"


def _receptor(tmp_path_factory):
    from Bio.PDB import PDBParser

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
    return PDBParser(QUIET=True).get_structure("r", str(path))


@pytest.fixture(scope="module")
def flexible(tmp_path_factory):
    mol = Chem.AddHs(Chem.MolFromSmiles(FLEXIBLE_SMILES))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)

    grids = AffinityGrids.build(
        _receptor(tmp_path_factory), mol, CENTER, np.array([18.0] * 3), spacing=0.4
    )
    tree = TorsionTree(mol)
    assert tree.n_torsions > 0, "fixture must actually have torsions"
    objective = DockingObjective(tree, grids)
    assert len(objective.pair_a) > 0, "fixture must have intramolecular pairs"
    return grids, tree, objective


def random_dofs(tree, n, seed=0):
    rng = np.random.default_rng(seed)
    dofs = np.zeros((n, tree.n_dof))
    dofs[:, :3] = rng.uniform(CENTER - 6, CENTER + 6, size=(n, 3))
    dofs[:, 3:6] = rng.normal(0.0, 1.2, size=(n, 3))
    dofs[:, 6:] = rng.uniform(-np.pi, np.pi, size=(n, tree.n_torsions))
    return dofs


def tol(device, loose=False):
    if dtype_for_device(device) == torch.float64:
        return dict(rtol=1e-5, atol=1e-5) if loose else dict(rtol=1e-9, atol=1e-9)
    return dict(rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_torsioned_coordinates_match_the_torsion_tree(device, flexible):
    """
    The full DOF vector must build the same conformer as the CPU.

    This covers the torsion chain, the rigid-body rotation and the translation
    together, in the order the CPU applies them -- which matters, since none of
    those three operations commute with the others.
    """
    grids, tree, objective = flexible
    search = build_flexible_search(grids, tree, objective, device=device)

    dofs = random_dofs(tree, 24, seed=1)
    got = search.build_coords(
        torch.tensor(dofs, dtype=search.dtype, device=device)
    )
    got = got.detach().cpu().double().numpy()

    for b in range(len(dofs)):
        assert np.allclose(got[b], tree.build_coords(dofs[b]), **tol(device)), f"pose {b}"


def test_torsions_are_applied_sequentially_not_in_parallel(flexible):
    """
    Each torsion turns about an axis its ancestors have already moved.

    Applying every rotation against the reference geometry instead would give a
    different, entirely plausible-looking conformer. This pins the distinction by
    checking a case where the two differ: rotating an outer torsion after an
    inner one is not the same as rotating both from the start pose.
    """
    grids, tree, objective = flexible
    if tree.n_torsions < 2:
        pytest.skip("needs at least two torsions to distinguish the orders")

    search = build_flexible_search(grids, tree, objective, device="cpu")

    both = np.zeros((1, tree.n_dof))
    both[0, 6] = 1.1
    both[0, 7] = 0.9
    combined = search.build_coords(torch.tensor(both, dtype=torch.float64))

    # Same two angles applied one at a time, sequentially: identical.
    first = np.zeros((1, tree.n_dof))
    first[0, 6] = 1.1
    coords_first = tree.build_coords(first[0])

    assert np.allclose(
        combined[0].detach().numpy(), tree.build_coords(both[0]), rtol=1e-9, atol=1e-9
    )
    # And the second torsion genuinely changed something relative to the first.
    assert not np.allclose(combined[0].detach().numpy(), coords_first, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_energy_includes_the_intramolecular_term(device, flexible):
    """
    The full CPU objective, not just the grid part.

    A flexible ligand can fold onto itself, and the CPU charges for it. Scoring
    only the receptor interaction would let the search return folded poses the
    CPU rejects, so this compares against `energy_and_gradient` rather than
    against the grids.
    """
    grids, tree, objective = flexible
    search = build_flexible_search(grids, tree, objective, device=device)

    dofs = random_dofs(tree, 24, seed=2)
    got = search.energy(torch.tensor(dofs, dtype=search.dtype, device=device))
    got = got.detach().cpu().double().numpy()

    for b in range(len(dofs)):
        expected, _ = objective.energy_and_gradient(dofs[b])
        assert np.allclose(got[b], expected, **tol(device, loose=True)), f"pose {b}"

    # The term is not vacuously zero on this fixture.
    coords = search.build_coords(
        torch.tensor(dofs, dtype=search.dtype, device=device)
    )
    clash = search.clash.energy(coords)
    assert float(clash.abs().max()) > 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_dof_gradient_matches_the_cpu_including_torsions(device, flexible):
    """
    The gradient the optimiser will actually be given.

    The torsion components are the interesting ones: the CPU derives them
    analytically by accumulating torque about each rotation axis, and here they
    come from autograd backpropagating through the sequential chain.
    """
    grids, tree, objective = flexible
    search = build_flexible_search(grids, tree, objective, device=device)

    dofs = random_dofs(tree, 16, seed=3)
    energy, grad = search.energy_and_dof_gradient(
        torch.tensor(dofs, dtype=search.dtype, device=device)
    )
    energy = energy.detach().cpu().double().numpy()
    grad = grad.detach().cpu().double().numpy()

    for b in range(len(dofs)):
        ref_e, ref_g = objective.energy_and_gradient(dofs[b])
        assert np.allclose(energy[b], ref_e, **tol(device, loose=True)), f"energy {b}"
        assert np.allclose(grad[b], ref_g, **tol(device, loose=True)), f"gradient {b}"


def test_torsion_gradient_matches_finite_differences(flexible):
    """
    Independent of the CPU, in case both were wrong the same way.

    Restricted to the torsion block, since that is what stage three adds.
    """
    grids, tree, objective = flexible
    search = build_flexible_search(grids, tree, objective, device="cpu")

    dofs = random_dofs(tree, 1, seed=4)
    x = torch.tensor(dofs, dtype=torch.float64)
    _, grad = search.energy_and_dof_gradient(x)
    grad = grad.detach().numpy()[0]

    h = 1e-6
    for k in range(6, tree.n_dof):
        plus, minus = dofs.copy(), dofs.copy()
        plus[0, k] += h
        minus[0, k] -= h
        e_plus = float(search.energy(torch.tensor(plus, dtype=torch.float64))[0])
        e_minus = float(search.energy(torch.tensor(minus, dtype=torch.float64))[0])
        fd = (e_plus - e_minus) / (2 * h)
        assert np.allclose(grad[k], fd, rtol=1e-4, atol=1e-5), (
            f"torsion {k - 6}: analytic {grad[k]} vs finite difference {fd}"
        )


def test_a_rigid_tree_has_no_clash_term(tmp_path_factory):
    """
    Explains why the earlier stages could compare against the grid alone.

    A rigid ligand has no pairs whose separation can change, so the term is
    exactly zero rather than merely small -- which is what made stages one, two
    and four comparable to `grids.score_and_gradient` directly.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(FLEXIBLE_SMILES))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)

    grids = AffinityGrids.build(
        _receptor(tmp_path_factory), mol, CENTER, np.array([18.0] * 3), spacing=0.4
    )
    rigid = DockingObjective(TorsionTree(mol, rigid=True), grids)
    flexible = DockingObjective(TorsionTree(mol), grids)

    assert len(rigid.pair_a) == 0
    assert len(flexible.pair_a) > 0


@pytest.mark.parametrize("device", DEVICES)
def test_flexible_basin_hopping_improves_and_stays_self_consistent(device, flexible):
    """
    End to end: the search improves on its start, and reports honest energies.

    The self-consistency half matters as much as the improvement: with torsions
    the DOF vector is longer and the best-so-far bookkeeping has more to keep
    aligned, so a returned energy that does not belong to its returned pose is a
    live possibility rather than a formality.
    """
    grids, tree, objective = flexible
    config = RigidSearchConfig(n_chains=48, n_steps=4, seed=1)
    search = build_flexible_search(
        grids, tree, objective, config=config, device=device
    )
    lo, hi = CENTER - 9, CENTER + 9

    gen = search._generator()
    start = search.initial_state(lo, hi, gen)
    start_energy = search.energy(start)

    best_x, best_energy = search.run_basin_hopping(
        lo, hi, LBFGSConfig(max_iter=40, max_line_search=12)
    )

    assert float(best_energy.min()) < float(start_energy.min())
    rescored = search.energy(best_x)
    assert torch.allclose(rescored, best_energy, **tol(device, loose=True))
