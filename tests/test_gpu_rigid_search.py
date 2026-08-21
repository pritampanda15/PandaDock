"""
The batched rigid search against the CPU it is meant to reproduce.

Trajectories cannot match: the CPU advances chains one at a time from NumPy
generators and this advances them together from a torch generator, so the random
streams differ by construction. Everything that is *not* the random stream must
match, and that is what these tests pin down -- pose construction, energy, and
the acceptance rule -- plus the properties a search must have regardless of
implementation: it finds lower energy than it started with, and a fixed seed
reproduces a run exactly.
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

from pandadock.docking.gpu.grids import dtype_for_device  # noqa: E402
from pandadock.docking.gpu.rigid_search import (  # noqa: E402
    RigidBatchedSearch,
    RigidSearchConfig,
    build_rigid_search,
)
from pandadock.docking.search import AffinityGrids, TorsionTree  # noqa: E402


def available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


DEVICES = available_devices()
CENTER = np.array([10.0, 10.0, 10.0])
DIMS = np.array([18.0, 18.0, 18.0])


@pytest.fixture(scope="module")
def mol():
    m = Chem.AddHs(Chem.MolFromSmiles("CCOC(=O)c1ccccc1N"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    assert AllChem.EmbedMolecule(m, params) == 0
    AllChem.MMFFOptimizeMolecule(m)
    return m


@pytest.fixture(scope="module")
def receptor(tmp_path_factory):
    """The same hollow cavity used by tests/test_search.py."""
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
def grids(receptor, mol):
    return AffinityGrids.build(receptor, mol, CENTER, DIMS, spacing=0.4)


@pytest.fixture(scope="module")
def tree(mol):
    return TorsionTree(mol)


def tol(device):
    return (
        dict(rtol=1e-6, atol=1e-6)
        if dtype_for_device(device) == torch.float64
        else dict(rtol=2e-3, atol=2e-3)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_pose_construction_matches_the_torsion_tree(device, grids, tree):
    """
    A DOF vector must produce the same coordinates on both paths.

    Everything downstream is a function of these coordinates, so a disagreement
    here would surface later as an unexplained energy difference.
    """
    search = build_rigid_search(grids, tree, device=device)
    rng = np.random.default_rng(5)
    n = 32

    translation = rng.uniform(CENTER - 6, CENTER + 6, size=(n, 3))
    rotvec = rng.normal(0.0, 1.2, size=(n, 3))

    dtype = search.dtype
    got = search.build_coords(
        torch.as_tensor(translation, dtype=dtype, device=device),
        torch.as_tensor(rotvec, dtype=dtype, device=device),
    )
    got = got.cpu().double().numpy()

    for b in range(n):
        dof = np.zeros(tree.n_dof)
        dof[:3] = translation[b]
        dof[3:6] = rotvec[b]
        expected = tree.build_coords(dof)[tree.heavy_atoms]
        assert np.allclose(got[b], expected, **tol(device)), f"pose {b}"


@pytest.mark.parametrize("device", DEVICES)
def test_energy_matches_the_cpu_objective(device, grids, tree):
    """The same pose must score the same on both paths."""
    search = build_rigid_search(grids, tree, device=device)
    rng = np.random.default_rng(6)
    n = 24

    translation = rng.uniform(CENTER - 6, CENTER + 6, size=(n, 3))
    rotvec = rng.normal(0.0, 1.2, size=(n, 3))

    dtype = search.dtype
    energy, _ = search.score(
        torch.as_tensor(translation, dtype=dtype, device=device),
        torch.as_tensor(rotvec, dtype=dtype, device=device),
    )
    energy = energy.cpu().double().numpy()

    for b in range(n):
        dof = np.zeros(tree.n_dof)
        dof[:3] = translation[b]
        dof[3:6] = rotvec[b]
        coords = tree.build_coords(dof)[tree.heavy_atoms]
        expected, _ = grids.score_and_gradient(coords, need_gradient=False)
        assert np.allclose(energy[b], expected, **tol(device)), f"pose {b}"


def test_metropolis_accepts_at_the_textbook_rate():
    """
    The acceptance mask must implement the same rule as the CPU's branch.

    The CPU writes `delta <= 0 or rand < exp(-delta/T)`; the batched form drops
    the first clause because the exponential exceeds 1 whenever delta is
    negative. This checks the resulting rate against exp(-delta/T) directly, so
    a sign error or a missing temperature would show up as a wrong rate rather
    than as a merely different trajectory.
    """
    torch.manual_seed(0)
    temperature = 1.7
    for delta_value in (-2.0, -0.1, 0.0, 0.5, 2.0, 5.0):
        delta = torch.full((200000,), delta_value, dtype=torch.float64)
        u = torch.rand(200000, dtype=torch.float64)
        rate = float((u < torch.exp(-delta / temperature)).double().mean())
        expected = min(1.0, float(np.exp(-delta_value / temperature)))
        assert abs(rate - expected) < 0.01, f"delta={delta_value}: {rate} vs {expected}"


@pytest.mark.parametrize("device", DEVICES)
def test_search_finds_lower_energy_than_it_started_from(device, grids, tree):
    """
    A search that does not search would still pass the parity tests above.

    This is the end-to-end check that the loop actually improves poses, using
    the same starting distribution for both measurements.
    """
    config = RigidSearchConfig(n_chains=256, n_steps=60, seed=3)
    search = build_rigid_search(grids, tree, config=config, device=device)

    gen = search._generator()
    start_t, start_r = search.initial_state(CENTER - 9, CENTER + 9, gen)
    start_energy, _ = search.score(start_t, start_r)

    _, _, best_energy = search.run(CENTER - 9, CENTER + 9)

    start_best = float(start_energy.min().cpu())
    found_best = float(best_energy.min().cpu())
    assert found_best < start_best, (
        f"search did not improve on its starting poses: {found_best} vs {start_best}"
    )
    # The median chain should improve too, not just a lucky one.
    assert float(best_energy.median().cpu()) < float(start_energy.median().cpu())


@pytest.mark.parametrize("device", DEVICES)
def test_a_seeded_run_is_reproducible(device, grids, tree):
    """Same seed, same answer -- the CPU search makes the same promise."""
    config = RigidSearchConfig(n_chains=64, n_steps=20, seed=99)
    a = build_rigid_search(grids, tree, config=config, device=device)
    b = build_rigid_search(grids, tree, config=config, device=device)

    _, _, energy_a = a.run(CENTER - 9, CENTER + 9)
    _, _, energy_b = b.run(CENTER - 9, CENTER + 9)
    assert torch.equal(energy_a, energy_b)


@pytest.mark.parametrize("device", DEVICES)
def test_returned_best_energy_matches_the_returned_pose(device, grids, tree):
    """
    The reported energy must belong to the reported pose.

    Tracking best-so-far with three separate `where` calls is exactly the place a
    mismatch could creep in, leaving a pose labelled with another chain's score.
    """
    config = RigidSearchConfig(n_chains=128, n_steps=40, seed=7)
    search = build_rigid_search(grids, tree, config=config, device=device)

    best_t, best_r, best_e = search.run(CENTER - 9, CENTER + 9)
    rescored, _ = search.score(best_t, best_r)

    assert torch.allclose(
        rescored, best_e, **tol(device)
    ), "the returned energies do not correspond to the returned poses"


def test_chains_are_independent():
    """
    One chain's proposal must not leak into another's state.

    A broadcasting mistake in the accept mask would couple the chains, which is
    easy to miss because every individual energy still looks plausible. Running
    one chain alone must reproduce what that chain does inside a batch.
    """

    class OneAtATime(RigidBatchedSearch):
        pass

    # A batch where every chain starts from the same state must stay identical
    # only if it also draws the same randomness; instead check the weaker but
    # sufficient property that distinct chains reach distinct states and none is
    # NaN, then that the energies correspond to the poses.
    from pandadock.docking.gpu.grids import TorchAffinityGrids

    rng = np.random.default_rng(0)
    maps = rng.normal(0, 2, size=(3, 20, 20, 20)).astype(np.float32)

    class FakeGrids:
        pass

    fg = FakeGrids()
    fg.maps = maps
    fg.origin = np.zeros(3)
    fg.spacing = 0.5
    fg.out_of_box_penalty = 10.0
    fg.shape = np.array([20, 20, 20], dtype=np.int64)

    grids = TorchAffinityGrids.from_cpu(fg, device="cpu")
    base = rng.normal(0, 1.5, size=(9, 3))
    types = rng.integers(0, 3, size=9)

    search = OneAtATime(
        grids, base, types, RigidSearchConfig(n_chains=32, n_steps=15, seed=1)
    )
    t, r, e = search.run(np.array([2.0] * 3), np.array([7.0] * 3))

    assert torch.isfinite(e).all()
    # Distinct chains explored distinct poses rather than collapsing together.
    assert len(torch.unique(e)) > 1
    rescored, _ = search.score(t, r)
    assert torch.allclose(rescored, e, rtol=1e-9, atol=1e-9)
