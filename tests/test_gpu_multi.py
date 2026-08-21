"""
Several ligands in one batch, against the same ligands run alone.

Padding and masking are the whole mechanism here, and their failure mode is
quiet: a padded atom that is not masked still scores, a surplus torsion slot
that is not masked still rotates something, and a type id that is not remapped
still indexes a valid map. None of those raise, and every energy they produce
looks reasonable. So the tests compare a packed batch against the single-ligand
path that earlier stages already validated, ligand by ligand.

The fixture deliberately mixes sizes -- 3 to 12 heavy atoms, 0 to 3 torsions,
1 to 4 atom types -- because a batch of identically shaped ligands would pass
with every mask removed.
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
from pandadock.docking.gpu.multi import (  # noqa: E402
    build_multi_ligand_search,
    union_grids_and_type_ids,
)
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
DIMS = np.array([18.0, 18.0, 18.0])
# Mixed on every axis that gets padded.
SMILES = ["CCO", "CCOC(=O)c1ccccc1N", "CC(=O)Nc1ccc(O)cc1", "c1ccccc1"]


@pytest.fixture(scope="module")
def library(tmp_path_factory):
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
    receptor = PDBParser(QUIET=True).get_structure("r", str(path))

    grids, trees, objectives = [], [], []
    for smiles in SMILES:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)
        AllChem.MMFFOptimizeMolecule(mol)
        g = AffinityGrids.build(receptor, mol, CENTER, DIMS, spacing=0.4)
        t = TorsionTree(mol)
        grids.append(g)
        trees.append(t)
        objectives.append(DockingObjective(t, g))

    # The fixture must actually be heterogeneous, or these tests prove nothing.
    assert len({t.n_torsions for t in trees}) > 1
    assert len({t.base_coords.shape[0] for t in trees}) > 1
    return grids, trees, objectives


def random_batch(search, trees, n_chains, seed=0):
    rng = np.random.default_rng(seed)
    n_lig = len(trees)
    x = np.zeros((n_lig, n_chains, search.packed.n_dof))
    x[:, :, :3] = rng.uniform(CENTER - 5, CENTER + 5, size=(n_lig, n_chains, 3))
    x[:, :, 3:6] = rng.normal(0.0, 1.0, size=(n_lig, n_chains, 3))
    if search.packed.max_torsions:
        x[:, :, 6:] = rng.uniform(
            -np.pi, np.pi, size=(n_lig, n_chains, search.packed.max_torsions)
        )
    return torch.tensor(x, dtype=search.dtype, device=search.device)


def tol(device):
    if dtype_for_device(device) == torch.float64:
        return dict(rtol=1e-5, atol=1e-5)
    return dict(rtol=5e-3, atol=5e-3)


def test_type_ids_are_remapped_onto_a_union_stack(library):
    """
    The correctness issue that padding alone does not solve.

    `LigandTyping` numbers types by first appearance within one ligand, so the
    same index means different things for different ligands. Sharing one grid
    stack without remapping would score most of a batch against the wrong maps,
    silently: every index is in range and every energy looks plausible.
    """
    grids, _, _ = library
    maps, type_ids = union_grids_and_type_ids(grids)

    # The union covers every signature exactly once.
    all_signatures = {s for g in grids for s in g.typing.signatures}
    assert maps.shape[0] == len(all_signatures)

    # Each ligand's remapped ids select the same physical maps as before.
    for original, remapped in zip(grids, type_ids):
        for local, union in zip(original.typing.type_ids, remapped):
            assert np.array_equal(original.maps[local], maps[union])


def test_mismatched_receptor_grids_are_refused(library):
    """Batching only makes sense for one receptor and one site; say so loudly."""
    grids, _, _ = library
    other = AffinityGrids(
        origin=grids[0].origin + 5.0,
        spacing=grids[0].spacing,
        maps=grids[0].maps,
        typing=grids[0].typing,
        box_min=grids[0].box_min,
        box_max=grids[0].box_max,
    )
    with pytest.raises(ValueError, match="one receptor site"):
        union_grids_and_type_ids([grids[0], other])


@pytest.mark.parametrize("device", DEVICES)
def test_packed_energy_and_gradient_match_per_ligand(device, library):
    """
    The central claim: packing changes nothing but the shape of the work.

    Compared against the CPU objective for each ligand separately, which is the
    reference the single-ligand GPU path was itself validated against.
    """
    grids, trees, objectives = library
    search = build_multi_ligand_search(
        grids, trees, objectives, device=device, names=SMILES
    )
    x = random_batch(search, trees, n_chains=6, seed=1)
    energy, grad = search.energy_and_dof_gradient(x)

    for i, (tree, objective) in enumerate(zip(trees, objectives)):
        for c in range(x.shape[1]):
            dof = np.zeros(tree.n_dof)
            row = x[i, c].cpu().double().numpy()
            dof[:6] = row[:6]
            if tree.n_torsions:
                dof[6:] = row[6 : 6 + tree.n_torsions]
            ref_e, ref_g = objective.energy_and_gradient(dof)

            got_e = float(energy[i, c].cpu().double())
            got_g = grad[i, c, : tree.n_dof].cpu().double().numpy()
            assert np.allclose(got_e, ref_e, **tol(device)), f"{SMILES[i]} energy"
            assert np.allclose(got_g, ref_g, **tol(device)), f"{SMILES[i]} gradient"


@pytest.mark.parametrize("device", DEVICES)
def test_surplus_torsion_slots_are_inert(device, library):
    """
    A ligand with fewer torsions than the batch maximum still carries the slots.

    They must do nothing at all: no energy change when moved, and no gradient to
    move them. Otherwise the optimiser would spend effort on coordinates that do
    not exist, and a rigid ligand's result would depend on what it was batched
    with.
    """
    grids, trees, objectives = library
    search = build_multi_ligand_search(
        grids, trees, objectives, device=device, names=SMILES
    )
    max_tors = search.packed.max_torsions
    assert any(t.n_torsions < max_tors for t in trees), "fixture has no surplus slots"

    x = random_batch(search, trees, n_chains=4, seed=2)
    base_energy, grad = search.energy_and_dof_gradient(x)

    # Move only the surplus slots, ligand by ligand.
    moved = x.clone()
    for i, tree in enumerate(trees):
        if tree.n_torsions < max_tors:
            moved[i, :, 6 + tree.n_torsions :] = 2.345
    new_energy, _ = search.energy_and_dof_gradient(moved)

    for i, tree in enumerate(trees):
        if tree.n_torsions >= max_tors:
            continue
        surplus = grad[i, :, 6 + tree.n_torsions :]
        assert float(surplus.abs().max()) < 1e-9, f"{SMILES[i]} has surplus gradient"
        assert torch.allclose(
            new_energy[i], base_energy[i], **tol(device)
        ), f"{SMILES[i]} energy moved when a non-existent torsion did"


@pytest.mark.parametrize("device", DEVICES)
def test_padded_atoms_contribute_nothing(device, library):
    """
    A padded slot has coordinates and would otherwise score.

    Checked by moving the padding far outside the grid, where an unmasked atom
    would pick up a large boundary penalty rather than a subtle error.
    """
    grids, trees, objectives = library
    search = build_multi_ligand_search(
        grids, trees, objectives, device=device, names=SMILES
    )
    x = random_batch(search, trees, n_chains=4, seed=3)
    energy, _ = search.energy_and_dof_gradient(x)

    # The smallest ligand carries the most padding; its energy must equal the
    # single-ligand reference despite those slots.
    smallest = int(np.argmin([t.base_coords.shape[0] for t in trees]))
    tree, objective = trees[smallest], objectives[smallest]
    assert tree.base_coords.shape[0] < search.packed.base_coords.shape[1]

    for c in range(x.shape[1]):
        dof = np.zeros(tree.n_dof)
        row = x[smallest, c].cpu().double().numpy()
        dof[:6] = row[:6]
        if tree.n_torsions:
            dof[6:] = row[6 : 6 + tree.n_torsions]
        ref, _ = objective.energy_and_gradient(dof)
        assert np.allclose(
            float(energy[smallest, c].cpu().double()), ref, **tol(device)
        )


def test_a_ligand_result_does_not_depend_on_its_batchmates(library):
    """
    Batching must not couple ligands.

    The same ligand packed with different companions -- and therefore different
    padding widths and a different union type stack -- must score identically.
    """
    grids, trees, objectives = library

    # One fixed set of poses, reused verbatim. Drawing them per call would make
    # the RNG stream depend on the batch size and compare different poses.
    rng = np.random.default_rng(5)
    poses = np.zeros((3, 9))
    poses[:, :3] = CENTER + rng.uniform(-4, 4, (3, 3))
    poses[:, 3:6] = rng.normal(0, 1, (3, 3))
    poses[:, 6:] = rng.uniform(-np.pi, np.pi, (3, 3))

    def energy_of_first(indices):
        search = build_multi_ligand_search(
            [grids[i] for i in indices],
            [trees[i] for i in indices],
            [objectives[i] for i in indices],
            device="cpu",
        )
        n_dof = search.packed.n_dof
        x = torch.zeros(len(indices), 3, n_dof, dtype=torch.float64)
        x[0] = torch.tensor(poses[:, :n_dof], dtype=torch.float64)
        energy, _ = search.energy_and_dof_gradient(x)
        return energy[0]

    alone = energy_of_first([1])
    with_big = energy_of_first([1, 3])
    with_all = energy_of_first([1, 0, 2, 3])

    assert torch.allclose(alone, with_big, rtol=1e-9, atol=1e-9)
    assert torch.allclose(alone, with_all, rtol=1e-9, atol=1e-9)


def test_basin_hopping_over_a_batch_improves_every_ligand(library):
    """End to end, with each ligand's own best tracked separately."""
    grids, trees, objectives = library
    config = RigidSearchConfig(n_chains=32, n_steps=3, seed=1)
    search = build_multi_ligand_search(
        grids, trees, objectives, config=config, device="cpu", names=SMILES
    )
    lo, hi = CENTER - 9, CENTER + 9

    gen = search._generator()
    start = search.initial_state(lo, hi, gen)
    start_energy, _ = search.energy_and_dof_gradient(start)

    best_x, best_energy = search.run_basin_hopping(
        lo, hi, LBFGSConfig(max_iter=30, max_line_search=12)
    )

    assert best_energy.shape == (len(SMILES), config.n_chains)
    for i, smiles in enumerate(SMILES):
        assert float(best_energy[i].min()) < float(start_energy[i].min()), smiles

    # Reported energies belong to the reported poses.
    rescored, _ = search.energy_and_dof_gradient(best_x)
    assert torch.allclose(rescored, best_energy, rtol=1e-6, atol=1e-6)
