"""
Tests for the conformational search stack.

These cover the properties that silently break docking when wrong: torsions must
preserve covalent geometry, gradients must match the energy they claim to
differentiate, orientation sampling must actually cover SO(3), and grid scores
must agree with the scoring function they approximate.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

# RDKit and BioPython are required for docking but are not pip-installable
# alongside conda in a way that works everywhere, so an environment can
# legitimately lack them. Skip rather than let an ImportError at collection time
# abort the entire suite, including tests that have nothing to do with docking.
pytest.importorskip("rdkit", reason="RDKit is required for the docking tests")
pytest.importorskip("Bio", reason="BioPython is required to parse receptors")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from pandadock.docking.search import (  # noqa: E402
    AffinityGrids,
    DockingObjective,
    MonteCarloSearch,
    SearchConfig,
    TorsionTree,
    cluster_poses,
)
from pandadock.docking.search.monte_carlo import SearchResult
from pandadock.docking.search.rotations import (
    compose_rotvecs,
    matrix_to_rotvec,
    random_rotvec,
    rodrigues_matrix,
    wrap_rotvec,
)

pytestmark = pytest.mark.filterwarnings("ignore")


# ------------------------------------------------------------------- fixtures


def build_mol(smiles="CCOC(=O)c1ccccc1N", seed=42):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


@pytest.fixture(scope="module")
def mol():
    return build_mol()


@pytest.fixture(scope="module")
def receptor(tmp_path_factory):
    """A hollow shell of atoms forming a cavity, written as a minimal PDB."""
    from Bio.PDB import PDBParser

    rng = np.random.default_rng(7)
    center = np.array([10.0, 10.0, 10.0])
    lines = []
    for i in range(400):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        pos = center + v * rng.uniform(7.5, 12.0)
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
def box():
    return np.array([10.0, 10.0, 10.0]), np.array([18.0, 18.0, 18.0])


@pytest.fixture(scope="module")
def objective(mol, receptor, box):
    center, dims = box
    tree = TorsionTree(mol)
    grids = AffinityGrids.build(receptor, mol, center, dims, spacing=0.4)
    return DockingObjective(tree, grids)


# ---------------------------------------------------------------- torsion tree


def test_torsion_tree_finds_rotatable_bonds(mol):
    tree = TorsionTree(mol)
    assert tree.n_torsions > 0
    assert tree.n_dof == 6 + tree.n_torsions
    assert tree.root_atom in range(mol.GetNumAtoms())


def test_rigid_mode_has_no_torsions(mol):
    tree = TorsionTree(mol, rigid=True)
    assert tree.n_torsions == 0
    assert tree.n_dof == 6


def test_torsions_preserve_bond_lengths(mol):
    """Rotating about a bond must not stretch any bond in the molecule."""
    tree = TorsionTree(mol)
    rng = np.random.default_rng(0)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]

    ref = tree.base_coords
    ref_lengths = np.array([np.linalg.norm(ref[i] - ref[j]) for i, j in bonds])

    for _ in range(10):
        angles = rng.uniform(-np.pi, np.pi, tree.n_torsions)
        moved = tree.apply_torsions(angles)
        lengths = np.array([np.linalg.norm(moved[i] - moved[j]) for i, j in bonds])
        assert np.allclose(lengths, ref_lengths, atol=1e-9)


def test_torsions_preserve_ring_geometry(mol):
    """Ring bonds are not rotatable, so aromatic rings must stay rigid."""
    tree = TorsionTree(mol)
    ring_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.IsInRing()]
    if len(ring_atoms) < 3:
        pytest.skip("molecule has no ring")

    ref = tree.base_coords[ring_atoms]
    ref_dists = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=2)

    angles = np.random.default_rng(1).uniform(-np.pi, np.pi, tree.n_torsions)
    moved = tree.apply_torsions(angles)[ring_atoms]
    dists = np.linalg.norm(moved[:, None, :] - moved[None, :, :], axis=2)
    assert np.allclose(dists, ref_dists, atol=1e-9)


def test_zero_dof_reproduces_reference(mol):
    tree = TorsionTree(mol)
    dof = np.zeros(tree.n_dof)
    coords = tree.build_coords(dof)
    assert np.allclose(coords, tree.base_coords, atol=1e-12)


# -------------------------------------------------------------------- rotations


def test_rodrigues_is_a_rotation():
    rng = np.random.default_rng(3)
    for _ in range(20):
        matrix = rodrigues_matrix(rng.normal(0, 2, 3))
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(matrix), 1.0, atol=1e-12)


def test_matrix_rotvec_roundtrip():
    rng = np.random.default_rng(4)
    for _ in range(50):
        rotvec = random_rotvec(rng)
        recovered = matrix_to_rotvec(rodrigues_matrix(rotvec))
        # Compare rotations, not parameters: r and its 2*pi complement agree.
        assert np.allclose(rodrigues_matrix(recovered), rodrigues_matrix(rotvec), atol=1e-8)


def test_random_rotvec_covers_so3():
    """
    Orientation sampling must be uniform over SO(3), not merely non-constant.

    For the uniform (Haar) measure the rotation angle has density
    (1 - cos t) / pi on [0, pi], with mean pi/2 + 2/pi ~ 2.207 rad. Checking the
    whole density rather than just the spread catches samplers that reach large
    angles but still concentrate near the identity -- which is exactly the failure
    mode of drawing axis-angle components independently from a Gaussian.
    """
    rng = np.random.default_rng(5)
    angles = np.array([np.linalg.norm(random_rotvec(rng)) for _ in range(20000)])

    # Canonical axis-angle form keeps the angle in [0, pi].
    assert angles.min() >= 0.0
    assert angles.max() <= np.pi + 1e-9

    assert np.isclose(angles.mean(), np.pi / 2 + 2 / np.pi, atol=0.05)

    observed, edges = np.histogram(angles, bins=10, range=(0.0, np.pi), density=True)
    midpoints = (edges[:-1] + edges[1:]) / 2
    expected = (1.0 - np.cos(midpoints)) / np.pi
    assert np.max(np.abs(observed - expected)) < 0.03


def test_wrap_rotvec_preserves_rotation():
    rng = np.random.default_rng(6)
    for _ in range(20):
        rotvec = rng.normal(0, 8, 3)
        wrapped = wrap_rotvec(rotvec)
        assert np.linalg.norm(wrapped) <= np.pi + 1e-9
        assert np.allclose(rodrigues_matrix(wrapped), rodrigues_matrix(rotvec), atol=1e-8)


def test_compose_rotvecs_matches_matrix_product():
    rng = np.random.default_rng(8)
    a, b = random_rotvec(rng), random_rotvec(rng)
    composed = rodrigues_matrix(compose_rotvecs(a, b))
    assert np.allclose(composed, rodrigues_matrix(a) @ rodrigues_matrix(b), atol=1e-8)


# -------------------------------------------------------------------- objective


def test_gradient_matches_finite_differences(objective):
    """
    The analytic gradient must agree with the energy it differentiates.

    A mismatch here does not raise; it silently makes the local optimizer descend
    the wrong direction, which is the kind of defect that shows up only as poor
    RMSD much later.
    """
    tree = objective.tree
    rng = np.random.default_rng(11)
    box_min, box_max = np.array([5.0] * 3), np.array([15.0] * 3)

    for _ in range(4):
        dof = tree.random_dof(rng, box_min, box_max)
        _, analytic = objective.energy_and_gradient(dof)

        numeric = np.zeros_like(analytic)
        h = 1e-6
        for k in range(len(dof)):
            plus, minus = dof.copy(), dof.copy()
            plus[k] += h
            minus[k] -= h
            numeric[k] = (objective.energy(plus) - objective.energy(minus)) / (2 * h)

        scale = np.maximum(np.abs(numeric), 1.0)
        assert np.max(np.abs(analytic - numeric) / scale) < 1e-4


def test_energy_is_translation_and_rotation_consistent(objective):
    """Moving the ligand must change the receptor term but not the internal one."""
    tree = objective.tree
    rng = np.random.default_rng(12)
    dof = tree.random_dof(rng, np.array([8.0] * 3), np.array([12.0] * 3))

    coords = objective.coords(dof)
    intra_a, _ = objective._intramolecular(coords, need_gradient=False)

    moved = dof.copy()
    moved[:3] += np.array([1.0, -2.0, 0.5])
    moved[3:6] = compose_rotvecs(random_rotvec(rng), dof[3:6])
    intra_b, _ = objective._intramolecular(objective.coords(moved), need_gradient=False)

    assert np.isclose(intra_a, intra_b, atol=1e-8)


def test_out_of_box_is_penalised(objective, box):
    """A ligand pushed far outside the grid must score worse than one inside."""
    center, dims = box
    tree = objective.tree
    inside = np.zeros(tree.n_dof)
    inside[:3] = center
    outside = inside.copy()
    outside[:3] = center + dims * 5.0
    assert objective.energy(outside) > objective.energy(inside)


# ------------------------------------------------------------------------ grids


def test_grid_score_tracks_direct_scoring(objective, mol, receptor):
    """
    Grid interpolation must approximate the underlying Vina function.

    Compared on relative terms: the absolute error scales with the magnitude of
    the energy, which is dominated by repulsion for clashing poses.
    """
    from scipy.spatial.distance import cdist

    from pandadock.docking.scoring.vina_scoring import VinaScoring

    vina = VinaScoring()
    tree = objective.tree
    grids = objective.grids
    rng = np.random.default_rng(13)

    heavy = tree.heavy_atoms
    rec_atoms = [a for a in receptor.get_atoms() if a.element.strip() != "H"]
    rec_coords = np.array([a.get_coord() for a in rec_atoms])
    lig_types = vina._get_ligand_atom_types(mol)
    rec_types = vina._get_receptor_atom_types(rec_atoms)
    lig_radii = np.array([vina._get_vdw_radius(lig_types[i]) for i in heavy])
    rec_radii = np.array([vina._get_vdw_radius(t) for t in rec_types])

    for _ in range(5):
        dof = tree.random_dof(rng, np.array([7.0] * 3), np.array([13.0] * 3))
        coords = objective.coords(dof)

        grid_energy = grids.score(coords[heavy])

        surf = cdist(coords[heavy], rec_coords) - lig_radii[:, None] - rec_radii[None, :]
        active = surf < vina.cutoff
        w = vina.weights
        direct = w["gauss1"] * np.exp(-((surf - vina.gauss1_offset) / vina.gauss1_width) ** 2)
        direct += w["gauss2"] * np.exp(-((surf - vina.gauss2_offset) / vina.gauss2_width) ** 2)
        direct += w["repulsion"] * np.where(surf < vina.repulsion_cutoff, surf**2, 0.0)
        steric = float(np.sum(np.where(active, direct, 0.0)))

        # Hydrophobic and H-bond terms are omitted from `steric`, so allow a
        # tolerance proportional to the total magnitude.
        assert abs(grid_energy - steric) < 0.15 * max(abs(steric), 10.0)


# ----------------------------------------------------------------------- search


def test_search_finds_negative_energy_poses(objective, box):
    center, dims = box
    search = MonteCarloSearch(objective, SearchConfig(exhaustiveness=2, n_steps=25, seed=1))
    minima = search.run(center - dims / 2, center + dims / 2)

    assert minima
    assert minima[0].energy < 0.0
    assert all(minima[i].energy <= minima[i + 1].energy for i in range(len(minima) - 1))


def test_search_is_reproducible_with_a_seed(objective, box):
    center, dims = box
    config = SearchConfig(exhaustiveness=2, n_steps=15, seed=99)

    first = MonteCarloSearch(objective, config).run(center - dims / 2, center + dims / 2)
    second = MonteCarloSearch(objective, config).run(center - dims / 2, center + dims / 2)

    assert np.isclose(first[0].energy, second[0].energy, atol=1e-9)
    assert np.allclose(first[0].coords, second[0].coords, atol=1e-9)


def test_search_explores_orientations(objective, box):
    """
    Sampled orientations must span SO(3).

    The algorithm this replaced drew rotations from a narrow Gaussian about the
    input conformer's orientation, so every pose it produced was within roughly
    30 degrees of however the ligand file happened to be written. Guard against a
    regression to that behaviour.
    """
    center, dims = box
    search = MonteCarloSearch(objective, SearchConfig(exhaustiveness=4, n_steps=15, seed=2))
    minima = search.run(center - dims / 2, center + dims / 2)

    angles = np.array([np.linalg.norm(m.dof[3:6]) for m in minima])
    assert angles.max() > 1.5, "orientation search is not covering SO(3)"


def test_search_explores_the_whole_box(objective, box):
    """Starting positions must cover the box, not cluster at its centre."""
    center, dims = box
    tree = objective.tree
    rng = np.random.default_rng(21)
    starts = np.array(
        [tree.random_dof(rng, center - dims / 2, center + dims / 2)[:3] for _ in range(500)]
    )
    spread = starts.max(axis=0) - starts.min(axis=0)
    assert np.all(spread > 0.8 * dims), "translation sampling is biased toward the centre"


def test_clustering_returns_distinct_modes(objective, box):
    center, dims = box
    search = MonteCarloSearch(objective, SearchConfig(exhaustiveness=3, n_steps=20, seed=3))
    minima = search.run(center - dims / 2, center + dims / 2)
    heavy = objective.tree.heavy_atoms

    clustered = cluster_poses(minima, heavy, rmsd_cutoff=2.0, max_poses=5)
    assert 1 <= len(clustered) <= 5

    for i, a in enumerate(clustered):
        for b in clustered[i + 1 :]:
            rmsd = np.sqrt(np.mean(np.sum((a.coords[heavy] - b.coords[heavy]) ** 2, axis=1)))
            assert rmsd >= 2.0 - 1e-9


def test_automorphisms_identify_ring_symmetry():
    """A benzene ring can be labelled twelve ways that are the same structure."""
    from pandadock.analysis.rmsd import heavy_atom_automorphisms

    assert len(heavy_atom_automorphisms(build_mol("c1ccccc1", seed=1))) == 12
    # No symmetry: the identity alone, so callers can use the result unguarded.
    assert len(heavy_atom_automorphisms(build_mol("CCO", seed=1))) == 1


def test_clustering_treats_symmetry_equivalent_poses_as_duplicates():
    """
    A ring flipped onto itself is the same physical pose and must not consume a
    second slot in the returned ensemble. Under fixed atom indices the flip
    reports a large RMSD, so without symmetry correction both copies are kept.
    """
    from pandadock.analysis.rmsd import heavy_atom_automorphisms

    mol = build_mol("c1ccccc1", seed=1)
    heavy_idx = np.array(
        [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1], dtype=np.int64
    )
    coords = np.asarray(mol.GetConformer().GetPositions())

    flipped = coords.copy()
    flipped[heavy_idx] = coords[heavy_idx][[3, 4, 5, 0, 1, 2]]

    original = SearchResult(coords=coords, energy=-9.0, dof=np.zeros(6))
    relabelled = SearchResult(coords=flipped, energy=-8.9, dof=np.zeros(6))
    results = [original, relabelled]

    plain = cluster_poses(results, heavy_idx, rmsd_cutoff=2.0, max_poses=5)
    assert len(plain) == 2, "precondition: the flip is far apart under fixed indices"

    corrected = cluster_poses(
        results,
        heavy_idx,
        rmsd_cutoff=2.0,
        max_poses=5,
        automorphisms=heavy_atom_automorphisms(mol),
    )
    assert len(corrected) == 1
    assert corrected[0].energy == -9.0, "the lower-energy copy is the one kept"
