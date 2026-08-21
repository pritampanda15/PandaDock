"""
The affinity grid must not depend on how the grid is traversed.

Building the grid is roughly half the cost of a docking run, and it was made
several times faster by evaluating each block of grid points against only the
receptor atoms that can reach it. That selection is exact -- an atom contributes
only where `d - r_ligand - r_receptor < cutoff`, so anything further away
contributes zero, not a small number -- and these tests hold it to that.

A silent change here would shift every energy the search sees without failing
anything else.
"""

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from pandadock.docking.scoring.vina_scoring import VinaScoring  # noqa: E402
from pandadock.docking.search.grid_maps import AffinityGrids  # noqa: E402


class FakeAtom:
    """Minimal stand-in for a BioPython atom."""

    def __init__(self, name, element, coord, residue_name="ALA"):
        self.name = name
        self.element = element
        self._coord = np.asarray(coord, dtype=float)
        self.parent = None
        self._residue_name = residue_name

    def get_coord(self):
        return self._coord

    def get_name(self):
        return self.name

    def get_parent(self):
        return self.parent


class FakeResidue:
    def __init__(self, name):
        self.resname = name

    def get_resname(self):
        return self.resname


class FakeStructure:
    def __init__(self, atoms):
        self._atoms = atoms

    def get_atoms(self):
        return iter(self._atoms)


@pytest.fixture
def receptor():
    """A slab of atoms wide enough that most are out of reach of any one block."""
    rng = np.random.default_rng(3)
    residue = FakeResidue("ALA")
    atoms = []
    for i, xyz in enumerate(rng.uniform(-14, 14, size=(400, 3))):
        element = ["C", "N", "O", "S"][i % 4]
        atom = FakeAtom(["CA", "N", "O", "SD"][i % 4], element, xyz)
        atom.parent = residue
        atoms.append(atom)
    return FakeStructure(atoms)


@pytest.fixture
def ligand():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1C(=O)NCCO"))
    AllChem.EmbedMolecule(mol, randomSeed=11)
    return mol


@pytest.mark.parametrize("block_size", [1, 3, 6, 13, 64])
def test_grid_is_independent_of_block_size(receptor, ligand, block_size):
    """
    Traversal order must not change a single value.

    Block size is a pure performance knob. If a grid differs between block
    sizes, the neighbour selection is dropping atoms that do contribute, and
    every docking energy computed from that grid is wrong.
    """
    center = np.zeros(3)
    dimensions = np.array([10.0, 10.0, 10.0])

    reference = AffinityGrids.build(
        receptor, ligand, center, dimensions, scoring=VinaScoring(), block_size=6
    )
    candidate = AffinityGrids.build(
        receptor, ligand, center, dimensions,
        scoring=VinaScoring(), block_size=block_size,
    )

    assert candidate.maps.shape == reference.maps.shape
    assert np.array_equal(candidate.maps, reference.maps), (
        f"block_size={block_size} produced different grids; the neighbour "
        f"selection is not exact (max difference "
        f"{np.abs(candidate.maps - reference.maps).max():.3g})"
    )


def test_grid_has_real_structure(receptor, ligand):
    """
    Guard against the grids being trivially equal because they are all zero.

    Every test above compares grids to each other, so a build that silently
    produced nothing would pass all of them.
    """
    grids = AffinityGrids.build(
        receptor, ligand, np.zeros(3), np.array([10.0, 10.0, 10.0]),
        scoring=VinaScoring(),
    )

    assert grids.maps.size > 0
    assert np.isfinite(grids.maps).all()
    assert np.any(grids.maps != 0.0), "grids are entirely zero"
    assert grids.maps.std() > 0, "grids are constant"


def test_distant_receptor_atoms_do_not_change_the_grid(receptor, ligand):
    """
    Adding atoms far beyond the cutoff must leave the grid untouched.

    This is the assumption the whole optimisation rests on, tested directly
    rather than inferred from the block-size invariance.
    """
    center = np.zeros(3)
    dimensions = np.array([8.0, 8.0, 8.0])

    before = AffinityGrids.build(
        receptor, ligand, center, dimensions, scoring=VinaScoring()
    )

    residue = FakeResidue("ALA")
    far = list(receptor.get_atoms())
    for offset in (300.0, 400.0, 500.0):
        atom = FakeAtom("CA", "C", [offset, offset, offset])
        atom.parent = residue
        far.append(atom)

    after = AffinityGrids.build(
        FakeStructure(far), ligand, center, dimensions, scoring=VinaScoring()
    )

    assert np.array_equal(before.maps, after.maps)
