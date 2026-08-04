"""
Affinity grids may be reused across ligands, but only when they are genuinely
the same grid.

A grid depends on the receptor, the box and one ligand atom signature. Reuse is
what makes virtual screening tractable -- docking a million ligands into one
receptor should build each grid once, not a million times -- but a key that is
too loose serves one receptor's field for another and silently corrupts every
energy downstream. Nothing else in the pipeline would notice.

These tests hold the key to exactly the things a grid depends on.
"""

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from pandadock.docking.scoring.vina_scoring import VinaScoring  # noqa: E402
from pandadock.docking.search.grid_maps import AffinityGrids, GridCache  # noqa: E402
from tests.test_grid_maps_blocking import (  # noqa: E402
    FakeAtom, FakeResidue, FakeStructure,
)

CENTER = np.zeros(3)
BOX = np.array([9.0, 9.0, 9.0])


def build_receptor(seed=5, shift=0.0):
    rng = np.random.default_rng(seed)
    residue = FakeResidue("ALA")
    atoms = []
    for i, xyz in enumerate(rng.uniform(-12, 12, size=(250, 3))):
        atom = FakeAtom(
            ["CA", "N", "O", "SD"][i % 4], ["C", "N", "O", "S"][i % 4], xyz + shift
        )
        atom.parent = residue
        atoms.append(atom)
    return FakeStructure(atoms)


def build_ligand(smiles, seed=11):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    return mol


@pytest.fixture
def receptor():
    return build_receptor()


@pytest.fixture
def ligand():
    return build_ligand("c1ccccc1C(=O)NCCO")


def build(receptor, ligand, cache=None, center=CENTER, box=BOX):
    return AffinityGrids.build(
        receptor, ligand, center, box, scoring=VinaScoring(), cache=cache
    )


def test_cache_returns_identical_grids(receptor, ligand):
    """Caching is an optimisation; it must not change a single value."""
    uncached = build(receptor, ligand)

    cache = GridCache()
    first = build(receptor, ligand, cache)
    second = build(receptor, ligand, cache)

    assert np.array_equal(first.maps, uncached.maps)
    assert np.array_equal(second.maps, uncached.maps)


def test_repeat_build_is_served_entirely_from_cache(receptor, ligand):
    cache = GridCache()
    grids = build(receptor, ligand, cache)
    n_types = grids.maps.shape[0]

    assert cache.stats()["misses"] == n_types
    assert cache.stats()["hits"] == 0

    build(receptor, ligand, cache)
    assert cache.stats()["hits"] == n_types
    assert cache.stats()["misses"] == n_types  # unchanged: nothing rebuilt


def test_a_different_receptor_is_not_reused(ligand):
    """
    The key must follow the receptor, not just the box.

    Screening scripts reuse one cache object across targets; a key that ignored
    the receptor would score every target against the first one's field.
    """
    cache = GridCache()
    first = build(build_receptor(seed=5), ligand, cache)
    second = build(build_receptor(seed=99), ligand, cache)

    assert not np.array_equal(first.maps, second.maps)

    fresh = build(build_receptor(seed=99), ligand)
    assert np.array_equal(second.maps, fresh.maps)


def test_a_moved_receptor_is_not_reused(ligand):
    """Same atoms, translated: a different field, and must not be shared."""
    cache = GridCache()
    first = build(build_receptor(shift=0.0), ligand, cache)
    second = build(build_receptor(shift=3.0), ligand, cache)

    assert not np.array_equal(first.maps, second.maps)


def test_a_different_box_is_not_reused(receptor, ligand):
    cache = GridCache()
    first = build(receptor, ligand, cache)
    second = build(receptor, ligand, cache, center=np.array([2.0, 0.0, 0.0]))

    assert not np.array_equal(first.maps, second.maps)
    fresh = build(receptor, ligand, center=np.array([2.0, 0.0, 0.0]))
    assert np.array_equal(second.maps, fresh.maps)


def test_shared_signatures_are_reused_across_different_ligands(receptor):
    """
    The point of keying per signature rather than per ligand.

    A second ligand sharing atom types with the first should build only the
    types it adds, not start over.
    """
    cache = GridCache()

    first = build(receptor, build_ligand("c1ccccc1C(=O)NCCO"), cache)
    after_first = cache.stats()["misses"]
    assert after_first == first.maps.shape[0]

    second = build(receptor, build_ligand("c1ccccc1C(=O)NCC"), cache)

    assert cache.stats()["hits"] > 0, "no signature was reused between ligands"
    assert cache.stats()["misses"] - after_first < second.maps.shape[0], (
        "every signature was rebuilt; the key is too specific to be useful"
    )

    fresh = build(receptor, build_ligand("c1ccccc1C(=O)NCC"))
    assert np.array_equal(second.maps, fresh.maps)


def test_cached_grids_are_not_aliased(receptor, ligand):
    """
    A caller mutating its grids must not corrupt the cache.

    The maps array is handed out directly, so the cache has to hold its own
    copy or the next hit returns whatever the previous caller left behind.
    """
    cache = GridCache()
    first = build(receptor, ligand, cache)
    first.maps[:] = 1234.0

    second = build(receptor, ligand, cache)
    assert not np.allclose(second.maps, 1234.0)


def test_eviction_keeps_the_cache_bounded(receptor):
    cache = GridCache(max_entries=2)
    for smiles in ("CCO", "CCN", "CCS", "CCCl", "c1ccccc1"):
        build(receptor, build_ligand(smiles), cache)
    assert cache.stats()["entries"] <= 2
