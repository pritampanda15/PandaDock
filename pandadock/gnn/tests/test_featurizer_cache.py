"""
The atom featurizer memoizes the discrete part of its feature vector.

That cache is the hot path of the dataloader, and a wrong cache key would not
raise -- it would silently hand back another atom's features and quietly degrade
every model trained afterwards. These tests pin the behaviour by recomputing the
features from first principles and comparing.
"""

import numpy as np
import pytest

from pandadock.gnn.data.featurizer import (
    AA_TO_IDX,
    AMINO_ACIDS,
    ELEMENT_TO_IDX,
    ELEMENTS,
    HYBRIDIZATION_MAP,
    SYBYL_TYPES,
    AtomFeaturizer,
)
from pandadock.gnn.data.mol2_parser import Atom


def atom(atom_type="C.3", name="CA", residue_name="ALA", charge=0.0):
    return Atom(
        id=1, name=name, x=0.0, y=0.0, z=0.0,
        atom_type=atom_type, charge=charge,
        residue_name=residue_name, residue_id=1,
    )


CONFIGS = [
    {},
    {"include_charge": False},
    {"include_residue": False},
    {"normalize_charge": False},
    {"include_charge": False, "include_residue": False},
]


@pytest.mark.parametrize("config", CONFIGS)
def test_feature_vector_matches_an_independent_computation(config):
    """
    Recompute each feature block directly and compare against the featurizer.

    This is deliberately a reimplementation rather than a stored golden array:
    it checks the values mean what the docstring says they mean, not merely that
    they have not changed.
    """
    featurizer = AtomFeaturizer(**config)
    rng = np.random.default_rng(0)

    for atom_type in SYBYL_TYPES:
        for element_name in ["CA", "N", "O", "S", "H1"]:
            for residue in ["ALA", "TRP", "UNK", "LIG"]:
                for is_protein in (True, False):
                    charge = float(rng.uniform(-2, 2))
                    a = atom(atom_type, element_name, residue, charge)
                    got = featurizer.featurize_atom(a, is_protein)

                    assert got.shape == (featurizer.feature_dim,)

                    element = a.element.upper()
                    if element not in ELEMENT_TO_IDX:
                        element = "OTHER"
                    assert got[ELEMENT_TO_IDX[element]] == 1.0
                    assert got[:len(ELEMENTS)].sum() == 1.0

                    offset = len(ELEMENTS) + 16
                    hyb = got[offset:offset + 4]
                    expected_hyb = HYBRIDIZATION_MAP.get(a.hybridization, 2)
                    assert hyb[expected_hyb] == 1.0
                    assert hyb.sum() == 1.0

                    if featurizer.include_charge:
                        charge_at = offset + 4 + 2 + 2
                        expected = charge
                        if featurizer.normalize_charge:
                            expected = max(-1.0, min(1.0, charge))
                        assert got[charge_at] == pytest.approx(expected, abs=1e-6)

                    if featurizer.include_residue:
                        residues = got[-len(AMINO_ACIDS):]
                        if is_protein:
                            idx = AA_TO_IDX.get(residue[:3].upper(), AA_TO_IDX["UNK"])
                            assert residues[idx] == 1.0
                            assert residues.sum() == 1.0
                        else:
                            assert residues.sum() == 0.0


def test_cache_key_covers_every_attribute_that_changes_the_features():
    """
    Atoms differing in any keyed attribute must not collide in the cache.

    A key that dropped an attribute would return the first atom's features for
    the second, which no shape or dtype check would catch.
    """
    featurizer = AtomFeaturizer()

    # Atom.element and Atom.hybridization are both derived from atom_type, so
    # element is varied through atom_type rather than through name. name matters
    # only via the 'H' test in the H-bond donor rule, so it is varied on an
    # oxygen, where that rule is live.
    variants = {
        "baseline": atom(atom_type="O.3", name="O"),
        "atom_type": atom(atom_type="N.ar", name="O"),
        "element": atom(atom_type="C.3", name="O"),
        "name_has_h": atom(atom_type="O.3", name="OH"),
        "residue": atom(atom_type="O.3", name="O", residue_name="TRP"),
    }

    vectors = {k: featurizer.featurize_atom(v, True) for k, v in variants.items()}
    for name, vector in vectors.items():
        if name == "baseline":
            continue
        assert not np.array_equal(vector, vectors["baseline"]), (
            f"{name} produced identical features to the baseline: the cache key "
            "is missing an attribute that affects the result"
        )

    # is_protein is keyed too: the residue block is zeroed for ligand atoms.
    assert not np.array_equal(
        featurizer.featurize_atom(atom(), True),
        featurizer.featurize_atom(atom(), False),
    )


def test_repeated_calls_are_stable_and_independent():
    """
    A cached vector must not be aliased into the caller's result.

    featurize_atom returns a fresh array; if it handed back the cached buffer,
    a caller mutating one atom's features would corrupt every other atom sharing
    that key.
    """
    featurizer = AtomFeaturizer()
    first = featurizer.featurize_atom(atom(), True)
    reference = first.copy()

    first[0] = 99.0
    second = featurizer.featurize_atom(atom(), True)

    assert np.array_equal(second, reference)


def test_charge_is_not_cached():
    """Charge is continuous; two atoms alike but for charge must still differ."""
    featurizer = AtomFeaturizer()
    low = featurizer.featurize_atom(atom(charge=-0.5), True)
    high = featurizer.featurize_atom(atom(charge=0.5), True)

    assert not np.array_equal(low, high)
    # Exactly one position may differ: the charge slot.
    assert int((low != high).sum()) == 1


def test_nan_charge_passes_through_like_np_clip():
    """
    Clipping is done with comparisons rather than np.clip for speed.

    min/max would turn a NaN charge into 1.0; np.clip leaves it NaN. Keep the
    original behaviour so a corrupt input stays visible as NaN rather than being
    silently rewritten to a plausible value.
    """
    featurizer = AtomFeaturizer()
    vector = featurizer.featurize_atom(atom(charge=float("nan")), True)
    assert np.isnan(vector).sum() == 1


def test_molecule_features_match_per_atom_features():
    from pandadock.gnn.data.mol2_parser import ParsedMolecule

    featurizer = AtomFeaturizer()
    atoms = [atom("C.3"), atom("N.ar", name="N"), atom("O.2", name="O")]
    molecule = ParsedMolecule(name="m", atoms=atoms, bonds=[])
    molecule.num_atoms = len(atoms)

    matrix = featurizer.featurize_molecule(molecule, is_protein=True)
    assert matrix.shape == (3, featurizer.feature_dim)
    for i, a in enumerate(atoms):
        assert np.array_equal(matrix[i], featurizer.featurize_atom(a, True))
