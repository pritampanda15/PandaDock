"""
Inference must prepare inputs the way training did.

The GNN is trained on a binding site -- a few hundred atoms cut around the
ligand -- and on heavy atoms only when the model comes from SAIR. Handed a whole
protonated protein it still returns a number, which is why the mismatch has to
be caught here rather than by a user noticing implausible affinities.
"""

import numpy as np
import pytest

from pandadock.gnn.data.graph_builder import (
    GraphConfig,
    drop_hydrogens,
    extract_binding_site,
)
from pandadock.gnn.data.mol2_parser import Atom, ParsedMolecule


def molecule(spec, name="m"):
    atoms = [
        Atom(id=i + 1, name=n, x=x, y=y, z=z, atom_type=t,
             charge=0.0, residue_name=r, residue_id=i + 1)
        for i, (n, t, r, x, y, z) in enumerate(spec)
    ]
    mol = ParsedMolecule(name=name, atoms=atoms, bonds=[])
    mol.num_atoms = len(atoms)
    return mol


def test_site_cut_is_inclusive_at_the_radius():
    protein = molecule([("CA", "C.3", "ALA", 0.0, 0.0, d)
                        for d in (0.0, 5.0, 9.9, 10.0, 10.1, 50.0)])
    site = extract_binding_site(protein, [0.0, 0.0, 0.0], radius=10.0)

    assert [round(a.z, 1) for a in site.atoms] == [0.0, 5.0, 9.9, 10.0]


def test_site_cut_matches_the_radius_it_is_given():
    rng = np.random.default_rng(0)
    coords = rng.normal(0, 8, (400, 3))
    protein = molecule([("CA", "C.3", "ALA", *xyz) for xyz in coords])

    previous = 0
    for radius in (5.0, 10.0, 15.0, 20.0):
        site = extract_binding_site(protein, [0.0, 0.0, 0.0], radius=radius)
        expected = int((np.linalg.norm(coords, axis=1) <= radius).sum())
        assert len(site.atoms) == expected
        assert len(site.atoms) >= previous
        previous = len(site.atoms)


def test_site_cut_falls_back_rather_than_returning_nothing():
    """An empty site would build a graph with no protein at all."""
    protein = molecule([("CA", "C.3", "ALA", 100.0, 100.0, 100.0)])
    site = extract_binding_site(protein, [0.0, 0.0, 0.0], radius=10.0)
    assert len(site.atoms) == 1


def test_drop_hydrogens_removes_and_renumbers():
    mol = molecule([
        ("CA", "C.3", "ALA", 0.0, 0.0, 0.0),
        ("H1", "H", "ALA", 1.0, 0.0, 0.0),
        ("N", "N.3", "ALA", 2.0, 0.0, 0.0),
        ("HD", "D", "ALA", 3.0, 0.0, 0.0),
    ])
    trimmed = drop_hydrogens(mol)

    assert [a.name for a in trimmed.atoms] == ["CA", "N"]
    assert [a.id for a in trimmed.atoms] == [1, 2]
    assert trimmed.num_atoms == 2
    # Renumbering must not mutate the input.
    assert len(mol.atoms) == 4


def test_drop_hydrogens_is_a_noop_without_hydrogens():
    mol = molecule([("CA", "C.3", "ALA", 0.0, 0.0, 0.0)])
    assert drop_hydrogens(mol) is mol


def test_graph_config_defaults_preserve_existing_behaviour():
    """
    Hydrogens are kept by default.

    ULVSH and PDBbind models were trained with hydrogens present; stripping by
    default would silently change what every existing checkpoint is given.
    """
    config = GraphConfig()
    assert config.strip_hydrogens is False
    assert config.site_radius == 10.0
    assert config.use_site_only is True
