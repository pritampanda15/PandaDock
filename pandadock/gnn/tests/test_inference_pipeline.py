"""
End-to-end inference on real files: parse, cut the site, build the graph.

Every other GNN test builds molecules in memory. These run the path a user
actually takes -- a receptor PDB and a ligand MOL2 off disk, through the same
`build_from_files` that `pandadock-gnn predict` calls.

The receptor fixture places six residues at known distances from the ligand
centroid (PHE 4.0, SER 5.5, ASP 7.0, ALA 8.5 A inside a 10 A site; TRP 14.0 and
LYS 22.0 A outside it), so the cut has an exact expected answer rather than a
plausible-looking one.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
pytest.importorskip("Bio", reason="BioPython is required to read the receptor PDB")

from pandadock.gnn.data.graph_builder import (  # noqa: E402
    GraphConfig,
    HeterogeneousGraphBuilder,
    drop_hydrogens,
    extract_binding_site,
    parse_molecule_file,
)

DATA = Path(__file__).parent / "data"
RECEPTOR = DATA / "receptor.pdb"
LIGAND = DATA / "ligand.mol2"

INSIDE = {"PHE", "SER", "ASP", "ALA"}
OUTSIDE = {"TRP", "LYS"}


def ligand_centroid(ligand):
    """Heavy-atom centroid, matching how the SAIR cache was cut."""
    heavy = np.array([
        [a.x, a.y, a.z] for a in ligand.atoms if a.element.upper() != "H"
    ])
    return heavy.mean(axis=0)


def test_fixtures_parse_without_silent_loss():
    """
    Every ATOM record must survive parsing.

    BioPython discards atoms whose names collide within a residue, which shrinks
    a fixture without any error -- the first version of this receptor lost an
    atom that way.
    """
    written = sum(
        1 for line in RECEPTOR.read_text().splitlines() if line.startswith("ATOM")
    )
    receptor = parse_molecule_file(str(RECEPTOR))
    assert len(receptor.atoms) == written

    ligand = parse_molecule_file(str(LIGAND))
    assert len(ligand.atoms) == 16
    assert sum(1 for a in ligand.atoms if a.element.upper() != "H") == 9


def test_residue_aware_typing_runs_on_a_real_receptor():
    """
    Protein atoms must not all collapse to sp3.

    A flat element-to-type mapping types every backbone carbonyl as C.3 and
    every aromatic ring carbon as C.3, and the featurizer embeds the SYBYL type
    into 16 of its 56 node features.
    """
    receptor = parse_molecule_file(str(RECEPTOR))
    types = {a.atom_type for a in receptor.atoms}

    assert "C.2" in types      # backbone carbonyl
    assert "C.ar" in types     # PHE/TRP ring
    assert "N.am" in types     # backbone amide
    assert "O.co2" in types    # ASP carboxylate
    assert "N.4" in types      # LYS NZ


def test_site_cut_keeps_near_residues_and_drops_far_ones():
    receptor = parse_molecule_file(str(RECEPTOR))
    ligand = parse_molecule_file(str(LIGAND))

    site = extract_binding_site(
        receptor, ligand_centroid(ligand), radius=10.0
    )

    kept = {a.residue_name for a in site.atoms}
    assert kept == INSIDE
    assert not kept & OUTSIDE
    assert len(site.atoms) < len(receptor.atoms)

    # Every kept atom really is inside the radius.
    centroid = ligand_centroid(ligand)
    for atom in site.atoms:
        distance = np.linalg.norm(np.array([atom.x, atom.y, atom.z]) - centroid)
        assert distance <= 10.0 + 1e-6


def test_build_from_files_cuts_the_site_without_being_asked():
    """
    The whole point of the fix: no site file, but the graph is still a site.

    Passing an entire receptor to a model trained on binding sites produces a
    confident number from an out-of-distribution input, so the cut has to happen
    by default rather than on request.
    """
    receptor = parse_molecule_file(str(RECEPTOR))
    builder = HeterogeneousGraphBuilder(GraphConfig())

    graph = builder.build_from_files(str(RECEPTOR), str(LIGAND))

    assert graph["protein"].x.shape[0] < len(receptor.atoms)
    assert graph["protein"].x.shape[0] == len(
        extract_binding_site(
            receptor, ligand_centroid(parse_molecule_file(str(LIGAND))), 10.0
        ).atoms
    )


def test_site_radius_is_honoured_end_to_end():
    builder_small = HeterogeneousGraphBuilder(GraphConfig(site_radius=5.0))
    builder_large = HeterogeneousGraphBuilder(GraphConfig(site_radius=25.0))

    small = builder_small.build_from_files(str(RECEPTOR), str(LIGAND))
    large = builder_large.build_from_files(str(RECEPTOR), str(LIGAND))

    assert small["protein"].x.shape[0] < large["protein"].x.shape[0]
    # 25 A reaches every residue in the fixture.
    assert large["protein"].x.shape[0] == len(
        parse_molecule_file(str(RECEPTOR)).atoms
    )


def test_strip_hydrogens_matches_what_a_sair_model_expects():
    """
    SAIR CIFs carry no hydrogens, so models trained on them need heavy atoms.

    Off by default: ULVSH and PDBbind models were trained with hydrogens
    present, and stripping for those would change their input distribution.
    """
    plain = HeterogeneousGraphBuilder(GraphConfig())
    heavy = HeterogeneousGraphBuilder(GraphConfig(strip_hydrogens=True))

    with_h = plain.build_from_files(str(RECEPTOR), str(LIGAND))
    without_h = heavy.build_from_files(str(RECEPTOR), str(LIGAND))

    assert without_h["ligand"].x.shape[0] == 9
    assert with_h["ligand"].x.shape[0] == 16
    assert without_h["protein"].x.shape[0] < with_h["protein"].x.shape[0]

    receptor = parse_molecule_file(str(RECEPTOR))
    ligand = parse_molecule_file(str(LIGAND))
    expected = drop_hydrogens(
        extract_binding_site(receptor, ligand_centroid(ligand), 10.0)
    )
    assert without_h["protein"].x.shape[0] == len(expected.atoms)


def test_graph_is_connected_and_well_formed():
    """A site with no protein-ligand edges would score on nothing."""
    builder = HeterogeneousGraphBuilder(GraphConfig(strip_hydrogens=True))
    graph = builder.build_from_files(str(RECEPTOR), str(LIGAND))

    edges = graph["protein", "interacts", "ligand"].edge_index
    assert edges.shape[1] > 0
    assert int(edges[0].max()) < graph["protein"].x.shape[0]
    assert int(edges[1].max()) < graph["ligand"].x.shape[0]

    for node in ("protein", "ligand"):
        assert graph[node].x.shape[1] == 56
        assert bool((graph[node].x == graph[node].x).all()), "NaN in node features"
        assert graph[node].pos.shape == (graph[node].x.shape[0], 3)
