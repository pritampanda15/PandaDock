"""
Tests for the files a docking run produces.

These cover defects that produce plausible-looking output rather than an error,
which is the category most likely to reach a user unnoticed.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

pytest.importorskip("rdkit", reason="RDKit is required for output tests")
pytest.importorskip("Bio", reason="BioPython is required for output tests")

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from pandadock.docking.core import Pose  # noqa: E402


def build_mol(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    assert AllChem.EmbedMolecule(mol, params) == 0
    return mol


def pose_from(mol):
    coords = np.asarray(mol.GetConformer().GetPositions())
    return Pose(
        coordinates=coords,
        center=coords.mean(axis=0),
        rotation=np.array([0.0, 0.0, 0.0, 1.0]),
        conformer_id=0,
        energy=-9.0,
    )


# ------------------------------------------------------------------ PDB output


@pytest.mark.parametrize(
    "smiles,element",
    [
        ("c1ccc(Br)cc1", "BR"),
        ("c1ccc(Cl)cc1", "CL"),
        ("c1ccc(F)cc1", "F"),
        ("c1ccc(I)cc1", "I"),
        ("c1ccsc1", "S"),
    ],
)
def test_halogen_ligands_write_pose_pdb(tmp_path, smiles, element):
    """
    Two-letter elements must not prevent the pose file from being written.

    BioPython asserts upper-case element symbols, while RDKit returns "Br" and
    "Cl". Passing the symbol through unchanged raised AssertionError inside a
    caught block, so no file was produced while the CLI still reported success.
    Chlorine and bromine are common in drug-like ligands.
    """
    from pandadock.docking.visualization.visualizer import DockingVisualizer

    mol = build_mol(smiles)
    out = tmp_path / "pose.pdb"

    DockingVisualizer().save_pose_pdb(pose_from(mol), out, mol)

    assert out.exists(), f"no pose PDB written for a ligand containing {element}"
    text = out.read_text()
    n_atoms = len([l for l in text.splitlines() if l.startswith(("ATOM", "HETATM"))])
    assert n_atoms == mol.GetNumAtoms()
    assert element in text, f"element {element} missing from the PDB"


def test_pose_pdb_preserves_coordinates(tmp_path):
    from pandadock.docking.visualization.visualizer import DockingVisualizer

    mol = build_mol("c1ccc(Br)cc1O")
    pose = pose_from(mol)
    out = tmp_path / "pose.pdb"
    DockingVisualizer().save_pose_pdb(pose, out, mol)

    written = []
    for line in out.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            written.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))

    assert np.allclose(np.array(written), pose.coordinates, atol=1e-2)


# ------------------------------------------------------------ interaction analysis


def make_receptor_pdb(path, ligand_coords):
    """A few backbone atoms placed within hydrogen-bonding distance."""
    lines = []
    serial = 1
    target = ligand_coords[0]
    for i, (name, element) in enumerate(
        [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]
    ):
        pos = target + np.array([3.0 + i * 0.05, 0.0, 0.0])
        lines.append(
            f"ATOM  {serial:5d}  {name:<3s} VAL A{1:4d}    "
            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00 20.00          {element:>2s}"
        )
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_backbone_hydrogen_bonds_are_detected(tmp_path):
    """
    Main-chain hydrogen bonds must be found.

    Receptor typing previously listed only a handful of sidechain atom names and
    omitted backbone N and O entirely. Kinase hinge binding is predominantly
    backbone-mediated, so a correctly docked kinase inhibitor reported zero
    hydrogen bonds.
    """
    from Bio.PDB import PDBParser

    from pandadock.docking.analysis.interaction_analysis import InteractionAnalyzer

    mol = build_mol("OCCO")
    coords = np.asarray(mol.GetConformer().GetPositions())
    receptor_path = make_receptor_pdb(tmp_path / "rec.pdb", coords)
    receptor = PDBParser(QUIET=True).get_structure("r", str(receptor_path))

    analyzer = InteractionAnalyzer()
    hb_atoms = analyzer._get_receptor_hb_atoms(list(receptor.get_atoms()))
    roles = {info["type"] for info in hb_atoms.values()}

    assert hb_atoms, "no receptor hydrogen bonding atoms identified"
    assert "donor" in roles, "backbone amide N not typed as a donor"
    assert "acceptor" in roles, "backbone carbonyl O not typed as an acceptor"


def test_interaction_analysis_reports_no_fabricated_affinity(tmp_path):
    """
    The analyzer must not emit an invented binding affinity.

    It previously applied hand-chosen weights to miscounted interactions and
    clamped the result to [-15, 5], so it saturated at -15.0 and sat alongside
    the real docking score, inviting confusion between the two.
    """
    from Bio.PDB import PDBParser

    from pandadock.docking.analysis.interaction_analysis import InteractionAnalyzer

    mol = build_mol("OCCO")
    coords = np.asarray(mol.GetConformer().GetPositions())
    receptor_path = make_receptor_pdb(tmp_path / "rec.pdb", coords)
    receptor = PDBParser(QUIET=True).get_structure("r", str(receptor_path))

    analysis = InteractionAnalyzer().analyze_pose_interactions(coords, receptor, mol)

    assert "binding_affinity_estimate" not in analysis
    assert "interaction_types" in analysis
    assert analysis["total_interactions"] == sum(analysis["interaction_types"].values())


# -------------------------------------------------------------------- ensemble


def test_ensemble_energy_stays_on_the_pose_energy_scale():
    """
    Ensemble dG must be interpretable against the pose scores it summarises.

    An unfitted "default calibration" used to remap the value through a
    hand-written piecewise function clamped to [-15, 10]. On a run whose poses
    spanned -16.3 to -14.5 kcal/mol it reported -8.3, outside the range entirely.
    Only the entropy penalty should now separate them.
    """
    from pandadock.docking.scoring.ensemble import BoltzmannEnsemble

    energies = [-16.3, -15.0, -14.9, -14.5]
    poses = []
    for energy in energies:
        pose = Pose(
            coordinates=np.zeros((5, 3)),
            center=np.zeros(3),
            rotation=np.array([0.0, 0.0, 0.0, 1.0]),
            conformer_id=0,
            energy=energy,
        )
        pose.torsion_angles = np.zeros(4)
        poses.append(pose)

    ensemble = BoltzmannEnsemble()
    energy, weights = ensemble.calculate_ensemble_binding_energy(poses)

    assert np.isfinite(energy)
    # Boltzmann free energy is at or below the best pose; the entropy penalty
    # then raises it. The result must stay within a few kcal/mol of the scores.
    assert min(energies) - 2.0 < energy < min(energies) + 8.0
    assert np.isclose(sum(weights), 1.0, atol=1e-6)
    # The best pose must dominate a 1.3 kcal/mol gap at room temperature.
    assert weights[0] > 0.5


def test_entropy_penalty_scales_with_flexibility():
    """
    The penalty must depend on the ligand.

    The rotatable bond count was hardcoded to 5 with a "Placeholder" comment, so
    every ligand received the same correction regardless of flexibility.
    """
    from pandadock.docking.scoring.ensemble import BoltzmannEnsemble

    ensemble = BoltzmannEnsemble()
    pose = Pose(
        coordinates=np.zeros((3, 3)),
        center=np.zeros(3),
        rotation=np.array([0.0, 0.0, 0.0, 1.0]),
        conformer_id=0,
    )

    rigid = ensemble._calculate_entropy_penalty(pose, n_rotatable_bonds=0)
    flexible = ensemble._calculate_entropy_penalty(pose, n_rotatable_bonds=10)

    assert flexible > rigid, "entropy penalty does not depend on rotatable bonds"


# ------------------------------------------------------------------ induced fit


def test_refinement_cost_is_never_negative():
    """
    Induced-fit refinement cost must be a penalty.

    The IFD score is `binding_energy + refinement_cost * refinement_penalty_weight`
    with a positive weight, so a negative cost improves the score. The cost was
    taken from a rotamer energy that combined a clash penalty with a contact
    reward, and with no clashes it was purely the negative reward -- so a pose
    demanding more side-chain rearrangement scored better than one needing none.
    On a real kinase run this turned a -16.2 kcal/mol binding energy into a
    -24.8 kcal/mol IFD score.
    """
    pytest.importorskip("rdkit")
    from pandadock.flex_docking.phases import ReceptorRefiner

    refiner = ReceptorRefiner()
    try:
        ligand = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        cases = {
            "no contact": np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]),
            "favourable": np.array([[3.0, 0.0, 0.0], [3.5, 1.0, 0.0]]),
            "clashing": np.array([[0.3, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        }
        strains = {}
        for name, rotamer in cases.items():
            _, strain = refiner._calculate_rotamer_energy(
                rotamer, ligand, return_strain=True
            )
            assert strain >= 0.0, f"{name}: refinement strain is negative ({strain})"
            strains[name] = strain

        assert strains["clashing"] > strains["no contact"], (
            "a clashing rotamer must carry more strain than a distant one"
        )
    finally:
        refiner.cleanup()


def test_rotamer_selection_still_prefers_favourable_contacts():
    """
    Separating strain from the selection energy must not change which rotamer
    wins: ranking still uses clash penalty plus contact reward.
    """
    pytest.importorskip("rdkit")
    from pandadock.flex_docking.phases import ReceptorRefiner

    refiner = ReceptorRefiner()
    try:
        ligand = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        # 3.5 A from the nearest ligand atom: inside the favourable 2.5-4.0 A
        # window rather than clashing with it.
        distant, _ = refiner._calculate_rotamer_energy(
            np.array([[20.0, 0.0, 0.0]]), ligand, return_strain=True
        )
        contacting, _ = refiner._calculate_rotamer_energy(
            np.array([[1.5, 3.5, 0.0]]), ligand, return_strain=True
        )
        assert contacting < distant, (
            "rotamer ranking no longer rewards favourable contacts"
        )
    finally:
        refiner.cleanup()
