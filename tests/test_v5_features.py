"""
Comprehensive unit and integration tests for PandaDock v5.0.0 capabilities:
1. PoseBusters chemistry & geometry validity audit
2. pH-aware protonation / tautomer / stereoisomer state enumeration
3. Differentiable flat-bottom geometric / metal / MCS restraints (CPU + PyTorch gradient parity)
4. Displaceable & bridging structural water analysis
5. Covalent warhead reaction SMARTS & rooted covalent torsion-tree docking
6. Per-target few-shot weight calibration (Ridge LOOCV)
7. One-command Dock-to-MD handoff (OpenMM / GROMACS / PandaInsight)
8. Dual-head PandaDockGNN v5 (predict_pose_rmsd + within_target_ranking_loss)
9. Congeneric series MCS helper & CLI smoke checks
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from rdkit import Chem
from rdkit.Chem import AllChem


def _make_3d_mol(smiles: str, name: str = "test_lig", seed: int = 42) -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    mol.SetProp("_Name", name)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    assert AllChem.EmbedMolecule(mol, params) == 0
    AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
    return mol


def _write_minimal_receptor(path: Path, include_water: bool = False, include_cys: bool = False) -> Path:
    lines = [
        "ATOM      1  N   ALA A  10       0.000   0.000   0.000  1.00 20.00           N",
        "ATOM      2  CA  ALA A  10       1.458   0.000   0.000  1.00 20.00           C",
        "ATOM      3  C   ALA A  10       2.009   1.420   0.000  1.00 20.00           C",
        "ATOM      4  O   ALA A  10       1.251   2.390   0.000  1.00 20.00           O",
        "ATOM      5  CB  ALA A  10       1.986  -0.760  -1.220  1.00 20.00           C",
        "HETATM    6 ZN    ZN A 301       5.000   5.000   5.000  1.00 20.00          ZN",
    ]
    if include_cys:
        lines.extend([
            "ATOM      7  N   CYS A 145       8.000   0.000   0.000  1.00 20.00           N",
            "ATOM      8  CA  CYS A 145       9.450   0.000   0.000  1.00 20.00           C",
            "ATOM      9  CB  CYS A 145      10.000   1.400   0.000  1.00 20.00           C",
            "ATOM     10  SG  CYS A 145      11.800   1.400   0.000  1.00 20.00           S",
        ])
    if include_water:
        lines.append(
            "HETATM   11  O   HOH A 501       1.500   2.500   2.200  1.00 20.00           O"
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. PoseBusters chemistry & geometry audit
# ---------------------------------------------------------------------------

def test_posebusters_passes_clean_conformer_and_catches_clash(tmp_path):
    from pandadock.docking.validation.posebusters import audit_pose

    mol = _make_3d_mol("CC(=O)Oc1ccccc1C(=O)O", "aspirin")
    report = audit_pose(mol, reference_mol=mol)
    assert report.all_passed, f"Clean conformer failed: {report.failed_checks}"
    assert report.n_passed == report.n_total

    # Induce severe internal steric clash by collapsing two non-bonded atoms together
    bad = Chem.Mol(mol)
    conf = bad.GetConformer(0)
    p0 = conf.GetAtomPosition(0)
    conf.SetAtomPosition(5, (p0.x + 0.1, p0.y, p0.z))
    bad_report = audit_pose(bad, reference_mol=mol)
    assert not bad_report.all_passed
    assert "internal_steric_clash" in bad_report.failed_checks or "bond_lengths" in bad_report.failed_checks


# ---------------------------------------------------------------------------
# 2. pH-aware state enumeration
# ---------------------------------------------------------------------------

def test_state_enumeration_ph74_carboxylic_and_amine():
    from pandadock.preprocessing.states import enumerate_ligand_states

    # Ibuprofen carboxylic acid -> deprotonated benzoate (-1) favored at pH 7.4
    ibu = Chem.MolFromSmiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
    states_acid = enumerate_ligand_states(ibu, ph=7.4, enumerate_stereoisomers=False)
    assert len(states_acid) >= 1
    charges = {st.formal_charge for st in states_acid}
    assert -1 in charges

    # Aliphatic secondary amine (Ketamine-like) -> protonated (+1) + neutral microspecies
    ket = Chem.MolFromSmiles("CNC1(c2ccccc2Cl)CCCCC1=O")
    states_base = enumerate_ligand_states(ket, ph=7.4, enumerate_stereoisomers=False)
    assert len(states_base) >= 1
    assert all(st.mol.GetNumConformers() == 1 for st in states_base)
    assert all(st.state_penalty >= 0.0 for st in states_base)


# ---------------------------------------------------------------------------
# 3. Differentiable restraints (CPU analytic gradient + PyTorch parity)
# ---------------------------------------------------------------------------

def test_restraint_analytic_gradient_and_torch_parity(tmp_path):
    from pandadock.docking.search.restraints import (
        DistanceRestraint,
        RestraintSet,
        build_mcs_restraints,
        parse_restraint_spec,
    )

    rec = _write_minimal_receptor(tmp_path / "rec.pdb")
    mol = _make_3d_mol("CC(=O)Nc1nnc(S(N)(=O)=O)s1", "AZM")
    r = parse_restraint_spec("A:ZN:301:ZN<2.2:donor", str(rec), mol, force_constant=10.0)
    rset = RestraintSet([r])

    coords = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
    e0, g0 = rset.energy_and_gradient(coords, need_gradient=True)
    assert e0 > 0.0  # AZM is centered near origin while ZN is at (5,5,5)

    # Finite-difference check on analytic gradient
    eps = 1e-5
    max_err = 0.0
    for idx in r.ligand_indices:
        for ax in range(3):
            c_plus = coords.copy()
            c_minus = coords.copy()
            c_plus[idx, ax] += eps
            c_minus[idx, ax] -= eps
            ep, _ = rset.energy_and_gradient(c_plus, need_gradient=False)
            em, _ = rset.energy_and_gradient(c_minus, need_gradient=False)
            fd = (ep - em) / (2 * eps)
            max_err = max(max_err, abs(fd - g0[idx, ax]))
    assert max_err < 1e-4

    # PyTorch batched parity
    torch = pytest.importorskip("torch")
    coords_t = torch.tensor(coords[None, :, :], dtype=torch.float64)
    et, gt = rset.torch_energy_and_gradient(coords_t, need_gradient=True)
    assert abs(float(et[0].item()) - e0) < 1e-6
    assert np.max(np.abs(gt[0].numpy() - g0)) < 1e-6

    # MCS core tether builder
    ref = _make_3d_mol("c1ccc(CC(=O)O)cc1", "ref")
    analog = _make_3d_mol("Fc1ccc(CC(=O)O)cc1", "analog")
    mcs_rset, smarts, n_atoms = build_mcs_restraints(analog, ref, tolerance=0.5)
    assert n_atoms >= 6
    assert len(mcs_rset) == n_atoms
    assert smarts is not None


# ---------------------------------------------------------------------------
# 4. Structural pocket waters detection & disposition
# ---------------------------------------------------------------------------

def test_pocket_waters_detection_and_disposition(tmp_path):
    from pandadock.docking.search.waters import (
        analyze_pose_water_disposition,
        find_pocket_waters,
        write_receptor_with_waters,
    )

    rec = _write_minimal_receptor(tmp_path / "rec_wat.pdb", include_water=True)
    waters = find_pocket_waters(str(rec), np.array([1.5, 2.0, 2.0]), np.array([10.0, 10.0, 10.0]))
    assert len(waters) == 1
    assert waters[0].receptor_polar_contacts >= 1

    stripped = write_receptor_with_waters(str(rec), set(), str(tmp_path / "stripped.pdb"))
    assert "HOH" not in Path(stripped).read_text()

    # Pose right on top of the water -> classified as displaced
    mol = _make_3d_mol("CCO", "ethanol")
    coords = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float64)
    coords = coords - coords.mean(axis=0) + waters[0].coord
    recs, dg = analyze_pose_water_disposition(coords, mol, waters, kept_water_keys=set())
    assert recs[0]["status"] == "displaced"
    assert dg < 0.0


# ---------------------------------------------------------------------------
# 5. Covalent warheads & rooted covalent docking
# ---------------------------------------------------------------------------

def test_covalent_warhead_detection_and_docking(tmp_path):
    from pandadock.covalent import (
        CovalentDocker,
        CovalentDockingConfig,
        detect_warhead,
        react_ligand_with_warhead,
    )

    # Auto-detect nitrile and Michael acceptor warheads
    nitrile = Chem.MolFromSmiles("CC(C)NC(=O)C(C)C#N")
    assert detect_warhead(nitrile).name == "nitrile"
    acrylamide = Chem.MolFromSmiles("C=CC(=O)Nc1ccccc1")
    assert detect_warhead(acrylamide).name == "michael_acceptor"

    adduct, anc, att, wh = react_ligand_with_warhead(acrylamide, warhead="auto", nucleophile_element="S")
    assert wh.name == "michael_acceptor"
    assert adduct.GetAtomWithIdx(anc).GetSymbol() == "S"
    assert adduct.GetBondBetweenAtoms(anc, att) is not None

    rec = _write_minimal_receptor(tmp_path / "rec_cys.pdb", include_cys=True)
    docker = CovalentDocker(CovalentDockingConfig(exhaustiveness=2, n_starts=4, num_poses=3, seed=42))
    res, adduct_out, meta = docker.dock(str(rec), acrylamide, reactive_residue="A:CYS:145")
    assert len(res.poses) >= 1
    assert meta["warhead"] == "michael_acceptor"
    assert 1.5 <= meta["measured_bond_length_A"] <= 2.1

    complex_pdb = tmp_path / "cov_complex1.pdb"
    CovalentDocker.write_covalent_complex_pdb(str(rec), adduct_out, res.poses[0], meta, str(complex_pdb))
    txt = complex_pdb.read_text()
    assert txt.startswith("LINK ")
    assert "COV L" in txt


# ---------------------------------------------------------------------------
# 6. Few-shot weight calibration
# ---------------------------------------------------------------------------

def test_few_shot_calibration_ridge_loocv(tmp_path):
    from pandadock.docking.calibration import calibrate_scoring_weights, load_calibrated_weights

    rec = _write_minimal_receptor(tmp_path / "rec.pdb")
    sdf_p = tmp_path / "poses.sdf"
    w = Chem.SDWriter(str(sdf_p))
    smiles_list = [
        ("lig1", "c1ccccc1", -5.1),
        ("lig2", "Cc1ccccc1O", -6.3),
        ("lig3", "CCc1ccc(O)cc1", -7.0),
        ("lig4", "CCCc1ccc(OC)cc1", -7.8),
    ]
    labels_p = tmp_path / "labels.csv"
    with open(labels_p, "w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh)
        cw.writerow(["name", "dg"])
        for nm, smi, dg in smiles_list:
            m = _make_3d_mol(smi, nm)
            # Shift near receptor atoms
            conf = m.GetConformer(0)
            for i in range(m.GetNumAtoms()):
                p = conf.GetAtomPosition(i)
                conf.SetAtomPosition(i, (p.x + 3.0, p.y + 3.0, p.z + 3.0))
            w.write(m)
            cw.writerow([nm, dg])
    w.close()

    out_json = tmp_path / "calibrated_weights.json"
    res = calibrate_scoring_weights(str(rec), str(sdf_p), str(labels_p), str(out_json))
    assert res["n_ligands"] == 4
    assert out_json.exists()
    loaded = load_calibrated_weights(str(out_json))
    assert set(("gauss1", "gauss2", "repulsion", "hydrophobic", "hydrogen")).issubset(loaded.keys())
    assert loaded["repulsion"] >= 0.25


# ---------------------------------------------------------------------------
# 7. One-command Dock-to-MD handoff
# ---------------------------------------------------------------------------

def test_md_handoff_package_export(tmp_path):
    from pandadock.md_handoff import export_md_package

    rec = _write_minimal_receptor(tmp_path / "rec.pdb")
    sdf_p = tmp_path / "poses.sdf"
    m = _make_3d_mol("CC(=O)Oc1ccccc1C(=O)O", "aspirin")
    w = Chem.SDWriter(str(sdf_p))
    w.write(m)
    w.close()

    md_dir = tmp_path / "md_out"
    manifest = export_md_package(
        output_dir=str(md_dir),
        receptor_file=str(rec),
        ligand_sdf=str(sdf_p),
        engine="both",
        duration_ns=15.0,
    )
    assert manifest["duration_ns"] == 15.0
    for expected in (
        "receptor.pdb",
        "ligand.sdf",
        "complex.pdb",
        "run_openmm.py",
        "run_gromacs.sh",
        "analyze_with_pandainsight.sh",
        "mdp/md.mdp",
        "md_handoff_manifest.json",
    ):
        assert (md_dir / expected).exists(), f"Missing {expected}"


# ---------------------------------------------------------------------------
# 8. Dual-head PandaDockGNN v5 (predict_pose_rmsd + within_target_ranking_loss)
# ---------------------------------------------------------------------------

def test_gnn_v5_dual_head_pose_rmsd_and_ranking_loss(tmp_path):
    torch = pytest.importorskip("torch")
    HeteroData = pytest.importorskip("torch_geometric.data").HeteroData
    from pandadock.gnn.models.pandadock_gnn import (
        ModelConfig,
        PandaDockGNN,
        within_target_ranking_loss,
    )

    cfg = ModelConfig(
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        affinity_hidden=32,
        predict_activity=True,
        predict_pose_rmsd=True,
    )
    model = PandaDockGNN(cfg)
    g = HeteroData()
    g["protein"].x = torch.randn(8, 56)
    g["protein"].pos = torch.randn(8, 3)
    g["protein"].batch = torch.zeros(8, dtype=torch.long)
    g["ligand"].x = torch.randn(6, 56)
    g["ligand"].pos = torch.randn(6, 3)
    g["ligand"].batch = torch.zeros(6, dtype=torch.long)
    g["protein", "interacts", "ligand"].edge_index = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    g["protein", "interacts", "ligand"].edge_attr = torch.randn(4, 23)

    out = model(g)
    assert "affinity" in out and "pose_rmsd" in out
    assert float(out["pose_rmsd"].item()) >= 0.0

    # Within-target ranking loss decreases when order matches
    pred_good = torch.tensor([5.0, 7.0, 9.0])
    pred_bad = torch.tensor([9.0, 7.0, 5.0])
    target = torch.tensor([5.0, 7.0, 9.0])
    groups = torch.tensor([1, 1, 1])
    assert float(within_target_ranking_loss(pred_good, target, groups)) < float(
        within_target_ranking_loss(pred_bad, target, groups)
    )

    # Save & reload preserves predict_pose_rmsd head
    ckpt = tmp_path / "v5_gnn.pt"
    model.save(str(ckpt))
    reloaded = PandaDockGNN.load(str(ckpt))
    assert reloaded.config.predict_pose_rmsd is True


# ---------------------------------------------------------------------------
# 9. CLI smoke check for all new v5.0 commands
# ---------------------------------------------------------------------------

def test_v5_cli_commands_help_and_version():
    from pandadock import __version__
    from pandadock.docking_cli import main

    assert __version__ == "5.0.0"
    runner = CliRunner()
    for sub in ("dock", "series", "covalent", "validate-poses", "calibrate", "to-md"):
        res = runner.invoke(main, [sub, "--help"])
        assert res.exit_code == 0, f"{sub} --help failed: {res.output}"
