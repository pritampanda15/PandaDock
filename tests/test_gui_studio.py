"""
Unit and API integration tests for PandaDock Studio GUI (`pandadock-gui`).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from rdkit import Chem
from rdkit.Chem import AllChem

from pandadock.gui.server import RunJobRequest, _build_cli_command, create_app


def _make_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    rec = tmp_path / "rec.pdb"
    rec.write_text(
        "\n".join([
            "ATOM      1  N   CYS A 145       0.000   0.000   0.000  1.00 20.00           N",
            "ATOM      2  CA  CYS A 145       1.458   0.000   0.000  1.00 20.00           C",
            "ATOM      3  CB  CYS A 145       2.000   1.400   0.000  1.00 20.00           C",
            "ATOM      4  SG  CYS A 145       3.800   1.400   0.000  1.00 20.00           S",
            "HETATM    5 ZN    ZN A 301       6.000   6.000   6.000  1.00 20.00          ZN",
            "HETATM    6  O   HOH A 501       4.500   5.500   5.500  1.00 20.00           O",
            "END",
        ]) + "\n",
        encoding="utf-8",
    )

    lig = tmp_path / "lig.sdf"
    mol = Chem.AddHs(Chem.MolFromSmiles("C=CC(=O)Nc1ccccc1"))
    mol.SetProp("_Name", "acrylamide_test")
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    w = Chem.SDWriter(str(lig))
    w.write(mol)
    w.close()

    # Create a mock completed run folder with PDF, HTML, PNG, PDB, SDF
    run_dir = tmp_path / "gui_runs" / "20261124_demo_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (run_dir / "report.html").write_text("<html><body>Mock Report</body></html>", encoding="utf-8")
    (run_dir / "pose_scores.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (run_dir / "complex1.pdb").write_text(rec.read_text(), encoding="utf-8")
    (run_dir / "poses.sdf").write_text(lig.read_text(), encoding="utf-8")
    (run_dir / "demo_summary.json").write_text(
        '{"best_pose_energy": -8.42, "ensemble_binding_energy": -7.91, "runtime_seconds": 12.3, '
        '"scoring_function": "vina_grid", "grid_center": [1.0, 2.0, 3.0], "grid_dimensions": [20.0, 20.0, 20.0]}',
        encoding="utf-8",
    )
    return tmp_path, rec, lig


def test_gui_health_index_logo_and_runs(tmp_path):
    ws, rec, lig = _make_workspace(tmp_path)
    app = create_app(workspace_dir=str(ws))
    client = TestClient(app)

    # Index HTML renders Apple-HIG Studio
    r_idx = client.get("/")
    assert r_idx.status_code == 200
    assert "Panda" in r_idx.text and "3Dmol" in r_idx.text

    # Health & Logo
    r_h = client.get("/api/health")
    assert r_h.status_code == 200
    assert r_h.json()["status"] == "ok"
    assert r_h.json()["version"] == "5.0.0"

    r_logo = client.get("/api/logo.png")
    assert r_logo.status_code == 200
    assert r_logo.headers["content-type"].startswith("image/png")

    # Runs discovery & summary
    r_runs = client.get("/api/runs")
    assert r_runs.status_code == 200
    runs = r_runs.json()["runs"]
    assert len(runs) >= 1
    rid = runs[0]["run_id"]

    r_sum = client.get(f"/api/runs/{rid}/summary")
    assert r_sum.status_code == 200
    s = r_sum.json()
    assert s["kpis"]["best_score"] == -8.42
    assert s["pdf_url"].endswith("/report.pdf")
    assert s["html_url"].endswith("/report.html")
    assert len(s["plots"]) == 1

    # Inline PDF & HTML serving
    r_pdf = client.get(s["pdf_url"])
    assert r_pdf.status_code == 200
    assert r_pdf.headers["content-type"] == "application/pdf"
    assert "inline" in r_pdf.headers.get("content-disposition", "")

    r_html = client.get(s["html_url"])
    assert r_html.status_code == 200
    assert "Mock Report" in r_html.text


def test_gui_upload_inspect_and_security(tmp_path):
    ws, rec, lig = _make_workspace(tmp_path)
    app = create_app(workspace_dir=str(ws))
    client = TestClient(app)

    # Upload receptor & ligand
    with open(rec, "rb") as fh:
        up_r = client.post("/api/upload?role=receptor", files={"file": ("rec.pdb", fh, "chemical/x-pdb")})
    assert up_r.status_code == 200
    rec_uploaded = up_r.json()["path"]
    assert up_r.json()["inspection"]["waters_count"] == 1
    assert len(up_r.json()["inspection"]["metals"]) == 1

    with open(lig, "rb") as fh:
        up_l = client.post("/api/upload?role=ligand", files={"file": ("lig.sdf", fh, "chemical/x-mdl-sdfile")})
    assert up_l.status_code == 200
    lig_uploaded = up_l.json()["path"]
    assert up_l.json()["inspection"]["detected_warhead"]["name"] == "michael_acceptor"

    # Inspect endpoint
    r_insp = client.post("/api/inspect", json={"receptor_path": rec_uploaded, "ligand_path": lig_uploaded, "ph": 7.4})
    assert r_insp.status_code == 200
    d = r_insp.json()
    assert d["receptor"]["n_residues"] == 1
    assert d["ligand"]["name"] == "acrylamide_test"

    # Structure text preview works for uploaded files & rejects outside-workspace traversal
    r_txt = client.get("/api/structure-text", params={"path": rec_uploaded})
    assert r_txt.status_code == 200
    assert "CYS A 145" in r_txt.text

    r_bad = client.get("/api/structure-text", params={"path": "/etc/passwd"})
    assert r_bad.status_code == 403


def test_gui_cli_command_builders_all_modes(tmp_path):
    out = tmp_path / "out"
    for mode, extra in [
        ("dock", {"restraints": ["A:ZN:301:ZN<2.2:donor"], "waters": "displace", "enumerate_states": True}),
        ("series", {"core_tolerance": 0.6}),
        ("covalent", {"reactive_residue": "A:CYS:145", "warhead": "auto"}),
        ("metal", {}),
        ("flex", {"flexible_residues": "A:145"}),
        ("validate-poses", {"poses_path": "/tmp/p.sdf"}),
        ("calibrate", {"poses_path": "/tmp/p.sdf", "labels_path": "/tmp/y.csv"}),
        ("to-md", {"poses_path": "/tmp/p.sdf", "md_engine": "both", "duration_ns": 20.0}),
    ]:
        req = RunJobRequest(
            mode=mode,
            receptor_path="/tmp/r.pdb",
            ligand_path="/tmp/l.sdf",
            center=[1.0, 2.0, 3.0],
            box=[20.0, 20.0, 20.0],
            **extra,
        )
        cmd = _build_cli_command(req, out)
        assert isinstance(cmd, list) and len(cmd) >= 4

    # Verify CLI help for both `pandadock gui` and `pandadock-gui`
    from pandadock.docking_cli import main as dock_main
    from pandadock.gui.server import cli as gui_cli

    runner = CliRunner()
    assert runner.invoke(dock_main, ["gui", "--help"]).exit_code == 0
    assert runner.invoke(gui_cli, ["--help"]).exit_code == 0


def test_gui_prepare_and_autogrid_endpoints(tmp_path):
    ws, rec, lig = _make_workspace(tmp_path)
    app = create_app(workspace_dir=str(ws))
    client = TestClient(app)

    # 1. Prepare receptor + ligand (with pH 7.4 state enumeration + minimization)
    r_prep = client.post(
        "/api/prepare",
        json={
            "receptor_path": str(rec),
            "ligand_path": str(lig),
            "ph": 7.4,
            "add_hydrogens": True,
            "minimize_ligand": True,
            "enumerate_states": True,
        },
    )
    assert r_prep.status_code == 200
    pd = r_prep.json()
    assert Path(pd["receptor"]["path"]).exists()
    assert Path(pd["ligand"]["path"]).exists()
    assert pd["ligand"]["n_output_molecules"] >= 1

    # 2. Auto-Gridbox: protein_center, metal, residues
    r_pc = client.post("/api/gridbox", json={"receptor_path": str(rec), "mode": "protein_center", "box_size": 20.0})
    assert r_pc.status_code == 200
    assert len(r_pc.json()["boxes"]) == 1

    r_met = client.post("/api/gridbox", json={"receptor_path": str(rec), "mode": "metal", "box_size": 18.0})
    assert r_met.status_code == 200
    assert r_met.json()["boxes"][0]["center"] == [6.0, 6.0, 6.0]

    r_res = client.post(
        "/api/gridbox",
        json={"receptor_path": str(rec), "mode": "residues", "residues": ["A:CYS:145"], "padding": 5.0},
    )
    assert r_res.status_code == 200
    assert r_res.json()["boxes"][0]["method"] == "residues"


def test_maestro_style_cocrystal_ligand_exclusion(tmp_path):
    from pandadock.gui.server import _strip_het_ligands_from_pdb

    complex_pdb = tmp_path / "complex.pdb"
    complex_pdb.write_text(
        "\n".join([
            "ATOM      1  CA  ALA A  10       0.000   0.000   0.000  1.00 20.00           C",
            "HETATM    2 ZN    ZN A 301       5.000   5.000   5.000  1.00 20.00          ZN",
            "HETATM    3  O   HOH A 501       4.000   5.000   5.000  1.00 20.00           O",
            "HETATM    4  C1  AZM A 701       6.000   6.000   6.000  1.00 20.00           C",
            "HETATM    5  N1  AZM A 701       6.500   6.500   6.500  1.00 20.00           N",
            "END",
        ]) + "\n",
        encoding="utf-8",
    )
    clean_pdb = tmp_path / "clean.pdb"
    _, n_removed = _strip_het_ligands_from_pdb(complex_pdb, ["AZM A:701"], clean_pdb)
    assert n_removed == 2
    txt = clean_pdb.read_text()
    assert "AZM" not in txt
    assert "ALA A  10" in txt
    assert "ZN A 301" in txt
    assert "HOH A 501" in txt


def test_pose_pocket_with_hydrogens_and_interactions(tmp_path):
    ws, rec, lig = _make_workspace(tmp_path)
    app = create_app(workspace_dir=str(ws))
    client = TestClient(app)

    runs = client.get("/api/runs").json()["runs"]
    rid = runs[0]["run_id"]
    r = client.get(f"/api/runs/{rid}/pose/0/pocket?cutoff=8.0")
    assert r.status_code == 200
    data = r.json()
    assert data["pocket_pdb_with_h"] is not None
    assert " H" in data["pocket_pdb_with_h"]  # explicit H added to pocket residues/waters
    assert data["ligand_sdf_with_h"] is not None
    assert " H " in data["ligand_sdf_with_h"]  # explicit H present on ligand
    assert isinstance(data["interactions"], list)
    assert isinstance(data["counts"], dict)


def test_pandalens_static_js_modules_served(tmp_path):
    ws, _, lig = _make_workspace(tmp_path)
    app = create_app(workspace_dir=str(ws))
    client = TestClient(app)

    for fname in ("bridge.js", "vec.js", "chem.js", "structure.js", "interactions.js", "labels.js", "map2d.js"):
        resp = client.get(f"/static/pandalens/{fname}")
        assert resp.status_code == 200, f"Failed to serve {fname}"
        assert "javascript" in resp.headers.get("content-type", "")
        assert len(resp.text) > 100

    # Path traversal blocked
    assert client.get("/static/../../pyproject.toml").status_code in (403, 404)

    # RDKit 2D ligand SVG rendering endpoint
    r2d = client.post("/api/ligand-2d", json={"path": str(lig), "dark": False})
    assert r2d.status_code == 200
    body = r2d.json()
    assert "<svg" in body["svg"]
    assert body["smiles"]
