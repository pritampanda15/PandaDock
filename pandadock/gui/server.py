"""
PandaDock Studio — FastAPI Backend Server (v5.0).

Provides a local-first REST + SSE API powering the Apple-HIG PandaDock Studio
web application:
- Structure & ligand auto-inspector (chains, metals, crystal waters, reactive
  nucleophiles, ligand properties, auto warhead detection, pH 7.4 state preview)
- Drag-and-drop file upload and curated workspace presets
- Managed subprocess job runner with live Server-Sent Events (SSE) log streaming
  covering all v5.0 engines: `dock`, `series`, `covalent`, `metal`, `flex`,
  `validate-poses`, `calibrate`, and `to-md`
- Unified run discovery & artifact serving for 3Dmol.js WebGL structures,
  300-DPI plots, embedded A4 vector PDF reports, editorial HTML reports, and
  downloadable complexes/SDFs/MD packages
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from .. import __version__

logger = logging.getLogger("pandadock.gui.server")

STATIC_DIR = Path(__file__).resolve().parent / "static"
LOGO_PATH = Path(__file__).resolve().parent.parent / "visualization" / "assets" / "pandadock_mark.png"
FALLBACK_LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "PandaDock.png"

METAL_RESNAMES = frozenset({
    "ZN", "MG", "MN", "FE", "CA", "CU", "CO", "NI", "CD", "HG", "PT", "PD", "MO", "NA", "K",
})
WATER_RESNAMES = frozenset({"HOH", "WAT", "DOD", "TIP3", "SOL"})
STANDARD_AA = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HID", "HIE", "HIP", "CYX", "MSE",
})
REACTIVE_AA = {"CYS": "SG", "SER": "OG", "THR": "OG1", "LYS": "NZ", "TYR": "OH", "HIS": "NE2"}

PLOT_METADATA: Dict[str, Tuple[str, str]] = {
    "pose_scores.png": ("Scoring Funnel & Boltzmann Weights", "Energetics"),
    "energy_components.png": ("Per-Pose Energy Decomposition", "Energetics"),
    "pose_diversity.png": ("Pairwise RMSD & MDS Landscape", "Conformational"),
    "interaction_fingerprint.png": ("Residue Interaction Fingerprint", "Interactions"),
    "residue_hotspots.png": ("Pocket Residue Hotspots", "Interactions"),
    "ligand_dynamics.png": ("Ligand Atom RMSF & Torsions", "Geometry"),
    "protein_silhouette.png": ("Receptor Architecture & Pocket", "Structure"),
    "binding_affinities.png": ("Binding Affinity Distribution", "Energetics"),
    "pandamap_2d_complex_1.png": ("PandaMap 2D Interaction Diagram", "2D Map"),
    "pandamap_2d_complex_1_3d.png": ("PandaMap 3D Contact Map", "3D Contacts"),
    "series_grid_2d.png": ("2D Congeneric Series & MCS Core Grid", "Series SAR"),
}


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class InspectRequest(BaseModel):
    receptor_path: Optional[str] = None
    ligand_path: Optional[str] = None
    ph: float = 7.4


class Ligand2DRequest(BaseModel):
    path: Optional[str] = None
    text: Optional[str] = None
    format: str = "sdf"
    dark: bool = False
    width: int = 380
    height: int = 260


class PrepareRequest(BaseModel):
    receptor_path: Optional[str] = None
    ligand_path: Optional[str] = None
    ph: float = 7.4
    add_hydrogens: bool = True
    add_charges: bool = True
    minimize_ligand: bool = True
    fix_missing_residues: bool = False
    enumerate_states: bool = False
    strip_waters: bool = False
    exclude_ligands: List[str] = Field(default_factory=list)  # e.g. ["AZM A:701"]


class GridBoxRequest(BaseModel):
    receptor_path: str
    mode: str = "cavities"  # cavities | residues | ligand | metal | protein_center
    ligand_path: Optional[str] = None
    residues: List[str] = Field(default_factory=list)  # e.g. ["A:145", "A:166"]
    padding: float = 5.0
    box_size: float = 22.0
    max_sites: int = 5
    min_volume: float = 50.0


class RunJobRequest(BaseModel):
    mode: str = Field(..., description="dock | series | covalent | metal | flex | validate-poses | calibrate | to-md")
    job_name: Optional[str] = None
    receptor_path: Optional[str] = None
    ligand_path: Optional[str] = None
    poses_path: Optional[str] = None
    labels_path: Optional[str] = None
    reference_pose_path: Optional[str] = None
    weights_file: Optional[str] = None
    exclude_ligands: List[str] = Field(default_factory=list)  # co-crystallized HETATM ligands excluded from receptor grid

    # Grid box
    center: Optional[List[float]] = None
    box: Optional[List[float]] = None
    box_size: float = 22.0

    # Docking options
    scoring: str = "vina"
    exhaustiveness: int = 8
    num_poses: int = 9
    workers: int = 4
    device: str = "cpu"
    seed: int = 42
    rigid_ligand: bool = False
    sample_rings: bool = False
    rescoring: str = "none"
    fast: bool = False

    # v5.0 differentiators
    restraints: List[str] = Field(default_factory=list)
    waters: str = "strip"
    enumerate_states: bool = False
    ph: float = 7.4
    export_md: str = "none"

    # Series
    core_smarts: Optional[str] = None
    core_tolerance: float = 0.75

    # Covalent
    reactive_residue: str = "A:CYS:145"
    warhead: str = "auto"

    # Metal / Flex
    flexible_residues: Optional[str] = None

    # Validate / Calibrate / MD
    max_strain: float = 15.0
    higher_is_better: bool = False
    md_engine: str = "both"
    duration_ns: float = 25.0
    pose_index: int = 0


# ---------------------------------------------------------------------------
# Managed background job state
# ---------------------------------------------------------------------------

@dataclass
class ManagedJob:
    job_id: str
    name: str
    mode: str
    command: List[str]
    output_dir: Path
    status: str = "queued"  # queued | running | completed | failed | cancelled
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    exit_code: Optional[int] = None
    logs: List[str] = field(default_factory=list)
    process: Optional[subprocess.Popen] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def append_log(self, line: str) -> None:
        with self.lock:
            self.logs.append(line.rstrip("\r\n"))
            if len(self.logs) > 4000:
                self.logs = self.logs[-3000:]

    def to_dict(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.job_id,
                "name": self.name,
                "mode": self.mode,
                "command": " ".join(self.command),
                "output_dir": str(self.output_dir),
                "run_id": self.job_id,
                "status": self.status,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "exit_code": self.exit_code,
                "log_lines": len(self.logs),
                "tail": self.logs[-15:],
            }


# ---------------------------------------------------------------------------
# Structure & ligand inspection helpers
# ---------------------------------------------------------------------------

def _strip_het_ligands_from_pdb(
    pdb_path: Path,
    exclude_labels: List[str],
    output_pdb: Path,
) -> Tuple[Path, int]:
    """Remove specified co-crystallized HETATM ligand residue(s) from a complex PDB.

    Preserves all standard protein ATOM records, catalytic metal ions, and
    crystallographic waters (which are managed separately by `--waters`).
    `exclude_labels` accepts labels like `"AZM A:701"` or `"*"` (strip all
    non-metal, non-water organic HETATM ligands).
    """
    strip_all = any(lbl.strip() == "*" for lbl in exclude_labels)
    targets = set()
    for lbl in exclude_labels:
        lbl = lbl.strip()
        if not lbl or lbl == "*":
            continue
        # Parse "RESNAME CHAIN:RESNUM" or "CHAIN:RESNAME:RESNUM"
        parts = lbl.replace(":", " ").split()
        if len(parts) >= 3:
            if parts[0].upper() not in STANDARD_AA and len(parts[1]) == 1:
                targets.add((parts[0].upper(), parts[1], int(parts[2])))
            else:
                targets.add((parts[1].upper(), parts[0], int(parts[2])))
        elif len(parts) == 1:
            targets.add((parts[0].upper(), "*", -1))

    kept_lines: List[str] = []
    removed_atoms = 0
    for line in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True):
        if line.startswith("HETATM") and len(line) >= 26:
            resn = line[17:20].strip().upper()
            chain = line[21:22].strip()
            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                resseq = -999999
            # Never strip catalytic metals or structural waters here
            if resn in METAL_RESNAMES or resn in WATER_RESNAMES:
                kept_lines.append(line)
                continue
            hit = strip_all or any(
                (t_rn == resn and (t_ch == "*" or t_ch == chain) and (t_num == -1 or t_num == resseq))
                for (t_rn, t_ch, t_num) in targets
            )
            if hit:
                removed_atoms += 1
                continue
        kept_lines.append(line)

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text("".join(kept_lines), encoding="utf-8")
    return output_pdb, removed_atoms


def _inspect_receptor(pdb_path: Path) -> Dict[str, Any]:
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("rec", str(pdb_path))
    model = next(struct.get_models())

    chains: List[str] = []
    n_residues = 0
    n_atoms = 0
    all_coords = []
    metals: List[Dict[str, Any]] = []
    waters_count = 0
    het_ligands: Dict[str, List[np.ndarray]] = {}
    reactive_residues: List[Dict[str, Any]] = []

    for chain in model:
        chains.append(chain.id)
        for res in chain:
            rn = res.get_resname().strip().upper()
            atoms = list(res.get_atoms())
            n_atoms += len(atoms)
            if res.id[0] == " ":
                n_residues += 1
                for a in atoms:
                    all_coords.append(a.get_coord())
                if rn in REACTIVE_AA:
                    target_atom = REACTIVE_AA[rn]
                    amap = {a.get_name().strip().upper(): a for a in atoms}
                    if target_atom in amap:
                        c = amap[target_atom].get_coord()
                        reactive_residues.append({
                            "spec": f"{chain.id}:{rn}:{int(res.id[1])}",
                            "atom": target_atom,
                            "coord": [round(float(x), 3) for x in c],
                        })
            elif rn in WATER_RESNAMES:
                waters_count += 1
            elif rn in METAL_RESNAMES:
                for a in atoms:
                    c = a.get_coord()
                    metals.append({
                        "label": f"{rn} {chain.id}:{int(res.id[1])}",
                        "restraint_spec": f"{chain.id}:{rn}:{int(res.id[1])}:{a.get_name().strip().upper()}<2.2:donor",
                        "coord": [round(float(x), 3) for x in c],
                    })
            else:
                key = f"{rn} {chain.id}:{int(res.id[1])}"
                pts = [np.asarray(a.get_coord(), dtype=float) for a in atoms]
                if len(pts) >= 4:
                    het_ligands[key] = pts

    centroid = [0.0, 0.0, 0.0]
    if all_coords:
        arr = np.asarray(all_coords, dtype=float)
        centroid = [round(float(x), 3) for x in arr.mean(axis=0)]

    lig_list = []
    for k, pts in het_ligands.items():
        arr = np.vstack(pts)
        lig_list.append({
            "label": k,
            "n_atoms": len(pts),
            "center": [round(float(x), 3) for x in arr.mean(axis=0)],
        })

    return {
        "filename": pdb_path.name,
        "path": str(pdb_path),
        "n_chains": len(chains),
        "chains": chains,
        "n_residues": n_residues,
        "n_atoms": n_atoms,
        "centroid": centroid,
        "metals": metals,
        "waters_count": waters_count,
        "het_ligands": lig_list[:12],
        "reactive_residues": reactive_residues[:40],
    }


def _inspect_ligand(lig_path: Path, ph: float = 7.4) -> Dict[str, Any]:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    from ..covalent.warheads import detect_warhead
    from ..preprocessing.states import enumerate_ligand_states

    ext = lig_path.suffix.lower()
    mols: List[Chem.Mol] = []
    if ext == ".sdf":
        mols = [m for m in Chem.SDMolSupplier(str(lig_path), removeHs=False) if m is not None]
    elif ext == ".mol2":
        m = Chem.MolFromMol2File(str(lig_path), removeHs=False)
        if m:
            mols = [m]
    elif ext == ".pdb":
        m = Chem.MolFromPDBFile(str(lig_path), removeHs=False)
        if m:
            mols = [m]
    elif ext in (".smi", ".smiles", ".txt"):
        for line in lig_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            m = Chem.MolFromSmiles(parts[0])
            if m:
                if len(parts) > 1:
                    m.SetProp("_Name", parts[1])
                mols.append(m)

    if not mols:
        return {"filename": lig_path.name, "path": str(lig_path), "n_molecules": 0}

    m0 = mols[0]
    name = m0.GetProp("_Name").strip() if m0.HasProp("_Name") and m0.GetProp("_Name").strip() else lig_path.stem
    center = None
    if m0.GetNumConformers() > 0:
        pts = np.asarray(m0.GetConformer(0).GetPositions(), dtype=float)
        if len(pts):
            center = [round(float(x), 3) for x in pts.mean(axis=0)]

    wh = detect_warhead(m0)
    states_preview = []
    try:
        sts = enumerate_ligand_states(m0, ph=ph, max_states=4, enumerate_stereoisomers=False)
        states_preview = [s.to_dict() for s in sts]
    except Exception:
        pass

    return {
        "filename": lig_path.name,
        "path": str(lig_path),
        "n_molecules": len(mols),
        "names": [
            (m.GetProp("_Name").strip() if m.HasProp("_Name") and m.GetProp("_Name").strip() else f"mol_{i+1}")
            for i, m in enumerate(mols[:20])
        ],
        "name": name,
        "smiles": Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(m0))),
        "formula": rdMolDescriptors.CalcMolFormula(m0),
        "mw": round(float(Descriptors.MolWt(m0)), 2),
        "clogp": round(float(Crippen.MolLogP(m0)), 2),
        "tpsa": round(float(rdMolDescriptors.CalcTPSA(m0)), 1),
        "hbd": int(Lipinski.NumHDonors(m0)),
        "hba": int(Lipinski.NumHAcceptors(m0)),
        "rot_bonds": int(Lipinski.NumRotatableBonds(m0)),
        "formal_charge": int(Chem.GetFormalCharge(m0)),
        "center": center,
        "detected_warhead": {"name": wh.name, "description": wh.description} if wh else None,
        "ph_states_preview": states_preview,
    }


# ---------------------------------------------------------------------------
# Run directory discovery & summary parsing
# ---------------------------------------------------------------------------

def _is_result_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    markers = (
        "report.pdf", "series_report.pdf", "report.html", "series_report.html",
        "poses.sdf", "series_poses.sdf", "posebusters_audit.json",
        "calibrated_weights.json", "md_handoff_manifest.json", "covalent_metadata.json",
    )
    if any((d / m).exists() for m in markers):
        return True
    if list(d.glob("*_summary.json")) or list(d.glob("audit_*.json")):
        return True
    return False


def _extract_pose_pocket_with_h_and_interactions(
    run_dir: Path,
    run_summary: Dict[str, Any],
    pose_idx: int = 0,
    cutoff: float = 5.0,
) -> Dict[str, Any]:
    """Build 3D binding-site pocket PDB with explicit H, ligand SDF with H, and 3D interaction vectors."""
    from Bio.PDB import PDBParser
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # 1. Load selected pose ligand molecule (with 3D coords + explicit H)
    lig_mol = None
    poses_sdf = None
    for cand in ("series_poses.sdf", "poses.sdf", "ligand.sdf"):
        if (run_dir / cand).exists():
            poses_sdf = run_dir / cand
            break
    if poses_sdf is not None:
        mols = [m for m in Chem.SDMolSupplier(str(poses_sdf), removeHs=False) if m is not None]
        if mols:
            idx = min(max(0, int(pose_idx)), len(mols) - 1)
            lig_mol = Chem.AddHs(Chem.Mol(mols[idx]), addCoords=True)

    # Locate receptor/complex PDB
    complex_files = sorted(
        [p for p in run_dir.glob("covalent_complex*.pdb")] + [p for p in run_dir.glob("complex*.pdb")],
        key=lambda p: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", p.name)],
    )
    rec_pdb_path: Optional[Path] = None
    if complex_files:
        idx_c = min(max(0, int(pose_idx)), len(complex_files) - 1)
        rec_pdb_path = complex_files[idx_c]
    elif run_summary.get("receptor_path") and Path(run_summary["receptor_path"]).exists():
        rec_pdb_path = Path(run_summary["receptor_path"])

    if rec_pdb_path is None or not rec_pdb_path.exists():
        return {"pocket_residues": [], "interactions": [], "counts": {}, "ligand_sdf_with_h": None, "pocket_pdb_with_h": None}

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("cpx", str(rec_pdb_path))
    model = next(struct.get_models())

    # If no SDF was found, build ligand heavy coordinates from LIG/COV residue in complex PDB
    lig_atoms_info: List[Dict[str, Any]] = []
    if lig_mol is not None and lig_mol.GetNumConformers() > 0:
        conf = lig_mol.GetConformer(0)
        for a in lig_mol.GetAtoms():
            p = conf.GetAtomPosition(a.GetIdx())
            sym = a.GetSymbol().upper()
            lig_atoms_info.append({
                "idx": a.GetIdx(),
                "elem": sym,
                "name": f"{sym}{a.GetIdx() + 1}",
                "coord": np.array([p.x, p.y, p.z], dtype=float),
                "is_h": sym == "H",
                "is_donor": sym in ("N", "O", "S") and a.GetTotalNumHs() > 0,
                "is_acceptor": sym in ("N", "O", "F", "S"),
                "is_halogen": sym in ("F", "CL", "BR", "I"),
                "is_hydrophobic": sym == "C" and not any(n.GetAtomicNum() in (7, 8) for n in a.GetNeighbors()),
                "charge": a.GetFormalCharge(),
            })
    else:
        for chain in model:
            for res in chain:
                if res.get_resname().strip().upper() in ("LIG", "COV", "AZM", "UNL"):
                    for i, atom in enumerate(res.get_atoms()):
                        el = (atom.element or atom.get_name()[0]).strip().upper()
                        lig_atoms_info.append({
                            "idx": i,
                            "elem": el,
                            "name": atom.get_name().strip(),
                            "coord": np.asarray(atom.get_coord(), dtype=float),
                            "is_h": el == "H",
                            "is_donor": el in ("N", "O", "S"),
                            "is_acceptor": el in ("N", "O", "F", "S"),
                            "is_halogen": el in ("F", "CL", "BR", "I"),
                            "is_hydrophobic": el == "C",
                            "charge": 0,
                        })

    lig_heavy = [a for a in lig_atoms_info if not a["is_h"]]
    if not lig_heavy:
        return {"pocket_residues": [], "interactions": [], "counts": {}, "ligand_sdf_with_h": None, "pocket_pdb_with_h": None}
    lig_heavy_xyz = np.vstack([a["coord"] for a in lig_heavy])

    # 2. Identify binding-site protein residues + catalytic metals + bridging waters within `cutoff` Å
    pocket_residues: List[Dict[str, Any]] = []
    pocket_res_objs = []
    pocket_metals = []
    pocket_waters = []
    for chain in model:
        for res in chain:
            rn = res.get_resname().strip().upper()
            if rn in ("LIG", "COV", "UNL"):
                continue
            atoms = list(res.get_atoms())
            if not atoms:
                continue
            pts = np.asarray([a.get_coord() for a in atoms], dtype=float)
            dmat = np.linalg.norm(pts[:, None, :] - lig_heavy_xyz[None, :, :], axis=2)
            min_d = float(np.min(dmat))
            if min_d <= cutoff:
                if rn in METAL_RESNAMES:
                    pocket_metals.append((chain.id, res))
                elif rn in WATER_RESNAMES:
                    pocket_waters.append((chain.id, res))
                elif res.id[0] == " ":
                    ca = next((a.get_coord() for a in atoms if a.get_name().strip() == "CA"), pts.mean(axis=0))
                    pocket_residues.append({
                        "chain": chain.id,
                        "resn": rn,
                        "resi": int(res.id[1]),
                        "label": f"{rn} {chain.id}:{int(res.id[1])}",
                        "min_dist": round(min_d, 2),
                        "ca": [round(float(x), 3) for x in ca],
                    })
                    pocket_res_objs.append((chain.id, res))

    pocket_residues.sort(key=lambda r: r["min_dist"])

    # 3. Build a clean pocket-only PDB block WITH explicit hydrogens (via RDKit AddHs when PDB lacks H)
    pocket_pdb_lines: List[str] = []
    atom_serial = 1
    has_rec_h = any((a.element or "").strip().upper() == "H" for _, res in pocket_res_objs for a in res.get_atoms())

    def _emit_res_lines(chain_id: str, res, rec_type: str = "ATOM  "):
        nonlocal atom_serial
        rn = res.get_resname().strip()[:3]
        rnum = int(res.id[1])
        res_lines = []
        for a in res.get_atoms():
            el = (a.element or a.get_name()[0]).strip().upper()
            an = a.get_name().strip()
            c = a.get_coord()
            name_fmt = (
                f"{an:<4s}" if (len(el) == 2 and an.upper() == el)
                else (f" {an:<3s}" if len(an) < 4 else f"{an[:4]}")
            )
            res_lines.append(
                f"{rec_type}{atom_serial:5d} {name_fmt} {rn:>3s} {chain_id:1s}{rnum:4d}    "
                f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00 20.00          {el:>2s}\n"
            )
            atom_serial += 1
        return res_lines

    for cid, res in pocket_res_objs:
        raw_res_lines = _emit_res_lines(cid, res, "ATOM  ")
        if has_rec_h:
            pocket_pdb_lines.extend(raw_res_lines)
        else:
            # Add accurate 3D hydrogens to this pocket residue using RDKit while preserving exact PDB residue metadata
            try:
                rmol = Chem.MolFromPDBBlock("".join(raw_res_lines) + "END\n", sanitize=False, removeHs=False)
                if rmol is not None:
                    try:
                        Chem.SanitizeMol(rmol)
                    except Exception:
                        rmol.UpdatePropertyCache(strict=False)
                    rmol_h = Chem.AddHs(rmol, addCoords=True)
                    h_block = Chem.MolToPDBBlock(rmol_h)
                    rn = res.get_resname().strip()[:3]
                    rnum = int(res.id[1])
                    # Keep original heavy lines + append newly added H atoms with exact chain/resnum
                    pocket_pdb_lines.extend(raw_res_lines)
                    h_idx = 1
                    for ln in h_block.splitlines(keepends=True):
                        if ln.startswith(("ATOM  ", "HETATM")) and ln[76:78].strip().upper() == "H":
                            hname = f" H{h_idx:<2d}"[:4]
                            pocket_pdb_lines.append(
                                f"ATOM  {atom_serial:5d} {hname} {rn:>3s} {cid:1s}{rnum:4d}    "
                                f"{ln[30:54]}  1.00 20.00           H\n"
                            )
                            atom_serial += 1
                            h_idx += 1
                else:
                    pocket_pdb_lines.extend(raw_res_lines)
            except Exception:
                pocket_pdb_lines.extend(raw_res_lines)

    for cid, res in pocket_metals:
        pocket_pdb_lines.extend(_emit_res_lines(cid, res, "HETATM"))
    for cid, res in pocket_waters:
        # Also add 2 H atoms to pocket waters if missing so water bridges look complete
        w_lines = _emit_res_lines(cid, res, "HETATM")
        pocket_pdb_lines.extend(w_lines)
        if not any((a.element or "").strip().upper() == "H" for a in res.get_atoms()):
            o_coord = next(iter(res.get_atoms())).get_coord()
            rnum = int(res.id[1])
            for hi, delta in enumerate((np.array([0.76, 0.59, 0.0]), np.array([-0.76, 0.59, 0.0])), 1):
                hc = o_coord + delta
                pocket_pdb_lines.append(
                    f"HETATM{atom_serial:5d}  H{hi}  HOH {cid:1s}{rnum:4d}    "
                    f"{hc[0]:8.3f}{hc[1]:8.3f}{hc[2]:8.3f}  1.00 20.00           H\n"
                )
                atom_serial += 1

    pocket_pdb_lines.append("END\n")

    # 4. Compute exact 3D binding-site interaction vectors
    interactions: List[Dict[str, Any]] = []
    seen_pairs = set()

    # Covalent bond highlight (if covalent run)
    cov_meta = run_summary.get("covalent")
    if isinstance(cov_meta, dict) and cov_meta.get("reactive_residue") and lig_mol is not None:
        try:
            att_idx = int(cov_meta.get("attached_atom_idx", 1))
            nuc_xyz = np.asarray(cov_meta["nucleophile_coord"], dtype=float)
            if att_idx < lig_mol.GetNumAtoms():
                p_att = lig_mol.GetConformer(0).GetAtomPosition(att_idx)
                att_xyz = np.array([p_att.x, p_att.y, p_att.z], dtype=float)
                d_cov = float(np.linalg.norm(att_xyz - nuc_xyz))
                interactions.append({
                    "type": "covalent",
                    "color": "#F59E0B",
                    "dashed": False,
                    "start": [round(float(x), 3) for x in nuc_xyz],
                    "end": [round(float(x), 3) for x in att_xyz],
                    "distance": round(d_cov, 2),
                    "residue": cov_meta["reactive_residue"],
                    "rec_atom": cov_meta["reactive_residue"].split(":")[-1],
                    "lig_atom": f"{lig_mol.GetAtomWithIdx(att_idx).GetSymbol()}{att_idx + 1}",
                })
        except Exception:
            pass

    # Metal coordination interactions
    for cid, mres in pocket_metals:
        mrn = mres.get_resname().strip().upper()
        for matom in mres.get_atoms():
            mxyz = np.asarray(matom.get_coord(), dtype=float)
            for la in lig_heavy:
                if la["elem"] in ("N", "O", "S", "P"):
                    d = float(np.linalg.norm(mxyz - la["coord"]))
                    if 1.5 <= d <= 2.9:
                        interactions.append({
                            "type": "metal",
                            "color": "#9333EA",
                            "dashed": True,
                            "start": [round(float(x), 3) for x in mxyz],
                            "end": [round(float(x), 3) for x in la["coord"]],
                            "distance": round(d, 2),
                            "residue": f"{mrn} {cid}:{int(mres.id[1])}",
                            "rec_atom": matom.get_name().strip(),
                            "lig_atom": la["name"],
                        })

    # Polar H-bonds, Salt bridges, Halogen bonds, and Hydrophobic contacts
    neg_res = {"ASP", "GLU"}
    pos_res = {"LYS", "ARG", "HIS", "HIP"}
    for cid, res in pocket_res_objs + pocket_waters:
        rn = res.get_resname().strip().upper()
        r_label = f"{rn} {cid}:{int(res.id[1])}"
        best_hydro: Optional[Tuple[float, Dict[str, Any]]] = None
        for ratom in res.get_atoms():
            rel = (ratom.element or ratom.get_name()[0]).strip().upper()
            if rel == "H":
                continue
            rname = ratom.get_name().strip().upper()
            rxyz = np.asarray(ratom.get_coord(), dtype=float)

            for la in lig_heavy:
                d = float(np.linalg.norm(rxyz - la["coord"]))
                if d > 4.2 or d < 1.3:
                    continue
                key = (r_label, rname, la["name"])
                # Salt bridge
                if (
                    d <= 4.0
                    and ((rn in neg_res and rname in ("OD1", "OD2", "OE1", "OE2") and la["charge"] > 0)
                         or (rn in pos_res and rname in ("NZ", "NH1", "NH2", "NE", "ND1", "NE2") and la["charge"] < 0))
                ):
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        interactions.append({
                            "type": "ionic",
                            "color": "#2563EB",
                            "dashed": True,
                            "start": [round(float(x), 3) for x in rxyz],
                            "end": [round(float(x), 3) for x in la["coord"]],
                            "distance": round(d, 2),
                            "residue": r_label,
                            "rec_atom": rname,
                            "lig_atom": la["name"],
                        })
                    continue
                # Hydrogen bond (N/O/S polar heavy atom pair 2.2 - 3.5 A)
                if 2.2 <= d <= 3.5 and rel in ("N", "O", "S") and la["elem"] in ("N", "O", "S"):
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        interactions.append({
                            "type": "hbond",
                            "color": "#10B981",
                            "dashed": True,
                            "start": [round(float(x), 3) for x in rxyz],
                            "end": [round(float(x), 3) for x in la["coord"]],
                            "distance": round(d, 2),
                            "residue": r_label,
                            "rec_atom": rname,
                            "lig_atom": la["name"],
                        })
                    continue
                # Halogen bond (ligand F/Cl/Br/I to receptor O/N 2.5 - 3.7 A)
                if 2.4 <= d <= 3.7 and la["is_halogen"] and rel in ("O", "N", "S"):
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        interactions.append({
                            "type": "halogen",
                            "color": "#06B6D4",
                            "dashed": True,
                            "start": [round(float(x), 3) for x in rxyz],
                            "end": [round(float(x), 3) for x in la["coord"]],
                            "distance": round(d, 2),
                            "residue": r_label,
                            "rec_atom": rname,
                            "lig_atom": la["name"],
                        })
                    continue
                # Keep at most 1 closest hydrophobic C-C contact per pocket residue to avoid visual clutter
                if d <= 4.0 and rel == "C" and la["is_hydrophobic"] and rn not in WATER_RESNAMES:
                    if best_hydro is None or d < best_hydro[0]:
                        best_hydro = (
                            d,
                            {
                                "type": "hydrophobic",
                                "color": "#D97706",
                                "dashed": True,
                                "start": [round(float(x), 3) for x in rxyz],
                                "end": [round(float(x), 3) for x in la["coord"]],
                                "distance": round(d, 2),
                                "residue": r_label,
                                "rec_atom": rname,
                                "lig_atom": la["name"],
                            },
                        )
        if best_hydro is not None:
            interactions.append(best_hydro[1])

    counts = {
        t: sum(1 for it in interactions if it["type"] == t)
        for t in ("covalent", "metal", "hbond", "ionic", "halogen", "hydrophobic")
    }
    lig_sdf_block = (Chem.MolToMolBlock(lig_mol) + "\n$$$$\n") if lig_mol is not None else None
    return {
        "pose_index": int(pose_idx),
        "cutoff_A": float(cutoff),
        "pocket_residues": pocket_residues,
        "pocket_pdb_with_h": "".join(pocket_pdb_lines),
        "ligand_sdf_with_h": lig_sdf_block,
        "interactions": interactions,
        "counts": counts,
    }


def _discover_runs(roots: List[Tuple[str, Path]]) -> Dict[str, Path]:
    """Return mapping `run_id -> Path` for all discoverable run folders."""
    found: Dict[str, Path] = {}
    for prefix, root in roots:
        if not root.exists() or not root.is_dir():
            continue
        # Walk up to depth 2 to discover nested dirs like 02_restraints_3hs4_zn/zn_restrained
        for item in sorted(root.iterdir()):
            if not item.is_dir() or item.name.startswith(".") or item.name == "inputs":
                continue
            if _is_result_dir(item):
                rid = f"{prefix}__{item.name}" if prefix else item.name
                found[rid] = item
            else:
                for sub in sorted(item.iterdir()):
                    if sub.is_dir() and _is_result_dir(sub):
                        rid = f"{prefix}__{item.name}__{sub.name}" if prefix else f"{item.name}__{sub.name}"
                        found[rid] = sub
    return found


def _summarize_run_dir(run_id: str, run_dir: Path, active_job: Optional[ManagedJob] = None) -> Dict[str, Any]:
    """Extract unified summary, poses, plots, reports, and downloadable artifacts for a run folder."""
    # Detect mode and read summary files
    summary_data: Dict[str, Any] = {}
    for sp in sorted(run_dir.glob("*_summary.json")):
        try:
            summary_data = json.loads(sp.read_text(encoding="utf-8"))
            break
        except Exception:
            pass

    series_data: Optional[Dict[str, Any]] = None
    if (run_dir / "series_results.json").exists():
        try:
            series_data = json.loads((run_dir / "series_results.json").read_text(encoding="utf-8"))
        except Exception:
            pass

    cov_meta: Optional[Dict[str, Any]] = None
    if (run_dir / "covalent_metadata.json").exists():
        try:
            cov_meta = json.loads((run_dir / "covalent_metadata.json").read_text(encoding="utf-8"))
        except Exception:
            pass

    pb_data: List[Dict[str, Any]] = []
    if (run_dir / "posebusters_audit.json").exists():
        try:
            pb_data = json.loads((run_dir / "posebusters_audit.json").read_text(encoding="utf-8"))
        except Exception:
            pass
    else:
        for ap in sorted(run_dir.glob("audit_*.json")):
            try:
                pb_data.extend(json.loads(ap.read_text(encoding="utf-8")))
            except Exception:
                pass

    calib_data: Optional[Dict[str, Any]] = None
    if (run_dir / "calibrated_weights.json").exists():
        try:
            calib_data = json.loads((run_dir / "calibrated_weights.json").read_text(encoding="utf-8"))
        except Exception:
            pass

    md_manifest: Optional[Dict[str, Any]] = None
    if (run_dir / "md_handoff_manifest.json").exists():
        try:
            md_manifest = json.loads((run_dir / "md_handoff_manifest.json").read_text(encoding="utf-8"))
        except Exception:
            pass

    states_data: Optional[Dict[str, Any]] = None
    if (run_dir / "ligand_states_summary.json").exists():
        try:
            states_data = json.loads((run_dir / "ligand_states_summary.json").read_text(encoding="utf-8"))
        except Exception:
            pass

    # Determine run mode label
    if series_data is not None:
        mode = "series"
    elif cov_meta is not None:
        mode = "covalent"
    elif md_manifest is not None:
        mode = "to-md"
    elif calib_data is not None and not summary_data:
        mode = "calibrate"
    elif pb_data and not summary_data and not (run_dir / "poses.sdf").exists():
        mode = "validate-poses"
    elif summary_data.get("parameters", {}).get("water_mode") == "displace":
        mode = "waters"
    elif summary_data.get("parameters", {}).get("restraints"):
        mode = "restraints"
    elif states_data is not None:
        mode = "states"
    else:
        mode = active_job.mode if active_job else "dock"

    # Reports
    pdf_file = next((p.name for p in (run_dir / "series_report.pdf", run_dir / "report.pdf") if p.exists()), None)
    html_file = next((p.name for p in (run_dir / "series_report.html", run_dir / "report.html") if p.exists()), None)

    # Plots
    plots = []
    for png in sorted(run_dir.glob("*.png")):
        if png.name.endswith("_classic.png"):
            continue
        title, category = PLOT_METADATA.get(
            png.name,
            (png.stem.replace("_", " ").title(), "Diagnostic"),
        )
        plots.append({
            "filename": png.name,
            "title": title,
            "category": category,
            "url": f"/api/runs/{run_id}/file/{png.name}",
        })

    # 3D Structures (receptor + poses)
    complex_pdbs = sorted(
        [p.name for p in run_dir.glob("covalent_complex*.pdb")]
        + [p.name for p in run_dir.glob("complex*.pdb")],
        key=lambda s: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)],
    )
    # Deduplicate preserving order
    seen_c = set()
    complex_pdbs = [c for c in complex_pdbs if not (c in seen_c or seen_c.add(c))]

    pose_pdbs = sorted(
        [p.name for p in run_dir.glob("pose*.pdb")],
        key=lambda s: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)],
    )
    poses_sdf = "series_poses.sdf" if (run_dir / "series_poses.sdf").exists() else (
        "poses.sdf" if (run_dir / "poses.sdf").exists() else (
            "ligand.sdf" if (run_dir / "ligand.sdf").exists() else None
        )
    )

    receptor_path = (
        (str(run_dir / "receptor.pdb") if (run_dir / "receptor.pdb").exists() else None)
        or summary_data.get("receptor_file")
        or (series_data.get("receptor") if series_data else None)
        or (md_manifest.get("receptor_source") if md_manifest else None)
    )

    # Pose table
    poses_rows: List[Dict[str, Any]] = []
    for pp in sorted(run_dir.glob("*_poses.json")):
        try:
            raw_poses = json.loads(pp.read_text(encoding="utf-8"))
            if isinstance(raw_poses, dict) and "poses" in raw_poses:
                raw_poses = raw_poses["poses"]
            if isinstance(raw_poses, list):
                best_e = float(raw_poses[0].get("energy", 0.0)) if raw_poses else 0.0
                for i, p in enumerate(raw_poses):
                    e = float(p.get("energy", 0.0))
                    pb_entry = pb_data[i] if i < len(pb_data) else None
                    poses_rows.append({
                        "rank": i + 1,
                        "energy": round(e, 3),
                        "delta_e": round(e - best_e, 3),
                        "confidence": round(float(p.get("confidence", 0.0)), 3),
                        "internal_strain": round(float(p.get("internal_strain", 0.0)), 3),
                        "pb_valid": pb_entry.get("all_passed") if pb_entry else None,
                        "pb_passed": f"{pb_entry['n_passed']}/{pb_entry['n_total']}" if pb_entry else None,
                        "complex_pdb": complex_pdbs[i] if i < len(complex_pdbs) else None,
                        "pose_pdb": pose_pdbs[i] if i < len(pose_pdbs) else None,
                    })
                break
        except Exception:
            pass

    # If covalent or series didn't have *_poses.json, extract pose rows from SDF properties
    if not poses_rows and poses_sdf and (run_dir / poses_sdf).exists():
        try:
            from rdkit import Chem

            sup = Chem.SDMolSupplier(str(run_dir / poses_sdf), removeHs=False)
            mols = [m for m in sup if m is not None]
            for i, m in enumerate(mols):
                e_str = (
                    m.GetProp("score_kcal_per_mol") if m.HasProp("score_kcal_per_mol")
                    else (m.GetProp("Vina_kcal") if m.HasProp("Vina_kcal") else "0")
                )
                pb_entry = pb_data[i] if i < len(pb_data) else None
                poses_rows.append({
                    "rank": i + 1,
                    "name": m.GetProp("_Name") if m.HasProp("_Name") else f"Pose {i+1}",
                    "energy": round(float(e_str), 3),
                    "delta_e": 0.0,
                    "confidence": round(float(m.GetProp("confidence")), 3) if m.HasProp("confidence") else 1.0,
                    "internal_strain": 0.0,
                    "pb_valid": pb_entry.get("all_passed") if pb_entry else None,
                    "pb_passed": f"{pb_entry['n_passed']}/{pb_entry['n_total']}" if pb_entry else None,
                    "complex_pdb": complex_pdbs[i] if i < len(complex_pdbs) else None,
                    "pose_pdb": pose_pdbs[i] if i < len(pose_pdbs) else None,
                })
            if poses_rows:
                e0 = poses_rows[0]["energy"]
                for row in poses_rows:
                    row["delta_e"] = round(row["energy"] - e0, 3)
        except Exception:
            pass

    # Downloadable files list
    artifacts = []
    for f in sorted(run_dir.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            rel = f.relative_to(run_dir).as_posix()
            artifacts.append({
                "name": rel,
                "size_kb": round(f.stat().st_size / 1024.0, 1),
                "url": f"/api/runs/{run_id}/file/{rel}",
            })

    # Headline KPIs
    best_score = None
    if series_data and series_data.get("records"):
        best_score = series_data["records"][0].get("vina_kcal")
    elif summary_data.get("best_pose_energy") is not None:
        best_score = round(float(summary_data["best_pose_energy"]), 3)
    elif poses_rows:
        best_score = poses_rows[0]["energy"]

    pb_pass = sum(1 for r in pb_data if r.get("all_passed"))
    pb_total = len(pb_data)
    if series_data is not None and pb_total == 0:
        pb_pass = int(series_data.get("posebusters_pass_count", 0))
        pb_total = int(series_data.get("n_analogs", 0))

    display_title = (
        (active_job.name if active_job else None)
        or run_id.replace("validation__", "").replace("gui__", "").replace("__", " / ").replace("_", " ")
    )

    return {
        "run_id": run_id,
        "title": display_title,
        "mode": mode,
        "path": str(run_dir),
        "updated_at": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds"),
        "status": active_job.status if active_job else "completed",
        "kpis": {
            "best_score": best_score,
            "ensemble_dg": round(float(summary_data["ensemble_binding_energy"]), 3)
            if summary_data.get("ensemble_binding_energy") is not None else None,
            "n_poses": len(poses_rows) or (series_data.get("n_analogs") if series_data else 0),
            "pb_pass": pb_pass,
            "pb_total": pb_total,
            "runtime_s": round(
                float(summary_data.get("runtime_seconds") or (series_data.get("runtime_seconds") if series_data else 0.0)),
                2,
            ),
            "scoring": summary_data.get("scoring_function") or ("consensus" if series_data else mode),
        },
        "grid_center": summary_data.get("grid_center"),
        "grid_dimensions": summary_data.get("grid_dimensions"),
        "receptor_path": receptor_path,
        "has_complex_pdb": bool(complex_pdbs),
        "complex_pdbs": [f"/api/runs/{run_id}/file/{c}" for c in complex_pdbs],
        "poses_sdf_url": f"/api/runs/{run_id}/file/{poses_sdf}" if poses_sdf else None,
        "pdf_url": f"/api/runs/{run_id}/file/{pdf_file}" if pdf_file else None,
        "html_url": f"/api/runs/{run_id}/file/{html_file}" if html_file else None,
        "plots": plots,
        "poses": poses_rows,
        "series": series_data,
        "covalent": cov_meta,
        "water_analysis": summary_data.get("parameters", {}).get("water_analysis"),
        "restraints": summary_data.get("parameters", {}).get("restraints"),
        "states": states_data,
        "posebusters": pb_data,
        "calibration": calib_data,
        "md_handoff": md_manifest,
        "artifacts": artifacts,
    }


# ---------------------------------------------------------------------------
# CLI builder for job execution
# ---------------------------------------------------------------------------

def _build_cli_command(req: RunJobRequest, out_dir: Path) -> List[str]:
    mode = req.mode.lower().strip()
    if mode == "dock":
        if not req.receptor_path or not req.ligand_path:
            raise ValueError("Docking requires both receptor and ligand files")
        if not req.center or len(req.center) != 3:
            raise ValueError("Docking requires grid center [X, Y, Z]")
        box = req.box if (req.box and len(req.box) == 3) else [req.box_size] * 3
        cmd = [
            "pandadock", "dock",
            "-r", req.receptor_path,
            "-l", req.ligand_path,
            "--center", *(f"{v:.3f}" for v in req.center),
            "--box", *(f"{v:.3f}" for v in box),
            "--scoring", req.scoring,
            "--exhaustiveness", str(req.exhaustiveness),
            "--num-poses", str(req.num_poses),
            "--workers", str(req.workers),
            "--device", req.device,
            "--seed", str(req.seed),
            "--rescoring", req.rescoring,
            "--waters", req.waters,
            "--export-md", req.export_md,
            "-o", str(out_dir),
        ]
        if req.rigid_ligand:
            cmd.append("--rigid-ligand")
        if req.sample_rings:
            cmd.append("--sample-rings")
        if req.fast:
            cmd.append("--fast")
        if req.enumerate_states:
            cmd.extend(["--enumerate-states", "--ph", f"{req.ph:.2f}"])
        if req.weights_file:
            cmd.extend(["--weights-file", req.weights_file])
        for spec in req.restraints:
            if spec.strip():
                cmd.extend(["--restraint", spec.strip()])
        return cmd

    if mode == "series":
        if not req.receptor_path or not req.ligand_path:
            raise ValueError("Series docking requires receptor PDB and multi-ligand SDF/SMI")
        if not req.center or len(req.center) != 3:
            raise ValueError("Series docking requires grid center [X, Y, Z]")
        box = req.box if (req.box and len(req.box) == 3) else [req.box_size] * 3
        cmd = [
            "pandadock", "series",
            "-r", req.receptor_path,
            "-l", req.ligand_path,
            "--center", *(f"{v:.3f}" for v in req.center),
            "--box", *(f"{v:.3f}" for v in box),
            "--scoring", req.scoring if req.scoring in ("vina", "vinardo") else "vina",
            "--exhaustiveness", str(req.exhaustiveness),
            "--workers", str(req.workers),
            "--device", req.device,
            "--seed", str(req.seed),
            "--core-tolerance", f"{req.core_tolerance:.2f}",
            "-o", str(out_dir),
        ]
        if req.reference_pose_path:
            cmd.extend(["--reference-pose", req.reference_pose_path])
        if req.core_smarts:
            cmd.extend(["--core-smarts", req.core_smarts])
        if req.weights_file:
            cmd.extend(["--weights-file", req.weights_file])
        return cmd

    if mode == "covalent":
        if not req.receptor_path or not req.ligand_path:
            raise ValueError("Covalent docking requires receptor PDB and ligand file")
        cmd = [
            "pandadock", "covalent",
            "-r", req.receptor_path,
            "-l", req.ligand_path,
            "--residue", req.reactive_residue,
            "--warhead", req.warhead,
            "--box-size", f"{req.box_size:.1f}",
            "--scoring", req.scoring if req.scoring in ("vina", "vinardo") else "vina",
            "--exhaustiveness", str(req.exhaustiveness),
            "--num-poses", str(req.num_poses),
            "--seed", str(req.seed),
            "-o", str(out_dir),
        ]
        if req.center and len(req.center) == 3:
            cmd.extend(["--center", *(f"{v:.3f}" for v in req.center)])
        return cmd

    if mode == "metal":
        if not req.receptor_path or not req.ligand_path:
            raise ValueError("Metal docking requires receptor PDB and ligand file")
        if not req.center or len(req.center) != 3:
            raise ValueError("Metal docking requires grid center [X, Y, Z]")
        box = req.box if (req.box and len(req.box) == 3) else [req.box_size] * 3
        return [
            "pandadock-metal", "dock",
            "-r", req.receptor_path,
            "-l", req.ligand_path,
            "--center", *(f"{v:.3f}" for v in req.center),
            "--box", *(f"{v:.3f}" for v in box),
            "--num-poses", str(req.num_poses),
            "-o", str(out_dir),
        ]

    if mode == "flex":
        if not req.receptor_path or not req.ligand_path:
            raise ValueError("Flexible docking requires receptor PDB and ligand file")
        if not req.center or len(req.center) != 3:
            raise ValueError("Flexible docking requires grid center [X, Y, Z]")
        box = req.box if (req.box and len(req.box) == 3) else [req.box_size] * 3
        cmd = [
            "pandadock-flex",
            "-r", req.receptor_path,
            "-l", req.ligand_path,
            "--center", *(f"{v:.3f}" for v in req.center),
            "--box", *(f"{v:.3f}" for v in box),
            "--num-poses", str(req.num_poses),
            "-o", str(out_dir),
        ]
        if req.flexible_residues:
            cmd.extend(["--flexible-residues", req.flexible_residues])
        return cmd

    if mode == "validate-poses":
        poses_in = req.poses_path or req.ligand_path
        if not poses_in:
            raise ValueError("PoseBusters validation requires a poses SDF file")
        cmd = [
            "pandadock", "validate-poses",
            "-p", poses_in,
            "--max-strain", f"{req.max_strain:.1f}",
            "-o", str(out_dir / "posebusters_audit.json"),
        ]
        if req.receptor_path:
            cmd.extend(["-r", req.receptor_path])
        return cmd

    if mode == "calibrate":
        poses_in = req.poses_path or req.ligand_path
        if not req.receptor_path or not poses_in or not req.labels_path:
            raise ValueError("Calibration requires receptor PDB, multi-ligand poses SDF, and labels CSV")
        cmd = [
            "pandadock", "calibrate",
            "-r", req.receptor_path,
            "-p", poses_in,
            "-y", req.labels_path,
            "--preset", req.scoring if req.scoring in ("vina", "vinardo") else "vina",
            "-o", str(out_dir / "calibrated_weights.json"),
        ]
        if req.higher_is_better:
            cmd.append("--higher-is-better")
        return cmd

    if mode == "to-md":
        poses_in = req.poses_path or req.ligand_path
        if not req.receptor_path or not poses_in:
            raise ValueError("MD Handoff requires receptor PDB and docked poses SDF")
        return [
            "pandadock", "to-md",
            "-r", req.receptor_path,
            "-p", poses_in,
            "--pose-index", str(req.pose_index),
            "--engine", req.md_engine,
            "--duration-ns", f"{req.duration_ns:.1f}",
            "-o", str(out_dir),
        ]

    raise ValueError(f"Unsupported mode: {mode}")


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------

def create_app(workspace_dir: Optional[str] = None) -> FastAPI:
    workspace = Path(workspace_dir or os.getcwd()).resolve()
    gui_runs_root = workspace / "gui_runs"
    uploads_root = gui_runs_root / "_uploads"
    prepared_root = gui_runs_root / "_prepared"
    gui_runs_root.mkdir(parents=True, exist_ok=True)
    uploads_root.mkdir(parents=True, exist_ok=True)
    prepared_root.mkdir(parents=True, exist_ok=True)

    validation_root = workspace / "test" / "v5_validation"
    allowed_roots = [workspace, gui_runs_root, uploads_root, prepared_root]

    jobs: Dict[str, ManagedJob] = {}

    app = FastAPI(
        title="PandaDock Studio",
        version=__version__,
        description="Apple-HIG Molecular Docking, 3D Visualization, Analytics & Publication Report Suite",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _safe_resolve(raw_path: str) -> Path:
        p = Path(raw_path).expanduser().resolve()
        for root in allowed_roots:
            try:
                p.relative_to(root)
                return p
            except ValueError:
                continue
        raise HTTPException(status_code=403, detail=f"Path outside allowed workspace: {raw_path}")

    def _all_run_dirs() -> Dict[str, Path]:
        roots: List[Tuple[str, Path]] = [("gui", gui_runs_root)]
        if validation_root.exists():
            roots.append(("validation", validation_root))
        found = _discover_runs(roots)
        # Also include currently running GUI jobs whose output dir is newly created
        for jid, job in jobs.items():
            found.setdefault(f"gui__{job.output_dir.name}", job.output_dir)
        return found

    @app.get("/", response_class=HTMLResponse)
    def index():
        html_path = STATIC_DIR / "index.html"
        if not html_path.exists():
            return HTMLResponse("<h1>PandaDock Studio UI missing</h1>", status_code=500)
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/static/{filepath:path}")
    def get_static_asset(filepath: str):
        target = (STATIC_DIR / filepath).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Invalid static asset path")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Static asset not found")
        suffix = target.suffix.lower()
        if suffix in (".js", ".mjs"):
            media_type = "application/javascript; charset=utf-8"
        elif suffix == ".css":
            media_type = "text/css; charset=utf-8"
        else:
            media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        return FileResponse(str(target), media_type=media_type)

    @app.get("/api/logo.png")
    def get_logo():
        for cand in (LOGO_PATH, FALLBACK_LOGO_PATH):
            if cand.exists():
                return FileResponse(str(cand), media_type="image/png")
        raise HTTPException(status_code=404, detail="Logo not found")

    @app.get("/api/health")
    def health():
        devices = ["cpu"]
        try:
            import torch

            if torch.cuda.is_available():
                devices.append("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                devices.append("mps")
        except Exception:
            pass
        return {
            "status": "ok",
            "version": __version__,
            "workspace": str(workspace),
            "devices": devices,
            "cpu_count": os.cpu_count() or 4,
        }

    @app.get("/api/presets")
    def list_presets():
        presets = []
        t_dir = workspace / "test"
        v_inp = validation_root / "inputs"

        if (t_dir / "6X3V.pdb").exists() and (t_dir / "reference_ligands.sdf").exists():
            presets.append({
                "id": "6x3v_series",
                "title": "6X3V GABA_A · Congeneric Series (5 Analogs)",
                "badge": "Series SAR",
                "mode": "series",
                "receptor_path": str(t_dir / "6X3V.pdb"),
                "ligand_path": str(t_dir / "reference_ligands.sdf"),
                "center": [126.504, 119.182, 146.110],
                "box": [21.782, 21.782, 21.782],
                "scoring": "vina",
            })
        if (t_dir / "6X3V.pdb").exists() and (v_inp / "Ketamine.sdf").exists():
            presets.append({
                "id": "6x3v_ketamine_ph",
                "title": "6X3V · Ketamine (pH 7.4 State Enumeration)",
                "badge": "pH 7.4 States",
                "mode": "dock",
                "receptor_path": str(t_dir / "6X3V.pdb"),
                "ligand_path": str(v_inp / "Ketamine.sdf"),
                "center": [126.504, 119.182, 146.110],
                "box": [21.782, 21.782, 21.782],
                "enumerate_states": True,
                "ph": 7.4,
            })
        if (v_inp / "3HS4_rec_zn_waters.pdb").exists() and (v_inp / "AZM.sdf").exists():
            presets.append({
                "id": "3hs4_zn_waters",
                "title": "3HS4 Carbonic Anhydrase II · Zn + Waters + AZM",
                "badge": "Metal + Waters",
                "mode": "dock",
                "receptor_path": str(v_inp / "3HS4_rec_zn_waters.pdb"),
                "ligand_path": str(v_inp / "AZM.sdf"),
                "center": [-4.221, -4.077, 17.036],
                "box": [18.0, 18.0, 18.0],
                "restraints": ["A:ZN:301:ZN<2.2:donor"],
                "waters": "displace",
            })
        if (v_inp / "6LU7_mpro.pdb").exists() and (v_inp / "nirmatrelvir.smi").exists():
            presets.append({
                "id": "6lu7_covalent",
                "title": "6LU7 SARS-CoV-2 Mpro · Nirmatrelvir (CYS A:145)",
                "badge": "Covalent",
                "mode": "covalent",
                "receptor_path": str(v_inp / "6LU7_mpro.pdb"),
                "ligand_path": str(v_inp / "nirmatrelvir.smi"),
                "reactive_residue": "A:CYS:145",
                "warhead": "auto",
                "box_size": 22.0,
            })
        return {"presets": presets}

    @app.post("/api/upload")
    async def upload_file(file: UploadFile = File(...), role: str = Query("generic")):
        orig_name = Path(file.filename or "upload.dat").name
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", orig_name)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = uploads_root / f"{stamp}_{uuid.uuid4().hex[:6]}_{safe_name}"
        content = await file.read()
        dest.write_bytes(content)

        info: Dict[str, Any] = {"path": str(dest), "filename": orig_name, "size_kb": round(len(content) / 1024.0, 1)}
        try:
            if role == "receptor" or dest.suffix.lower() in (".pdb", ".ent"):
                info["inspection"] = _inspect_receptor(dest)
            elif role in ("ligand", "poses") or dest.suffix.lower() in (".sdf", ".mol2", ".smi", ".smiles"):
                info["inspection"] = _inspect_ligand(dest)
        except Exception as exc:
            info["inspection_warning"] = str(exc)
        return info

    @app.post("/api/inspect")
    def inspect_inputs(req: InspectRequest):
        out: Dict[str, Any] = {}
        if req.receptor_path:
            rp = _safe_resolve(req.receptor_path)
            if not rp.exists():
                raise HTTPException(status_code=404, detail="Receptor file not found")
            out["receptor"] = _inspect_receptor(rp)
        if req.ligand_path:
            lp = _safe_resolve(req.ligand_path)
            if not lp.exists():
                raise HTTPException(status_code=404, detail="Ligand file not found")
            out["ligand"] = _inspect_ligand(lp, ph=req.ph)
        return out

    @app.post("/api/ligand-2d")
    def render_ligand_2d_svg(req: Ligand2DRequest):
        """Render a crisp 2D chemical depiction SVG via RDKit for any SDF/MOL2/PDB/SMILES ligand."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import rdMolDraw2D

        mol = None
        if req.text:
            fmt = (req.format or "sdf").lower()
            if fmt in ("sdf", "mol"):
                mol = Chem.MolFromMolBlock(req.text, sanitize=True, removeHs=True)
                if mol is None:
                    mol = Chem.MolFromMolBlock(req.text, sanitize=False, removeHs=True)
            elif fmt == "pdb":
                mol = Chem.MolFromPDBBlock(req.text, sanitize=True, removeHs=True)
            elif fmt == "mol2":
                mol = Chem.MolFromMol2Block(req.text, sanitize=True, removeHs=True)
            elif fmt in ("smi", "smiles"):
                first = next(
                    (ln.strip().split()[0] for ln in req.text.splitlines() if ln.strip() and not ln.startswith("#")),
                    "",
                )
                if first:
                    mol = Chem.MolFromSmiles(first)
        elif req.path:
            lp = _safe_resolve(req.path)
            if not lp.exists():
                raise HTTPException(status_code=404, detail=f"Ligand not found: {req.path}")
            ext = lp.suffix.lower()
            if ext in (".sdf", ".mol"):
                supp = Chem.SDMolSupplier(str(lp), removeHs=True, sanitize=True)
                mol = next((m for m in supp if m is not None), None)
            elif ext == ".mol2":
                mol = Chem.MolFromMol2File(str(lp), removeHs=True)
            elif ext == ".pdb":
                mol = Chem.MolFromPDBFile(str(lp), removeHs=True)
            elif ext in (".smi", ".smiles", ".txt"):
                first = next(
                    (ln.strip().split()[0] for ln in lp.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip() and not ln.startswith("#")),
                    "",
                )
                if first:
                    mol = Chem.MolFromSmiles(first)

        if mol is None:
            raise HTTPException(status_code=400, detail="Could not parse ligand into a 2D RDKit molecule")

        mol_2d = Chem.RemoveHs(mol)
        AllChem.Compute2DCoords(mol_2d)
        try:
            rdMolDraw2D.PrepareMolForDrawing(mol_2d, kekulize=True)
        except Exception:
            pass

        w = max(200, min(900, int(req.width)))
        h = max(160, min(700, int(req.height)))
        drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
        opts = drawer.drawOptions()
        opts.bondLineWidth = 1.9
        opts.padding = 0.12
        opts.clearBackground = True
        if req.dark:
            opts.setBackgroundColour((0.063, 0.071, 0.098, 1.0))
            opts.setSymbolColour((0.95, 0.96, 0.98, 1.0))
            opts.updateAtomPalette({
                6: (0.92, 0.94, 0.97),
                7: (0.45, 0.68, 1.0),
                8: (1.0, 0.42, 0.38),
                16: (0.98, 0.80, 0.25),
                9: (0.38, 0.86, 0.48),
                17: (0.38, 0.86, 0.48),
            })
        else:
            opts.setBackgroundColour((0.984, 0.984, 0.992, 1.0))
        drawer.DrawMolecule(mol_2d)
        drawer.FinishDrawing()
        return {"svg": drawer.GetDrawingText(), "smiles": Chem.MolToSmiles(mol_2d)}

    @app.post("/api/prepare")
    def prepare_structures(req: PrepareRequest):
        """Prepare receptor (protonation, hydrogens, missing atoms) and/or ligand (3D, H, MMFF94, pH states)."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from ..preprocessing.receptor import ReceptorPreprocessor
        from ..preprocessing.states import enumerate_ligand_states

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_prep_dir = prepared_root / f"{stamp}_{uuid.uuid4().hex[:5]}"
        job_prep_dir.mkdir(parents=True, exist_ok=True)
        resp: Dict[str, Any] = {"ph": req.ph, "output_dir": str(job_prep_dir)}

        # 1. Receptor preparation
        if req.receptor_path:
            rp = _safe_resolve(req.receptor_path)
            if not rp.exists():
                raise HTTPException(status_code=404, detail="Receptor file not found")

            rec_in = rp
            removed_het_atoms = 0
            if req.exclude_ligands:
                rec_in, removed_het_atoms = _strip_het_ligands_from_pdb(
                    rec_in, req.exclude_ligands, job_prep_dir / f"{rp.stem}_nolig.pdb"
                )
            if req.strip_waters:
                from ..docking.search.waters import write_receptor_with_waters

                rec_in = Path(
                    write_receptor_with_waters(str(rec_in), set(), str(job_prep_dir / f"{rp.stem}_nowat.pdb"))
                )

            out_rec = job_prep_dir / f"{rp.stem}_prepared.pdb"
            proc = ReceptorPreprocessor(ph=req.ph)
            try:
                rec_info = proc.process(
                    str(rec_in),
                    str(out_rec),
                    add_hydrogens=req.add_hydrogens,
                    add_charges=req.add_charges,
                    fix_missing_residues=req.fix_missing_residues,
                )
                prep_backend = "openmm_pdbfixer"
            except (RuntimeError, ImportError) as exc:
                logger.info("OpenMM/PDBFixer unavailable (%s); using clean PDB fallback", exc)
                shutil.copy2(str(rec_in), str(out_rec))
                n_atoms = sum(
                    1 for ln in out_rec.read_text(encoding="utf-8", errors="ignore").splitlines()
                    if ln.startswith(("ATOM  ", "HETATM"))
                )
                rec_info = {
                    "original_atoms": n_atoms,
                    "final_atoms": n_atoms,
                    "hydrogens_added": 0,
                    "charges_assigned": False,
                }
                prep_backend = "pdb_clean_fallback"
            resp["receptor"] = {
                "path": str(out_rec),
                "filename": out_rec.name,
                "original_atoms": rec_info.get("original_atoms", 0),
                "final_atoms": rec_info.get("final_atoms", 0),
                "hydrogens_added": rec_info.get("hydrogens_added", 0),
                "excluded_ligand_atoms": removed_het_atoms,
                "charges_assigned": bool(rec_info.get("charges_assigned", False)),
                "backend": prep_backend,
                "ph": req.ph,
                "inspection": _inspect_receptor(out_rec),
            }

        # 2. Ligand / series preparation (handles single SDF/MOL2/PDB/SMI and multi-ligand series)
        if req.ligand_path:
            lp = _safe_resolve(req.ligand_path)
            if not lp.exists():
                raise HTTPException(status_code=404, detail="Ligand file not found")

            ext = lp.suffix.lower()
            raw_mols: List[Chem.Mol] = []
            if ext == ".sdf":
                raw_mols = [m for m in Chem.SDMolSupplier(str(lp), removeHs=False) if m is not None]
            elif ext == ".mol2":
                m = Chem.MolFromMol2File(str(lp), removeHs=False)
                if m:
                    raw_mols = [m]
            elif ext == ".pdb":
                m = Chem.MolFromPDBFile(str(lp), removeHs=False)
                if m:
                    raw_mols = [m]
            elif ext in (".smi", ".smiles", ".txt"):
                for idx, line in enumerate(lp.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    m = Chem.MolFromSmiles(parts[0])
                    if m:
                        m.SetProp("_Name", parts[1] if len(parts) > 1 else f"lig_{idx}")
                        raw_mols.append(m)

            if not raw_mols:
                raise HTTPException(status_code=400, detail=f"No valid ligand molecules parsed from {lp.name}")

            prepared_mols: List[Chem.Mol] = []
            state_count = 0
            for idx, m in enumerate(raw_mols):
                name = m.GetProp("_Name").strip() if m.HasProp("_Name") and m.GetProp("_Name").strip() else f"{lp.stem}_{idx+1}"
                if req.enumerate_states:
                    states = enumerate_ligand_states(m, ph=req.ph, max_states=4 if len(raw_mols) == 1 else 2)
                    for st in states:
                        sm = Chem.Mol(st.mol)
                        if req.minimize_ligand:
                            try:
                                AllChem.MMFFOptimizeMolecule(sm, maxIters=300)
                            except Exception:
                                try:
                                    AllChem.UFFOptimizeMolecule(sm, maxIters=300)
                                except Exception:
                                    pass
                        sm.SetProp("_Name", st.state_id)
                        sm.SetProp("state_penalty_kcal", f"{st.state_penalty:.3f}")
                        sm.SetProp("formal_charge", str(st.formal_charge))
                        prepared_mols.append(sm)
                        state_count += 1
                else:
                    pm = Chem.Mol(m)
                    if req.add_hydrogens:
                        pm = Chem.AddHs(pm, addCoords=(pm.GetNumConformers() > 0))
                    if pm.GetNumConformers() == 0:
                        params = AllChem.ETKDGv3()
                        params.randomSeed = 42 + idx
                        if AllChem.EmbedMolecule(pm, params) != 0:
                            AllChem.EmbedMolecule(pm, useRandomCoords=True, randomSeed=42 + idx)
                    if req.minimize_ligand:
                        try:
                            AllChem.MMFFOptimizeMolecule(pm, maxIters=400)
                        except Exception:
                            try:
                                AllChem.UFFOptimizeMolecule(pm, maxIters=400)
                            except Exception:
                                pass
                    pm.SetProp("_Name", name)
                    prepared_mols.append(pm)

            out_lig = job_prep_dir / f"{lp.stem}_prepared.sdf"
            w = Chem.SDWriter(str(out_lig))
            for pm in prepared_mols:
                w.write(pm)
            w.close()

            resp["ligand"] = {
                "path": str(out_lig),
                "filename": out_lig.name,
                "n_input_molecules": len(raw_mols),
                "n_output_molecules": len(prepared_mols),
                "states_enumerated": state_count,
                "minimized": bool(req.minimize_ligand),
                "ph": req.ph,
                "inspection": _inspect_ligand(out_lig, ph=req.ph),
            }

        return resp

    @app.post("/api/gridbox")
    def compute_gridbox(req: GridBoxRequest):
        """Auto-generate grid box(es) via PandaDock's cavity, residue, ligand, metal, or protein-center generators."""
        rp = _safe_resolve(req.receptor_path)
        if not rp.exists():
            raise HTTPException(status_code=404, detail="Receptor file not found")

        mode = req.mode.lower().strip()
        boxes_out: List[Dict[str, Any]] = []

        if mode == "protein_center":
            from Bio.PDB import PDBParser

            struct = PDBParser(QUIET=True).get_structure("rec", str(rp))
            pts = np.asarray(
                [a.get_coord() for a in struct.get_atoms() if a.get_parent().id[0] == " "],
                dtype=float,
            )
            if len(pts) == 0:
                pts = np.asarray([a.get_coord() for a in struct.get_atoms()], dtype=float)
            center = pts.mean(axis=0)
            span = np.clip(pts.max(axis=0) - pts.min(axis=0), 16.0, max(req.box_size, 36.0))
            dims = np.array([req.box_size, req.box_size, req.box_size], dtype=float)
            boxes_out.append({
                "label": "Protein Center",
                "method": "protein_center",
                "center": [round(float(x), 3) for x in center],
                "dimensions": [round(float(x), 1) for x in dims],
                "full_span": [round(float(x), 1) for x in span],
                "volume": round(float(np.prod(dims)), 1),
                "confidence": 1.0,
            })

        elif mode == "metal":
            rec_info = _inspect_receptor(rp)
            metals = rec_info.get("metals") or []
            if not metals:
                raise HTTPException(status_code=400, detail="No metal ions detected in this receptor")
            for m in metals:
                dims = [round(float(req.box_size), 1)] * 3
                boxes_out.append({
                    "label": f"Metal Site · {m['label']}",
                    "method": "metal",
                    "center": m["coord"],
                    "dimensions": dims,
                    "volume": round(float(req.box_size ** 3), 1),
                    "confidence": 1.0,
                    "restraint_spec": m.get("restraint_spec"),
                })

        elif mode == "ligand":
            center = None
            dims = [round(float(req.box_size), 1)] * 3
            label = "Ligand Center"
            if req.ligand_path:
                lp = _safe_resolve(req.ligand_path)
                if lp.exists():
                    try:
                        from ..gridbox.ligand_based import LigandBasedGridBoxGenerator

                        gen = LigandBasedGridBoxGenerator()
                        gbs = gen.generate(str(rp), reference_ligand=str(lp), padding=req.padding)
                        for gb in gbs:
                            d = [round(float(max(16.0, min(34.0, x))), 1) for x in gb.dimensions]
                            boxes_out.append({
                                "label": f"Reference Ligand ({lp.name})",
                                "method": "similarity",
                                "center": [round(float(x), 3) for x in gb.center],
                                "dimensions": d,
                                "volume": round(float(np.prod(d)), 1),
                                "confidence": round(float(gb.confidence_score), 3),
                            })
                    except Exception:
                        pass
            if not boxes_out:
                rec_info = _inspect_receptor(rp)
                hets = rec_info.get("het_ligands") or []
                if hets:
                    for h in hets[:4]:
                        boxes_out.append({
                            "label": f"Co-crystal {h['label']}",
                            "method": "hetatm_ligand",
                            "center": h["center"],
                            "dimensions": dims,
                            "volume": round(float(np.prod(dims)), 1),
                            "confidence": 0.95,
                        })
                else:
                    raise HTTPException(status_code=400, detail="No 3D ligand or co-crystallized HETATM found")

        elif mode == "residues":
            if not req.residues:
                raise HTTPException(status_code=400, detail="Select or enter at least one residue (e.g. A:145)")
            from ..gridbox.core import ResidueBasedGridBoxGenerator

            # Normalize specs like "A:CYS:145" -> "A:145" for ResidueBasedGridBoxGenerator
            norm_res = []
            for r in req.residues:
                parts = [p.strip() for p in r.split(":") if p.strip()]
                if len(parts) >= 3:
                    norm_res.append(f"{parts[0]}:{parts[-1]}")
                elif len(parts) == 2:
                    norm_res.append(f"{parts[0]}:{parts[1]}")
                elif len(parts) == 1:
                    norm_res.append(f"A:{parts[0]}")
            gen = ResidueBasedGridBoxGenerator()
            gbs = gen.generate(str(rp), residues=norm_res, padding=req.padding)
            if not gbs:
                raise HTTPException(status_code=400, detail=f"None of the specified residues were found: {', '.join(norm_res)}")
            for gb in gbs:
                d = [round(float(max(16.0, min(36.0, x))), 1) for x in gb.dimensions]
                boxes_out.append({
                    "label": f"Selected Residues ({len(norm_res)})",
                    "method": "residues",
                    "center": [round(float(x), 3) for x in gb.center],
                    "dimensions": d,
                    "volume": round(float(np.prod(d)), 1),
                    "confidence": round(float(gb.confidence_score), 3),
                    "residues": norm_res,
                })

        elif mode == "cavities":
            from ..gridbox.cavity_detection import CavityDetectionGridBoxGenerator

            gen = CavityDetectionGridBoxGenerator()
            gbs = gen.generate(str(rp), max_cavities=req.max_sites, min_volume=req.min_volume)
            for idx, gb in enumerate(gbs, 1):
                d = [round(float(max(18.0, min(32.0, x))), 1) for x in gb.dimensions]
                boxes_out.append({
                    "label": f"Pocket #{idx}",
                    "method": "cavities",
                    "center": [round(float(x), 3) for x in gb.center],
                    "dimensions": d,
                    "volume": round(float(gb.volume()), 1),
                    "confidence": round(float(gb.confidence_score), 3),
                })
            if not boxes_out:
                raise HTTPException(status_code=404, detail="No cavities above threshold detected; try Protein Center or click residues in 3D")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported gridbox mode: {mode}")

        return {"mode": mode, "boxes": boxes_out}

    @app.get("/api/structure-text")
    def get_structure_text(path: str = Query(...)):
        p = _safe_resolve(path)
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=404, detail="Structure file not found")
        # Convert SMILES to quick 3D SDF text so 3Dmol.js can preview even SMILES inputs!
        if p.suffix.lower() in (".smi", ".smiles", ".txt"):
            from rdkit import Chem
            from rdkit.Chem import AllChem

            first = next(
                (ln.strip() for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip() and not ln.startswith("#")),
                "",
            )
            if first:
                m = Chem.MolFromSmiles(first.split()[0])
                if m is not None:
                    m = Chem.AddHs(m)
                    AllChem.EmbedMolecule(m, AllChem.ETKDGv3())
                    return PlainTextResponse(Chem.MolToMolBlock(m), media_type="text/plain")
        return PlainTextResponse(p.read_text(encoding="utf-8", errors="ignore"), media_type="text/plain")

    @app.post("/api/jobs/run")
    def start_job(req: RunJobRequest):
        # Validate paths
        for attr in ("receptor_path", "ligand_path", "poses_path", "labels_path", "reference_pose_path", "weights_file"):
            val = getattr(req, attr)
            if val:
                p = _safe_resolve(val)
                if not p.exists():
                    raise HTTPException(status_code=400, detail=f"File not found for {attr}: {val}")
                setattr(req, attr, str(p))

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^A-Za-z0-9_-]", "_", (req.job_name or req.mode).strip())[:32] or req.mode
        folder_name = f"{stamp}_{slug}"
        out_dir = gui_runs_root / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"gui__{folder_name}"

        # Maestro-style co-crystallized ligand exclusion: if the receptor PDB is a
        # complex and the user marked co-crystal ligand(s) (e.g. used to define the
        # pocket center), strip ONLY those ligand atoms into a clean docking receptor
        # while preserving protein residues, catalytic metals, and structural waters.
        excluded_note = None
        if req.receptor_path and req.exclude_ligands and req.mode in ("dock", "series", "covalent", "metal", "flex"):
            clean_rec = out_dir / f"{Path(req.receptor_path).stem}_pocket_clean.pdb"
            _, n_removed = _strip_het_ligands_from_pdb(Path(req.receptor_path), req.exclude_ligands, clean_rec)
            if n_removed > 0:
                req.receptor_path = str(clean_rec)
                excluded_note = (
                    f"[Grid Prep] Excluded co-crystal ligand(s) {', '.join(req.exclude_ligands)} "
                    f"({n_removed} HETATM atoms removed; protein & metals preserved) -> {clean_rec.name}"
                )

        try:
            cmd = _build_cli_command(req, out_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        job = ManagedJob(
            job_id=run_id,
            name=req.job_name or f"{req.mode.title()} ({stamp[-6:]})",
            mode=req.mode,
            command=cmd,
            output_dir=out_dir,
            status="running",
            started_at=datetime.now().isoformat(timespec="seconds"),
        )
        if excluded_note:
            job.append_log(excluded_note)
        jobs[run_id] = job

        def _runner():
            job.append_log(f"$ {' '.join(cmd)}")
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(workspace),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                job.process = proc
                assert proc.stdout is not None
                for line in proc.stdout:
                    job.append_log(line)
                rc = proc.wait()
                job.exit_code = rc
                if job.status != "cancelled":
                    job.status = "completed" if rc == 0 else "failed"
            except Exception as exc:
                job.append_log(f"ERROR: {exc}")
                job.status = "failed"
                job.exit_code = -1
            finally:
                job.finished_at = datetime.now().isoformat(timespec="seconds")
                try:
                    (out_dir / "gui_job.log").write_text("\n".join(job.logs) + "\n", encoding="utf-8")
                except Exception:
                    pass

        threading.Thread(target=_runner, daemon=True).start()
        return job.to_dict()

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": [j.to_dict() for j in sorted(jobs.values(), key=lambda x: x.created_at, reverse=True)]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        d = job.to_dict()
        d["logs"] = list(job.logs)
        return d

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.process and job.process.poll() is None:
            job.status = "cancelled"
            job.process.terminate()
            job.append_log("[Cancelled by user]")
        return job.to_dict()

    @app.get("/api/jobs/{job_id}/stream")
    async def stream_job(job_id: str):
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        async def _gen():
            cursor = 0
            while True:
                with job.lock:
                    new_lines = job.logs[cursor:]
                    cursor = len(job.logs)
                    status = job.status
                    exit_code = job.exit_code
                for ln in new_lines:
                    payload = json.dumps({"type": "log", "line": ln})
                    yield f"data: {payload}\n\n"
                if status in ("completed", "failed", "cancelled"):
                    done_payload = json.dumps({"type": "status", "status": status, "exit_code": exit_code, "run_id": job_id})
                    yield f"data: {done_payload}\n\n"
                    break
                await asyncio.sleep(0.35)

        return StreamingResponse(_gen(), media_type="text/event-stream")

    @app.get("/api/runs")
    def list_runs():
        run_dirs = _all_run_dirs()
        items = []
        for rid, rdir in run_dirs.items():
            job = jobs.get(rid)
            try:
                s = _summarize_run_dir(rid, rdir, active_job=job)
                items.append({
                    "run_id": rid,
                    "title": s["title"],
                    "mode": s["mode"],
                    "status": s["status"],
                    "updated_at": s["updated_at"],
                    "best_score": s["kpis"]["best_score"],
                    "pb_pass": s["kpis"]["pb_pass"],
                    "pb_total": s["kpis"]["pb_total"],
                    "has_pdf": bool(s["pdf_url"]),
                    "has_html": bool(s["html_url"]),
                    "n_plots": len(s["plots"]),
                    "has_3d": bool(s["complex_pdbs"] or s["poses_sdf_url"]),
                })
            except Exception as exc:
                logger.debug("Skipping run %s: %s", rid, exc)
        # Active/GUI runs first, then validation runs sorted cleanly
        items.sort(key=lambda x: (0 if x["run_id"].startswith("gui__") else 1, x["updated_at"]), reverse=False)
        items.sort(key=lambda x: x["updated_at"], reverse=True)
        return {"runs": items}

    @app.get("/api/runs/{run_id}/summary")
    def get_run_summary(run_id: str):
        run_dirs = _all_run_dirs()
        rdir = run_dirs.get(run_id)
        if not rdir or not rdir.exists():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return _summarize_run_dir(run_id, rdir, active_job=jobs.get(run_id))

    @app.get("/api/runs/{run_id}/pose/{pose_idx}/pocket")
    def get_pose_pocket_interactions(
        run_id: str,
        pose_idx: int,
        cutoff: float = Query(5.0, ge=3.0, le=8.0),
    ):
        run_dirs = _all_run_dirs()
        rdir = run_dirs.get(run_id)
        if not rdir or not rdir.exists():
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        summary = _summarize_run_dir(run_id, rdir, active_job=jobs.get(run_id))
        return _extract_pose_pocket_with_h_and_interactions(rdir, summary, pose_idx=pose_idx, cutoff=cutoff)

    @app.get("/api/runs/{run_id}/file/{relpath:path}")
    def get_run_file(run_id: str, relpath: str):
        run_dirs = _all_run_dirs()
        rdir = run_dirs.get(run_id)
        if not rdir or not rdir.exists():
            raise HTTPException(status_code=404, detail="Run not found")
        target = (rdir / relpath).resolve()
        try:
            target.relative_to(rdir.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Invalid file path")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        mime, _ = mimetypes.guess_type(str(target))
        if target.suffix.lower() == ".pdf":
            return FileResponse(
                str(target),
                media_type="application/pdf",
                headers={"Content-Disposition": f'inline; filename="{target.name}"'},
            )
        if target.suffix.lower() in (".pdb", ".sdf", ".mol2", ".smi", ".mdp", ".sh", ".py", ".log", ".txt"):
            mime = "text/plain; charset=utf-8"
        return FileResponse(str(target), media_type=mime or "application/octet-stream")

    return app


def launch_gui(
    host: str = "127.0.0.1",
    port: int = 8765,
    workspace: Optional[str] = None,
    open_browser: bool = True,
) -> None:
    """Launch PandaDock Studio via Uvicorn."""
    import uvicorn

    url = f"http://{host}:{port}"
    app = create_app(workspace_dir=workspace)
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    click.echo(f"PandaDock Studio v{__version__} running at {url}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind host address")
@click.option("--port", default=8765, type=int, show_default=True, help="Bind port")
@click.option("--workspace", default=".", type=click.Path(exists=True, file_okay=False),
              help="Workspace directory containing structures & results")
@click.option("--no-browser", is_flag=True, default=False, help="Do not auto-open the web browser")
def cli(host: str, port: int, workspace: str, no_browser: bool):
    """Launch PandaDock Studio — Apple-designed molecular docking GUI."""
    launch_gui(host=host, port=port, workspace=workspace, open_browser=not no_browser)


if __name__ == "__main__":
    cli()
