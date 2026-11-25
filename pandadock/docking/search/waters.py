"""
Crystallographic structural water detection and displaceable/bridging water docking.

Supports three water modes in `pandadock dock --waters <mode>`:
- `strip` (default): removes all crystallographic waters (`HOH`/`WAT`) before grid construction.
- `keep`: retains crystallographic pocket waters inside the binding box so the ligand can
  form water-mediated hydrogen-bond bridges.
- `displace`: explicitly compares docking against the stripped pocket vs. the conserved-water
  pocket, evaluates whether each pocket water is displaced (heavy-atom overlap < 2.2 Å) or
  retained as a bridging water, applies a conservative thermodynamic displacement adjustment
  based on protein H-bond burial, and returns the preferred complex plus a per-water audit.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem

logger = logging.getLogger("pandadock.docking.search.waters")

WATER_RESNAMES = frozenset({"HOH", "WAT", "DOD", "TIP3", "SOL"})


@dataclass
class PocketWater:
    """A crystallographic water molecule located inside the docking box."""

    chain_id: str
    resseq: int
    resname: str
    coord: np.ndarray
    receptor_polar_contacts: int = 0
    nearest_receptor_dist: float = 99.0

    @property
    def key(self) -> Tuple[str, int]:
        return (self.chain_id, self.resseq)

    @property
    def label(self) -> str:
        return f"{self.resname} {self.chain_id}:{self.resseq}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "chain": self.chain_id,
            "resseq": int(self.resseq),
            "coord": [round(float(x), 3) for x in self.coord],
            "receptor_polar_contacts": int(self.receptor_polar_contacts),
            "nearest_receptor_dist": round(float(self.nearest_receptor_dist), 2),
        }


def find_pocket_waters(
    receptor_file: str,
    grid_center: np.ndarray,
    grid_dimensions: np.ndarray,
    max_waters: int = 6,
    margin: float = 0.0,
) -> List[PocketWater]:
    """Identify crystallographic waters inside the docking box and score their burial."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", receptor_file)
    model = next(structure.get_models())

    center = np.asarray(grid_center, dtype=np.float64)
    half = np.asarray(grid_dimensions, dtype=np.float64) / 2.0 + float(margin)
    box_min = center - half
    box_max = center + half

    # Collect receptor polar heavy atoms (N, O, S, metals) for burial/H-bond scoring
    polar_coords = []
    raw_waters: List[Tuple[str, int, str, np.ndarray]] = []
    for chain in model:
        for res in chain:
            rn = res.get_resname().strip().upper()
            if rn in WATER_RESNAMES:
                o_atom = None
                for atom in res.get_atoms():
                    if atom.element.strip().upper() == "O" or atom.get_name().strip().upper().startswith("O"):
                        o_atom = atom
                        break
                if o_atom is not None:
                    xyz = np.asarray(o_atom.get_coord(), dtype=np.float64)
                    if np.all(xyz >= box_min) and np.all(xyz <= box_max):
                        raw_waters.append((chain.id, int(res.id[1]), rn, xyz))
            elif res.id[0] == " ":
                for atom in res.get_atoms():
                    el = (atom.element or atom.get_name()[0]).strip().upper()
                    if el in ("N", "O", "S"):
                        polar_coords.append(atom.get_coord())
            else:
                # Metal ions count strongly as water-coordinating partners
                for atom in res.get_atoms():
                    el = (atom.element or "").strip().upper()
                    if el in ("ZN", "MG", "MN", "FE", "CA", "CU", "CO", "NI"):
                        polar_coords.append(atom.get_coord())

    polar_arr = np.asarray(polar_coords, dtype=np.float64) if polar_coords else np.zeros((0, 3))
    waters: List[PocketWater] = []
    for chain_id, resseq, rn, xyz in raw_waters:
        if len(polar_arr):
            dists = np.linalg.norm(polar_arr - xyz[None, :], axis=1)
            n_polar = int(np.sum(dists <= 3.5))
            nearest = float(np.min(dists))
        else:
            n_polar = 0
            nearest = 99.0
        waters.append(
            PocketWater(
                chain_id=chain_id,
                resseq=resseq,
                resname=rn,
                coord=xyz,
                receptor_polar_contacts=n_polar,
                nearest_receptor_dist=nearest,
            )
        )

    # Prioritize conserved structural waters (most receptor polar contacts, then closest to pocket center)
    waters.sort(
        key=lambda w: (-w.receptor_polar_contacts, float(np.linalg.norm(w.coord - center)))
    )
    return waters[:max_waters]


def write_receptor_with_waters(
    receptor_file: str,
    keep_water_keys: Set[Tuple[str, int]],
    output_pdb: str,
) -> str:
    """Write a copy of `receptor_file` stripping all waters except those in `keep_water_keys`."""
    out_lines: List[str] = []
    with open(receptor_file, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                resname = line[17:20].strip().upper()
                if resname in WATER_RESNAMES:
                    chain_id = line[21:22]
                    try:
                        resseq = int(line[22:26].strip())
                    except ValueError:
                        continue
                    if (chain_id, resseq) not in keep_water_keys:
                        continue
            out_lines.append(line)

    Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdb, "w", encoding="utf-8") as fh:
        fh.writelines(out_lines)
    return str(output_pdb)


def analyze_pose_water_disposition(
    pose_coords: np.ndarray,
    ligand_mol: Optional[Chem.Mol],
    pocket_waters: Sequence[PocketWater],
    kept_water_keys: Set[Tuple[str, int]],
) -> Tuple[List[Dict[str, Any]], float]:
    """Classify each pocket water as displaced, bridging, or spectator and compute ΔG_water.

    Thermodynamic heuristic (conservative):
    - Displacing a weakly anchored pocket water (`receptor_polar_contacts <= 2`) earns a
      favourable entropic release credit of `-0.50 kcal/mol`.
    - Displacing a tightly bound structural water (`>= 3` protein polar H-bonds) incurs
      a desolvation penalty of `+0.60 kcal/mol`.
    - Forming a bridging H-bond (`2.4 <= d <= 3.5 Å` to a ligand N/O/S atom) with a
      retained water earns `-0.35 kcal/mol` stabilization.
    """
    if not pocket_waters:
        return [], 0.0

    coords = np.asarray(pose_coords, dtype=np.float64)
    if ligand_mol is not None:
        heavy_idx = [a.GetIdx() for a in ligand_mol.GetAtoms() if a.GetAtomicNum() > 1]
        polar_idx = [a.GetIdx() for a in ligand_mol.GetAtoms() if a.GetAtomicNum() in (7, 8, 16)]
    else:
        heavy_idx = list(range(len(coords)))
        polar_idx = heavy_idx

    heavy_xyz = coords[heavy_idx] if heavy_idx else coords
    polar_xyz = coords[polar_idx] if polar_idx else heavy_xyz

    records: List[Dict[str, Any]] = []
    delta_g_water = 0.0
    for w in pocket_waters:
        d_heavy = float(np.min(np.linalg.norm(heavy_xyz - w.coord[None, :], axis=1)))
        d_polar = float(np.min(np.linalg.norm(polar_xyz - w.coord[None, :], axis=1))) if len(polar_xyz) else d_heavy

        if w.key not in kept_water_keys:
            if d_heavy < 2.2:
                status = "displaced"
                if w.receptor_polar_contacts >= 3:
                    contrib = +0.60
                elif w.receptor_polar_contacts >= 1:
                    contrib = -0.50
                else:
                    contrib = -0.25
            else:
                status = "stripped_unused"
                contrib = 0.0
        else:
            if 2.3 <= d_polar <= 3.5 and w.receptor_polar_contacts >= 1:
                status = "bridging"
                contrib = -0.35
            elif d_heavy < 2.2:
                status = "clashing_kept"
                contrib = +0.80
            else:
                status = "kept_spectator"
                contrib = 0.0

        delta_g_water += contrib
        rec = w.to_dict()
        rec.update(
            {
                "status": status,
                "min_ligand_heavy_dist_A": round(d_heavy, 2),
                "min_ligand_polar_dist_A": round(d_polar, 2),
                "delta_g_water_kcal": round(contrib, 3),
            }
        )
        records.append(rec)

    return records, float(delta_g_water)


def dock_with_water_mode(
    engine,
    receptor_file: str,
    ligand_mol: Chem.Mol,
    grid_center: np.ndarray,
    grid_dimensions: np.ndarray,
    water_mode: str = "strip",
    max_waters: int = 5,
    dock_kwargs: Optional[Dict[str, Any]] = None,
):
    """Run docking under `strip`, `keep`, or `displace` structural-water protocol."""
    mode = (water_mode or "strip").lower()
    kwargs = dict(dock_kwargs or {})

    pocket_waters = find_pocket_waters(
        receptor_file, grid_center, grid_dimensions, max_waters=max_waters
    )

    if mode == "strip" or not pocket_waters:
        with tempfile.TemporaryDirectory(prefix="pandadock_water_") as tmpdir:
            stripped_pdb = write_receptor_with_waters(
                receptor_file, set(), str(Path(tmpdir) / "receptor_stripped.pdb")
            )
            result = engine.dock_ligand(
                stripped_pdb, ligand_mol, grid_center, grid_dimensions, **kwargs
            )
            result.receptor_file = receptor_file
            result.parameters["water_mode"] = mode
            result.parameters["pocket_waters_found"] = len(pocket_waters)
            if pocket_waters and result.poses:
                recs, dg_w = analyze_pose_water_disposition(
                    result.poses[0].coordinates, ligand_mol, pocket_waters, set()
                )
                result.parameters["water_analysis"] = {
                    "mode": mode,
                    "selected_state": "stripped",
                    "delta_g_water": round(dg_w, 3),
                    "waters": recs,
                }
            return result

    with tempfile.TemporaryDirectory(prefix="pandadock_water_") as tmpdir:
        kept_keys = {w.key for w in pocket_waters if w.receptor_polar_contacts >= 1} or {
            w.key for w in pocket_waters
        }
        water_pdb = write_receptor_with_waters(
            receptor_file, kept_keys, str(Path(tmpdir) / "receptor_waters.pdb")
        )

        if mode == "keep":
            result = engine.dock_ligand(
                water_pdb, ligand_mol, grid_center, grid_dimensions, **kwargs
            )
            result.receptor_file = receptor_file
            result.parameters["water_mode"] = "keep"
            result.parameters["pocket_waters_found"] = len(pocket_waters)
            if result.poses:
                recs, dg_w = analyze_pose_water_disposition(
                    result.poses[0].coordinates, ligand_mol, pocket_waters, kept_keys
                )
                result.parameters["water_analysis"] = {
                    "mode": "keep",
                    "selected_state": "water_retained",
                    "kept_waters": [w.label for w in pocket_waters if w.key in kept_keys],
                    "delta_g_water": round(dg_w, 3),
                    "waters": recs,
                }
            return result

        # mode == "displace": compare stripped vs conserved-water docking
        stripped_pdb = write_receptor_with_waters(
            receptor_file, set(), str(Path(tmpdir) / "receptor_stripped.pdb")
        )
        res_stripped = engine.dock_ligand(
            stripped_pdb, ligand_mol, grid_center, grid_dimensions, **kwargs
        )
        res_water = engine.dock_ligand(
            water_pdb, ligand_mol, grid_center, grid_dimensions, **kwargs
        )

        e_strip_raw = res_stripped.poses[0].energy if res_stripped.poses else 0.0
        e_water_raw = res_water.poses[0].energy if res_water.poses else 0.0

        rec_strip, dg_strip = (
            analyze_pose_water_disposition(
                res_stripped.poses[0].coordinates, ligand_mol, pocket_waters, set()
            )
            if res_stripped.poses
            else ([], 0.0)
        )
        rec_water, dg_water = (
            analyze_pose_water_disposition(
                res_water.poses[0].coordinates, ligand_mol, pocket_waters, kept_keys
            )
            if res_water.poses
            else ([], 0.0)
        )

        eff_strip = e_strip_raw + dg_strip
        eff_water = e_water_raw + dg_water

        if res_water.poses and (not res_stripped.poses or eff_water < eff_strip):
            chosen = res_water
            chosen_state = "water_bridged"
            chosen_recs = rec_water
            chosen_dg = dg_water
        else:
            chosen = res_stripped
            chosen_state = "water_displaced"
            chosen_recs = rec_strip
            chosen_dg = dg_strip

        chosen.receptor_file = receptor_file
        chosen.parameters["water_mode"] = "displace"
        chosen.parameters["pocket_waters_found"] = len(pocket_waters)
        chosen.parameters["water_analysis"] = {
            "mode": "displace",
            "selected_state": chosen_state,
            "stripped_raw_score": round(float(e_strip_raw), 3),
            "stripped_effective_score": round(float(eff_strip), 3),
            "water_bridged_raw_score": round(float(e_water_raw), 3),
            "water_bridged_effective_score": round(float(eff_water), 3),
            "delta_g_water": round(float(chosen_dg), 3),
            "displaced_count": sum(1 for r in chosen_recs if r["status"] == "displaced"),
            "bridging_count": sum(1 for r in chosen_recs if r["status"] == "bridging"),
            "waters": chosen_recs,
        }
        return chosen
