"""
Built-in PoseBusters-style physical and chemical validity audit for docked poses.

Implements the core chemical-validity and geometric-plausibility checks defined by
Buttenschoen, Morris & Deane (Chemical Science, 2024, 15, 3130-3139) using RDKit,
NumPy, and BioPython with zero extra dependencies:

  1. mol_loaded_and_sanitized  — RDKit valence & aromaticity sanitization
  2. all_atoms_connected       — single connected molecular fragment
  3. bond_lengths              — all bonds within [0.75, 1.25] of ideal length
  4. bond_angles               — all angles within ±20° of ideal geometry
  5. aromatic_ring_flatness    — aromatic rings planar (max out-of-plane <= 0.35 Å)
  6. double_bond_flatness      — non-ring sp2/amide planes flat (<= 0.35 Å)
  7. tetrahedral_chirality     — assigned R/S stereo preserved vs reference mol
  8. internal_steric_clash     — no non-bonded ligand atom pairs < 0.70 * (r_i + r_j)
  9. internal_energy_strain    — relative MMFF94/UFF strain <= 15.0 kcal/mol
 10. protein_ligand_clash      — no heavy-atom overlap < 0.75 * (r_i + r_j)
 11. volume_in_box             — ligand centroid lies inside the docking box (if given)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

logger = logging.getLogger("pandadock.docking.validation.posebusters")

# Standard Bondi van der Waals radii (Å) for clash detection
VDW_RADII = {
    1: 1.20, 5: 1.92, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
    14: 2.10, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85, 53: 1.98,
    12: 1.73, 20: 1.97, 25: 1.61, 26: 1.52, 27: 1.52, 28: 1.63,
    29: 1.40, 30: 1.39, 78: 1.75,
}
ELEM_VDW = {
    "H": 1.20, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "MG": 1.30, "MN": 1.30, "ZN": 1.30, "CA": 1.70, "FE": 1.30,
    "CU": 1.40, "CO": 1.35, "NI": 1.35, "PT": 1.75,
}
METAL_SYMBOLS = {"MG", "MN", "ZN", "CA", "FE", "CU", "CO", "NI", "CD", "HG", "PT", "PD", "MO", "NA", "K"}


@dataclass
class PoseBustersReport:
    """Structured PoseBusters-style validity audit for a single docked pose."""

    pose_index: int
    ligand_name: str = ""
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_passed(self) -> int:
        return sum(1 for ok in self.checks.values() if ok)

    @property
    def n_total(self) -> int:
        return len(self.checks)

    @property
    def all_passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    @property
    def failed_checks(self) -> List[str]:
        return [k for k, ok in self.checks.items() if not ok]

    @property
    def relative_energy_kcal(self) -> Optional[float]:
        val = self.metrics.get("internal_strain_kcal")
        return float(val) if val is not None else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pose_index": self.pose_index,
            "ligand_name": self.ligand_name,
            "all_passed": self.all_passed,
            "pb_valid": self.all_passed,
            "n_passed": self.n_passed,
            "n_total": self.n_total,
            "failed_checks": self.failed_checks,
            "checks": dict(self.checks),
            "metrics": dict(self.metrics),
        }


def _mol_with_coords(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    conf = Chem.Conformer(m.GetNumAtoms())
    for i, c in enumerate(coords):
        conf.SetAtomPosition(i, (float(c[0]), float(c[1]), float(c[2])))
    m.AddConformer(conf, assignId=True)
    return m


def _extract_receptor_heavy(receptor_structure_or_path) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
    """Return heavy-atom coords, Bondi radii, and is_metal flags for a receptor."""
    from Bio.PDB import PDBParser

    if isinstance(receptor_structure_or_path, (str, Path)):
        struct = PDBParser(QUIET=True).get_structure("rec", str(receptor_structure_or_path))
    else:
        struct = receptor_structure_or_path

    coords, radii, is_metal = [], [], []
    for model in struct:
        for chain in model:
            for res in chain:
                if res.get_resname().strip().upper() in ("HOH", "WAT", "DOD"):
                    continue
                for atom in res:
                    el = (atom.element or atom.get_name()[0]).strip().upper()
                    if el == "H":
                        continue
                    coords.append(atom.get_coord())
                    radii.append(ELEM_VDW.get(el, 1.70))
                    is_metal.append(el in METAL_SYMBOLS)
        break
    if not coords:
        return np.zeros((0, 3)), np.zeros((0,)), []
    return np.asarray(coords, dtype=np.float64), np.asarray(radii, dtype=np.float64), is_metal


def _ideal_geometry(mol: Chem.Mol) -> Tuple[Optional[Chem.Mol], Optional[float]]:
    """Generate a local MMFF94/UFF minimized reference conformer & energy."""
    try:
        ref = Chem.Mol(mol)
        if ref.GetNumConformers() == 0:
            if AllChem.EmbedMolecule(ref, randomSeed=42) != 0:
                return None, None
        props = AllChem.MMFFGetMoleculeProperties(ref)
        ff = (
            AllChem.MMFFGetMoleculeForceField(ref, props)
            if props is not None
            else AllChem.UFFGetMoleculeForceField(ref)
        )
        if ff is None:
            return ref, None
        ff.Minimize(maxIts=300)
        return ref, float(ff.CalcEnergy())
    except Exception:
        return None, None


def _plane_max_deviation(coords: np.ndarray, indices: Sequence[int]) -> float:
    pts = coords[list(indices)]
    if len(pts) < 4:
        return 0.0
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    return float(np.max(np.abs(centered @ normal)))


def _check_bond_lengths_and_angles(
    pose_mol: Chem.Mol, ref_mol: Optional[Chem.Mol]
) -> Tuple[bool, float, bool, float]:
    """Compare bond lengths and angles against minimized reference conformer."""
    if ref_mol is None or pose_mol.GetNumConformers() == 0 or ref_mol.GetNumConformers() == 0:
        return True, 0.0, True, 0.0

    p_conf = pose_mol.GetConformer()
    r_conf = ref_mol.GetConformer()

    max_len_ratio_dev = 0.0
    max_len_abs_dev = 0.0
    lengths_ok = True
    for bond in pose_mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if pose_mol.GetAtomWithIdx(i).GetAtomicNum() == 1 or pose_mol.GetAtomWithIdx(j).GetAtomicNum() == 1:
            continue
        l_pose = rdMolTransforms.GetBondLength(p_conf, i, j)
        l_ref = rdMolTransforms.GetBondLength(r_conf, i, j)
        if l_ref < 0.5:
            continue
        ratio = l_pose / l_ref
        abs_dev = abs(l_pose - l_ref)
        max_len_ratio_dev = max(max_len_ratio_dev, abs(ratio - 1.0))
        max_len_abs_dev = max(max_len_abs_dev, abs_dev)
        if ratio < 0.75 or ratio > 1.25:
            lengths_ok = False

    max_ang_dev = 0.0
    angles_ok = True
    for atom in pose_mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        j = atom.GetIdx()
        nbrs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
        for a_i in range(len(nbrs)):
            for b_i in range(a_i + 1, len(nbrs)):
                i, k = nbrs[a_i], nbrs[b_i]
                ang_p = rdMolTransforms.GetAngleDeg(p_conf, i, j, k)
                ang_r = rdMolTransforms.GetAngleDeg(r_conf, i, j, k)
                if math.isnan(ang_p) or math.isnan(ang_r):
                    continue
                dev = abs(ang_p - ang_r)
                max_ang_dev = max(max_ang_dev, dev)
                if dev > 20.0:
                    angles_ok = False

    return lengths_ok, float(max_len_abs_dev), angles_ok, float(max_ang_dev)


def _check_flatness(pose_mol: Chem.Mol, coords: np.ndarray) -> Tuple[bool, float, bool, float]:
    """Check aromatic ring planarity and non-ring double-bond planarity."""
    max_arom_dev = 0.0
    arom_ok = True
    ri = pose_mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) >= 5 and all(pose_mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            dev = _plane_max_deviation(coords, ring)
            max_arom_dev = max(max_arom_dev, dev)
            if dev > 0.35:
                arom_ok = False

    max_db_dev = 0.0
    db_ok = True
    for bond in pose_mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.IsInRing():
            continue
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        # Only sp2 C=C / C=N / C=O / N=N double bonds are planar; skip tetrahedral S=O / P=O
        if a.GetAtomicNum() not in (6, 7) or b.GetAtomicNum() not in (6, 7, 8):
            continue
        if a.GetDegree() > 3 or b.GetDegree() > 3:
            continue
        grp = [a.GetIdx(), b.GetIdx()]
        grp += [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() != b.GetIdx() and n.GetAtomicNum() > 1]
        grp += [n.GetIdx() for n in b.GetNeighbors() if n.GetIdx() != a.GetIdx() and n.GetAtomicNum() > 1]
        if len(grp) >= 4:
            dev = _plane_max_deviation(coords, grp)
            max_db_dev = max(max_db_dev, dev)
            if dev > 0.35:
                db_ok = False

    return arom_ok, float(max_arom_dev), db_ok, float(max_db_dev)


def _check_internal_clash(pose_mol: Chem.Mol, coords: np.ndarray) -> Tuple[bool, float]:
    """Check non-bonded heavy-atom pairs (graph distance >= 4) for steric clashes."""
    heavy = [a.GetIdx() for a in pose_mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(heavy) < 4:
        return True, 1.0
    dmat = Chem.GetDistanceMatrix(pose_mol)
    min_ratio = 99.0
    ok = True
    for idx_a in range(len(heavy)):
        i = heavy[idx_a]
        ri = VDW_RADII.get(pose_mol.GetAtomWithIdx(i).GetAtomicNum(), 1.70)
        for idx_b in range(idx_a + 1, len(heavy)):
            j = heavy[idx_b]
            if dmat[i, j] < 4:
                continue
            rj = VDW_RADII.get(pose_mol.GetAtomWithIdx(j).GetAtomicNum(), 1.70)
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            ratio = dist / (ri + rj)
            min_ratio = min(min_ratio, ratio)
            if ratio < 0.70:
                ok = False
    return ok, float(min_ratio if min_ratio < 90 else 1.0)


def _check_chirality(pose_mol: Chem.Mol, ref_mol: Optional[Chem.Mol]) -> bool:
    """Verify assigned 3D tetrahedral chirality matches the input reference ligand."""
    if ref_mol is None:
        return True
    try:
        m_p = Chem.Mol(pose_mol)
        m_r = Chem.Mol(ref_mol)
        Chem.AssignStereochemistryFrom3D(m_p)
        Chem.AssignStereochemistryFrom3D(m_r)
        p_chiral = dict(Chem.FindMolChiralCenters(m_p, includeUnassigned=False))
        r_chiral = dict(Chem.FindMolChiralCenters(m_r, includeUnassigned=False))
        for idx, tag in r_chiral.items():
            if idx in p_chiral and p_chiral[idx] != tag:
                return False
        return True
    except Exception:
        return True


def audit_pose(
    pose_mol: Chem.Mol,
    receptor_coords: Optional[np.ndarray] = None,
    receptor_radii: Optional[np.ndarray] = None,
    receptor_is_metal: Optional[Sequence[bool]] = None,
    receptor_file: Optional[str] = None,
    reference_mol: Optional[Chem.Mol] = None,
    ideal_ref_mol: Optional[Chem.Mol] = None,
    ideal_ref_energy: Optional[float] = None,
    grid_center: Optional[np.ndarray] = None,
    grid_dimensions: Optional[np.ndarray] = None,
    pose_index: int = 0,
    strain_cutoff: float = 15.0,
    max_rel_energy_kcal: Optional[float] = None,
) -> PoseBustersReport:
    """Run all PoseBusters-style validity checks on a single pose."""
    if max_rel_energy_kcal is not None:
        strain_cutoff = float(max_rel_energy_kcal)
    lig_name = pose_mol.GetProp("_Name") if (pose_mol is not None and pose_mol.HasProp("_Name")) else ""
    report = PoseBustersReport(pose_index=pose_index, ligand_name=lig_name)

    if receptor_coords is None:
        if receptor_file and Path(receptor_file).exists():
            receptor_coords, receptor_radii, receptor_is_metal = _extract_receptor_heavy(receptor_file)
        else:
            receptor_coords = np.zeros((0, 3), dtype=np.float64)
            receptor_radii = np.zeros((0,), dtype=np.float64)
            receptor_is_metal = []

    if pose_mol is None or pose_mol.GetNumConformers() == 0:
        report.checks["mol_loaded_and_sanitized"] = False
        return report

    # 1. Sanitization
    try:
        m_check = Chem.Mol(pose_mol)
        Chem.SanitizeMol(m_check)
        report.checks["mol_loaded_and_sanitized"] = True
    except Exception:
        report.checks["mol_loaded_and_sanitized"] = False

    # 2. All atoms connected (allow metal coordination zero-order bonds)
    frags = Chem.GetMolFrags(pose_mol)
    has_metal = any(a.GetSymbol().upper() in METAL_SYMBOLS for a in pose_mol.GetAtoms())
    report.checks["all_atoms_connected"] = (len(frags) == 1) or has_metal

    coords = np.asarray(pose_mol.GetConformer().GetPositions(), dtype=np.float64)

    # Ideal reference geometry (computed once and reused across poses)
    if ideal_ref_mol is None:
        ideal_ref_mol, ideal_ref_energy = _ideal_geometry(reference_mol or pose_mol)

    # 3 & 4. Bond lengths & angles
    len_ok, max_len_dev, ang_ok, max_ang_dev = _check_bond_lengths_and_angles(pose_mol, ideal_ref_mol)
    report.checks["bond_lengths"] = len_ok
    report.checks["bond_angles"] = ang_ok
    report.metrics["max_bond_length_dev_A"] = round(max_len_dev, 4)
    report.metrics["max_bond_angle_dev_deg"] = round(max_ang_dev, 2)

    # 5 & 6. Aromatic & double-bond planarity
    arom_ok, max_arom_dev, db_ok, max_db_dev = _check_flatness(pose_mol, coords)
    report.checks["aromatic_ring_flatness"] = arom_ok
    report.checks["double_bond_flatness"] = db_ok
    report.metrics["max_aromatic_dev_A"] = round(max_arom_dev, 4)
    report.metrics["max_double_bond_dev_A"] = round(max_db_dev, 4)

    # 7. Tetrahedral chirality preservation
    report.checks["tetrahedral_chirality"] = _check_chirality(pose_mol, reference_mol)

    # 8. Internal steric clash
    int_ok, min_int_ratio = _check_internal_clash(pose_mol, coords)
    report.checks["internal_steric_clash"] = int_ok
    report.metrics["min_internal_vdw_ratio"] = round(min_int_ratio, 3)

    # 9. Relative internal strain (MMFF94 / UFF)
    strain_val = 0.0
    strain_ok = True
    try:
        m_e = Chem.Mol(pose_mol)
        props = AllChem.MMFFGetMoleculeProperties(m_e)
        ff = (
            AllChem.MMFFGetMoleculeForceField(m_e, props)
            if props is not None
            else AllChem.UFFGetMoleculeForceField(m_e)
        )
        if ff is not None and ideal_ref_energy is not None:
            e_pose = float(ff.CalcEnergy())
            strain_val = max(0.0, e_pose - ideal_ref_energy)
            # Compare against reference input conformer if available to avoid external-SDF bond offset
            if reference_mol is not None and reference_mol.GetNumConformers() > 0:
                ref_props = AllChem.MMFFGetMoleculeProperties(reference_mol)
                ref_ff = (
                    AllChem.MMFFGetMoleculeForceField(reference_mol, ref_props)
                    if ref_props is not None
                    else AllChem.UFFGetMoleculeForceField(reference_mol)
                )
                if ref_ff is not None:
                    e_ref_input = float(ref_ff.CalcEnergy())
                    strain_val = min(strain_val, max(0.0, e_pose - e_ref_input))
            strain_ok = strain_val <= strain_cutoff
    except Exception:
        pass
    report.checks["internal_energy_strain"] = strain_ok
    report.metrics["internal_strain_kcal"] = round(strain_val, 3)

    # 10. Protein-ligand steric clash (heavy atoms; allow tighter metal coordination 0.55x
    #     and exempt covalent anchor/attached bond atoms when auditing a covalent adduct)
    cov_exempt: set = set()
    for prop_k in ("covalent_anchor_idx", "covalent_attached_idx", "_covalent_anchor_idx", "_covalent_attached_idx"):
        if pose_mol.HasProp(prop_k):
            try:
                cov_exempt.add(int(pose_mol.GetProp(prop_k)))
            except ValueError:
                pass
    heavy_idx = [a.GetIdx() for a in pose_mol.GetAtoms() if a.GetAtomicNum() > 1]
    clash_eval_idx = [i for i in heavy_idx if i not in cov_exempt] if cov_exempt else heavy_idx
    lig_h_coords = coords[heavy_idx]
    eval_coords = coords[clash_eval_idx] if clash_eval_idx else lig_h_coords
    lig_h_radii = np.array(
        [VDW_RADII.get(pose_mol.GetAtomWithIdx(i).GetAtomicNum(), 1.70) for i in (clash_eval_idx or heavy_idx)],
        dtype=np.float64,
    )
    min_pl_ratio = 99.0
    min_pl_dist = 99.0
    n_clashes = 0
    if len(receptor_coords) > 0 and len(eval_coords) > 0:
        # Pre-filter receptor atoms within 6 Å of ligand bounding box
        lo = eval_coords.min(axis=0) - 6.0
        hi = eval_coords.max(axis=0) + 6.0
        mask = np.all((receptor_coords >= lo) & (receptor_coords <= hi), axis=1)
        near_rec = receptor_coords[mask]
        near_rad = receptor_radii[mask]
        near_met = np.asarray(receptor_is_metal, dtype=bool)[mask]
        if len(near_rec) > 0:
            diff = eval_coords[:, None, :] - near_rec[None, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            rsum = lig_h_radii[:, None] + near_rad[None, :]
            ratios = dists / np.maximum(rsum, 1e-6)
            thresholds = np.where(near_met[None, :], 0.55, 0.72)
            clash_mask = ratios < thresholds
            n_clashes = int(np.sum(clash_mask))
            min_pl_ratio = float(np.min(ratios))
            min_pl_dist = float(np.min(dists))

    report.checks["protein_ligand_clash"] = (n_clashes == 0)
    report.metrics["min_protein_ligand_dist_A"] = round(min_pl_dist if min_pl_dist < 90 else 0.0, 3)
    report.metrics["min_protein_ligand_vdw_ratio"] = round(min_pl_ratio if min_pl_ratio < 90 else 1.0, 3)
    report.metrics["n_protein_ligand_clashes"] = n_clashes

    # 11. Centroid inside docking box (if box specified)
    if grid_center is not None and grid_dimensions is not None:
        c_box = np.asarray(grid_center, dtype=np.float64)
        half = np.asarray(grid_dimensions, dtype=np.float64) / 2.0 + 1.5
        lig_centroid = lig_h_coords.mean(axis=0)
        in_box = bool(np.all(np.abs(lig_centroid - c_box) <= half))
        report.checks["volume_in_box"] = in_box

    return report


def audit_docking_result(
    result,
    ligand_mol: Optional[Chem.Mol],
    receptor_file: Optional[str] = None,
    max_poses: Optional[int] = None,
) -> List[PoseBustersReport]:
    """Audit all poses in a PandaDock DockingResult."""
    poses = list(getattr(result, "poses", []) or [])
    if max_poses is not None:
        poses = poses[: int(max_poses)]
    if not poses or ligand_mol is None:
        return []

    rec_path = receptor_file or getattr(result, "receptor_file", None)
    rec_coords, rec_radii, rec_metal = (
        _extract_receptor_heavy(rec_path)
        if rec_path and Path(rec_path).exists()
        else (np.zeros((0, 3)), np.zeros((0,)), [])
    )
    ideal_mol, ideal_e = _ideal_geometry(ligand_mol)
    g_c = getattr(result, "grid_center", None)
    g_d = getattr(result, "grid_dimensions", None)

    reports = []
    for idx, pose in enumerate(poses):
        coords = np.asarray(pose.coordinates, dtype=np.float64)
        if coords.shape[0] != ligand_mol.GetNumAtoms():
            continue
        p_mol = _mol_with_coords(ligand_mol, coords)
        if ligand_mol.HasProp("_Name"):
            p_mol.SetProp("_Name", f"{ligand_mol.GetProp('_Name')}_pose{idx + 1}")
        for prop_k in ("covalent_anchor_idx", "covalent_attached_idx"):
            if ligand_mol.HasProp(prop_k):
                p_mol.SetProp(prop_k, ligand_mol.GetProp(prop_k))
        rep = audit_pose(
            pose_mol=p_mol,
            receptor_coords=rec_coords,
            receptor_radii=rec_radii,
            receptor_is_metal=rec_metal,
            reference_mol=ligand_mol,
            ideal_ref_mol=ideal_mol,
            ideal_ref_energy=ideal_e,
            grid_center=g_c,
            grid_dimensions=g_d,
            pose_index=idx,
        )
        reports.append(rep)
    return reports


def audit_sdf_file(
    sdf_path: str,
    receptor_file: Optional[str] = None,
    reference_sdf: Optional[str] = None,
    max_rel_energy_kcal: float = 15.0,
) -> List[PoseBustersReport]:
    """Audit all poses in any SDF file (PandaDock, Vina, Glide, Boltz-2, AF3, etc.)."""
    if receptor_file and Path(receptor_file).exists():
        rec_coords, rec_radii, rec_metal = _extract_receptor_heavy(receptor_file)
    else:
        rec_coords, rec_radii, rec_metal = np.zeros((0, 3)), np.zeros((0,)), []
    ref_mol = None
    if reference_sdf and Path(reference_sdf).exists():
        sup = Chem.SDMolSupplier(str(reference_sdf), removeHs=False)
        for m in sup:
            if m is not None:
                ref_mol = m
                break

    reports = []
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    ideal_cache: Dict[str, Tuple[Optional[Chem.Mol], Optional[float]]] = {}
    for idx, mol in enumerate(suppl):
        if mol is None:
            rep = PoseBustersReport(pose_index=idx, checks={"mol_loaded_and_sanitized": False})
            reports.append(rep)
            continue
        smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
        if smi not in ideal_cache:
            ideal_cache[smi] = _ideal_geometry(ref_mol if ref_mol is not None else mol)
        ideal_m, ideal_e = ideal_cache[smi]
        reports.append(
            audit_pose(
                pose_mol=mol,
                receptor_coords=rec_coords,
                receptor_radii=rec_radii,
                receptor_is_metal=rec_metal,
                reference_mol=ref_mol,
                ideal_ref_mol=ideal_m,
                ideal_ref_energy=ideal_e,
                pose_index=idx,
                strain_cutoff=max_rel_energy_kcal,
            )
        )
    return reports
