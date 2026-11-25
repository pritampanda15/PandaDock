"""
Differentiable geometric, H-bond, metal, pharmacophore, and MCS core restraints.

All restraints use a flat-bottom harmonic potential:
- Zero energy and zero gradient inside `[min_dist, max_dist]`, so a satisfied
  restraint never distorts the underlying Vina/Vinardo energy landscape.
- Smooth quadratic penalty `k * excess^2` outside `[min_dist, max_dist]` with
  exact analytic gradients on both CPU (NumPy) and GPU/MPS (PyTorch).

CLI specification syntax (repeatable via `--restraint`):
    CHAIN:RESNAME:RESNUM:ATOM<MAXDIST[:SELECTOR]
    CHAIN:RESNAME:RESNUM:ATOM=MINDIST-MAXDIST[:SELECTOR]
    point:X,Y,Z<MAXDIST[:SELECTOR]

Examples:
    A:ZN:301:ZN<2.3:donor
    A:ASP:128:OD1<3.2:hbond
    point:126.5,119.2,146.1<2.0:aromatic
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import rdFMCS

logger = logging.getLogger("pandadock.docking.search.restraints")


@dataclass
class DistanceRestraint:
    """Single flat-bottom distance restraint between ligand atom(s) and a 3D target point."""

    target_xyz: np.ndarray
    ligand_indices: np.ndarray
    min_dist: float = 0.0
    max_dist: float = 3.5
    force_constant: float = 15.0
    mode: str = "min"  # "min" | "atom" | "centroid"
    label: str = ""

    def __post_init__(self) -> None:
        self.target_xyz = np.asarray(self.target_xyz, dtype=np.float64).reshape(3)
        self.ligand_indices = np.asarray(self.ligand_indices, dtype=np.int64).reshape(-1)
        if self.ligand_indices.size == 0:
            raise ValueError(f"Restraint '{self.label}' matched zero ligand atoms")
        self.min_dist = float(max(0.0, self.min_dist))
        self.max_dist = float(max(self.min_dist, self.max_dist))
        self.force_constant = float(max(0.0, self.force_constant))
        if self.mode not in ("min", "atom", "centroid"):
            raise ValueError(f"Unsupported restraint mode '{self.mode}'")

    def distance(self, coords: np.ndarray) -> float:
        """Current effective distance (Å) for this restraint."""
        pts = coords[self.ligand_indices]
        if self.mode == "centroid":
            delta = pts.mean(axis=0) - self.target_xyz
            return float(np.linalg.norm(delta))
        deltas = pts - self.target_xyz[None, :]
        dists = np.linalg.norm(deltas, axis=1)
        if self.mode == "atom" or len(dists) == 1:
            return float(dists[0])
        return float(np.min(dists))

    def energy_and_gradient(
        self, coords: np.ndarray, need_gradient: bool = True
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Compute flat-bottom harmonic penalty (kcal/mol) and per-atom Cartesian gradient."""
        if self.force_constant <= 0.0:
            return 0.0, (np.zeros_like(coords) if need_gradient else None)

        pts = coords[self.ligand_indices]
        if self.mode == "centroid":
            offset = pts.mean(axis=0) - self.target_xyz
            dist = float(np.linalg.norm(offset))
            atom_targets = self.ligand_indices
            weight = 1.0 / len(atom_targets)
        else:
            deltas = pts - self.target_xyz[None, :]
            dists = np.linalg.norm(deltas, axis=1)
            pick = 0 if (self.mode == "atom" or len(dists) == 1) else int(np.argmin(dists))
            offset = deltas[pick]
            dist = float(dists[pick])
            atom_targets = self.ligand_indices[pick : pick + 1]
            weight = 1.0

        if dist > self.max_dist:
            excess = dist - self.max_dist
            sign = 1.0
        elif dist < self.min_dist:
            excess = self.min_dist - dist
            sign = -1.0
        else:
            return 0.0, (np.zeros_like(coords) if need_gradient else None)

        energy = float(self.force_constant * excess * excess)
        if not need_gradient:
            return energy, None

        grad = np.zeros_like(coords, dtype=np.float64)
        if dist > 1e-9:
            direction = offset / dist
        else:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        per_atom = (2.0 * self.force_constant * excess * sign * weight) * direction
        grad[atom_targets] += per_atom
        return energy, grad


class RestraintSet:
    """Collection of differentiable restraints evaluated on CPU or GPU."""

    def __init__(self, restraints: Optional[Sequence[DistanceRestraint]] = None):
        self.restraints: List[DistanceRestraint] = list(restraints or [])

    def __len__(self) -> int:
        return len(self.restraints)

    def __bool__(self) -> bool:
        return bool(self.restraints)

    def add(self, restraint: DistanceRestraint) -> None:
        self.restraints.append(restraint)

    def extend(self, other: "RestraintSet") -> None:
        self.restraints.extend(other.restraints)

    def energy_and_gradient(
        self, coords: np.ndarray, need_gradient: bool = True
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Sum flat-bottom restraint energies and Cartesian gradients on CPU."""
        if not self.restraints:
            return 0.0, (np.zeros_like(coords) if need_gradient else None)

        total_e = 0.0
        total_g = np.zeros_like(coords, dtype=np.float64) if need_gradient else None
        for r in self.restraints:
            e, g = r.energy_and_gradient(coords, need_gradient=need_gradient)
            total_e += e
            if need_gradient and g is not None:
                total_g += g
        return total_e, total_g

    def evaluate_summary(self, coords: np.ndarray) -> List[Dict[str, Any]]:
        """Return per-restraint distances, violations, penalties, and satisfaction flags."""
        summary: List[Dict[str, Any]] = []
        for r in self.restraints:
            d = r.distance(coords)
            e, _ = r.energy_and_gradient(coords, need_gradient=False)
            if d > r.max_dist:
                viol = d - r.max_dist
            elif d < r.min_dist:
                viol = r.min_dist - d
            else:
                viol = 0.0
            summary.append(
                {
                    "label": r.label or f"restraint_{len(summary) + 1}",
                    "mode": r.mode,
                    "min_dist": r.min_dist,
                    "max_dist": r.max_dist,
                    "distance_A": round(d, 3),
                    "violation_A": round(viol, 3),
                    "penalty_kcal": round(e, 3),
                    "satisfied": bool(viol <= 0.15),
                }
            )
        return summary

    def torch_energy_and_gradient(
        self, coords_batch, need_gradient: bool = True
    ):
        """Batched PyTorch evaluation matching CPU `energy_and_gradient` exactly.

        Args:
            coords_batch: Tensor of shape `(B, N_atoms, 3)`.
        Returns:
            `(energy [B], grad_atoms [B, N_atoms, 3] or None)`.
        """
        import torch

        batch_size = coords_batch.shape[0]
        device = coords_batch.device
        dtype = coords_batch.dtype
        total_e = torch.zeros(batch_size, device=device, dtype=dtype)
        total_g = (
            torch.zeros_like(coords_batch) if need_gradient else None
        )
        if not self.restraints:
            return total_e, total_g

        for r in self.restraints:
            if r.force_constant <= 0.0:
                continue
            target = torch.as_tensor(r.target_xyz, device=device, dtype=dtype)
            idx = torch.as_tensor(r.ligand_indices, device=device, dtype=torch.long)
            pts = coords_batch[:, idx, :]  # (B, M, 3)

            if r.mode == "centroid":
                offset = pts.mean(dim=1) - target.unsqueeze(0)  # (B, 3)
                dist = torch.linalg.norm(offset, dim=-1)  # (B,)
                upper = torch.clamp(dist - r.max_dist, min=0.0)
                lower = torch.clamp(r.min_dist - dist, min=0.0)
                excess = upper + lower
                total_e = total_e + r.force_constant * excess * excess
                if need_gradient:
                    sign = torch.where(
                        upper > 0.0,
                        torch.ones_like(dist),
                        torch.where(lower > 0.0, -torch.ones_like(dist), torch.zeros_like(dist)),
                    )
                    safe_dist = dist.clamp_min(1e-9).unsqueeze(-1)
                    direction = offset / safe_dist
                    per_atom = (
                        (2.0 * r.force_constant * excess * sign / float(len(r.ligand_indices)))
                        .unsqueeze(-1)
                        * direction
                    )  # (B, 3)
                    expanded = per_atom.unsqueeze(1).expand(-1, len(r.ligand_indices), -1)
                    total_g.index_add_(1, idx, expanded)
            else:
                deltas = pts - target.view(1, 1, 3)  # (B, M, 3)
                dists = torch.linalg.norm(deltas, dim=-1)  # (B, M)
                if r.mode == "atom" or len(r.ligand_indices) == 1:
                    pick = torch.zeros(batch_size, dtype=torch.long, device=device)
                else:
                    pick = torch.argmin(dists, dim=1)  # (B,)
                batch_idx = torch.arange(batch_size, device=device)
                dist = dists[batch_idx, pick]  # (B,)
                offset = deltas[batch_idx, pick, :]  # (B, 3)

                upper = torch.clamp(dist - r.max_dist, min=0.0)
                lower = torch.clamp(r.min_dist - dist, min=0.0)
                excess = upper + lower
                total_e = total_e + r.force_constant * excess * excess

                if need_gradient:
                    sign = torch.where(
                        upper > 0.0,
                        torch.ones_like(dist),
                        torch.where(lower > 0.0, -torch.ones_like(dist), torch.zeros_like(dist)),
                    )
                    safe_dist = dist.clamp_min(1e-9).unsqueeze(-1)
                    direction = offset / safe_dist
                    atom_grad = (2.0 * r.force_constant * excess * sign).unsqueeze(-1) * direction
                    chosen_atom = idx[pick]  # (B,)
                    total_g[batch_idx, chosen_atom, :] = (
                        total_g[batch_idx, chosen_atom, :] + atom_grad
                    )

        return total_e, total_g


# ---------------------------------------------------------------------------
# Selectors and parsers
# ---------------------------------------------------------------------------

def select_ligand_atoms(mol: Chem.Mol, selector: str = "heavy") -> np.ndarray:
    """Resolve a human-readable ligand atom selector into 0-based atom indices."""
    sel = (selector or "heavy").strip()
    low = sel.lower()

    if low.startswith("atom:"):
        raw = sel.split(":", 1)[1]
        indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
        for idx in indices:
            if idx < 0 or idx >= mol.GetNumAtoms():
                raise ValueError(f"Ligand atom index {idx} out of range (0..{mol.GetNumAtoms()-1})")
        return np.array(indices, dtype=np.int64)

    if low.startswith("smarts:"):
        pat = Chem.MolFromSmarts(sel.split(":", 1)[1])
        if pat is None:
            raise ValueError(f"Invalid SMARTS selector: {sel}")
        matches = mol.GetSubstructMatches(pat)
        indices = sorted({i for m in matches for i in m if mol.GetAtomWithIdx(i).GetAtomicNum() > 1})
        if not indices:
            raise ValueError(f"SMARTS selector '{sel}' matched no ligand atoms")
        return np.array(indices, dtype=np.int64)

    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]

    if low in ("heavy", "any", "all"):
        return np.array(heavy, dtype=np.int64)

    if low in ("donor", "metal", "coord", "coordinating"):
        # Metal-coordinating heteroatoms (N, O, S, P)
        idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8, 15, 16)]
        return np.array(idx or heavy, dtype=np.int64)

    if low in ("hbd", "hbond_donor"):
        idx = [
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() in (7, 8, 16) and a.GetTotalNumHs() > 0
        ]
        return np.array(idx or [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8, 16)] or heavy, dtype=np.int64)

    if low in ("hba", "hbond_acceptor"):
        idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8, 9, 16)]
        return np.array(idx or heavy, dtype=np.int64)

    if low in ("hbond", "polar"):
        idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() in (7, 8, 16)]
        return np.array(idx or heavy, dtype=np.int64)

    if low in ("aromatic", "ring"):
        idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic() and a.GetAtomicNum() > 1]
        return np.array(idx or heavy, dtype=np.int64)

    if low in ("hydrophobic", "lipophilic"):
        idx = [
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 6
            and not any(nb.GetAtomicNum() in (7, 8) for nb in a.GetNeighbors())
        ]
        return np.array(idx or heavy, dtype=np.int64)

    raise ValueError(
        f"Unknown ligand selector '{selector}'. Supported: heavy, donor, hbd, hba, "
        "hbond, aromatic, hydrophobic, atom:<idx>, smarts:<pattern>"
    )


def find_receptor_atom_coord(
    receptor_file: str,
    chain_id: str,
    resname: str,
    resnum: int,
    atom_name: str,
) -> np.ndarray:
    """Locate a receptor atom (standard residue, HETATM, or metal ion) in a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("receptor", receptor_file)
    model = next(structure.get_models())
    target_chain = chain_id.strip()
    target_resname = resname.strip().upper()
    target_atom = atom_name.strip().upper()

    for chain in model:
        if target_chain and target_chain != "*" and chain.id.strip() != target_chain:
            continue
        for res in chain:
            if int(res.id[1]) != int(resnum):
                continue
            rn = res.get_resname().strip().upper()
            if target_resname and target_resname != "*" and rn != target_resname:
                continue
            for atom in res.get_atoms():
                if atom.get_name().strip().upper() == target_atom:
                    return np.asarray(atom.get_coord(), dtype=np.float64)

    raise ValueError(
        f"Receptor atom {chain_id}:{resname}:{resnum}:{atom_name} not found in {receptor_file}"
    )


_RESTRAINT_RE = re.compile(
    r"^(?P<target>[^<=]+?)(?P<op><|=)(?P<bounds>[0-9.\-]+)(?::(?P<selector>.+))?$"
)


def parse_restraint_spec(
    spec: str,
    receptor_file: str,
    ligand_mol: Chem.Mol,
    force_constant: float = 15.0,
) -> DistanceRestraint:
    """Parse a CLI restraint string into a `DistanceRestraint`.

    Supported forms:
      - `CHAIN:RESNAME:RESNUM:ATOM<MAXDIST[:SELECTOR]`
      - `CHAIN:RESNAME:RESNUM:ATOM=MINDIST-MAXDIST[:SELECTOR]`
      - `point:X,Y,Z<MAXDIST[:SELECTOR]`
      - `point:X,Y,Z=MINDIST-MAXDIST[:SELECTOR]`
    """
    m = _RESTRAINT_RE.match(spec.strip())
    if not m:
        raise ValueError(
            f"Invalid restraint specification '{spec}'. Expected e.g. "
            "'A:ZN:301:ZN<2.3:donor' or 'point:10.0,20.0,30.0<3.0:hbond'"
        )

    target_str = m.group("target").strip()
    op = m.group("op")
    bounds = m.group("bounds").strip()
    selector = (m.group("selector") or "heavy").strip()

    if op == "<":
        min_d, max_d = 0.0, float(bounds)
    else:
        if "-" in bounds:
            lo_s, hi_s = bounds.split("-", 1)
            min_d, max_d = float(lo_s), float(hi_s)
        else:
            val = float(bounds)
            min_d, max_d = max(0.0, val - 0.3), val + 0.3

    if target_str.lower().startswith("point:"):
        xyz_parts = target_str.split(":", 1)[1].split(",")
        if len(xyz_parts) != 3:
            raise ValueError(f"Expected 3 coordinates in '{target_str}'")
        target_xyz = np.array([float(v) for v in xyz_parts], dtype=np.float64)
    else:
        parts = target_str.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Expected CHAIN:RESNAME:RESNUM:ATOM in '{target_str}'"
            )
        chain_id, resname, resnum_s, atom_name = parts
        target_xyz = find_receptor_atom_coord(
            receptor_file, chain_id, resname, int(resnum_s), atom_name
        )

    lig_idx = select_ligand_atoms(ligand_mol, selector)
    mode = "atom" if selector.lower().startswith("atom:") and len(lig_idx) == 1 else "min"
    return DistanceRestraint(
        target_xyz=target_xyz,
        ligand_indices=lig_idx,
        min_dist=min_d,
        max_dist=max_d,
        force_constant=force_constant,
        mode=mode,
        label=spec.strip(),
    )


def load_restraints_json(
    json_path: str,
    receptor_file: str,
    ligand_mol: Chem.Mol,
) -> RestraintSet:
    """Load restraints from a JSON file (list of spec strings or dicts)."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    entries = data.get("restraints", data) if isinstance(data, dict) else data
    rset = RestraintSet()
    for entry in entries:
        if isinstance(entry, str):
            rset.add(parse_restraint_spec(entry, receptor_file, ligand_mol))
            continue
        if not isinstance(entry, dict):
            continue
        if "spec" in entry:
            rset.add(
                parse_restraint_spec(
                    entry["spec"],
                    receptor_file,
                    ligand_mol,
                    force_constant=float(entry.get("force_constant", 15.0)),
                )
            )
            continue
        if "target_xyz" in entry:
            xyz = np.asarray(entry["target_xyz"], dtype=np.float64)
        else:
            xyz = find_receptor_atom_coord(
                receptor_file,
                str(entry.get("chain", "*")),
                str(entry.get("resname", "*")),
                int(entry["resnum"]),
                str(entry["atom"]),
            )
        lig_idx = select_ligand_atoms(ligand_mol, str(entry.get("selector", "heavy")))
        rset.add(
            DistanceRestraint(
                target_xyz=xyz,
                ligand_indices=lig_idx,
                min_dist=float(entry.get("min_dist", 0.0)),
                max_dist=float(entry.get("max_dist", 3.5)),
                force_constant=float(entry.get("force_constant", 15.0)),
                mode=str(entry.get("mode", "min")),
                label=str(entry.get("label", entry.get("atom", "restraint"))),
            )
        )
    return rset


def build_mcs_restraints(
    ligand_mol: Chem.Mol,
    reference_mol: Chem.Mol,
    tolerance: float = 0.6,
    force_constant: float = 12.0,
    min_mcs_atoms: int = 5,
    core_smarts: Optional[str] = None,
) -> Tuple[RestraintSet, Optional[str], int]:
    """Build per-atom flat-bottom MCS core restraints toward a 3D reference ligand.

    Returns:
        `(restraint_set, mcs_smarts, n_mapped_atoms)`.
        If the MCS is smaller than `min_mcs_atoms`, returns an empty `RestraintSet`.
    """
    if reference_mol.GetNumConformers() == 0:
        raise ValueError("Reference ligand must have 3D coordinates for MCS tethering")

    if core_smarts:
        pattern = Chem.MolFromSmarts(core_smarts)
        smarts = core_smarts
    else:
        mcs = rdFMCS.FindMCS(
            [Chem.RemoveHs(reference_mol), Chem.RemoveHs(ligand_mol)],
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,
            timeout=5,
        )
        smarts = mcs.smartsString if mcs.numAtoms >= min_mcs_atoms else None
        pattern = Chem.MolFromSmarts(smarts) if smarts else None

    if pattern is None:
        return RestraintSet(), smarts, 0

    ref_match = reference_mol.GetSubstructMatch(pattern)
    lig_match = ligand_mol.GetSubstructMatch(pattern)
    if not ref_match or not lig_match or len(ref_match) < min_mcs_atoms:
        return RestraintSet(), smarts, 0

    ref_conf = reference_mol.GetConformer(0)
    rset = RestraintSet()
    for r_idx, l_idx in zip(ref_match, lig_match):
        if ligand_mol.GetAtomWithIdx(l_idx).GetAtomicNum() <= 1:
            continue
        pos = ref_conf.GetAtomPosition(int(r_idx))
        rset.add(
            DistanceRestraint(
                target_xyz=np.array([pos.x, pos.y, pos.z], dtype=np.float64),
                ligand_indices=np.array([int(l_idx)], dtype=np.int64),
                min_dist=0.0,
                max_dist=float(tolerance),
                force_constant=float(force_constant),
                mode="atom",
                label=f"mcs_core_{l_idx}->{r_idx}",
            )
        )
    return rset, smarts, len(rset)
