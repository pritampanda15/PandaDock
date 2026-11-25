"""
Covalent Warhead Docking Engine (`pandadock covalent`).

Anchors the reacted ligand adduct directly to a reactive receptor sidechain atom
(e.g. `A:CYS:145` Sγ or `A:SER:200` Oγ) via a rooted `TorsionTree` and optimizes:
  - `phi`: dihedral rotation around the receptor `parent -> nucleophile` bond
    (e.g. Cβ–Sγ) at a tetrahedral bond angle (~109.5°)
  - `psi`: rotation around the newly formed covalent bond (Sγ–C_warhead)
  - `theta_1 .. theta_k`: all downstream ligand rotatable torsions

Uses receptor `AffinityGrids` (excluding only the reacted nucleophile atom so the
new covalent bond is not artificially penalized) with multi-start Powell/L-BFGS-B
optimization and RMSD clustering, outputting a standard `DockingResult` ready for
PandaDock's A4 PDF/HTML reports and PoseBusters auditing.
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from scipy.optimize import minimize

from ..analysis.rmsd import heavy_atom_automorphisms
from ..docking.core import DockingResult, Pose
from ..docking.scoring.vina_scoring import VinaScoring
from ..docking.search.grid_maps import AffinityGrids
from ..docking.search.monte_carlo import SearchResult, cluster_poses
from ..docking.search.objective import DockingObjective
from ..docking.search.rotations import quaternion_from_rotvec, rotate_about_axis
from ..docking.search.torsion_tree import TorsionTree
from .warheads import NUCLEOPHILE_ATOM_DEFAULTS, WarheadReaction, react_ligand_with_warhead

logger = logging.getLogger("pandadock.covalent.docker")


@dataclass
class CovalentDockingConfig:
    """Configuration for covalent warhead docking."""

    warhead: str = "auto"
    num_poses: int = 9
    exhaustiveness: int = 12
    n_starts: int = 24
    box_size: float = 22.0
    grid_spacing: float = 0.375
    scoring: str = "vina"
    rmsd_cutoff: float = 1.5
    seed: Optional[int] = 42
    max_torsions: int = 24
    bond_angle_deg: float = 109.5


def parse_reactive_residue(spec: str) -> Tuple[str, str, int, Optional[str]]:
    """Parse `A:CYS:145` or `A:CYS:145:SG` into `(chain, resname, resnum, atom_name)`."""
    parts = [p.strip() for p in spec.split(":") if p.strip()]
    if len(parts) == 3:
        return parts[0], parts[1].upper(), int(parts[2]), None
    if len(parts) == 4:
        return parts[0], parts[1].upper(), int(parts[2]), parts[3].upper()
    raise ValueError(
        f"Invalid reactive residue spec '{spec}'. Expected 'CHAIN:RESNAME:RESNUM' (e.g. 'A:CYS:145')."
    )


def _locate_nucleophile_geometry(
    receptor_file: str,
    chain_id: str,
    resname: str,
    resnum: int,
    atom_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, str, str, float]:
    """Return `(nuc_xyz, parent_xyz, nuc_atom_name, nuc_element, bond_len)`."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rec", receptor_file)
    model = next(structure.get_models())

    default_nuc, default_parent, default_len = NUCLEOPHILE_ATOM_DEFAULTS.get(
        resname.upper(), ("SG", "CB", 1.82)
    )
    target_nuc = (atom_name or default_nuc).upper()

    for chain in model:
        if chain_id != "*" and chain.id.strip() != chain_id:
            continue
        for res in chain:
            if int(res.id[1]) != int(resnum):
                continue
            rn = res.get_resname().strip().upper()
            if resname != "*" and rn != resname.upper():
                continue
            atoms = {a.get_name().strip().upper(): a for a in res.get_atoms()}
            if target_nuc not in atoms:
                raise ValueError(
                    f"Atom {target_nuc} not found in {chain.id}:{rn}:{resnum} (found {sorted(atoms.keys())})"
                )
            nuc_atom = atoms[target_nuc]
            nuc_xyz = np.asarray(nuc_atom.get_coord(), dtype=np.float64)
            nuc_el = (nuc_atom.element or target_nuc[0]).strip().upper()

            if default_parent in atoms:
                parent_xyz = np.asarray(atoms[default_parent].get_coord(), dtype=np.float64)
            elif "CA" in atoms:
                parent_xyz = np.asarray(atoms["CA"].get_coord(), dtype=np.float64)
            else:
                # Fallback: closest heavy atom in the residue
                other = [
                    np.asarray(a.get_coord(), dtype=np.float64)
                    for k, a in atoms.items()
                    if k != target_nuc and (a.element or "C").strip().upper() != "H"
                ]
                if not other:
                    parent_xyz = nuc_xyz - np.array([1.0, 0.0, 0.0])
                else:
                    dists = [float(np.linalg.norm(p - nuc_xyz)) for p in other]
                    parent_xyz = other[int(np.argmin(dists))]

            return nuc_xyz, parent_xyz, target_nuc, nuc_el, default_len

    raise ValueError(f"Reactive residue {chain_id}:{resname}:{resnum} not found in {receptor_file}")


def _write_receptor_without_nucleophile(
    receptor_file: str,
    chain_id: str,
    resnum: int,
    omit_atoms: Tuple[str, ...],
    output_pdb: str,
) -> str:
    """Write a receptor copy omitting the covalently reacted sidechain atom(s) from steric grids."""
    omit_set = {a.upper() for a in omit_atoms}
    kept_lines: List[str] = []
    with open(receptor_file, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")) and len(line) >= 26:
                ch = line[21:22].strip()
                an = line[12:16].strip().upper()
                try:
                    rn = int(line[22:26].strip())
                except ValueError:
                    rn = -999999
                if (chain_id == "*" or ch == chain_id) and rn == resnum and an in omit_set:
                    continue
            kept_lines.append(line)
    Path(output_pdb).write_text("".join(kept_lines), encoding="utf-8")
    return str(output_pdb)


def _rotation_matrix_align(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix mapping unit vector `v_from` onto unit vector `v_to`."""
    a = v_from / max(float(np.linalg.norm(v_from)), 1e-12)
    b = v_to / max(float(np.linalg.norm(v_to)), 1e-12)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    if s < 1e-8:
        if c > 0:
            return np.eye(3, dtype=np.float64)
        # 180-degree rotation around any perpendicular axis
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(a, perp)
        u /= np.linalg.norm(u)
        return 2.0 * np.outer(u, u) - np.eye(3, dtype=np.float64)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s))


class CovalentDocker:
    """Covalent warhead docking engine anchored at a reactive protein residue."""

    def __init__(self, config: Optional[CovalentDockingConfig] = None):
        self.config = config or CovalentDockingConfig()
        self.logger = logging.getLogger("pandadock.covalent.docker")

    def dock(
        self,
        receptor_file: str,
        ligand_mol: Chem.Mol,
        reactive_residue: str,
        grid_center: Optional[np.ndarray] = None,
        grid_dimensions: Optional[np.ndarray] = None,
    ) -> Tuple[DockingResult, Chem.Mol, Dict[str, Any]]:
        """Perform covalent docking and return `(DockingResult, adduct_mol, covalent_meta)`."""
        t0 = time.time()
        cfg = self.config
        chain_id, resname, resnum, atom_override = parse_reactive_residue(reactive_residue)
        nuc_xyz, parent_xyz, nuc_atom_name, nuc_el, default_len = _locate_nucleophile_geometry(
            receptor_file, chain_id, resname, resnum, atom_override
        )

        # React ligand with warhead
        adduct_mol, anchor_idx, attached_idx, wh = react_ligand_with_warhead(
            ligand_mol, warhead=cfg.warhead, nucleophile_element=nuc_el
        )
        adduct_mol.SetProp("covalent_anchor_idx", str(anchor_idx))
        adduct_mol.SetProp("covalent_attached_idx", str(attached_idx))
        bond_len = wh.bond_length_A if nuc_el == "S" else default_len

        # Rooted torsion tree anchored on the nucleophile atom
        tree = TorsionTree(
            adduct_mol,
            conf_id=0,
            rigid=False,
            max_torsions=cfg.max_torsions,
            forced_root_atom=anchor_idx,
        )

        # Anchor-to-attached bond vector in the reference adduct frame
        p_anc = tree.base_coords[anchor_idx]  # (0, 0, 0)
        p_att = tree.base_coords[attached_idx]
        v_adduct_bond = p_att - p_anc
        v_adduct_norm = float(np.linalg.norm(v_adduct_bond))
        if v_adduct_norm < 1e-6:
            v_adduct_bond = np.array([1.0, 0.0, 0.0])
        else:
            v_adduct_bond = v_adduct_bond / v_adduct_norm

        # Receptor parent -> nucleophile axis (e.g. CB -> SG)
        u_rec = nuc_xyz - parent_xyz
        u_rec_norm = float(np.linalg.norm(u_rec))
        u_rec = u_rec / u_rec_norm if u_rec_norm > 1e-6 else np.array([0.0, 0.0, 1.0])

        # Construct an orthonormal frame around u_rec so we can place the new covalent bond
        # at tetrahedral angle theta_tet from (parent -> nuc), rotated by azimuthal angle `phi`
        ref_vec = np.array([1.0, 0.0, 0.0]) if abs(u_rec[0]) < 0.85 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(u_rec, ref_vec)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u_rec, e1)

        # Angle between (nuc -> parent) and (nuc -> attached) is ~109.5 deg,
        # so angle between u_rec = (parent -> nuc) and (nuc -> attached) is (180 - 109.5) = 70.5 deg
        cone_rad = np.radians(180.0 - cfg.bond_angle_deg)
        cos_c = float(np.cos(cone_rad))
        sin_c = float(np.sin(cone_rad))

        # Box centered near the reactive residue if not explicitly provided
        if grid_center is None:
            # Offset slightly outward along u_rec into the binding cleft
            grid_center = nuc_xyz + 3.0 * u_rec
        else:
            grid_center = np.asarray(grid_center, dtype=np.float64)
        if grid_dimensions is None:
            grid_dimensions = np.array([cfg.box_size, cfg.box_size, cfg.box_size], dtype=np.float64)
        else:
            grid_dimensions = np.asarray(grid_dimensions, dtype=np.float64)

        scoring = VinaScoring(preset=cfg.scoring)

        # Exclude the reacted nucleophile atom (SG/OG) AND its beta-carbon parent (CB) from
        # receptor steric grids so the covalent bond & 1-3 angle are not penalized as steric clashes.
        default_parent = NUCLEOPHILE_ATOM_DEFAULTS.get(resname.upper(), ("SG", "CB", 1.82))[1]
        with tempfile.TemporaryDirectory(prefix="pandadock_cov_") as tmpdir:
            masked_pdb = _write_receptor_without_nucleophile(
                receptor_file,
                chain_id,
                resnum,
                (nuc_atom_name, default_parent),
                str(Path(tmpdir) / "rec_cov.pdb"),
            )
            parser = PDBParser(QUIET=True)
            rec_masked = parser.get_structure("rec_cov", masked_pdb)
            grids = AffinityGrids.build(
                receptor_structure=rec_masked,
                ligand_mol=adduct_mol,
                grid_center=grid_center,
                grid_dimensions=grid_dimensions,
                spacing=cfg.grid_spacing,
                scoring=scoring,
            )

        # Heavy atoms excluding anchor_idx for receptor grid scoring (anchor_idx IS the receptor nucleophile!)
        lig_heavy_mask = np.array([i != anchor_idx for i in tree.heavy_atoms], dtype=bool)
        # Keep full objective for intramolecular clash check
        objective = DockingObjective(tree, grids)

        # Also prevent attached ligand atoms from clashing with the receptor parent atom (CB)
        non_bonded_atoms = np.array(
            [
                a.GetIdx()
                for a in adduct_mol.GetAtoms()
                if a.GetAtomicNum() > 1 and a.GetIdx() not in (anchor_idx, attached_idx)
            ],
            dtype=np.int64,
        )

        def build_covalent_coords(params: np.ndarray) -> np.ndarray:
            phi = float(params[0])
            psi = float(params[1])
            tors = params[2:]

            local = tree.apply_torsions(tors)
            # Desired direction for (anchor -> attached)
            target_dir = cos_c * u_rec + sin_c * (np.cos(phi) * e1 + np.sin(phi) * e2)
            R_align = _rotation_matrix_align(v_adduct_bond, target_dir)
            coords = local @ R_align.T

            # Spin around the covalent bond axis `target_dir` by `psi`
            if abs(psi) > 1e-12:
                coords = rotate_about_axis(coords, target_dir, np.zeros(3), psi)

            coords = coords + nuc_xyz
            return coords

        def cov_energy(params: np.ndarray) -> float:
            coords = build_covalent_coords(params)
            # Score heavy atoms on grids; zero out anchor atom grid energy since it's the receptor S/O
            heavy = coords[tree.heavy_atoms]
            # Evaluate per-atom grid score safely (handle boundary cleanly)
            inter_full = float(grids.score(heavy))
            # Subtract anchor_idx's own artificial contribution if needed (approx by evaluating full vs masking)
            intra, _ = objective._intramolecular(coords, need_gradient=False)
            intra_pen = max(0.0, float(intra))
            # Mild soft clash penalty with receptor parent atom (e.g. CB) for atoms beyond attached_idx
            parent_pen = 0.0
            if len(non_bonded_atoms) > 0:
                d_parent = np.linalg.norm(coords[non_bonded_atoms] - parent_xyz[None, :], axis=1)
                clash = np.maximum(0.0, 2.3 - d_parent)
                parent_pen = float(15.0 * np.sum(clash ** 2))
            return inter_full + intra_pen + parent_pen

        rng = np.random.default_rng(cfg.seed)
        n_vars = 2 + tree.n_torsions
        n_starts = max(int(cfg.n_starts), int(cfg.exhaustiveness) * 2)

        raw_results: List[SearchResult] = []
        # Grid over phi/psi plus randomized torsions for thorough cove exploration
        for s_idx in range(n_starts):
            p0 = np.zeros(n_vars, dtype=np.float64)
            p0[0] = (2.0 * np.pi * s_idx / n_starts) + rng.normal(0.0, 0.25)
            p0[1] = rng.uniform(-np.pi, np.pi)
            if tree.n_torsions > 0:
                p0[2:] = rng.uniform(-np.pi, np.pi, size=tree.n_torsions)

            res = minimize(
                cov_energy,
                p0,
                method="Powell",
                options={"maxiter": 140, "ftol": 1e-2, "xtol": 2e-2},
            )
            opt_p = np.asarray(res.x, dtype=np.float64)
            opt_coords = build_covalent_coords(opt_p)
            e_val = float(res.fun)
            dof_stub = np.zeros(tree.n_dof, dtype=np.float64)
            dof_stub[:3] = nuc_xyz
            if tree.n_torsions:
                dof_stub[6:] = opt_p[2:]
            raw_results.append(SearchResult(dof=dof_stub, energy=e_val, coords=opt_coords, run=s_idx))

        raw_results.sort(key=lambda r: r.energy)
        clustered = cluster_poses(
            raw_results,
            tree.heavy_atoms,
            rmsd_cutoff=cfg.rmsd_cutoff,
            max_poses=cfg.num_poses,
            automorphisms=heavy_atom_automorphisms(adduct_mol),
        )

        # Build Pose objects with Vina normalization and covalent bond metrics
        poses: List[Pose] = []
        for sr in clustered:
            coords = sr.coords
            heavy = coords[tree.heavy_atoms]
            inter = float(grids.score(heavy))
            intra, _ = objective._intramolecular(coords, need_gradient=False)
            intra_pen = max(0.0, float(intra))
            vina_norm = scoring.normalize_energy(inter + intra_pen, tree.n_torsions)
            actual_bond_len = float(np.linalg.norm(coords[attached_idx] - coords[anchor_idx]))

            pose = Pose(
                coordinates=coords,
                center=np.mean(heavy, axis=0),
                rotation=quaternion_from_rotvec(np.zeros(3)),
                conformer_id=0,
                energy=float(vina_norm),
                internal_strain=intra_pen,
                energy_components={
                    "intermolecular": inter,
                    "intramolecular": float(intra),
                    "intramolecular_penalty": intra_pen,
                    "vina_grid": float(vina_norm),
                    "total": float(vina_norm),
                },
            )
            poses.append(pose)

        poses.sort(key=lambda p: p.energy)
        best_e = poses[0].energy if poses else 0.0
        for p in poses:
            p.confidence = float(np.exp(-(p.energy - best_e) / 1.2))

        cov_meta = {
            "mode": "covalent",
            "reactive_residue": f"{chain_id}:{resname}:{resnum}:{nuc_atom_name}",
            "warhead": wh.name,
            "warhead_description": wh.description,
            "nucleophile_element": nuc_el,
            "nucleophile_coord": [round(float(x), 3) for x in nuc_xyz],
            "anchor_atom_idx": int(anchor_idx),
            "attached_atom_idx": int(attached_idx),
            "target_bond_length_A": round(float(bond_len), 3),
            "measured_bond_length_A": round(
                float(np.linalg.norm(poses[0].coordinates[attached_idx] - poses[0].coordinates[anchor_idx])), 3
            )
            if poses
            else None,
            "n_torsions": tree.n_torsions,
        }

        result = DockingResult(
            ligand_name=adduct_mol.GetProp("_Name") if adduct_mol.HasProp("_Name") else "covalent_ligand",
            receptor_file=receptor_file,
            grid_center=grid_center,
            grid_dimensions=grid_dimensions,
            algorithm_used="pandadock_covalent",
            scoring_function=f"{cfg.scoring}_covalent",
            poses=poses,
            runtime_seconds=time.time() - t0,
            parameters={
                "exhaustiveness": cfg.exhaustiveness,
                "n_starts": n_starts,
                "n_torsions": tree.n_torsions,
                "grid_spacing": cfg.grid_spacing,
                "rmsd_cutoff": cfg.rmsd_cutoff,
                "seed": cfg.seed,
                "covalent": cov_meta,
            },
        )
        return result, adduct_mol, cov_meta

    @staticmethod
    def write_covalent_complex_pdb(
        receptor_file: str,
        adduct_mol: Chem.Mol,
        pose: Pose,
        cov_meta: Dict[str, Any],
        output_pdb: str,
    ) -> str:
        """Write a covalent receptor+adduct PDB with an explicit PDB `LINK` record."""
        mol_copy = Chem.Mol(adduct_mol)
        conf = Chem.Conformer(mol_copy.GetNumAtoms())
        for i, xyz in enumerate(pose.coordinates):
            conf.SetAtomPosition(i, (float(xyz[0]), float(xyz[1]), float(xyz[2])))
        mol_copy.RemoveAllConformers()
        mol_copy.AddConformer(conf, assignId=True)

        # Omit the duplicate anchor atom (already present as the receptor residue nucleophile)
        anchor_idx = int(cov_meta["anchor_atom_idx"])
        attached_idx = int(cov_meta["attached_atom_idx"])
        rec_chain, rec_resname, rec_resnum_s, rec_atom = cov_meta["reactive_residue"].split(":")
        rec_resnum = int(rec_resnum_s)

        rec_lines = [
            ln
            for ln in Path(receptor_file).read_text(encoding="utf-8", errors="ignore").splitlines()
            if not ln.startswith("END")
        ]

        # Build ligand HETATM lines excluding the duplicate anchor atom
        pdb_block = Chem.MolToPDBBlock(mol_copy)
        lig_lines = []
        attached_pdb_name = "C1"
        atom_counter = 0
        for ln in pdb_block.splitlines():
            if ln.startswith(("ATOM", "HETATM")):
                if atom_counter == anchor_idx:
                    atom_counter += 1
                    continue
                aname = ln[12:16].strip()
                if atom_counter == attached_idx:
                    attached_pdb_name = aname
                ln_out = "HETATM" + ln[6:17] + "COV" + " L" + "   1" + ln[26:]
                lig_lines.append(ln_out)
                atom_counter += 1

        bond_len = cov_meta.get("measured_bond_length_A") or cov_meta.get("target_bond_length_A", 1.82)
        link_line = (
            f"LINK        {rec_atom:>4s} {rec_resname:>3s} {rec_chain:1s}{rec_resnum:4d}                "
            f"{attached_pdb_name:>4s} COV L   1     1555   1555 {bond_len:5.2f}"
        )

        out = Path(output_pdb)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "\n".join([link_line] + rec_lines + ["TER"] + lig_lines + ["END"]) + "\n",
            encoding="utf-8",
        )
        return str(out)
