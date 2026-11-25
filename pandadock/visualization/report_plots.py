"""
Publication-grade scientific figures and analytics for PandaDock reports.

Visual system inspired by editorial structural-biology reports:
- Warm stone figure canvases (#F3F0EC), deep indigo/navy ink (#141923, #2B4C8C)
- Colorblind-safe muted teal (#49A6B5), emerald (#1D9A78), coral (#E56B55),
  cobalt (#3B49B8), and amber (#D98A32)
- Clean top/right spine removal, bold lowercase panel letters (a, b, c), 300 DPI
- Zero fabricated affinity conversions (Kd/IC50) or unsubstantiated metrics
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("pandadock.visualization.report_plots")

# Shared PandaDock / PandaInsight-inspired editorial palette
PALETTE = {
    "indigo": "#3B49B8",
    "indigo_dark": "#26318A",
    "navy": "#141D33",
    "cobalt": "#2B4C8C",
    "teal": "#49A6B5",
    "emerald": "#1D9A78",
    "coral": "#E56B55",
    "salmon": "#EA907A",
    "amber": "#D98A32",
    "slate": "#64748B",
    "muted": "#94A3B8",
    "stone_bg": "#F3F0EC",
    "card_border": "#E5E0D8",
    "ink": "#141923",
    "ink_soft": "#4A5568",
    "grid": "#DFDBD4",
    "white": "#FFFFFF",
}

FAVOURABLE_TERMS = {
    "intermolecular", "gauss1", "gauss2", "hydrophobic", "hydrogen",
    "vdw", "vdW", "electrostatic", "SA_solvation", "hbond", "metal_coordination",
}
UNFAVOURABLE_TERMS = {
    "intramolecular", "repulsion", "clash", "strain", "entropy",
    "GB_solvation", "internal_strain", "refinement_cost",
}

INTERACTION_COLORS = {
    "hydrogen_bonds": PALETTE["teal"],
    "hydrophobic": PALETTE["salmon"],
    "van_der_waals_contacts": PALETTE["salmon"],
    "pi_stacking": PALETTE["emerald"],
    "electrostatic": PALETTE["cobalt"],
    "electrostatic_interactions": PALETTE["cobalt"],
    "salt_bridges": PALETTE["cobalt"],
    "metal_coordination": PALETTE["amber"],
    "halogen_bonds": "#8B5CF6",
}

INTERACTION_LABELS = {
    "hydrogen_bonds": "H-bonds",
    "hydrophobic": "Hydrophobic",
    "van_der_waals_contacts": "Hydrophobic / vdW",
    "pi_stacking": "π-Stacking",
    "electrostatic": "Ionic / Elec.",
    "electrostatic_interactions": "Ionic / Elec.",
    "salt_bridges": "Ionic / Salt",
    "metal_coordination": "Metal",
    "halogen_bonds": "Halogen",
}

# Empirical proxy weights (kcal/mol per residue-interaction-type per pose) for relative ranking
_PROXY_WEIGHTS = {
    "hydrogen_bonds": -0.95,
    "salt_bridges": -1.10,
    "electrostatic": -0.75,
    "electrostatic_interactions": -0.75,
    "metal_coordination": -1.60,
    "pi_stacking": -0.65,
    "halogen_bonds": -0.50,
    "hydrophobic": -0.40,
    "van_der_waals_contacts": -0.35,
}


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#94A3B8",
        "axes.linewidth": 0.8,
        "axes.labelcolor": PALETTE["ink"],
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.color": PALETTE["ink_soft"],
        "ytick.color": PALETTE["ink_soft"],
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.facecolor": PALETTE["stone_bg"],
        "axes.facecolor": PALETTE["stone_bg"],
        "savefig.facecolor": PALETTE["stone_bg"],
        "savefig.edgecolor": "none",
    })
    return plt


def _style_ax(ax, grid_axis: Optional[str] = "y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8C96A5")
    ax.spines["bottom"].set_color("#8C96A5")
    ax.tick_params(direction="out", length=3.5, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=PALETTE["grid"], linewidth=0.65, linestyle="-", alpha=0.8)
        ax.set_axisbelow(True)


def _panel_label(ax, letter: str, x: float = -0.14, y: float = 1.07):
    ax.text(
        x, y, letter, transform=ax.transAxes,
        fontsize=12, fontweight="bold", color=PALETTE["ink"],
        va="bottom", ha="left",
    )


def _custom_teal_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "pandadock_teal",
        ["#F8F6F2", "#CBE7EC", "#6DB8C8", "#2B7A9E", "#193A6F"],
    )


def pose_rmsd_matrix(poses: Sequence, heavy_atoms: Optional[np.ndarray] = None) -> np.ndarray:
    """Pairwise in-place heavy-atom RMSD between returned poses (Å)."""
    coords = [np.asarray(p.coordinates, dtype=np.float64) for p in poses]
    if heavy_atoms is not None and len(heavy_atoms) > 0:
        coords = [c[heavy_atoms] if len(c) > int(np.max(heavy_atoms)) else c for c in coords]

    n = len(coords)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if coords[i].shape != coords[j].shape:
                rmsd = float("nan")
            else:
                rmsd = float(np.sqrt(np.mean(np.sum((coords[i] - coords[j]) ** 2, axis=1))))
            matrix[i, j] = matrix[j, i] = rmsd
    return matrix


def _boltzmann_weights(scores: Sequence[float], temperature: float = 298.15) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=np.float64)
    rt = 0.001987204 * temperature
    shifted = np.clip(-(arr - np.min(arr)) / rt, -500.0, 0.0)
    w = np.exp(shifted)
    total = float(np.sum(w))
    return w / total if total > 0 else np.ones_like(w) / len(w)


def _cluster_modes(rmsd_mat: np.ndarray, cutoff: float = 2.0) -> Tuple[int, List[int]]:
    """Greedy leader clustering at `cutoff` Å RMSD (returns cluster count and per-pose cluster IDs)."""
    n = rmsd_mat.shape[0]
    if n == 0:
        return 0, []
    centers: List[int] = []
    labels = [-1] * n
    for i in range(n):
        assigned = None
        for cid, rep in enumerate(centers):
            d = rmsd_mat[i, rep]
            if not math.isnan(d) and d <= cutoff:
                assigned = cid
                break
        if assigned is None:
            assigned = len(centers)
            centers.append(i)
        labels[i] = assigned
    return len(centers), labels


def _receptor_summary(receptor_file: Optional[str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "n_residues": 0, "n_chains": 0, "n_atoms": 0, "chains": [], "metals": [],
    }
    if not receptor_file or not Path(receptor_file).exists():
        return info
    try:
        from Bio.PDB import PDBParser

        struct = PDBParser(QUIET=True).get_structure("rec", str(receptor_file))
        model = next(struct.get_models())
        chains = []
        n_res = 0
        n_atoms = 0
        metals = []
        metal_set = {
            "ZN", "MG", "FE", "CA", "MN", "CU", "CO", "NI", "CD", "HG", "PT", "PD", "MO", "NA", "K",
        }
        for chain in model:
            chains.append(chain.id)
            for res in chain:
                if res.id[0] == " ":
                    n_res += 1
                else:
                    rn = res.get_resname().strip().upper()
                    if rn in metal_set:
                        metals.append(f"{rn} {chain.id}:{res.id[1]}")
                n_atoms += sum(1 for _ in res.get_atoms())
        info.update({
            "n_residues": n_res,
            "n_chains": len(chains),
            "n_atoms": n_atoms,
            "chains": chains,
            "metals": metals,
        })
    except Exception as exc:
        logger.debug("Receptor summary inspection skipped: %s", exc)
    return info


def _ligand_summary(ligand_mol=None, poses: Sequence = ()) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "n_atoms": 0, "n_heavy": 0, "mw": None, "rot_bonds": None,
        "hbd": None, "hba": None, "logp": None, "tpsa": None, "formula": "",
    }
    if ligand_mol is not None:
        try:
            from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

            info["n_atoms"] = int(ligand_mol.GetNumAtoms())
            info["n_heavy"] = int(ligand_mol.GetNumHeavyAtoms())
            info["mw"] = float(Descriptors.MolWt(ligand_mol))
            info["rot_bonds"] = int(Lipinski.NumRotatableBonds(ligand_mol))
            info["hbd"] = int(Lipinski.NumHDonors(ligand_mol))
            info["hba"] = int(Lipinski.NumHAcceptors(ligand_mol))
            info["logp"] = float(Crippen.MolLogP(ligand_mol))
            info["tpsa"] = float(rdMolDescriptors.CalcTPSA(ligand_mol))
            info["formula"] = rdMolDescriptors.CalcMolFormula(ligand_mol)
            return info
        except Exception as exc:
            logger.debug("RDKit ligand property computation skipped: %s", exc)
    if poses:
        info["n_atoms"] = int(len(poses[0].coordinates))
        info["n_heavy"] = info["n_atoms"]
    return info


def _aggregate_interactions(
    per_pose_interactions: Optional[List[Dict]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[Dict[str, Any]]]:
    """
    Aggregate per-pose interaction analyses into:
    - persistent_contacts: sorted list of per-residue occupancy, types, mean distance
    - event_totals: dict of total interaction events by type
    - hotspots: sorted list of per-residue empirical proxy energy contributions (kcal/mol)
    """
    if not per_pose_interactions:
        return [], {}, []

    n_poses = len(per_pose_interactions)
    res_poses: Dict[str, set] = {}
    res_types: Dict[str, Dict[str, int]] = {}
    res_dists: Dict[str, List[float]] = {}
    res_proxy: Dict[str, float] = {}
    event_totals: Dict[str, int] = {}

    for p_idx, pose_data in enumerate(per_pose_interactions):
        details = pose_data.get("interaction_details", {}) or {}
        seen_type_in_pose: Dict[Tuple[str, str], bool] = {}
        for itype, items in details.items():
            if not isinstance(items, list):
                continue
            event_totals[itype] = event_totals.get(itype, 0) + len(items)
            base_w = _PROXY_WEIGHTS.get(itype, -0.25)
            for item in items:
                residue = item.get("residue")
                if not residue:
                    continue
                res_poses.setdefault(residue, set()).add(p_idx)
                dist = item.get("distance")
                if dist is not None:
                    try:
                        d_val = float(dist)
                        res_dists.setdefault(residue, []).append(d_val)
                    except (TypeError, ValueError):
                        d_val = 3.5
                else:
                    d_val = 3.5

                # Count pose occupancy per (residue, type) and add bounded per-residue proxy score
                if not seen_type_in_pose.get((residue, itype)):
                    seen_type_in_pose[(residue, itype)] = True
                    tmap = res_types.setdefault(residue, {})
                    tmap[itype] = tmap.get(itype, 0) + 1
                    attenuation = float(np.clip(1.0 - max(0.0, d_val - 2.5) / 3.5, 0.30, 1.0))
                    res_proxy[residue] = res_proxy.get(residue, 0.0) + (base_w * attenuation) / n_poses
                else:
                    # Diminishing contribution for additional contacts of the same type
                    attenuation = float(np.clip(1.0 - max(0.0, d_val - 2.5) / 3.5, 0.25, 1.0))
                    res_proxy[residue] = res_proxy.get(residue, 0.0) + (0.04 * base_w * attenuation) / n_poses

    persistent: List[Dict[str, Any]] = []
    for res, pset in res_poses.items():
        occ = len(pset) / n_poses
        tmap = res_types.get(res, {})
        type_names = [
            INTERACTION_LABELS.get(k, k.replace("_", " ").title())
            for k in sorted(tmap, key=lambda x: -tmap[x])
        ]
        # Deduplicate display labels preserving order
        seen_lbl = []
        for lbl in type_names:
            if lbl not in seen_lbl:
                seen_lbl.append(lbl)
        dists = res_dists.get(res, [])
        mean_d = float(np.mean(dists)) if dists else float("nan")
        persistent.append({
            "residue": res,
            "occupancy": occ,
            "occupancy_pct": 100.0 * occ,
            "types": ", ".join(seen_lbl),
            "type_fractions": {k: v / n_poses for k, v in tmap.items()},
            "mean_dist": mean_d,
            "pose_count": len(pset),
        })
    persistent.sort(key=lambda r: (-r["occupancy"], r["mean_dist"] if not math.isnan(r["mean_dist"]) else 99.0))

    hotspots = [
        {"residue": res, "proxy_score": float(score)}
        for res, score in sorted(res_proxy.items(), key=lambda kv: kv[1])
    ]
    return persistent, event_totals, hotspots


def compute_report_analytics(
    result,
    ligand_mol=None,
    receptor_file: Optional[str] = None,
    interaction_analyses: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Compute all summary KPIs, quality checks, tables, and executive prose."""
    poses = list(getattr(result, "poses", []) or [])
    scores = [float(p.energy) for p in poses]
    params = getattr(result, "parameters", None) or {}
    rec_path = receptor_file or getattr(result, "receptor_file", "")

    heavy = None
    if ligand_mol is not None:
        heavy = np.array(
            [a.GetIdx() for a in ligand_mol.GetAtoms() if a.GetAtomicNum() > 1],
            dtype=np.int64,
        )

    rmsd_mat = pose_rmsd_matrix(poses, heavy) if len(poses) >= 2 else np.zeros((len(poses), len(poses)))
    n_clusters, cluster_ids = _cluster_modes(rmsd_mat, cutoff=2.0)
    boltz = _boltzmann_weights(scores)

    best_score = scores[0] if scores else float("nan")
    mean_score = float(np.mean(scores)) if scores else float("nan")
    sd_score = float(np.std(scores)) if len(scores) > 1 else 0.0
    gap_1_2 = (scores[1] - scores[0]) if len(scores) > 1 else 0.0
    top1_boltz_pct = float(boltz[0] * 100.0) if len(boltz) > 0 else 0.0

    off_diag = rmsd_mat[np.triu_indices(len(poses), k=1)] if len(poses) >= 2 else np.array([])
    valid_off = off_diag[~np.isnan(off_diag)]
    mean_rmsd = float(np.mean(valid_off)) if valid_off.size else 0.0
    max_rmsd = float(np.max(valid_off)) if valid_off.size else 0.0
    rmsd_to_top1 = [float(rmsd_mat[0, i]) if len(poses) > i else 0.0 for i in range(len(poses))]

    strains = [float(getattr(p, "internal_strain", 0.0)) for p in poses]
    top1_strain = strains[0] if strains else 0.0

    persistent, event_totals, hotspots = _aggregate_interactions(interaction_analyses)
    n_persistent_30 = sum(1 for c in persistent if c["occupancy"] >= 0.30)
    n_hbonds_top1 = 0
    if interaction_analyses:
        n_hbonds_top1 = int(
            interaction_analyses[0].get("interaction_types", {}).get("hydrogen_bonds", 0)
        )
    total_contacts_top1 = 0
    if interaction_analyses:
        total_contacts_top1 = int(interaction_analyses[0].get("total_interactions", 0))

    rec_info = _receptor_summary(rec_path)
    lig_info = _ligand_summary(ligand_mol, poses)

    # Automated scientific quality checks (green 'ok' vs amber 'warn')
    checks: List[Tuple[str, str]] = []
    if not math.isnan(best_score) and best_score < -4.0:
        checks.append(("ok", f"Favourable top-ranked binding score ({best_score:.2f} kcal/mol; no net steric clash)."))
    elif not math.isnan(best_score) and best_score < 0.0:
        checks.append(("warn", f"Modest docking score ({best_score:.2f} kcal/mol): inspect pocket fit and protonation state."))
    else:
        checks.append(("warn", f"Unfavourable or positive top score ({best_score:.2f} kcal/mol): possible steric clash or mis-centred grid box."))

    if len(scores) > 1:
        if gap_1_2 >= 0.4:
            checks.append(("ok", f"Top pose is energetically separated from rank 2 (ΔE = {gap_1_2:.2f} kcal/mol; Boltzmann weight {top1_boltz_pct:.0f}%)."))
        else:
            checks.append(("warn", f"Narrow energy gap to rank 2 (ΔE = {gap_1_2:.2f} kcal/mol): top modes are nearly degenerate; inspect multiple poses."))

    if len(poses) >= 2:
        if n_clusters <= max(2, len(poses) // 2):
            checks.append(("ok", f"Search converges cleanly into {n_clusters} distinct conformational cluster(s) (mean pairwise RMSD {mean_rmsd:.2f} Å)."))
        else:
            checks.append(("warn", f"High pose dispersion ({n_clusters} clusters at 2.0 Å cutoff; mean pairwise RMSD {mean_rmsd:.2f} Å): binding mode may be ambiguous."))

    if interaction_analyses:
        if n_hbonds_top1 >= 1:
            checks.append(("ok", f"Top pose forms {n_hbonds_top1} hydrogen bond(s) and {total_contacts_top1} total pocket contacts."))
        else:
            checks.append(("warn", f"Top pose relies primarily on nonpolar contacts ({total_contacts_top1} total, 0 H-bonds)."))
        if n_persistent_30 >= 3:
            checks.append(("ok", f"{n_persistent_30} pocket residues contact the ligand in ≥30% of returned poses."))

    if top1_strain > 5.0:
        checks.append(("warn", f"Elevated internal ligand strain on rank 1 ({top1_strain:.2f} kcal/mol): verify ring/torsion geometry."))
    elif poses:
        checks.append(("ok", f"Low internal ligand strain on rank 1 ({top1_strain:.2f} kcal/mol)."))

    if lig_info["mw"] is not None:
        ro5_violations = sum([
            lig_info["mw"] > 500,
            (lig_info["logp"] or 0) > 5,
            (lig_info["hbd"] or 0) > 5,
            (lig_info["hba"] or 0) > 10,
        ])
        if ro5_violations == 0:
            checks.append(("ok", f"Drug-like physicochemical profile (MW {lig_info['mw']:.1f} Da, cLogP {lig_info['logp']:.2f}, 0 Lipinski violations)."))
        else:
            checks.append(("warn", f"Ligand has {ro5_violations} Lipinski Rule-of-5 flag(s) (MW {lig_info['mw']:.1f} Da, cLogP {lig_info['logp']:.2f})."))

    # Built-in PoseBusters-style chemistry & geometry audit (v5.0)
    pb_reports = []
    pb_pass_count = 0
    if ligand_mol is not None and poses:
        try:
            from ..docking.validation.posebusters import audit_docking_result

            pb_reports = audit_docking_result(
                result,
                ligand_mol=ligand_mol,
                receptor_file=rec_path or None,
                max_poses=min(len(poses), 10),
            )
            pb_pass_count = sum(1 for r in pb_reports if r.all_passed)
            if pb_reports:
                top_pb = pb_reports[0]
                if top_pb.all_passed:
                    checks.append((
                        "ok",
                        f"PoseBusters audit PASS on rank 1 ({top_pb.n_passed}/{top_pb.n_total} checks; "
                        f"{pb_pass_count}/{len(pb_reports)} audited poses valid; "
                        f"ΔE_strain = {top_pb.relative_energy_kcal if top_pb.relative_energy_kcal is not None else 0.0:.2f} kcal/mol).",
                    ))
                else:
                    checks.append((
                        "warn",
                        f"PoseBusters audit flagged rank 1 ({top_pb.n_passed}/{top_pb.n_total}; "
                        f"failed: {', '.join(top_pb.failed_checks[:3])}).",
                    ))
        except Exception as exc:
            logger.debug("PoseBusters report audit skipped: %s", exc)

    # Restraint satisfaction & structural water reporting (v5.0)
    restraint_summary = params.get("restraints")
    if isinstance(restraint_summary, list) and restraint_summary:
        n_sat = sum(1 for r in restraint_summary if r.get("satisfied"))
        if n_sat == len(restraint_summary):
            checks.append((
                "ok",
                f"All {n_sat}/{len(restraint_summary)} user geometric restraints satisfied on rank 1 "
                f"(first: {restraint_summary[0].get('label')} = {restraint_summary[0].get('distance_A')} Å).",
            ))
        else:
            checks.append((
                "warn",
                f"Restraints satisfied on rank 1: {n_sat}/{len(restraint_summary)} "
                f"(max violation {max(float(r.get('violation_A', 0.0)) for r in restraint_summary):.2f} Å).",
            ))

    water_info = params.get("water_analysis")
    if isinstance(water_info, dict) and water_info.get("waters"):
        checks.append((
            "ok",
            f"Structural water protocol ({water_info.get('mode', 'displace')}): "
            f"selected {water_info.get('selected_state', 'default')} "
            f"({water_info.get('bridging_count', 0)} bridging, {water_info.get('displaced_count', 0)} displaced, "
            f"ΔG_water = {water_info.get('delta_g_water', 0.0):+.2f} kcal/mol).",
        ))

    # Executive pull quote
    lig_name = getattr(result, "ligand_name", "ligand")
    rec_stem = Path(rec_path).stem if rec_path else "Receptor"
    if n_persistent_30 > 0:
        quote = (
            f"Top pose of {lig_name} scores {best_score:.2f} kcal/mol "
            f"(ΔE₁₋₂ = {gap_1_2:.2f} kcal/mol, {top1_boltz_pct:.0f}% Boltzmann weight), "
            f"anchored by {n_persistent_30} persistent binding-site residues across "
            f"{n_clusters} conformational mode{'s' if n_clusters != 1 else ''}."
        )
    else:
        quote = (
            f"Top pose of {lig_name} in {rec_stem} scores {best_score:.2f} kcal/mol "
            f"(ensemble ΔG = {getattr(result, 'ensemble_binding_energy', best_score):.2f} kcal/mol) "
            f"across {len(poses)} returned pose{'s' if len(poses) != 1 else ''} "
            f"spanning {n_clusters} conformational mode{'s' if n_clusters != 1 else ''}."
        )

    # Per-pose table rows
    pose_rows = []
    for i, pose in enumerate(poses[:10]):
        hb = hp = pi = mt = 0
        if interaction_analyses and i < len(interaction_analyses):
            itypes = interaction_analyses[i].get("interaction_types", {}) or {}
            details = interaction_analyses[i].get("interaction_details", {}) or {}
            hb = int(itypes.get("hydrogen_bonds", 0))
            hp = int(itypes.get("hydrophobic", 0))
            if hp == 0 and details.get("van_der_waals_contacts"):
                hp = len({it.get("residue") for it in details["van_der_waals_contacts"] if it.get("residue")})
            pi = int(itypes.get("pi_stacking", 0))
            mt = int(itypes.get("metal_coordination", 0))
        pose_rows.append({
            "rank": i + 1,
            "score": float(pose.energy),
            "delta_e": float(pose.energy - best_score) if not math.isnan(best_score) else 0.0,
            "boltz_pct": float(boltz[i] * 100.0) if i < len(boltz) else 0.0,
            "rmsd_top1": rmsd_to_top1[i],
            "cluster": f"M{cluster_ids[i] + 1}" if i < len(cluster_ids) else "M1",
            "strain": float(getattr(pose, "internal_strain", 0.0)),
            "hb": hb,
            "hp": hp,
            "pi": pi,
            "mt": mt,
        })

    return {
        "ligand_name": lig_name,
        "receptor_name": rec_stem,
        "receptor_file": rec_path,
        "algorithm": getattr(result, "algorithm_used", "pandadock"),
        "scoring": getattr(result, "scoring_function", "vina"),
        "grid_center": getattr(result, "grid_center", np.zeros(3)),
        "grid_dimensions": getattr(result, "grid_dimensions", np.zeros(3)),
        "runtime_seconds": float(getattr(result, "runtime_seconds", 0.0)),
        "parameters": params,
        "n_poses": len(poses),
        "best_score": best_score,
        "ensemble_dg": float(getattr(result, "ensemble_binding_energy", best_score)),
        "mean_score": mean_score,
        "sd_score": sd_score,
        "gap_1_2": gap_1_2,
        "top1_boltz_pct": top1_boltz_pct,
        "mean_rmsd": mean_rmsd,
        "max_rmsd": max_rmsd,
        "n_clusters": n_clusters,
        "cluster_ids": cluster_ids,
        "top1_strain": top1_strain,
        "rmsd_matrix": rmsd_mat,
        "boltz_weights": boltz,
        "receptor_info": rec_info,
        "ligand_info": lig_info,
        "persistent_contacts": persistent,
        "posebusters": [r.to_dict() for r in pb_reports],
        "posebusters_pass_count": pb_pass_count,
        "posebusters_total_audited": len(pb_reports),
        "n_persistent_30": n_persistent_30,
        "event_totals": event_totals,
        "hotspots": hotspots,
        "checks": checks,
        "quote": quote,
        "pose_rows": pose_rows,
    }


def plot_protein_silhouette(
    receptor_file: Optional[str],
    top_pose_coords: Optional[np.ndarray],
    output_file: Path,
) -> Optional[Path]:
    """
    Clean cover-page protein-ligand silhouette (2D PCA projection of receptor Cα
    backbone trace with translucent depth shading and highlighted top ligand pose).
    """
    if not receptor_file or not Path(receptor_file).exists():
        return None
    try:
        from Bio.PDB import PDBParser

        struct = PDBParser(QUIET=True).get_structure("rec", str(receptor_file))
        model = next(struct.get_models())
        chain_traces: List[np.ndarray] = []
        all_ca = []
        for chain in model:
            pts = []
            for res in chain:
                if res.id[0] == " " and "CA" in res:
                    c = res["CA"].get_coord()
                    pts.append(c)
                    all_ca.append(c)
            if len(pts) >= 2:
                chain_traces.append(np.asarray(pts, dtype=np.float64))
        if len(all_ca) < 5:
            return None

        all_ca_arr = np.asarray(all_ca, dtype=np.float64)
        center = all_ca_arr.mean(axis=0)
        centered = all_ca_arr - center
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:3]  # 2D projection + depth axis

        plt = _matplotlib()
        fig, ax = plt.subplots(figsize=(5.0, 3.6), facecolor="#F2F2F4")
        ax.set_facecolor("#F2F2F4")

        # Draw soft silhouette halo + backbone ribbon per chain
        for ci, trace in enumerate(chain_traces):
            proj = (trace - center) @ basis.T
            shade = "#6B7280" if ci % 2 == 0 else "#9CA3AF"
            # Outer soft surface envelope
            ax.plot(proj[:, 0], proj[:, 1], color="#4B5563", linewidth=10.0, alpha=0.14,
                    solid_capstyle="round", solid_joinstyle="round", zorder=1)
            ax.plot(proj[:, 0], proj[:, 1], color="#374151", linewidth=4.8, alpha=0.28,
                    solid_capstyle="round", solid_joinstyle="round", zorder=2)
            # Crisp inner trace
            ax.plot(proj[:, 0], proj[:, 1], color=shade, linewidth=1.6, alpha=0.82,
                    solid_capstyle="round", solid_joinstyle="round", zorder=3)

        # Highlight top docked ligand as crisp white/indigo CPK spheres
        if top_pose_coords is not None and len(top_pose_coords) > 0:
            lig_proj = (np.asarray(top_pose_coords, dtype=np.float64) - center) @ basis.T
            # Subtle indigo pocket glow
            cx, cy = float(np.mean(lig_proj[:, 0])), float(np.mean(lig_proj[:, 1]))
            ax.scatter([cx], [cy], s=950, color=PALETTE["indigo"], alpha=0.14, edgecolors="none", zorder=4)
            ax.scatter(
                lig_proj[:, 0], lig_proj[:, 1],
                s=44, facecolor="#FFFFFF", edgecolor=PALETTE["indigo"],
                linewidth=1.15, alpha=0.96, zorder=5,
            )

        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
        fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        fig.savefig(output_file, dpi=300, facecolor="#F2F2F4")
        plt.close(fig)
        return output_file
    except Exception as exc:
        logger.debug("Could not render protein silhouette: %s", exc)
        return None


def plot_pose_scores(
    poses: Sequence,
    output_file: Path,
    title: str = "",
    heavy_atoms: Optional[np.ndarray] = None,
) -> Optional[Path]:
    """
    3-panel editorial energetics figure:
      a) Score by pose rank with Boltzmann ensemble ΔG and rank 1->2 gap
      b) Boltzmann population share (%) at 298 K
      c) Binding funnel: Score vs heavy-atom RMSD from top pose
    """
    if not poses:
        return None
    plt = _matplotlib()

    scores = np.asarray([float(p.energy) for p in poses], dtype=np.float64)
    ranks = np.arange(1, len(scores) + 1)
    boltz = _boltzmann_weights(scores) * 100.0
    rmsd_mat = pose_rmsd_matrix(poses, heavy_atoms) if len(poses) >= 2 else np.zeros((len(poses), len(poses)))
    rmsd_to_top1 = rmsd_mat[0] if len(poses) >= 1 else np.zeros(len(poses))

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.85), facecolor=PALETTE["stone_bg"])
    fig.subplots_adjust(wspace=0.34, left=0.06, right=0.98, top=0.86, bottom=0.18)

    # Panel a: Ranked pose scores
    ax = axes[0]
    _style_ax(ax, "y")
    _panel_label(ax, "a")
    ax.plot(ranks, scores, "-", color=PALETTE["cobalt"], linewidth=1.7, zorder=2)
    ax.scatter(ranks[1:], scores[1:], color=PALETTE["cobalt"], s=34, zorder=3, label="Docked modes")
    ax.scatter([ranks[0]], [scores[0]], color=PALETTE["coral"], s=95, marker="*", zorder=4, label="Top pose")

    ens_dg = float(np.sum((boltz / 100.0) * scores))
    ax.axhline(ens_dg, color=PALETTE["teal"], linestyle="--", linewidth=1.1, label=f"Boltzmann ⟨E⟩ ({ens_dg:.2f})")
    if len(scores) > 1:
        gap = scores[1] - scores[0]
        span = max(float(np.ptp(scores)), 0.8)
        ax.annotate(
            f"ΔE₁₋₂ = {gap:.2f}",
            xy=(1.0, scores[0]),
            xytext=(min(len(scores), 2.2), scores[0] - 0.14 * span),
            fontsize=8.0, color=PALETTE["coral"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PALETTE["coral"], lw=0.8),
        )
    ax.set_xlabel("Pose rank")
    ax.set_ylabel("Docking score (kcal/mol)")
    ax.set_xticks(ranks if len(ranks) <= 10 else np.linspace(1, len(ranks), 6, dtype=int))
    ax.legend(loc="lower right", fontsize=7.5)

    # Panel b: Boltzmann population distribution
    ax = axes[1]
    _style_ax(ax, "y")
    _panel_label(ax, "b")
    bar_colors = [PALETTE["coral"] if i == 0 else PALETTE["teal"] for i in range(len(ranks))]
    ax.bar(ranks, boltz, color=bar_colors, width=0.68, edgecolor="none", zorder=3)
    for r, b in zip(ranks[:5], boltz[:5]):
        if b >= 3.0:
            ax.text(r, b + 1.8, f"{b:.0f}%", ha="center", va="bottom", fontsize=7.5, color=PALETTE["ink_soft"])
    ax.set_ylim(0, min(115, max(float(np.max(boltz)) * 1.18, 25.0)))
    ax.set_xlabel("Pose rank")
    ax.set_ylabel("Boltzmann population (%)")
    ax.set_xticks(ranks if len(ranks) <= 10 else np.linspace(1, len(ranks), 6, dtype=int))

    # Panel c: Score vs RMSD funnel
    ax = axes[2]
    _style_ax(ax, "both")
    _panel_label(ax, "c")
    if len(scores) >= 2:
        sc = ax.scatter(
            rmsd_to_top1[1:], scores[1:],
            c=ranks[1:], cmap=_custom_teal_cmap(), s=44,
            edgecolors=PALETTE["navy"], linewidths=0.5, zorder=3,
        )
        ax.scatter(
            [0.0], [scores[0]], color=PALETTE["coral"], s=110, marker="*",
            edgecolors=PALETTE["navy"], linewidths=0.4, zorder=4, label="Rank 1 (ref)",
        )
        ax.axvline(2.0, color=PALETTE["muted"], linestyle=":", linewidth=1.0, label="2.0 Å cutoff")
        ax.legend(loc="lower right", fontsize=7.5)
    else:
        ax.scatter([0.0], [scores[0]], color=PALETTE["coral"], s=110, marker="*")
    ax.set_xlabel("Heavy-atom RMSD to rank 1 (Å)")
    ax.set_ylabel("Docking score (kcal/mol)")

    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    return output_file


def plot_energy_components(poses: Sequence, output_file: Path) -> Optional[Path]:
    """
    Per-pose energy decomposition with explicit favourable (below 0, teal/emerald/cobalt)
    vs unfavourable penalty (above 0, coral/amber) visual separation.
    """
    components = [p.energy_components for p in poses if getattr(p, "energy_components", None)]
    if not components:
        return None
    names = sorted({k for c in components for k in c if k != "total"})
    if not names:
        return None

    plt = _matplotlib()
    ranks = np.arange(1, len(components) + 1)
    fav_palette = [PALETTE["cobalt"], PALETTE["teal"], PALETTE["emerald"], "#5B8DEF", "#38A3A5"]
    pen_palette = [PALETTE["coral"], PALETTE["amber"], "#C2410C", "#B45309"]

    fig, ax = plt.subplots(figsize=(10.5, 4.0), facecolor=PALETTE["stone_bg"])
    _style_ax(ax, "y")
    _panel_label(ax, "a", x=-0.07, y=1.05)

    n_terms = len(names)
    width = min(0.76 / max(n_terms, 1), 0.22)
    f_i, p_i = 0, 0
    for i, name in enumerate(names):
        vals = [float(c.get(name, 0.0)) for c in components]
        offset = (i - (n_terms - 1) / 2.0) * width
        is_fav = name in FAVOURABLE_TERMS or (name not in UNFAVOURABLE_TERMS and np.mean(vals) <= 0)
        if is_fav:
            col = fav_palette[f_i % len(fav_palette)]
            f_i += 1
            lbl = f"{name.replace('_', ' ')} (favourable)"
        else:
            col = pen_palette[p_i % len(pen_palette)]
            p_i += 1
            lbl = f"{name.replace('_', ' ')} (penalty)"
        ax.bar(ranks + offset, vals, width=width, label=lbl, color=col, edgecolor="none", zorder=3)

    ax.axhline(0.0, color=PALETTE["ink"], linewidth=0.9, zorder=4)
    ax.set_xlabel("Pose rank")
    ax.set_ylabel("Energy contribution (kcal/mol)")
    ax.set_xticks(ranks)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=min(4, max(2, n_terms)), fontsize=8.0)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.26)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    return output_file


def plot_pose_diversity(
    poses: Sequence,
    output_file: Path,
    heavy_atoms: Optional[np.ndarray] = None,
) -> Optional[Path]:
    """
    2-panel pose diversity & conformational landscape figure:
      a) Pairwise heavy-atom RMSD matrix (Å) in crisp cream-to-teal-to-navy scale
      b) 2D Classical MDS conformational embedding colored by docking score (kcal/mol)
    """
    if len(poses) < 2:
        return None
    plt = _matplotlib()

    matrix = pose_rmsd_matrix(poses, heavy_atoms)
    n = len(poses)
    scores = np.asarray([float(p.energy) for p in poses], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), facecolor=PALETTE["stone_bg"])
    fig.subplots_adjust(wspace=0.32, left=0.07, right=0.96, top=0.88, bottom=0.14)

    # Panel a: RMSD matrix
    ax = axes[0]
    _panel_label(ax, "a", x=-0.13, y=1.04)
    cmap = _custom_teal_cmap()
    im = ax.imshow(matrix, cmap=cmap, origin="upper", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(1, n + 1))
    ax.set_yticklabels(range(1, n + 1))
    ax.set_xlabel("Pose rank")
    ax.set_ylabel("Pose rank")
    for spine in ax.spines.values():
        spine.set_color("#94A3B8")
        spine.set_linewidth(0.7)

    vmax = float(np.nanmax(matrix)) if np.any(~np.isnan(matrix)) else 1.0
    if n <= 12:
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if not math.isnan(val):
                    ax.text(
                        j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7.2,
                        color="#FFFFFF" if val > vmax * 0.58 else PALETTE["ink"],
                    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Heavy-atom RMSD (Å)", fontsize=8.5)
    cbar.outline.set_linewidth(0.6)

    # Panel b: 2D Classical MDS conformational embedding
    ax = axes[1]
    _style_ax(ax, "both")
    _panel_label(ax, "b", x=-0.13, y=1.04)
    d_clean = np.nan_to_num(matrix, nan=0.0)
    d2 = d_clean ** 2
    j_mat = np.eye(n) - np.ones((n, n)) / n
    b_mat = -0.5 * j_mat @ d2 @ j_mat
    eigvals, eigvecs = np.linalg.eigh(b_mat)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    total_var = float(np.sum(eigvals)) + 1e-12
    pc1 = eigvecs[:, 0] * np.sqrt(eigvals[0])
    pc2 = eigvecs[:, 1] * np.sqrt(eigvals[1]) if n > 2 else np.zeros(n)
    var1 = 100.0 * eigvals[0] / total_var
    var2 = 100.0 * (eigvals[1] if n > 2 else 0.0) / total_var

    # Subtle radial density backdrop around the lowest-energy basin
    sc = ax.scatter(
        pc1, pc2, c=scores, cmap="RdYlBu_r", s=85,
        edgecolors=PALETTE["navy"], linewidths=0.75, zorder=4,
    )
    ax.scatter(
        [pc1[0]], [pc2[0]], s=185, marker="*", facecolor=PALETTE["coral"],
        edgecolor=PALETTE["navy"], linewidth=0.7, zorder=5,
    )
    for k in range(min(3, n)):
        ax.annotate(
            f"#{k + 1}", (pc1[k], pc2[k]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8.0, fontweight="bold", color=PALETTE["ink"],
        )
    ax.set_xlabel(f"Pose MDS-1 ({var1:.0f}% var)")
    ax.set_ylabel(f"Pose MDS-2 ({var2:.0f}% var)")
    cbar2 = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label("Score (kcal/mol)", fontsize=8.5)
    cbar2.outline.set_linewidth(0.6)

    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    return output_file


def plot_interaction_fingerprint(
    per_pose_interactions: List[Dict],
    output_file: Path,
    max_residues: int = 18,
) -> Optional[Path]:
    """
    2-panel binding mode figure mirroring PandaInsight Figures 11 & 12:
      a) Stacked contact occupancy per residue split by interaction type
      b) Residue x Pose rank contact intensity heatmap
    """
    if not per_pose_interactions:
        return None

    persistent, _, _ = _aggregate_interactions(per_pose_interactions)
    if not persistent:
        return None

    top_res = persistent[:max_residues]
    residues = [r["residue"] for r in top_res]
    n_poses = len(per_pose_interactions)

    # Collect present interaction types in deterministic order
    canonical_order = [
        "hydrogen_bonds", "hydrophobic", "van_der_waals_contacts", "pi_stacking",
        "salt_bridges", "electrostatic", "electrostatic_interactions",
        "metal_coordination", "halogen_bonds",
    ]
    present_types = []
    for k in canonical_order:
        if any(k in r["type_fractions"] for r in top_res):
            present_types.append(k)
    for r in top_res:
        for k in r["type_fractions"]:
            if k not in present_types:
                present_types.append(k)

    # Build Pose x Residue count matrix
    mat = np.zeros((len(residues), n_poses), dtype=np.float64)
    res_to_idx = {res: i for i, res in enumerate(residues)}
    for p_idx, pose_data in enumerate(per_pose_interactions):
        for items in (pose_data.get("interaction_details") or {}).values():
            if not isinstance(items, list):
                continue
            for item in items:
                res = item.get("residue")
                if res in res_to_idx:
                    mat[res_to_idx[res], p_idx] += 1.0

    plt = _matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, max(4.3, len(residues) * 0.25 + 1.4)),
                             facecolor=PALETTE["stone_bg"])
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.95, top=0.86, bottom=0.22)

    # Panel a: Stacked bar occupancy per residue
    ax = axes[0]
    _style_ax(ax, "y")
    _panel_label(ax, "a", x=-0.12, y=1.06)
    x = np.arange(len(residues))
    bottom = np.zeros(len(residues), dtype=np.float64)
    for itype in present_types:
        heights = np.array([r["type_fractions"].get(itype, 0.0) for r in top_res])
        col = INTERACTION_COLORS.get(itype, PALETTE["slate"])
        lbl = INTERACTION_LABELS.get(itype, itype.replace("_", " ").title())
        ax.bar(x, heights, bottom=bottom, width=0.72, color=col, label=lbl, edgecolor="none", zorder=3)
        bottom += heights

    ax.set_xticks(x)
    ax.set_xticklabels(residues, rotation=45, ha="right", fontsize=7.8)
    ax.set_ylabel("Interaction fraction across poses")
    ax.set_ylim(0, max(1.15, float(np.max(bottom)) * 1.15))
    ax.legend(loc="upper right", ncol=2, fontsize=7.8)

    # Panel b: Residue x Pose contact heatmap
    ax = axes[1]
    _panel_label(ax, "b", x=-0.18, y=1.06)
    im = ax.imshow(mat, cmap=_custom_teal_cmap(), aspect="auto", origin="upper", interpolation="nearest")
    ax.set_xticks(range(n_poses))
    ax.set_xticklabels(range(1, n_poses + 1), fontsize=8.0)
    ax.set_yticks(range(len(residues)))
    ax.set_yticklabels(residues, fontsize=7.8)
    ax.set_xlabel("Pose rank")
    for spine in ax.spines.values():
        spine.set_color("#94A3B8")
        spine.set_linewidth(0.7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Contacts", fontsize=8.5)
    cbar.outline.set_linewidth(0.6)

    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    return output_file


def plot_ligand_dynamics(
    poses: Sequence,
    ligand_mol,
    output_file: Path,
) -> Optional[Path]:
    """
    Ligand conformational variability across docked poses (mirrors PandaInsight Page 7):
      a) Per-heavy-atom RMSF (Å) across the docked ensemble (low = binding anchor)
      b) Rotatable-bond torsion angle distributions in [-180°, 180°]
    """
    if len(poses) < 2:
        return None

    coords_list = [np.asarray(p.coordinates, dtype=np.float64) for p in poses]
    n_atoms_pose = coords_list[0].shape[0]
    if any(c.shape[0] != n_atoms_pose for c in coords_list):
        return None

    # Determine heavy atom indices & labels
    if ligand_mol is not None and ligand_mol.GetNumAtoms() == n_atoms_pose:
        heavy_idx = [a.GetIdx() for a in ligand_mol.GetAtoms() if a.GetAtomicNum() > 1]
        atom_labels = [f"{ligand_mol.GetAtomWithIdx(i).GetSymbol()}{k + 1}" for k, i in enumerate(heavy_idx)]
    else:
        heavy_idx = list(range(n_atoms_pose))
        atom_labels = [f"A{i + 1}" for i in heavy_idx]

    if len(heavy_idx) < 2:
        return None

    stack = np.stack([c[heavy_idx] for c in coords_list], axis=0)  # (P, N_heavy, 3)
    mean_pos = stack.mean(axis=0, keepdims=True)
    rmsf = np.sqrt(np.mean(np.sum((stack - mean_pos) ** 2, axis=2), axis=0))

    # Extract rotatable-bond dihedrals if RDKit mol is available
    torsions: List[Tuple[str, np.ndarray]] = []
    if ligand_mol is not None and ligand_mol.GetNumAtoms() == n_atoms_pose:
        try:
            from rdkit import Chem

            rot_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
            matches = ligand_mol.GetSubstructMatches(rot_smarts) if rot_smarts else ()
            for b_idx, (a2, a3) in enumerate(matches[:8]):
                nbrs2 = [n.GetIdx() for n in ligand_mol.GetAtomWithIdx(a2).GetNeighbors()
                         if n.GetIdx() != a3 and n.GetAtomicNum() > 1]
                nbrs3 = [n.GetIdx() for n in ligand_mol.GetAtomWithIdx(a3).GetNeighbors()
                         if n.GetIdx() != a2 and n.GetAtomicNum() > 1]
                if not nbrs2 or not nbrs3:
                    continue
                a1, a4 = nbrs2[0], nbrs3[0]
                angles = []
                for coords in coords_list:
                    p0, p1, p2, p3 = coords[a1], coords[a2], coords[a3], coords[a4]
                    b0 = -(p1 - p0)
                    b1 = p2 - p1
                    b2 = p3 - p2
                    n1 = np.linalg.norm(b1)
                    if n1 < 1e-8:
                        continue
                    b1_u = b1 / n1
                    v = b0 - np.dot(b0, b1_u) * b1_u
                    w = b2 - np.dot(b2, b1_u) * b1_u
                    x = float(np.dot(v, w))
                    y = float(np.dot(np.cross(b1_u, v), w))
                    angles.append(math.degrees(math.atan2(y, x)))
                if angles:
                    s1 = ligand_mol.GetAtomWithIdx(a1).GetSymbol()
                    s2 = ligand_mol.GetAtomWithIdx(a2).GetSymbol()
                    s3 = ligand_mol.GetAtomWithIdx(a3).GetSymbol()
                    s4 = ligand_mol.GetAtomWithIdx(a4).GetSymbol()
                    lbl = f"{s1}{a1+1}-{s2}{a2+1}-{s3}{a3+1}-{s4}{a4+1}"
                    torsions.append((lbl, np.asarray(angles, dtype=np.float64)))
        except Exception as exc:
            logger.debug("Torsion calculation skipped: %s", exc)

    plt = _matplotlib()
    if torsions:
        fig = plt.figure(figsize=(11.6, 5.8), facecolor=PALETTE["stone_bg"])
        gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.45,
                              left=0.07, right=0.97, top=0.92, bottom=0.10)
        ax_rmsf = fig.add_subplot(gs[0])
    else:
        fig, ax_rmsf = plt.subplots(figsize=(11.0, 3.4), facecolor=PALETTE["stone_bg"])
        fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.22)

    # Panel a: Per-heavy-atom RMSF
    _style_ax(ax_rmsf, "y")
    _panel_label(ax_rmsf, "a", x=-0.06, y=1.06)
    xs = np.arange(len(rmsf))
    ax_rmsf.plot(xs, rmsf, "-o", color=PALETTE["coral"], linewidth=1.6, markersize=4.5, zorder=3)
    ax_rmsf.fill_between(xs, 0, rmsf, color=PALETTE["coral"], alpha=0.10, zorder=2)
    # Mark most anchored atom
    anchor_i = int(np.argmin(rmsf))
    ax_rmsf.scatter([anchor_i], [rmsf[anchor_i]], color=PALETTE["emerald"], s=65, zorder=4, label=f"Anchor ({atom_labels[anchor_i]}: {rmsf[anchor_i]:.2f} Å)")
    step = max(1, len(atom_labels) // 24)
    ax_rmsf.set_xticks(xs[::step])
    ax_rmsf.set_xticklabels(atom_labels[::step], rotation=75 if len(atom_labels) > 16 else 0, fontsize=7.5)
    ax_rmsf.set_ylabel("Atom RMSF (Å)")
    ax_rmsf.set_ylim(0, max(float(np.max(rmsf)) * 1.2, 0.5))
    ax_rmsf.legend(loc="upper right", fontsize=8.0)

    # Panel b: Torsion histograms
    if torsions:
        n_t = len(torsions)
        ncols = min(4, n_t)
        nrows = math.ceil(n_t / ncols)
        sub_gs = gs[1].subgridspec(nrows, ncols, wspace=0.28, hspace=0.55)
        t_colors = [PALETTE["teal"], PALETTE["coral"], PALETTE["emerald"], PALETTE["cobalt"],
                    PALETTE["amber"], PALETTE["salmon"], "#6366F1", "#0EA5E9"]
        for ti, (lbl, angs) in enumerate(torsions):
            r_i, c_i = divmod(ti, ncols)
            ax_t = fig.add_subplot(sub_gs[r_i, c_i])
            _style_ax(ax_t, None)
            if ti == 0:
                _panel_label(ax_t, "b", x=-0.18, y=1.18)
            ax_t.hist(angs, bins=np.linspace(-180, 180, 19), color=t_colors[ti % len(t_colors)],
                      alpha=0.88, edgecolor=PALETTE["stone_bg"], linewidth=0.4)
            ax_t.set_xlim(-180, 180)
            ax_t.set_xticks([-180, 0, 180])
            ax_t.set_yticks([])
            ax_t.set_title(lbl, fontsize=7.5, color=PALETTE["ink_soft"], pad=3)

    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    return output_file


def plot_residue_hotspots(
    per_pose_interactions: List[Dict],
    output_file: Path,
    max_residues: int = 12,
) -> Optional[Path]:
    """
    Horizontal bar chart of top residue empirical contact proxy scores (kcal/mol),
    mirroring PandaInsight Figure 16.
    """
    _, _, hotspots = _aggregate_interactions(per_pose_interactions)
    if not hotspots:
        return None

    top = hotspots[:max_residues]
    residues = [h["residue"] for h in top][::-1]
    scores = [h["proxy_score"] for h in top][::-1]

    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, max(3.4, len(residues) * 0.30 + 1.1)), facecolor=PALETTE["stone_bg"])
    _style_ax(ax, "x")
    _panel_label(ax, "a", x=-0.20, y=1.04)

    y = np.arange(len(residues))
    ax.barh(y, scores, height=0.68, color=PALETTE["teal"], edgecolor="none", zorder=3)
    ax.axvline(0.0, color=PALETTE["ink"], linewidth=0.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(residues, fontsize=8.3)
    ax.set_xlabel("Empirical contact proxy score (kcal/mol)")
    ax.set_xlim(min(scores) * 1.18, 0.08)
    for yi, sc in zip(y, scores):
        ax.text(sc - 0.03 * abs(min(scores)), yi, f"{sc:.2f}", va="center", ha="right",
                fontsize=7.6, color=PALETTE["ink_soft"])

    fig.subplots_adjust(left=0.22, right=0.96, top=0.88, bottom=0.18)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    return output_file
