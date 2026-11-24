"""
Publication-grade A4 PDF Molecular Docking Report for PandaDock.

Built with the Panda family brochure layout kit (`report_kit.py`) so PandaDock
and PandaInsight reports share one cohesive editorial visual language:
  Page 1  Cover (3-D backbone+ligand silhouette, 3-KPI indigo banner, pull-quote, fact table)
  Page 2  Docking Setup sidebar, narrative summary, quality metrics table & automated checks
  Page 3  Binding energetics, Boltzmann populations, score-vs-RMSD funnel & energy breakdown
  Page 4  Pose diversity (pairwise RMSD matrix & 2-D MDS conformational landscape) + mode table
  Page 5  Protein-ligand binding mode: persistent contacts table, stacked occupancy & heatmap
  Page 6  Ligand flexibility (per-atom RMSF, rotatable torsions) & residue energy hotspots
  Page 7* 2-D interaction map from PandaMap (when available)
  End     Analysis modules (4 indigo tiles), Methods & Parameters, References, branded footer
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .. import __version__  # noqa: E402
from .report_kit import (  # noqa: E402
    BEIGE, CW, H, INK, M, NIGHT, NPG_AMBER, NPG_BLUE, NPG_CATEGORICAL,
    NPG_NAVY, NPG_ORANGE, NPG_RED, NPG_SLATE, NPG_TEAL, PLOT_RC, RULE, SOFT,
    TEXT, W, Doc, Page, art_trace, cover, end_page, figure_block, seq_cmap,
    sidebar_page, style_axes,
)
from .report_plots import (  # noqa: E402
    FAVOURABLE_TERMS, INTERACTION_COLORS, INTERACTION_LABELS, UNFAVOURABLE_TERMS,
    compute_report_analytics,
)

logger = logging.getLogger("pandadock.visualization.report_pdf")

REFERENCES = [
    "Trott O, Olson AJ. AutoDock Vina: improving the speed and accuracy of docking with a new "
    "scoring function, efficient optimization, and multithreading. **J. Comput. Chem.** 2010, 31, 455–461.",
    "Quiroga R, Villarreal MA. Vinardo: a scoring function based on Vina improves scoring, "
    "docking, and virtual screening. **PLoS ONE** 2016, 11, e0155183.",
    "Morris GM, et al. AutoDock4 and AutoDockTools4: automated docking with selective receptor "
    "flexibility. **J. Comput. Chem.** 2009, 30, 2785–2791.",
    "Landrum G. RDKit: Open-source cheminformatics. https://www.rdkit.org.",
    "Cock PJA, et al. Biopython: freely available Python tools for computational molecular biology "
    "and bioinformatics. **Bioinformatics** 2009, 25, 1422–1423.",
]


def _extract_ca_coords(receptor_file: Optional[str]) -> Optional[np.ndarray]:
    if not receptor_file or not Path(receptor_file).exists():
        return None
    try:
        from Bio.PDB import PDBParser

        struct = PDBParser(QUIET=True).get_structure("rec", str(receptor_file))
        model = next(struct.get_models())
        pts = [
            res["CA"].get_coord()
            for chain in model
            for res in chain
            if res.id[0] == " " and "CA" in res
        ]
        return np.asarray(pts, dtype=float) if len(pts) >= 5 else None
    except Exception:
        return None


def _plot_ax(p: Page, x, y, w, h, pad=(0.50, 0.40, 0.18, 0.30), legend=False, grid=False,
             spines=("left", "bottom"), **kw):
    lp, b, r, t = pad
    if legend:
        b += 0.24
    ax = p.axes(x + lp, y + t, max(0.3, w - lp - r), max(0.3, h - t - b), **kw)
    style_axes(ax, grid=grid, spines=spines)
    return ax


def _extract_ligand_rmsf_and_torsions(poses: Sequence, ligand_mol=None):
    coords_list = [np.asarray(p.coordinates, dtype=np.float64) for p in poses]
    if len(coords_list) < 2:
        return None, None, []
    n_atoms = coords_list[0].shape[0]
    if any(c.shape[0] != n_atoms for c in coords_list):
        return None, None, []

    if ligand_mol is not None and ligand_mol.GetNumAtoms() == n_atoms:
        heavy_idx = [a.GetIdx() for a in ligand_mol.GetAtoms() if a.GetAtomicNum() > 1]
        labels = [f"{ligand_mol.GetAtomWithIdx(i).GetSymbol()}{k + 1}" for k, i in enumerate(heavy_idx)]
    else:
        heavy_idx = list(range(n_atoms))
        labels = [f"A{i + 1}" for i in heavy_idx]

    if len(heavy_idx) < 2:
        return None, None, []

    stack = np.stack([c[heavy_idx] for c in coords_list], axis=0)
    mean_pos = stack.mean(axis=0, keepdims=True)
    rmsf = np.sqrt(np.mean(np.sum((stack - mean_pos) ** 2, axis=2), axis=0))

    torsions: List[Tuple[str, np.ndarray]] = []
    if ligand_mol is not None and ligand_mol.GetNumAtoms() == n_atoms:
        try:
            from rdkit import Chem

            rot_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
            matches = ligand_mol.GetSubstructMatches(rot_smarts) if rot_smarts else ()
            for a2, a3 in matches[:8]:
                n2 = [n.GetIdx() for n in ligand_mol.GetAtomWithIdx(a2).GetNeighbors()
                      if n.GetIdx() != a3 and n.GetAtomicNum() > 1]
                n3 = [n.GetIdx() for n in ligand_mol.GetAtomWithIdx(a3).GetNeighbors()
                      if n.GetIdx() != a2 and n.GetAtomicNum() > 1]
                if not n2 or not n3:
                    continue
                a1, a4 = n2[0], n3[0]
                angs = []
                for c in coords_list:
                    b0 = -(c[a2] - c[a1])
                    b1 = c[a3] - c[a2]
                    b2 = c[a4] - c[a3]
                    n1 = np.linalg.norm(b1)
                    if n1 < 1e-8:
                        continue
                    b1_u = b1 / n1
                    v = b0 - np.dot(b0, b1_u) * b1_u
                    w_v = b2 - np.dot(b2, b1_u) * b1_u
                    angs.append(math.degrees(math.atan2(float(np.dot(np.cross(b1_u, v), w_v)), float(np.dot(v, w_v)))))
                if angs:
                    lbl = (
                        f"{ligand_mol.GetAtomWithIdx(a1).GetSymbol()}{a1+1}-"
                        f"{ligand_mol.GetAtomWithIdx(a2).GetSymbol()}{a2+1}-"
                        f"{ligand_mol.GetAtomWithIdx(a3).GetSymbol()}{a3+1}-"
                        f"{ligand_mol.GetAtomWithIdx(a4).GetSymbol()}{a4+1}"
                    )
                    torsions.append((lbl, np.asarray(angs, dtype=float)))
        except Exception:
            pass
    return labels, rmsf, torsions


def build_pdf_report(
    output_path: Path,
    result,
    ligand_mol=None,
    receptor_file: Optional[str] = None,
    interaction_analyses: Optional[List[Dict]] = None,
    pandamap_image: Optional[Path] = None,
    analytics: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """
    Build a multi-page publication A4 PDF report matching the Panda family brochure aesthetic.
    """
    try:
        data = analytics or compute_report_analytics(
            result, ligand_mol=ligand_mol, receptor_file=receptor_file,
            interaction_analyses=interaction_analyses,
        )
        poses = list(getattr(result, "poses", []) or [])
        if not poses:
            return None

        lig_name = data["ligand_name"]
        rec_name = data["receptor_name"]
        rec_info = data["receptor_info"]
        lig_info = data["ligand_info"]
        params = data["parameters"] or {}

        with plt.rc_context(PLOT_RC):
            doc = Doc(f"PandaDock Molecular Docking Report — {rec_name} · {lig_name}",
                      kind="Molecular Docking Report")

            # =================================================================
            # PAGE 1: COVER
            # =================================================================
            ca_xyz = _extract_ca_coords(data["receptor_file"])
            top_coords = np.asarray(poses[0].coordinates, dtype=float)
            art_fn = (lambda pg: art_trace(pg, ca_xyz, extra=top_coords)) if ca_xyz is not None else None

            h1 = f"Top score {data['best_score']:.2f} kcal/mol (ΔE₁₋₂ = {data['gap_1_2']:.2f})"
            h2 = (
                f"{data['n_persistent_30']} residues contact ligand in ≥30% of poses"
                if data["persistent_contacts"]
                else f"{data['n_poses']} binding modes ranked (Boltzmann {data['top1_boltz_pct']:.0f}%)"
            )
            h3 = f"{data['n_clusters']} conformational mode{'s' if data['n_clusters'] != 1 else ''} at 2.0 Å RMSD"

            rec_str = (
                f"{rec_info['n_residues']} residues, {rec_info['n_chains']} chain(s); {rec_info['n_atoms']:,} atoms"
                if rec_info["n_residues"] > 0
                else (Path(data["receptor_file"]).name if data["receptor_file"] else "Receptor PDB")
            )
            if rec_info["metals"]:
                rec_str += f" · Metals: {', '.join(rec_info['metals'][:3])}"

            lig_str = f"{lig_name}, {lig_info['n_atoms']} atoms ({lig_info['n_heavy']} heavy)"
            if lig_info["mw"] is not None:
                lig_str += f"; MW {lig_info['mw']:.1f} Da"

            g_c = np.asarray(data["grid_center"], dtype=float)
            g_d = np.asarray(data["grid_dimensions"], dtype=float)
            box_str = f"({g_c[0]:.1f}, {g_c[1]:.1f}, {g_c[2]:.1f}) Å  |  {g_d[0]:.1f} × {g_d[1]:.1f} × {g_d[2]:.1f} Å"

            facts = [
                ("Receptor", rec_str),
                ("Ligand", lig_str),
                ("Engine", f"{data['algorithm']}  ·  {data['scoring']} scoring"),
                ("Grid Box", box_str),
                ("Sampling", f"{data['n_poses']} poses in {data['runtime_seconds']:.1f} s "
                             f"(seed {params.get('seed', 'random')})"),
            ]

            cover(
                doc,
                title=f"{rec_name} · {lig_name}",
                subtitle="Automated analysis of binding energetics, conformational diversity, "
                         "protein–ligand contacts and ligand pose geometry from molecular docking.",
                highlights=[("chart", h1), ("molecule", h2), ("target", h3)],
                quote=data["quote"],
                quote_by="PandaDock summary | computed from docked ensemble",
                facts=facts,
                art=art_fn,
                kind="MOLECULAR DOCKING REPORT",
            )

            # =================================================================
            # PAGE 2: SETUP SIDEBAR, NARRATIVE, SUMMARY METRICS & QUALITY CHECKS
            # =================================================================
            setup_paras = [
                f"The receptor **{rec_name}** contains {rec_str}." if rec_info["n_residues"] > 0
                else f"Receptor structure: **{rec_name}**.",
                f"The ligand **{lig_name}** has {lig_info['n_atoms']} atoms "
                f"({lig_info['n_heavy']} heavy)"
                + (
                    f", molecular weight **{lig_info['mw']:.1f} Da**, "
                    f"**{lig_info['rot_bonds']}** rotatable bond(s), cLogP {lig_info['logp']:.2f} "
                    f"and TPSA {lig_info['tpsa']:.1f} Å²."
                    if lig_info["mw"] is not None
                    else f" and {params.get('n_torsions', '?')} rotatable torsion(s) searched."
                ),
                f"Search box centred at ({g_c[0]:.2f}, {g_c[1]:.2f}, {g_c[2]:.2f}) Å with "
                f"dimensions {g_d[0]:.1f} × {g_d[1]:.1f} × {g_d[2]:.1f} Å.",
                f"Algorithm: **{data['algorithm']}** with **{data['scoring']}** scoring "
                f"(exhaustiveness {params.get('exhaustiveness', 'default')}, "
                f"{int(params.get('energy_evaluations', 0)):,} evaluations, "
                f"runtime {data['runtime_seconds']:.1f} s).",
            ]

            narrative = [
                f"Across **{data['n_poses']}** returned binding modes, the top-ranked pose scores "
                f"**{data['best_score']:.2f} kcal/mol** against a Boltzmann ensemble average of "
                f"**{data['ensemble_dg']:.2f} kcal/mol** (mean {data['mean_score']:.2f} ± {data['sd_score']:.2f} kcal/mol). "
                f"The energy margin to rank 2 is **{data['gap_1_2']:.2f} kcal/mol**, assigning "
                f"**{data['top1_boltz_pct']:.1f}%** Boltzmann population weight at 298 K to the leading pose.",
                f"Pairwise heavy-atom RMSD between returned modes averages **{data['mean_rmsd']:.2f} Å** "
                f"(maximum {data['max_rmsd']:.2f} Å), grouping into **{data['n_clusters']}** "
                f"conformational cluster(s) at a 2.0 Å threshold.",
                "Docking scores are empirical estimates designed to rank poses and congeneric ligands; "
                "they are **not** experimental binding free energies and are intentionally not converted "
                "to Kd or IC50 values in this report.",
            ]

            p2, y_body = sidebar_page(
                doc,
                sidebar_title="DOCKING SETUP",
                sidebar_text=setup_paras,
                heading="Binding energetics and pose convergence",
                body=narrative,
            )

            y = max(y_body, p2.sidebar_bottom) + 0.25
            p2.t(M, y, "Quality metrics", size=12.5, bold=True, head=True, color=INK)
            y += 0.30

            metric_rows = [
                ("Top-1 docking score (kcal/mol)", f"{data['best_score']:.3f}",
                 f"{data['mean_score']:.2f} ± {data['sd_score']:.2f}", f"ΔE₁₋₂ = {data['gap_1_2']:.2f}"),
                ("Boltzmann ensemble ΔG (kcal/mol)", f"{data['ensemble_dg']:.3f}",
                 f"w₁ = {data['top1_boltz_pct']:.1f}%", "T = 298.15 K"),
                ("Pairwise heavy-atom RMSD (Å)", f"{data['mean_rmsd']:.2f}",
                 f"max {data['max_rmsd']:.2f} Å", f"{data['n_clusters']} cluster(s)"),
                ("Top-1 internal strain (kcal/mol)", f"{data['top1_strain']:.2f}",
                 f"{params.get('n_torsions', lig_info.get('rot_bonds') or 0)} torsions", "ligand intra"),
            ]
            if data["persistent_contacts"]:
                top_c = data["persistent_contacts"][0]
                metric_rows.append((
                    "Persistent pocket contacts (≥30%)", str(data["n_persistent_30"]),
                    f"{len(data['persistent_contacts'])} total residues",
                    f"top: {top_c['residue']} ({top_c['occupancy_pct']:.0f}%)",
                ))

            y = p2.table(
                M, y, [2.75, 1.35, 1.50, 1.43],
                ["Observable", "Top / Value", "Spread / Ensemble", "Notes"],
                metric_rows,
                size=7.9, row_h=0.30, head_h=0.33,
                align=["left", "center", "center", "center"], zebra=True,
            )

            y += 0.22
            p2.t(M, y, "Quality checks", size=12.5, bold=True, head=True, color=INK)
            y += 0.30
            for status, msg in data["checks"]:
                col = NPG_TEAL if status == "ok" else NPG_AMBER
                kind = "check" if status == "ok" else "flame"
                p2.icon(kind, M + 0.11, y + 0.09, s=0.095, color=col, lw=1.25)
                y = p2.flow(M + 0.32, y, CW - 0.32, msg, size=8.4, color=TEXT, lead=1.38) + 0.08

            # =================================================================
            # PAGE 3: BINDING ENERGETICS & ENERGY DECOMPOSITION
            # =================================================================
            p3 = doc.page()
            p3.heading("Binding energetics and scoring funnel", y=0.68, size=20)
            y3 = p3.y + 0.05

            scores = np.asarray([float(p.energy) for p in poses], dtype=float)
            ranks = np.arange(1, len(scores) + 1)
            boltz_pct = data["boltz_weights"] * 100.0
            rmsd_to_top1 = data["rmsd_matrix"][0] if len(poses) >= 2 else np.zeros(len(poses))

            gap_w = 0.18
            col_w = (CW - 2 * gap_w) / 3.0
            panel_h = 3.10

            boltz_mean_e = float(np.sum(data["boltz_weights"] * scores))

            def draw_rank_scores(pg: Page, x, y, w, h):
                pg.panel_label(x + 0.14, y + 0.10, "a")
                ax = _plot_ax(pg, x, y, w, h, pad=(0.50, 0.56, 0.14, 0.34), legend=True, grid=True)
                ax.plot(ranks, scores, "-", color=NPG_NAVY, lw=1.3, zorder=2)
                if len(ranks) > 1:
                    ax.scatter(ranks[1:], scores[1:], color=NPG_NAVY, s=18, zorder=3, label="Modes")
                ax.scatter([ranks[0]], [scores[0]], color=NPG_RED, s=50, marker="*", zorder=4, label="Top pose")
                ax.axhline(boltz_mean_e, color=NPG_TEAL, ls="--", lw=0.9, label="Boltzmann ⟨E⟩")
                ax.set_xlabel("Pose rank")
                ax.set_ylabel("Score (kcal/mol)")
                ax.set_xticks(ranks if len(ranks) <= 8 else np.linspace(1, len(ranks), 5, dtype=int))
                ax.legend(loc="upper center", bbox_to_anchor=(0.48, -0.25), ncol=2, fontsize=5.8,
                          columnspacing=0.9, handletextpad=0.4)

            def draw_boltz(pg: Page, x, y, w, h):
                pg.panel_label(x + 0.14, y + 0.10, "b")
                ax = _plot_ax(pg, x, y, w, h, pad=(0.48, 0.44, 0.14, 0.36), legend=False, grid=True)
                cols = [NPG_RED if i == 0 else NPG_BLUE for i in range(len(ranks))]
                ax.bar(ranks, boltz_pct, color=cols, width=0.66, zorder=3)
                ax.set_ylim(0, min(115.0, max(float(np.max(boltz_pct)) * 1.18, 25.0)))
                ax.set_xlabel("Pose rank")
                ax.set_ylabel("Population (%)")
                ax.set_xticks(ranks if len(ranks) <= 8 else np.linspace(1, len(ranks), 5, dtype=int))

            def draw_funnel(pg: Page, x, y, w, h):
                pg.panel_label(x + 0.14, y + 0.10, "c")
                ax = _plot_ax(pg, x, y, w, h, pad=(0.50, 0.44, 0.14, 0.36), legend=True, grid=True)
                if len(scores) >= 2:
                    ax.scatter(rmsd_to_top1[1:], scores[1:], c=ranks[1:], cmap=seq_cmap(), s=26,
                               edgecolors=NPG_NAVY, linewidths=0.4, zorder=3, label="Other modes")
                ax.scatter([0.0], [scores[0]], color=NPG_RED, s=60, marker="*", zorder=4, label="Rank 1")
                ax.axvline(2.0, color=SOFT, ls=":", lw=0.8)
                ax.set_xlabel("RMSD to rank 1 (Å)")
                ax.set_ylabel("Score (kcal/mol)")
                ax.legend(loc="upper center", bbox_to_anchor=(0.48, -0.24), ncol=2, fontsize=6.2)

            y3a = figure_block(doc, p3, M, y3, col_w, panel_h,
                               "Ranked docking scores (navy) with rank 1 (red) and Boltzmann ensemble average (teal).",
                               draw_rank_scores)
            y3b = figure_block(doc, p3, M + col_w + gap_w, y3, col_w, panel_h,
                               "Boltzmann population weights (%) at 298.15 K across returned binding modes.",
                               draw_boltz)
            y3c = figure_block(doc, p3, M + 2 * (col_w + gap_w), y3, col_w, panel_h,
                               "Score-versus-RMSD binding funnel relative to the top-ranked pose (dashed: 2.0 Å).",
                               draw_funnel)

            # Second row on Page 3: Energy decomposition (if components exist)
            components = [p.energy_components for p in poses if getattr(p, "energy_components", None)]
            comp_names = sorted({k for c in components for k in c if k != "total"}) if components else []
            if comp_names:
                y_comp = max(y3a, y3b, y3c) + 0.30
                comp_h = 3.55

                def draw_components(pg: Page, x, y, w, h):
                    pg.panel_label(x + 0.16, y + 0.12, "d")
                    ax = _plot_ax(pg, x, y, w, h, pad=(0.56, 0.56, 0.22, 0.36), legend=True, grid=True)
                    c_ranks = np.arange(1, len(components) + 1)
                    n_t = len(comp_names)
                    bw = min(0.76 / max(n_t, 1), 0.22)
                    fav_c = [NPG_NAVY, NPG_BLUE, NPG_TEAL, "#5B8DEF"]
                    pen_c = [NPG_RED, NPG_ORANGE, NPG_AMBER]
                    fi, pi = 0, 0
                    for idx, name in enumerate(comp_names):
                        vals = [float(c.get(name, 0.0)) for c in components]
                        off = (idx - (n_t - 1) / 2.0) * bw
                        is_fav = name in FAVOURABLE_TERMS or (name not in UNFAVOURABLE_TERMS and np.mean(vals) <= 0)
                        if is_fav:
                            col = fav_c[fi % len(fav_c)]
                            fi += 1
                            lbl = f"{name.replace('_', ' ')} (fav.)"
                        else:
                            col = pen_c[pi % len(pen_c)]
                            pi += 1
                            lbl = f"{name.replace('_', ' ')} (penalty)"
                        ax.bar(c_ranks + off, vals, width=bw, color=col, label=lbl, zorder=3)
                    ax.axhline(0.0, color=INK, lw=0.75, zorder=4)
                    ax.set_xlabel("Pose rank")
                    ax.set_ylabel("Contribution (kcal/mol)")
                    ax.set_xticks(c_ranks)
                    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                              ncol=min(4, max(2, n_t)), fontsize=6.4)

                figure_block(
                    doc, p3, M, y_comp, CW, comp_h,
                    "Per-pose energy decomposition. Stabilising terms lie below zero; steric/strain/entropy penalties lie above zero.",
                    draw_components,
                )

            # =================================================================
            # PAGE 4: POSE DIVERSITY & CONFORMATIONAL LANDSCAPE
            # =================================================================
            if len(poses) >= 2:
                p4 = doc.page()
                p4.heading("Conformational landscape and pose diversity", y=0.68, size=20)
                y4 = p4.columns(
                    M, p4.y + 0.02, CW,
                    [
                        f"Pairwise heavy-atom RMSD across the {len(poses)} returned modes averages "
                        f"**{data['mean_rmsd']:.2f} Å** (range 0.0–{data['max_rmsd']:.2f} Å). Low off-diagonal "
                        "values indicate tight convergence on a single binding mode; larger blocks reveal distinct orientations.",
                        f"Classical multidimensional scaling (MDS) projects the pairwise RMSD distance matrix "
                        f"onto two principal axes, separating **{data['n_clusters']}** cluster(s) at a 2.0 Å cutoff "
                        "and colouring each mode by its docking score.",
                    ],
                ) + 0.14

                rmsd_mat = data["rmsd_matrix"]
                n_p = len(poses)
                half_w = (CW - 0.24) / 2.0
                div_h = 3.25

                def draw_rmsd_mat(pg: Page, x, y, w, h):
                    pg.panel_label(x + 0.14, y + 0.10, "a")
                    ax = _plot_ax(pg, x, y, w, h, pad=(0.48, 0.42, 0.56, 0.34), spines=("left", "bottom", "top", "right"))
                    im = ax.imshow(rmsd_mat, cmap=seq_cmap(), origin="upper", interpolation="nearest")
                    ax.set_xticks(range(n_p))
                    ax.set_yticks(range(n_p))
                    ax.set_xticklabels(range(1, n_p + 1), fontsize=6.0)
                    ax.set_yticklabels(range(1, n_p + 1), fontsize=6.0)
                    ax.set_xlabel("Pose rank")
                    ax.set_ylabel("Pose rank")
                    vmax = float(np.nanmax(rmsd_mat)) if np.any(~np.isnan(rmsd_mat)) else 1.0
                    if n_p <= 10:
                        for i in range(n_p):
                            for j in range(n_p):
                                v = rmsd_mat[i, j]
                                if not math.isnan(v):
                                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.8,
                                            color="white" if v > vmax * 0.58 else INK)
                    cax = pg.axes(x + w - 0.44, y + 0.34, 0.10, h - 0.76)
                    cb = pg.fig.colorbar(im, cax=cax)
                    cb.set_label("RMSD (Å)", fontsize=6.5)
                    cb.ax.tick_params(labelsize=6.0, length=2)

                def draw_mds(pg: Page, x, y, w, h):
                    pg.panel_label(x + 0.14, y + 0.10, "b")
                    ax = _plot_ax(pg, x, y, w, h, pad=(0.50, 0.42, 0.56, 0.34), grid=True)
                    d_cl = np.nan_to_num(rmsd_mat, nan=0.0)
                    j_m = np.eye(n_p) - np.ones((n_p, n_p)) / n_p
                    b_m = -0.5 * j_m @ (d_cl ** 2) @ j_m
                    ev, evec = np.linalg.eigh(b_m)
                    ord_i = np.argsort(ev)[::-1]
                    ev = np.maximum(ev[ord_i], 0.0)
                    evec = evec[:, ord_i]
                    tot = float(np.sum(ev)) + 1e-12
                    pc1 = evec[:, 0] * np.sqrt(ev[0])
                    pc2 = evec[:, 1] * np.sqrt(ev[1]) if n_p > 2 else np.zeros(n_p)
                    sc = ax.scatter(pc1, pc2, c=scores, cmap="RdYlBu_r", s=46,
                                    edgecolors=NPG_NAVY, linewidths=0.55, zorder=4)
                    ax.scatter([pc1[0]], [pc2[0]], s=95, marker="*", facecolor=NPG_RED,
                               edgecolor=NPG_NAVY, lw=0.5, zorder=5)
                    for k in range(min(3, n_p)):
                        ax.annotate(f"#{k + 1}", (pc1[k], pc2[k]), xytext=(4, 4),
                                    textcoords="offset points", fontsize=6.8, fontweight="bold", color=INK)
                    ax.set_xlabel(f"MDS-1 ({100.0 * ev[0] / tot:.0f}% var)")
                    ax.set_ylabel(f"MDS-2 ({100.0 * (ev[1] if n_p > 2 else 0.0) / tot:.0f}% var)")
                    cax = pg.axes(x + w - 0.44, y + 0.34, 0.10, h - 0.76)
                    cb = pg.fig.colorbar(sc, cax=cax)
                    cb.set_label("Score (kcal/mol)", fontsize=6.5)
                    cb.ax.tick_params(labelsize=6.0, length=2)

                y4a = figure_block(doc, p4, M, y4, half_w, div_h,
                                   "Pairwise heavy-atom RMSD (Å) between returned poses.", draw_rmsd_mat)
                y4b = figure_block(doc, p4, M + half_w + 0.24, y4, half_w, div_h,
                                   "2-D conformational embedding coloured by docking score (star: rank 1).", draw_mds)

                yt = max(y4a, y4b) + 0.28
                p4.t(M, yt, "Top docked binding modes", size=12.5, bold=True, head=True, color=INK)
                yt += 0.28
                t_rows = [
                    (
                        f"#{r['rank']}",
                        f"{r['score']:.3f}",
                        f"+{r['delta_e']:.2f}" if r["delta_e"] > 0 else "0.00",
                        f"{r['boltz_pct']:.1f}%",
                        f"{r['rmsd_top1']:.2f}",
                        r["cluster"],
                        f"{r['strain']:.2f}",
                        f"HB:{r['hb']} HP:{r['hp']}" + (f" π:{r['pi']}" if r["pi"] else "") + (f" M:{r['mt']}" if r["mt"] else ""),
                    )
                    for r in data["pose_rows"][:10]
                ]
                p4.table(
                    M, yt, [0.55, 0.90, 0.72, 0.80, 0.85, 0.62, 0.74, 1.85],
                    ["Rank", "Score", "ΔE", "Pop.", "RMSD₁ (Å)", "Mode", "Strain", "Contacts"],
                    t_rows,
                    size=7.6, row_h=0.28, head_h=0.31,
                    align=["center", "center", "center", "center", "center", "center", "center", "left"],
                    zebra=True, highlight=0,
                )

            # =================================================================
            # PAGE 5: PROTEIN-LIGAND BINDING MODE & PERSISTENT CONTACTS
            # =================================================================
            persistent = data["persistent_contacts"]
            if persistent and interaction_analyses:
                p5 = doc.page()
                p5.heading("Protein–ligand binding mode", y=0.68, size=20)
                top_names = ", ".join(f"**{c['residue']}** ({c['occupancy_pct']:.0f}%)" for c in persistent[:5])
                ev_str = ", ".join(
                    f"{INTERACTION_LABELS.get(k, k).lower()} {v}"
                    for k, v in sorted(data["event_totals"].items())
                    if v > 0
                ) or "none detected"
                y5 = p5.columns(
                    M, p5.y + 0.02, CW,
                    [
                        f"**{len(persistent)}** pocket residues contact the ligand across the returned "
                        f"poses and **{data['n_persistent_30']}** in at least 30% of modes. The most "
                        f"persistent are {top_names}. Occupancy is the fraction of poses in which a "
                        "contact occurs, so stacked bars can exceed 1 when a residue forms multiple interaction types.",
                        f"Detected events across all poses: {ev_str}. "
                        "Criteria are geometric (H-bond < 3.5 Å, hydrophobic C–C < 4.5 Å, "
                        "π-stacking < 5.5 Å, ionic < 4.0 Å, metal < 2.8 Å); verify key contacts visually.",
                    ],
                ) + 0.12

                p5.t(M, y5, "Persistent contacts", size=12.5, bold=True, head=True, color=INK)
                y5 += 0.28
                c_rows = [
                    (
                        c["residue"],
                        f"{c['occupancy_pct']:.0f}%",
                        c["types"],
                        f"{c['mean_dist']:.2f}" if not math.isnan(c["mean_dist"]) else "—",
                    )
                    for c in persistent[:8]
                ]
                y5 = p5.table(
                    M, y5, [1.55, 1.15, 3.13, 1.20],
                    ["Residue", "Occupancy", "Interaction types", "Mean dist. (Å)"],
                    c_rows, size=7.8, row_h=0.27, head_h=0.31,
                    align=["left", "center", "left", "center"], zebra=True,
                ) + 0.16

                top_res = persistent[:15]
                res_list = [c["residue"] for c in top_res]
                n_int_poses = len(interaction_analyses)
                canon = [
                    "hydrogen_bonds", "hydrophobic", "van_der_waals_contacts",
                    "pi_stacking", "salt_bridges", "electrostatic",
                    "electrostatic_interactions", "metal_coordination", "halogen_bonds",
                ]
                present_t = [k for k in canon if any(k in c["type_fractions"] for c in top_res)]

                mat_c = np.zeros((len(res_list), n_int_poses), dtype=float)
                r2i = {r: i for i, r in enumerate(res_list)}
                for pi, pdata in enumerate(interaction_analyses):
                    for items in (pdata.get("interaction_details") or {}).values():
                        if isinstance(items, list):
                            for it in items:
                                rn = it.get("residue")
                                if rn in r2i:
                                    mat_c[r2i[rn], pi] += 1.0

                half_w = (CW - 0.24) / 2.0
                fig5_h = min(3.75, max(2.85, H - 0.95 - y5 - 0.55))

                def draw_occ_bars(pg: Page, x, y, w, h):
                    pg.panel_label(x + 0.14, y + 0.10, "a")
                    ax = _plot_ax(pg, x, y, w, h, pad=(0.46, 0.72, 0.14, 0.34), grid=True)
                    xs = np.arange(len(res_list))
                    bot = np.zeros(len(res_list))
                    for itype in present_t:
                        vals = np.array([c["type_fractions"].get(itype, 0.0) for c in top_res])
                        ax.bar(xs, vals, bottom=bot, width=0.72,
                               color=INTERACTION_COLORS.get(itype, NPG_SLATE),
                               label=INTERACTION_LABELS.get(itype, itype), zorder=3)
                        bot += vals
                    ax.set_xticks(xs)
                    ax.set_xticklabels(res_list, rotation=52, ha="right", fontsize=5.8)
                    ax.set_ylabel("Interaction fraction")
                    ax.set_ylim(0, max(1.15, float(np.max(bot)) * 1.18))
                    ax.legend(loc="upper right", ncol=2, fontsize=5.8)

                def draw_contact_hm(pg: Page, x, y, w, h):
                    pg.panel_label(x + 0.14, y + 0.10, "b")
                    ax = _plot_ax(pg, x, y, w, h, pad=(0.74, 0.44, 0.50, 0.34),
                                  spines=("left", "bottom", "top", "right"))
                    im = ax.imshow(mat_c, cmap=seq_cmap(), aspect="auto", origin="upper", interpolation="nearest")
                    ax.set_xticks(range(n_int_poses))
                    ax.set_xticklabels(range(1, n_int_poses + 1), fontsize=6.0)
                    ax.set_yticks(range(len(res_list)))
                    ax.set_yticklabels(res_list, fontsize=5.8)
                    ax.set_xlabel("Pose rank")
                    cax = pg.axes(x + w - 0.40, y + 0.34, 0.10, h - 0.78)
                    cb = pg.fig.colorbar(im, cax=cax)
                    cb.set_label("Contacts", fontsize=6.4)
                    cb.ax.tick_params(labelsize=5.8, length=2)

                figure_block(doc, p5, M, y5, half_w, fig5_h,
                             "Contact occupancy per residue across returned poses, split by interaction type.",
                             draw_occ_bars)
                figure_block(doc, p5, M + half_w + 0.24, y5, half_w, fig5_h,
                             "Contact count of each pocket residue across ranked poses.",
                             draw_contact_hm)

            # =================================================================
            # PAGE 6: LIGAND POSE FLEXIBILITY, TORSIONS & RESIDUE HOTSPOTS
            # =================================================================
            atom_labels, rmsf, torsions = _extract_ligand_rmsf_and_torsions(poses, ligand_mol)
            hotspots = data["hotspots"]
            if rmsf is not None or hotspots:
                p6 = doc.page()
                p6.heading("Ligand pose geometry and residue hotspots", y=0.68, size=20)
                y6 = p6.y + 0.04

                if rmsf is not None:
                    anc_i = int(np.argmin(rmsf))
                    mob_i = int(np.argmax(rmsf))
                    y6 = p6.columns(
                        M, y6, CW,
                        [
                            f"Heavy-atom RMSF across the docked ensemble spans **{rmsf.min():.2f}–{rmsf.max():.2f} Å**. "
                            f"Low values mark binding anchors (most anchored: **{atom_labels[anc_i]}** at {rmsf[anc_i]:.2f} Å); "
                            f"high values highlight mobile or solvent-exposed groups (**{atom_labels[mob_i]}** at {rmsf[mob_i]:.2f} Å).",
                            (f"**{len(torsions)}** rotatable ligand bond(s) were profiled across poses (panel b); "
                             "multi-modal histograms indicate distinct rotamer families.")
                            if torsions else
                            "No rotatable heavy-atom dihedrals were present in this rigid ligand scaffold.",
                        ],
                    ) + 0.10

                    if torsions:
                        # Row 1: RMSF (left) + Torsion histograms (right)
                        w_left = CW * 0.48
                        w_right = CW - w_left - 0.22
                        h_lig = 2.95

                        def draw_rmsf(pg: Page, x, y, w, h):
                            pg.panel_label(x + 0.14, y + 0.10, "a")
                            ax = _plot_ax(pg, x, y, w, h, pad=(0.48, 0.56, 0.14, 0.34), grid=True)
                            xs = np.arange(len(rmsf))
                            ax.plot(xs, rmsf, "-o", color=NPG_RED, lw=1.2, ms=3.2, zorder=3)
                            ax.fill_between(xs, 0, rmsf, color=NPG_RED, alpha=0.10, zorder=2)
                            ax.scatter([anc_i], [rmsf[anc_i]], color=NPG_TEAL, s=36, zorder=4)
                            step = max(1, len(atom_labels) // 14)
                            ax.set_xticks(xs[::step])
                            ax.set_xticklabels(atom_labels[::step], rotation=70, fontsize=5.6)
                            ax.set_ylabel("Atom RMSF (Å)")
                            ax.set_ylim(0, max(float(np.max(rmsf)) * 1.22, 0.4))

                        def draw_torsions(pg: Page, x, y, w, h):
                            pg.panel_label(x + 0.14, y + 0.10, "b")
                            nt = min(len(torsions), 6)
                            nc = min(3, nt)
                            nr = math.ceil(nt / nc)
                            pad_l, pad_r, pad_t, pad_b = 0.24, 0.20, 0.44, 0.30
                            hgap = 0.28
                            gx_w = (w - pad_l - pad_r - (nc - 1) * hgap) / nc
                            gy_h = (h - pad_t - pad_b - (nr - 1) * 0.40) / nr
                            for ti, (lbl, angs) in enumerate(torsions[:nt]):
                                r_i, c_i = divmod(ti, nc)
                                ax = pg.axes(
                                    x + pad_l + c_i * (gx_w + hgap),
                                    y + pad_t + r_i * (gy_h + 0.40),
                                    gx_w, gy_h,
                                )
                                style_axes(ax)
                                ax.hist(angs, bins=np.linspace(-180, 180, 17),
                                        color=NPG_CATEGORICAL[ti % len(NPG_CATEGORICAL)],
                                        alpha=0.88, edgecolor=BEIGE, lw=0.35)
                                ax.set_xlim(-180, 180)
                                ax.set_xticks([-180, 0, 180])
                                ax.set_yticks([])
                                ax.tick_params(labelsize=5.0)
                                ax.set_title(lbl, fontsize=5.3, color=SOFT, pad=2)

                        ya = figure_block(doc, p6, M, y6, w_left, h_lig,
                                          "Per-heavy-atom RMSF (Å) across docked poses (green: most anchored atom).",
                                          draw_rmsf)
                        yb = figure_block(doc, p6, M + w_left + 0.22, y6, w_right, h_lig,
                                          "Torsion-angle distributions (°) of rotatable ligand bonds across poses.",
                                          draw_torsions)
                        y6 = max(ya, yb) + 0.24
                    else:
                        def draw_rmsf_wide(pg: Page, x, y, w, h):
                            pg.panel_label(x + 0.14, y + 0.10, "a")
                            ax = _plot_ax(pg, x, y, w, h, pad=(0.50, 0.56, 0.18, 0.34), grid=True)
                            xs = np.arange(len(rmsf))
                            ax.plot(xs, rmsf, "-o", color=NPG_RED, lw=1.2, ms=3.5, zorder=3)
                            ax.fill_between(xs, 0, rmsf, color=NPG_RED, alpha=0.10, zorder=2)
                            ax.scatter([anc_i], [rmsf[anc_i]], color=NPG_TEAL, s=40, zorder=4)
                            step = max(1, len(atom_labels) // 20)
                            ax.set_xticks(xs[::step])
                            ax.set_xticklabels(atom_labels[::step], rotation=60, fontsize=6.0)
                            ax.set_ylabel("Atom RMSF (Å)")
                            ax.set_ylim(0, max(float(np.max(rmsf)) * 1.22, 0.4))

                        y6 = figure_block(doc, p6, M, y6, CW, 2.55,
                                          "Per-heavy-atom RMSF (Å) across returned poses.", draw_rmsf_wide) + 0.24

                if hotspots:
                    p6.t(M, y6, "Energy hotspots (empirical contact proxy)", size=12.0, bold=True, head=True, color=INK)
                    y6 += 0.24
                    p6.flow(
                        M, y6, CW,
                        "Residue contributions are **empirical contact-proxy scores** weighted by interaction type "
                        "and distance; they rank pocket residues but are not MM/PBSA or FEP free energies.",
                        size=8.0, color=SOFT, lead=1.35,
                    )
                    y6 += 0.38
                    top_hs = hotspots[:10]
                    w_bar = CW * 0.56
                    w_tbl = CW - w_bar - 0.24
                    h_hs = min(3.45, max(2.40, H - 0.85 - y6 - 0.42))

                    def draw_hotspots(pg: Page, x, y, w, h):
                        pg.panel_label(x + 0.14, y + 0.10, "c" if rmsf is not None else "a")
                        ax = _plot_ax(pg, x, y, w, h, pad=(0.78, 0.42, 0.18, 0.32), grid=True)
                        r_lbls = [h_i["residue"] for h_i in top_hs][::-1]
                        r_vals = [h_i["proxy_score"] for h_i in top_hs][::-1]
                        ys = np.arange(len(r_lbls))
                        ax.barh(ys, r_vals, height=0.66, color=NPG_BLUE, zorder=3)
                        ax.axvline(0.0, color=INK, lw=0.7)
                        ax.set_yticks(ys)
                        ax.set_yticklabels(r_lbls, fontsize=6.2)
                        ax.set_xlabel("Proxy score (kcal/mol)")
                        ax.set_xlim(min(r_vals) * 1.20, 0.05)

                    figure_block(doc, p6, M, y6, w_bar, h_hs,
                                 "Strongest pocket residue contributions to the empirical interaction proxy.",
                                 draw_hotspots)

                    hs_rows = [(h_i["residue"], f"{h_i['proxy_score']:.2f}") for h_i in top_hs[:9]]
                    p6.table(
                        M + w_bar + 0.24, y6, [w_tbl * 0.58, w_tbl * 0.42],
                        ["Residue", "Proxy (kcal/mol)"],
                        hs_rows, size=7.6, row_h=0.27, head_h=0.31,
                        align=["left", "center"], zebra=True,
                    )

            # =================================================================
            # OPTIONAL PAGE: PANDAMAP 2D INTERACTION DIAGRAM
            # =================================================================
            if pandamap_image is not None and Path(pandamap_image).exists():
                p_pm = doc.page()
                p_pm.heading("2D protein–ligand interaction map", y=0.68, size=20)
                ypm = p_pm.flow(
                    M, p_pm.y + 0.02, CW,
                    "Two-dimensional ligand-centric interaction diagram for the top-ranked docked complex, "
                    "highlighting hydrogen bonds, hydrophobic envelopes, π-stacking and metal contacts (PandaMap).",
                    size=9.0, color=TEXT, lead=1.45,
                ) + 0.18

                def draw_pandamap(pg: Page, x, y, w, h):
                    import matplotlib.image as mpimg

                    img = mpimg.imread(str(pandamap_image))
                    ax = pg.axes(x + 0.18, y + 0.18, w - 0.36, h - 0.36)
                    ax.imshow(img)
                    ax.axis("off")

                figure_block(
                    doc, p_pm, M, ypm, CW, 6.95,
                    f"PandaMap 2D interaction diagram for rank-1 pose of {lig_name} in {rec_name}.",
                    draw_pandamap,
                )

            # =================================================================
            # FINAL PAGE: ANALYSIS MODULES, METHODS, REFERENCES & FOOTER
            # =================================================================
            tiles = [
                ("Search", "Flexible-ligand Monte Carlo with local L-BFGS optimisation over translation, SO(3) rotation and torsions."),
                ("Scoring", f"{data['scoring']} empirical affinity grids with explicit favourable vs penalty energy decomposition."),
                ("Interactions", "Geometric H-bond, hydrophobic, π-stacking, ionic and metal-coordination occupancy profiling."),
                ("Geometry", "Pairwise heavy-atom RMSD clustering, 2-D MDS landscape, atom RMSF and torsion distributions."),
            ]
            methods_rows = [
                ("Software", f"PandaDock {__version__}; RDKit; BioPython; NumPy / SciPy; Matplotlib"),
                ("Engine", f"{data['algorithm']} ({data['scoring']} scoring); runtime {data['runtime_seconds']:.1f} s"),
                ("Grid Box", f"centre ({g_c[0]:.2f}, {g_c[1]:.2f}, {g_c[2]:.2f}) Å; "
                             f"size {g_d[0]:.1f} × {g_d[1]:.1f} × {g_d[2]:.1f} Å"),
                ("Sampling", f"exhaustiveness {params.get('exhaustiveness', 'default')}, "
                             f"{int(params.get('energy_evaluations', 0)):,} evaluations, "
                             f"seed {params.get('seed', 'random')}"),
                ("Clustering", f"greedy leader clustering at 2.0 Å heavy-atom RMSD; {data['n_clusters']} mode(s) returned"),
                ("Caveats", "docking scores are empirical ranking estimates, not experimental ΔG/Kd/IC50; "
                            "contact proxy scores rank pocket residues qualitatively and are not MM/PBSA or FEP."),
            ]
            end_page(
                doc,
                tiles=tiles,
                methods_rows=methods_rows,
                references=REFERENCES,
                contact="https://github.com/pritampanda15/PandaDock",
            )

            return doc.save(output_path)
    except Exception as exc:
        logger.error("Could not build PDF report: %s", exc, exc_info=True)
        return None
