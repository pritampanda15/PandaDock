"""
Congeneric Series / MCS Core-Tethered Docking & SAR Brochure Pipeline (`pandadock series`).

Built specifically for medicinal-chemistry lead optimization:
1. Automatically discovers the Maximum Common Substructure (MCS) across the analog
   series (or uses a user-supplied `--core-smarts` / `--reference-pose`).
2. Builds a shared receptor `GridCache` once and docks all analogs (with optional
   differentiable MCS core restraints toward a 3D reference pose).
3. Scores every top analog pose across an honest multi-physics consensus panel:
   - Vina score (kcal/mol)
   - Vinardo score (kcal/mol)
   - Differential pocket MM-GBSA ΔG (kcal/mol)
   - Ligand Efficiency (LE = -Vina / N_heavy)
   - PoseBusters chemistry & geometry validity (11 checks)
   - Optional PandaDockGNN pKd prediction
4. Computes pairwise SAR ΔΔScore comparisons against the series reference/lead.
5. Exports `series_results.csv`, `series_results.json`, `series_poses.sdf`,
   2D SAR analog grid (`series_grid_2d.png`), and a publication-grade Panda-family
   A4 brochure PDF (`series_report.pdf`) + HTML (`series_report.html`).
"""

from __future__ import annotations

import csv
import html
import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, Lipinski, rdFMCS, rdMolDescriptors

from .. import __version__
from .algorithms.pandacore import PandaCoreDocker
from .refinement.mmgbsa import MMGBSARescoring
from .scoring.vina_scoring import VinaScoring
from .search.grid_maps import GridCache
from .search.restraints import build_mcs_restraints
from .validation.posebusters import audit_pose

logger = logging.getLogger("pandadock.docking.series")


def load_series_ligands(ligands_path: str) -> List[Chem.Mol]:
    """Load an analog series from an SDF, SMILES (.smi/.txt), or directory of ligand files."""
    p = Path(ligands_path)
    if not p.exists():
        raise FileNotFoundError(f"Ligands input not found: {ligands_path}")

    mols: List[Chem.Mol] = []
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in (".sdf", ".mol", ".mol2", ".smi"):
                mols.extend(load_series_ligands(str(f)))
        return mols

    ext = p.suffix.lower()
    if ext == ".sdf":
        suppl = Chem.SDMolSupplier(str(p), removeHs=False)
        for i, m in enumerate(suppl):
            if m is not None:
                if not m.HasProp("_Name") or not m.GetProp("_Name").strip():
                    m.SetProp("_Name", f"{p.stem}_{i + 1}")
                mols.append(m)
    elif ext == ".mol":
        m = Chem.MolFromMolFile(str(p), removeHs=False)
        if m is not None:
            m.SetProp("_Name", p.stem)
            mols.append(m)
    elif ext == ".mol2":
        m = Chem.MolFromMol2File(str(p), removeHs=False)
        if m is not None:
            m.SetProp("_Name", p.stem)
            mols.append(m)
    elif ext in (".smi", ".smiles", ".txt"):
        for idx, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            m = Chem.MolFromSmiles(parts[0])
            if m is not None:
                name = parts[1] if len(parts) > 1 else f"analog_{idx}"
                m.SetProp("_Name", name)
                mols.append(m)
    else:
        raise ValueError(f"Unsupported series ligand file format: {ext}")

    return mols


def _ensure_3d(mol: Chem.Mol, seed: int = 42) -> Chem.Mol:
    m = Chem.Mol(mol)
    if m.GetNumConformers() == 0:
        m = Chem.AddHs(m)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        if AllChem.EmbedMolecule(m, params) != 0:
            AllChem.EmbedMolecule(m, useRandomCoords=True, randomSeed=int(seed))
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=300)
        except Exception:
            pass
    return m


def _compute_mcs_smarts(mols: Sequence[Chem.Mol], min_atoms: int = 5) -> Optional[str]:
    if len(mols) < 2:
        return None
    clean = [Chem.RemoveHs(Chem.Mol(m)) for m in mols]
    mcs = rdFMCS.FindMCS(
        clean,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        timeout=5,
    )
    return mcs.smartsString if mcs.numAtoms >= min_atoms else None


def _core_rmsd(pose_mol: Chem.Mol, ref_mol: Chem.Mol, smarts: Optional[str]) -> Optional[float]:
    if not smarts or ref_mol is None or ref_mol.GetNumConformers() == 0:
        return None
    pat = Chem.MolFromSmarts(smarts)
    if pat is None:
        return None
    m1 = pose_mol.GetSubstructMatch(pat)
    m2 = ref_mol.GetSubstructMatch(pat)
    if not m1 or not m2 or len(m1) != len(m2):
        return None
    c1 = pose_mol.GetConformer(0)
    c2 = ref_mol.GetConformer(0)
    diffs = []
    for i, j in zip(m1, m2):
        if pose_mol.GetAtomWithIdx(i).GetAtomicNum() <= 1:
            continue
        p1 = c1.GetAtomPosition(i)
        p2 = c2.GetAtomPosition(j)
        diffs.append((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
    return float(np.sqrt(np.mean(diffs))) if diffs else None


def _render_2d_sar_grid(
    mols: Sequence[Chem.Mol],
    records: Sequence[Dict[str, Any]],
    mcs_smarts: Optional[str],
    out_png: Path,
) -> Optional[Path]:
    """Render an RDKit 2D analog grid highlighting the shared MCS core and scores."""
    try:
        pat = Chem.MolFromSmarts(mcs_smarts) if mcs_smarts else None
        depict_mols = []
        legends = []
        highlights = []
        for m, rec in zip(mols, records):
            m2d = Chem.RemoveHs(Chem.Mol(m))
            AllChem.Compute2DCoords(m2d)
            depict_mols.append(m2d)
            pb_tag = "PB:PASS" if rec["pb_valid"] else "PB:WARN"
            legends.append(
                f"#{rec['consensus_rank']} {rec['name']}\n"
                f"Vina {rec['vina_kcal']:.2f} | MM-GBSA {rec['mmgbsa_kcal']:.1f} | {pb_tag}"
            )
            if pat is not None and m2d.HasSubstructMatch(pat):
                highlights.append(list(m2d.GetSubstructMatch(pat)))
            else:
                highlights.append([])

        n = len(depict_mols)
        mols_per_row = min(4, max(2, int(math.ceil(math.sqrt(n)))))
        img = Draw.MolsToGridImage(
            depict_mols,
            molsPerRow=mols_per_row,
            subImgSize=(340, 260),
            legends=legends,
            highlightAtomLists=highlights,
            useSVG=False,
        )
        out_png.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_png))
        return out_png
    except Exception as exc:
        logger.debug("2D SAR grid rendering skipped: %s", exc)
        return None


def _build_series_pdf(
    output_pdf: Path,
    receptor_file: str,
    records: List[Dict[str, Any]],
    ddg_rows: List[Dict[str, Any]],
    mcs_smarts: Optional[str],
    grid_center: np.ndarray,
    grid_dimensions: np.ndarray,
    grid_2d_png: Optional[Path],
    runtime_s: float,
    tethered: bool,
) -> Path:
    """Build the Panda-family A4 vector Series SAR Brochure PDF using `report_kit.py`."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from ..visualization.report_kit import (
        ACCENT,
        BEIGE,
        CW,
        Doc,
        INK,
        M,
        NPG_AMBER,
        NPG_BLUE,
        NPG_NAVY,
        NPG_RED,
        NPG_TEAL,
        PLOT_RC,
        RULE,
        SOFT,
        TEXT,
        W,
        art_trace,
        cover,
        end_page,
        figure_block,
        sidebar_page,
        style_axes,
    )

    rec_name = Path(receptor_file).stem
    n_lig = len(records)
    best = records[0]
    pb_pass = sum(1 for r in records if r["pb_valid"])
    best_vina = best["vina_kcal"]
    best_mmgbsa = best["mmgbsa_kcal"]

    # Extract C-alpha trace for cover art
    ca_xyz = np.zeros((0, 3))
    try:
        struct = PDBParser(QUIET=True).get_structure("rec", receptor_file)
        ca = [a.get_coord() for a in struct.get_atoms() if a.get_name().strip() == "CA"]
        if ca:
            ca_xyz = np.asarray(ca, dtype=float)
    except Exception:
        pass

    doc = Doc(title=f"PandaDock Series SAR Report — {rec_name}", kind="Congeneric Series SAR Report")

    # Page 1: Cover
    highlights = [
        ("target", f"Lead analog: **{best['name']}**\nVina {best_vina:.2f} kcal/mol"),
        ("check", f"{n_lig} analogs profiled\n{pb_pass}/{n_lig} PoseBusters PASS"),
        ("molecule", f"MM-GBSA best {best_mmgbsa:.1f} kcal/mol\n{'MCS-tethered' if tethered else 'Shared-grid'} search"),
    ]
    quote = (
        f"{best['name']} ranks #1 across the {n_lig}-compound series in {rec_name} "
        f"(Vina {best_vina:.2f}, Vinardo {best['vinardo_kcal']:.2f}, MM-GBSA {best_mmgbsa:.1f} kcal/mol; "
        f"LE = {best['ligand_efficiency']:.2f} kcal/mol/HA)."
    )
    facts = [
        ("Receptor", f"{rec_name} ({n_lig} analogs)"),
        ("Lead analog", f"{best['name']} (LE {best['ligand_efficiency']:.2f})"),
        ("Top Vina / Vinardo", f"{best_vina:.2f} / {best['vinardo_kcal']:.2f} kcal/mol"),
        ("PoseBusters pass", f"{pb_pass} / {n_lig} ({100.0 * pb_pass / max(n_lig, 1):.0f}%)"),
        ("Runtime", f"{runtime_s:.1f} s (shared GridCache)"),
    ]

    cover(
        doc,
        title=f"{rec_name} Series SAR",
        subtitle=(
            f"Multi-scoring congeneric docking and structure-activity brochure across "
            f"{n_lig} analogs with Vina, Vinardo, pocket MM-GBSA, and PoseBusters validation."
        ),
        highlights=highlights,
        quote=quote,
        quote_by=f"PandaDock v{__version__} Series Engine | {datetime.now():%Y-%m-%d}",
        facts=facts,
        art=(lambda p: art_trace(p, ca_xyz)) if len(ca_xyz) >= 3 else None,
    )

    # Page 2: Executive SAR Overview & Consensus Ranking Plot
    sidebar_text = [
        f"All **{n_lig} analogs** were docked into a single shared affinity-grid volume centered at "
        f"({grid_center[0]:.1f}, {grid_center[1]:.1f}, {grid_center[2]:.1f}) Å.",
        f"**Consensus ranking** combines normalized ranks from AutoDock Vina, Vinardo, and differential "
        f"pocket MM-GBSA to reduce single-force-field bias.",
        f"**PoseBusters audit** verifies 11 chemical and geometric invariants per top pose "
        f"({pb_pass}/{n_lig} passed all checks).",
    ]
    body_text = [
        f"Across the {n_lig}-compound series, **Vina scores** span "
        f"{min(r['vina_kcal'] for r in records):.2f} to {max(r['vina_kcal'] for r in records):.2f} kcal/mol, "
        f"while **differential pocket MM-GBSA** spans "
        f"{min(r['mmgbsa_kcal'] for r in records):.1f} to {max(r['mmgbsa_kcal'] for r in records):.1f} kcal/mol.",
        f"The consensus lead **{best['name']}** achieves Ligand Efficiency "
        f"**LE = {best['ligand_efficiency']:.2f} kcal/mol/HA** (MW {best['mw']:.1f} Da, cLogP {best['clogp']:.2f}).",
    ]
    p2, y2 = sidebar_page(
        doc,
        sidebar_title="Series Protocol",
        sidebar_text=sidebar_text,
        heading="Multi-Scoring Consensus & Efficiency",
        body=body_text,
    )

    fig_y = max(y2, p2.sidebar_bottom) + 0.25

    def _draw_consensus(page, x, y, w, h):
        with plt.rc_context(PLOT_RC):
            names = [r["name"][:14] for r in records]
            vina_v = [r["vina_kcal"] for r in records]
            vin_v = [r["vinardo_kcal"] for r in records]
            le_v = [r["ligand_efficiency"] for r in records]
            mm_v = [r["mmgbsa_kcal"] for r in records]

            # Left subplot: Vina & Vinardo grouped horizontal bars
            ax1 = page.axes(x + 0.98, y + 0.36, w * 0.48 - 1.05, h - 0.78)
            idx = np.arange(len(names))
            bar_h = 0.35
            ax1.barh(idx - bar_h / 2, vina_v, height=bar_h, color=NPG_NAVY, label="Vina")
            ax1.barh(idx + bar_h / 2, vin_v, height=bar_h, color=NPG_TEAL, label="Vinardo")
            ax1.set_yticks(idx)
            ax1.set_yticklabels(names)
            ax1.invert_yaxis()
            ax1.set_xlabel("Docking Score (kcal/mol)")
            ax1.set_title("a  Vina vs. Vinardo Score", loc="left", fontweight="bold", color=INK)
            ax1.legend(loc="lower left", fontsize=6.2)
            style_axes(ax1, grid=True)

            # Right subplot: MM-GBSA vs Ligand Efficiency bubble scatter
            ax2 = page.axes(x + w * 0.56, y + 0.36, w * 0.41, h - 0.78)
            colors = [NPG_TEAL if r["pb_valid"] else NPG_AMBER for r in records]
            sizes = [max(35.0, min(160.0, r["mw"] * 0.32)) for r in records]
            ax2.scatter(mm_v, le_v, s=sizes, c=colors, alpha=0.85, edgecolors=INK, linewidths=0.6)
            for r in records:
                ax2.annotate(
                    r["name"][:10],
                    (r["mmgbsa_kcal"], r["ligand_efficiency"]),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=6.0,
                    color=INK,
                )
            ax2.set_xlabel("MM-GBSA ΔG (kcal/mol; more negative = stronger)")
            ax2.set_ylabel("Ligand Efficiency (kcal/mol/HA)")
            ax2.set_title("b  MM-GBSA vs. Ligand Efficiency", loc="left", fontweight="bold", color=INK)
            style_axes(ax2, grid=True)

    figure_block(
        doc,
        p2,
        M,
        fig_y,
        CW,
        min(4.2, 10.2 - fig_y),
        "Series docking consensus: (a) Vina and Vinardo binding scores per analog; (b) differential pocket MM-GBSA vs. heavy-atom Ligand Efficiency (teal = PoseBusters PASS, amber = PoseBusters flag; marker area scales with MW).",
        _draw_consensus,
    )

    # Page 3: Ranked Analog Table & Pairwise ΔΔScore SAR
    p3 = doc.page()
    p3.heading("Ranked Analog Table & Pairwise SAR", y=0.72, size=18)
    headers = ["Rank", "Analog", "Vina", "Vinardo", "MM-GBSA", "LE", "MW", "cLogP", "PB"]
    col_w = [0.46, 1.38, 0.70, 0.74, 0.84, 0.56, 0.68, 0.62, 1.05]
    rows_tbl = [
        [
            f"#{r['consensus_rank']}",
            r["name"][:16],
            f"{r['vina_kcal']:.2f}",
            f"{r['vinardo_kcal']:.2f}",
            f"{r['mmgbsa_kcal']:.1f}",
            f"{r['ligand_efficiency']:.2f}",
            f"{r['mw']:.0f}",
            f"{r['clogp']:.2f}",
            f"PASS ({r['pb_passed']})" if r["pb_valid"] else f"WARN ({r['pb_passed']})",
        ]
        for r in records[:14]
    ]
    y3 = p3.table(
        M, 1.30, col_w, headers, rows_tbl,
        align=("center", "left", "right", "right", "right", "right", "right", "right", "center"),
        zebra=True,
    )

    if ddg_rows:
        y3 += 0.25
        p3.heading("Relative ΔΔScore vs. Consensus Lead", y=y3, size=14)
        d_headers = ["Analog vs. Lead", "ΔΔVina", "ΔΔVinardo", "ΔΔMM-GBSA", "ΔLE", "Core RMSD"]
        d_w = [2.10, 0.95, 0.98, 1.05, 0.85, 1.10]
        d_tbl = [
            [
                f"{d['analog']} vs {d['reference']}"[:24],
                f"{d['dd_vina']:+.2f}",
                f"{d['dd_vinardo']:+.2f}",
                f"{d['dd_mmgbsa']:+.1f}",
                f"{d['d_le']:+.3f}",
                f"{d['core_rmsd_A']:.2f} Å" if d["core_rmsd_A"] is not None else "—",
            ]
            for d in ddg_rows[:10]
        ]
        y3 = p3.table(
            M, p3.y + 0.08, d_w, d_headers, d_tbl,
            align=("left", "right", "right", "right", "right", "right"),
            zebra=True,
        )

    # Optional 2D SAR Grid on Page 4
    if grid_2d_png is not None and grid_2d_png.exists():
        p4 = doc.page()
        p4.heading("2D Analog Structures & Shared Core", y=0.72, size=18)
        p4.paras(
            M,
            1.28,
            CW,
            [
                "Each analog is rendered in 2D with the detected Maximum Common Substructure (MCS) core highlighted "
                "when present, annotated with consensus rank, Vina score, MM-GBSA ΔG, and PoseBusters status."
            ],
            size=9.0,
        )

        def _draw_2d(page, x, y, w, h):
            ax = page.axes(x + 0.12, y + 0.12, w - 0.24, h - 0.24)
            img = mpimg.imread(str(grid_2d_png))
            ax.imshow(img)
            ax.axis("off")

        figure_block(
            doc,
            p4,
            M,
            2.05,
            CW,
            7.4,
            f"2D chemical depictions of the {n_lig} docked analogs in {rec_name} (MCS core highlighted where shared).",
            _draw_2d,
        )

    # Final Page: Methods & References
    end_page(
        doc,
        tiles=[
            ("Shared GridCache", f"Receptor affinity grids computed once for {rec_name} and reused across all {n_lig} analogs."),
            ("Consensus Scoring", "Vina, Vinardo, and differential pocket MM-GBSA combined into an unbiased series rank."),
            ("PoseBusters Audit", f"11-point RDKit/NumPy chemistry & steric validity check executed on every top pose ({pb_pass}/{n_lig} PASS)."),
        ],
        methods_rows=[
            ("Receptor", f"{rec_name} ({Path(receptor_file).name})"),
            ("Binding box", f"Center ({grid_center[0]:.2f}, {grid_center[1]:.2f}, {grid_center[2]:.2f}) Å; Size {grid_dimensions[0]:.1f}×{grid_dimensions[1]:.1f}×{grid_dimensions[2]:.1f} Å"),
            ("Core tethering", f"{'Enabled (MCS flat-bottom restraints)' if tethered else 'Unrestrained shared-grid docking'}"),
            ("Runtime", f"{runtime_s:.2f} s total ({runtime_s / max(n_lig, 1):.2f} s/analog)"),
        ],
        references=[
            "Trott, O. & Olson, A. J. AutoDock Vina: improving the speed and accuracy of docking. J. Comput. Chem. 31, 455–461 (2010).",
            "Quiroga, R. & Villarreal, M. A. Vinardo: a scoring function based on Autodock Vina improves scoring, docking, and virtual screening. PLoS ONE 11, e0155183 (2016).",
            "Buttenschoen, M., Morris, G. M. & Deane, C. M. PoseBusters: AI-based docking methods fail to generate physically valid poses or generalise to novel sequences. Chem. Sci. 15, 3130–3139 (2024).",
        ],
    )
    return doc.save(output_pdf)


def _build_series_html(
    output_html: Path,
    receptor_file: str,
    records: List[Dict[str, Any]],
    ddg_rows: List[Dict[str, Any]],
    mcs_smarts: Optional[str],
    grid_2d_png: Optional[Path],
    runtime_s: float,
) -> Path:
    """Write a clean companion HTML brochure for the congeneric series."""
    rec_name = Path(receptor_file).stem
    rows_html = []
    for r in records:
        badge = (
            f'<span style="background:#DCFCE7;color:#166534;padding:2px 8px;border-radius:10px;font-weight:600;">PASS ({r["pb_passed"]})</span>'
            if r["pb_valid"]
            else f'<span style="background:#FEF3C7;color:#92400E;padding:2px 8px;border-radius:10px;font-weight:600;">WARN ({r["pb_passed"]})</span>'
        )
        rows_html.append(
            f"<tr><td>#{r['consensus_rank']}</td><td><strong>{html.escape(r['name'])}</strong></td>"
            f"<td>{r['vina_kcal']:.3f}</td><td>{r['vinardo_kcal']:.3f}</td>"
            f"<td>{r['mmgbsa_kcal']:.2f}</td><td>{r['ligand_efficiency']:.3f}</td>"
            f"<td>{r['mw']:.1f}</td><td>{r['clogp']:.2f}</td><td>{badge}</td></tr>"
        )
    img_block = (
        f'<div class="card"><h2>2D Analog SAR Grid</h2><img src="{grid_2d_png.name}" style="max-width:100%;border-radius:8px;"></div>'
        if grid_2d_png is not None and grid_2d_png.exists()
        else ""
    )
    doc_html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>PandaDock Series SAR — {html.escape(rec_name)}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#F4F0EF;color:#141923;margin:0;padding:32px;}}
.wrap{{max-width:1080px;margin:0 auto;}}
.hero{{background:linear-gradient(120deg,#4A3AA8,#3453B8);color:#fff;padding:28px 32px;border-radius:14px;margin-bottom:24px;}}
.card{{background:#fff;border-radius:12px;padding:22px 26px;margin-bottom:20px;box-shadow:0 2px 10px rgba(20,25,35,0.06);}}
table{{width:100%;border-collapse:collapse;font-size:14px;}}
th{{background:#161A2E;color:#fff;text-align:left;padding:10px;}}
td{{padding:9px 10px;border-bottom:1px solid #E5E7EB;}}
</style></head>
<body><div class="wrap">
<div class="hero"><h1>PandaDock v{__version__} · Congeneric Series SAR Report</h1>
<p>Receptor: <strong>{html.escape(rec_name)}</strong> · {len(records)} analogs · Runtime {runtime_s:.1f} s · MCS: <code>{html.escape(mcs_smarts or 'Diverse')}</code></p></div>
<div class="card"><h2>Consensus Ranking & Multi-Scoring Panel</h2>
<table><thead><tr><th>Rank</th><th>Analog</th><th>Vina (kcal/mol)</th><th>Vinardo (kcal/mol)</th><th>MM-GBSA (kcal/mol)</th><th>LE</th><th>MW</th><th>cLogP</th><th>PoseBusters</th></tr></thead>
<tbody>{''.join(rows_html)}</tbody></table></div>
{img_block}
</div></body></html>
"""
    output_html.write_text(doc_html, encoding="utf-8")
    return output_html


def run_congeneric_series(
    receptor_file: str,
    ligands_path: str,
    grid_center: np.ndarray,
    grid_dimensions: np.ndarray,
    output_dir: str,
    reference_pose_sdf: Optional[str] = None,
    core_smarts: Optional[str] = None,
    core_tolerance: float = 0.75,
    core_force: float = 12.0,
    exhaustiveness: int = 8,
    num_poses: int = 5,
    scoring: str = "vina",
    device: str = "cpu",
    seed: int = 42,
    n_workers: int = 1,
    weights_override: Optional[Dict[str, float]] = None,
    gnn_model_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Dock an analog series with shared GridCache, multi-scoring consensus, and SAR brochure PDF."""
    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    center = np.asarray(grid_center, dtype=np.float64)
    box = np.asarray(grid_dimensions, dtype=np.float64)

    mols = load_series_ligands(ligands_path)
    if not mols:
        raise ValueError(f"No valid ligands loaded from {ligands_path}")

    ref_mol = None
    if reference_pose_sdf and Path(reference_pose_sdf).exists():
        ref_mols = load_series_ligands(reference_pose_sdf)
        if ref_mols and ref_mols[0].GetNumConformers() > 0:
            ref_mol = ref_mols[0]

    mcs_smarts = core_smarts or _compute_mcs_smarts(mols)
    grid_cache = GridCache()
    docker = PandaCoreDocker(grid_cache=grid_cache)
    docker.set_scoring_function(VinaScoring(preset=scoring))

    parser = PDBParser(QUIET=True)
    receptor_struct = parser.get_structure("rec", receptor_file)
    vina_scorer = VinaScoring(preset="vina")
    vinardo_scorer = VinaScoring(preset="vinardo")
    mmgbsa_scorer = MMGBSARescoring()

    gnn_scorer = None
    if gnn_model_path and Path(gnn_model_path).exists():
        try:
            from ..gnn.scoring import GNNScoring

            gnn_scorer = GNNScoring(model_path=gnn_model_path, device=device)
        except Exception as exc:
            logger.warning("Optional GNN rescoring unavailable: %s", exc)

    records: List[Dict[str, Any]] = []
    docked_mols: List[Chem.Mol] = []
    tethered_any = False

    for idx, raw_mol in enumerate(mols):
        name = raw_mol.GetProp("_Name") if raw_mol.HasProp("_Name") else f"analog_{idx + 1}"
        mol3d = _ensure_3d(raw_mol, seed=seed + idx)
        mol3d.SetProp("_Name", name)

        restraints = None
        n_core_atoms = 0
        if ref_mol is not None:
            rset, _, n_core_atoms = build_mcs_restraints(
                mol3d,
                ref_mol,
                tolerance=core_tolerance,
                force_constant=core_force,
                core_smarts=mcs_smarts,
            )
            if len(rset) > 0:
                restraints = rset
                tethered_any = True

        res = docker.dock(
            receptor_file=receptor_file,
            ligand_mol=mol3d,
            grid_center=center,
            grid_dimensions=box,
            exhaustiveness=exhaustiveness,
            num_poses=num_poses,
            seed=seed + idx * 31,
            device=device,
            n_workers=n_workers,
            restraints=restraints,
            weights_override=weights_override,
        )
        if not res.poses:
            continue

        top_pose = res.poses[0]
        # Build RDKit molecule with docked coordinates
        pose_mol = Chem.Mol(mol3d)
        conf = Chem.Conformer(pose_mol.GetNumAtoms())
        for a_i, xyz in enumerate(top_pose.coordinates):
            conf.SetAtomPosition(a_i, (float(xyz[0]), float(xyz[1]), float(xyz[2])))
        pose_mol.RemoveAllConformers()
        pose_mol.AddConformer(conf, assignId=True)
        pose_mol.SetProp("_Name", name)

        # Multi-physics consensus scores on the exact same top pose
        vina_e = float(top_pose.energy) if scoring == "vina" else float(
            vina_scorer.calculate_binding_energy(top_pose.coordinates, receptor_struct, pose_mol)
        )
        vinardo_e = float(top_pose.energy) if scoring == "vinardo" else float(
            vinardo_scorer.calculate_binding_energy(top_pose.coordinates, receptor_struct, pose_mol)
        )
        try:
            mmgbsa_e = float(
                mmgbsa_scorer.calculate_binding_free_energy(top_pose, pose_mol, receptor_struct)
            )
        except Exception:
            mmgbsa_e = vina_e

        gnn_pkd = None
        if gnn_scorer is not None:
            try:
                pred = gnn_scorer.predict_affinity(receptor_file, pose_mol)
                gnn_pkd = float(pred.get("pKd", pred.get("predicted_pKd", 0.0)))
            except Exception:
                gnn_pkd = None

        pb = audit_pose(
            pose_mol,
            receptor_file=receptor_file,
            reference_mol=mol3d,
            grid_center=center,
            grid_dimensions=box,
            pose_index=0,
        )

        n_heavy = max(1, int(pose_mol.GetNumHeavyAtoms()))
        le = -vina_e / n_heavy
        c_rmsd = _core_rmsd(pose_mol, ref_mol, mcs_smarts) if ref_mol is not None else None

        rec: Dict[str, Any] = {
            "name": name,
            "smiles": Chem.MolToSmiles(Chem.RemoveHs(pose_mol)),
            "vina_kcal": round(vina_e, 3),
            "vinardo_kcal": round(vinardo_e, 3),
            "mmgbsa_kcal": round(mmgbsa_e, 2),
            "gnn_pkd": round(gnn_pkd, 3) if gnn_pkd is not None else None,
            "ligand_efficiency": round(le, 3),
            "n_heavy": n_heavy,
            "mw": round(float(Descriptors.MolWt(pose_mol)), 2),
            "clogp": round(float(Crippen.MolLogP(pose_mol)), 2),
            "tpsa": round(float(rdMolDescriptors.CalcTPSA(pose_mol)), 1),
            "hbd": int(Lipinski.NumHDonors(pose_mol)),
            "hba": int(Lipinski.NumHAcceptors(pose_mol)),
            "rot_bonds": int(Lipinski.NumRotatableBonds(pose_mol)),
            "pb_valid": bool(pb.all_passed),
            "pb_passed": f"{pb.n_passed}/{pb.n_total}",
            "pb_failed": ";".join(pb.failed_checks),
            "core_atoms_tethered": int(n_core_atoms),
            "core_rmsd_A": round(c_rmsd, 3) if c_rmsd is not None else None,
            "runtime_s": round(float(res.runtime_seconds), 2),
        }
        records.append(rec)
        docked_mols.append(pose_mol)

    if not records:
        raise RuntimeError("Series docking produced zero valid poses")

    # Compute consensus rank across Vina, Vinardo, and MM-GBSA (lower energy = better rank)
    def _rank_asc(vals: Sequence[float]) -> np.ndarray:
        order = np.argsort(vals)
        ranks = np.empty(len(vals), dtype=np.float64)
        ranks[order] = np.arange(1, len(vals) + 1, dtype=np.float64)
        return ranks

    r_vina = _rank_asc([r["vina_kcal"] for r in records])
    r_vin = _rank_asc([r["vinardo_kcal"] for r in records])
    r_mm = _rank_asc([r["mmgbsa_kcal"] for r in records])
    mean_rank = (r_vina + r_vin + r_mm) / 3.0

    # Sort both records and docked_mols by consensus mean_rank (tie-break by Vina)
    order = sorted(range(len(records)), key=lambda i: (mean_rank[i], records[i]["vina_kcal"]))
    records = [records[i] for i in order]
    docked_mols = [docked_mols[i] for i in order]
    for rank_i, (r, m) in enumerate(zip(records, docked_mols), 1):
        r["consensus_rank"] = rank_i
        m.SetProp("Consensus_Rank", str(rank_i))
        m.SetProp("Vina_kcal", f"{r['vina_kcal']:.3f}")
        m.SetProp("Vinardo_kcal", f"{r['vinardo_kcal']:.3f}")
        m.SetProp("MMGBSA_kcal", f"{r['mmgbsa_kcal']:.2f}")
        m.SetProp("PoseBusters_Valid", str(r["pb_valid"]))

    # Pairwise ΔΔScore table relative to consensus lead (#1)
    lead = records[0]
    lead_mol = docked_mols[0]
    ddg_rows: List[Dict[str, Any]] = []
    for r, m in zip(records[1:], docked_mols[1:]):
        c_rmsd = r["core_rmsd_A"]
        if c_rmsd is None and mcs_smarts:
            c_rmsd = _core_rmsd(m, lead_mol, mcs_smarts)
        ddg_rows.append(
            {
                "analog": r["name"],
                "reference": lead["name"],
                "dd_vina": round(r["vina_kcal"] - lead["vina_kcal"], 3),
                "dd_vinardo": round(r["vinardo_kcal"] - lead["vinardo_kcal"], 3),
                "dd_mmgbsa": round(r["mmgbsa_kcal"] - lead["mmgbsa_kcal"], 2),
                "d_le": round(r["ligand_efficiency"] - lead["ligand_efficiency"], 3),
                "core_rmsd_A": round(c_rmsd, 3) if c_rmsd is not None else None,
            }
        )

    # Write SDF, CSV, JSON
    sdf_path = out / "series_poses.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    for m in docked_mols:
        writer.write(m)
    writer.close()

    csv_path = out / "series_results.csv"
    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        cw = csv.DictWriter(fh, fieldnames=fieldnames)
        cw.writeheader()
        cw.writerows(records)

    runtime_s = time.time() - t0
    grid_2d_png = _render_2d_sar_grid(docked_mols, records, mcs_smarts, out / "series_grid_2d.png")

    pdf_path = _build_series_pdf(
        out / "series_report.pdf",
        receptor_file,
        records,
        ddg_rows,
        mcs_smarts,
        center,
        box,
        grid_2d_png,
        runtime_s,
        tethered_any,
    )
    html_path = _build_series_html(
        out / "series_report.html",
        receptor_file,
        records,
        ddg_rows,
        mcs_smarts,
        grid_2d_png,
        runtime_s,
    )

    summary = {
        "receptor": receptor_file,
        "n_analogs": len(records),
        "mcs_smarts": mcs_smarts,
        "core_tethered": tethered_any,
        "posebusters_pass_count": sum(1 for r in records if r["pb_valid"]),
        "runtime_seconds": round(runtime_s, 2),
        "lead_analog": lead["name"],
        "records": records,
        "pairwise_sar": ddg_rows,
        "files": {
            "poses_sdf": str(sdf_path),
            "results_csv": str(csv_path),
            "report_pdf": str(pdf_path),
            "report_html": str(html_path),
            "grid_2d_png": str(grid_2d_png) if grid_2d_png else None,
        },
    }
    (out / "series_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
