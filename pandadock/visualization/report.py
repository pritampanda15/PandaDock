"""
Docking run report: publication-grade plots, editorial HTML summary, and A4 vector PDF.

Produces the scientific diagnostics that inform decisions about a docking run:
- Cover summary with 3-D receptor-ligand silhouette, KPI banner, and automated quality checks
- Score by rank, Boltzmann population weights, and score-vs-RMSD binding funnel
- Energy terms with an explicit favourable/unfavourable split
- Pose diversity: pairwise heavy-atom RMSD matrix and 2-D MDS conformational landscape
- Protein-ligand binding mode: persistent contacts table, stacked occupancy, and contact map
- Ligand pose geometry: per-heavy-atom RMSF and rotatable-bond torsion angle histograms
- Empirical pocket residue hotspots and optional 2-D PandaMap interaction diagram

Deliberately omitted: plots of Kd or IC50 against the docking score. Those are a
fixed algebraic transform of one another (Kd = exp(dG/RT)), so converting an empirical
docking score to a potency prediction overstates what the number supports.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .. import __version__
from .report_pdf import REFERENCES, build_pdf_report
from .report_plots import (
    FAVOURABLE_TERMS,
    UNFAVOURABLE_TERMS,
    compute_report_analytics,
    plot_energy_components,
    plot_interaction_fingerprint,
    plot_ligand_dynamics,
    plot_pose_diversity,
    plot_pose_scores,
    plot_protein_silhouette,
    plot_residue_hotspots,
    pose_rmsd_matrix,
)

logger = logging.getLogger("pandadock.visualization.report")

__all__ = [
    "FAVOURABLE_TERMS",
    "UNFAVOURABLE_TERMS",
    "pose_rmsd_matrix",
    "plot_pose_scores",
    "plot_energy_components",
    "plot_pose_diversity",
    "plot_interaction_fingerprint",
    "plot_ligand_dynamics",
    "plot_residue_hotspots",
    "plot_protein_silhouette",
    "compute_report_analytics",
    "build_html_report",
    "build_pdf_report",
    "generate_report",
    "load_result_from_dir",
    "load_interaction_analyses",
]


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PandaDock Report — {receptor_name} · {ligand_name}</title>
<style>
  :root {{
    --ink: #12142B;
    --night: #161A2E;
    --indigo: #4A3AA8;
    --cobalt: #3453B8;
    --accent: #2F4FB5;
    --beige: #F4F0EF;
    --text: #2C2E3E;
    --soft: #6B6F83;
    --rule: #D9DCE8;
    --teal: #00A087;
    --amber: #D98A32;
    --spark: #7DD3FC;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: "Avenir Next", "Source Sans 3", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0; color: var(--text); background: #FFFFFF; line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }}
  .page {{ max-width: 980px; margin: 0 auto; padding: 0 28px 48px; }}
  .hero {{
    background: linear-gradient(180deg, #F6F5F5 0%, #ECEBEC 100%);
    padding: 40px 0 32px; border-bottom: 1px solid #EAE6E4;
  }}
  .hero-inner {{
    max-width: 980px; margin: 0 auto; padding: 0 28px;
    display: grid; grid-template-columns: 1.15fr 0.85fr; gap: 28px; align-items: center;
  }}
  .kicker {{ font-size: 11px; font-weight: 700; letter-spacing: 0.12em; color: var(--ink); text-transform: uppercase; }}
  h1 {{ font-size: 40px; line-height: 1.06; margin: 10px 0 18px; color: var(--ink); letter-spacing: -0.02em; }}
  .subtitle {{ border-left: 3px solid var(--indigo); padding-left: 14px; font-size: 14.5px; color: var(--text); margin: 0; max-width: 440px; }}
  .silhouette {{ width: 100%; border-radius: 12px; display: block; }}
  .kpi-banner {{
    max-width: 924px; margin: -22px auto 32px; padding: 22px 24px;
    border-radius: 14px;
    background: linear-gradient(115deg, var(--indigo) 0%, var(--cobalt) 100%);
    color: #EEF0FB;
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
    box-shadow: 0 10px 26px rgba(22,26,46,0.12);
  }}
  .kpi {{ text-align: center; padding: 4px 12px; border-right: 1px solid rgba(255,255,255,0.22); }}
  .kpi:last-child {{ border-right: none; }}
  .kpi-val {{ font-size: 19px; font-weight: 700; color: #FFFFFF; margin-bottom: 4px; }}
  .kpi-lbl {{ font-size: 12.5px; color: #E2E6FA; }}
  .cover-grid {{ display: grid; grid-template-columns: 1fr 1.05fr; gap: 32px; margin: 28px 0 38px; align-items: start; }}
  .quote-mark {{ font-family: Georgia, serif; font-size: 64px; line-height: 0.7; color: var(--accent); font-weight: 700; margin-bottom: 10px; }}
  .quote-text {{ font-size: 19px; font-weight: 700; line-height: 1.32; color: var(--ink); margin: 0 0 12px; }}
  .quote-by {{ font-size: 12px; font-weight: 700; color: var(--ink); text-align: right; }}
  .quote-sub {{ font-size: 11.5px; color: var(--soft); margin-top: 4px; }}
  .fact-card {{ background: var(--beige); border-radius: 16px; padding: 16px; }}
  .fact-row {{ display: grid; grid-template-columns: 115px 1fr; margin-bottom: 3px; border-radius: 4px; overflow: hidden; }}
  .fact-k {{
    background: linear-gradient(135deg, var(--indigo), var(--cobalt));
    color: #FFFFFF; font-weight: 700; font-size: 12.5px; padding: 11px 12px;
    display: flex; align-items: center;
  }}
  .fact-v {{ background: #FFFFFF; font-size: 12.5px; padding: 11px 14px; color: var(--text); }}
  h2 {{ font-size: 24px; color: var(--ink); margin: 38px 0 14px; letter-spacing: -0.01em; }}
  h3 {{ font-size: 15.5px; color: var(--ink); margin: 22px 0 10px; }}
  .setup-grid {{ display: grid; grid-template-columns: 0.92fr 1.08fr; gap: 28px; margin: 24px 0 30px; }}
  .setup-card {{
    background: linear-gradient(145deg, var(--indigo), var(--cobalt));
    color: #F0F0FA; border-radius: 16px; padding: 24px; font-size: 13.5px;
  }}
  .setup-card h3 {{ color: #FFFFFF; text-transform: uppercase; font-size: 12px; letter-spacing: 0.1em; margin-top: 0; }}
  .setup-card p {{ margin: 0 0 10px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0 20px; font-size: 13px; font-variant-numeric: tabular-nums; }}
  thead th {{ background: var(--night); color: #FFFFFF; text-align: left; padding: 9px 12px; font-weight: 600; }}
  tbody td {{ padding: 8px 12px; border-bottom: 1px solid var(--rule); }}
  tbody tr:nth-child(even) {{ background: #F7F8FC; }}
  tbody td:first-child {{ color: var(--accent); font-weight: 600; }}
  .checks {{ list-style: none; padding: 0; margin: 10px 0 22px; }}
  .checks li {{ display: flex; gap: 10px; align-items: flex-start; margin-bottom: 8px; font-size: 13.5px; }}
  .badge {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 20px; height: 20px; border-radius: 50%; flex: 0 0 20px; font-size: 12px; font-weight: 700; margin-top: 2px;
  }}
  .badge-ok {{ background: rgba(0,160,135,0.14); color: var(--teal); border: 1.5px solid var(--teal); }}
  .badge-warn {{ background: rgba(217,138,50,0.14); color: var(--amber); border: 1.5px solid var(--amber); }}
  .fig-card {{ background: var(--beige); border-radius: 16px; padding: 18px 18px 14px; margin: 18px 0 24px; }}
  .fig-card img {{ width: 100%; height: auto; display: block; border-radius: 8px; }}
  .fig-cap {{ font-size: 12.5px; color: var(--text); margin: 10px 4px 0; }}
  .fig-cap strong {{ color: var(--ink); }}
  .tiles {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 16px 0 26px; }}
  .tile {{
    background: linear-gradient(155deg, var(--indigo), var(--cobalt));
    color: #F0F0FA; border-radius: 12px; padding: 16px;
  }}
  .tile h4 {{ margin: 0 0 8px; color: #FFFFFF; font-size: 15px; }}
  .tile p {{ margin: 0; font-size: 12px; line-height: 1.45; }}
  .note {{
    background: #FFF9EB; border-left: 4px solid var(--amber);
    padding: 12px 16px; border-radius: 0 8px 8px 0; font-size: 13px; margin: 16px 0;
  }}
  .refs {{ background: var(--beige); border-radius: 16px; padding: 22px 26px; margin-top: 28px; }}
  .refs ol {{ margin: 10px 0 0; padding-left: 20px; font-size: 12.5px; }}
  .refs li {{ margin-bottom: 6px; }}
  .footer {{
    background: var(--night); color: #FFFFFF; margin-top: 40px; padding: 28px 0;
  }}
  .footer-inner {{
    max-width: 980px; margin: 0 auto; padding: 0 28px;
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;
  }}
  .brand {{ display: inline-flex; align-items: center; gap: 10px; font-size: 20px; font-weight: 800; letter-spacing: -0.01em; }}
  .brand img {{ width: 34px; height: 34px; border-radius: 8px; background: #FFFFFF; padding: 1px; }}
  .brand span {{ color: #48B9D0; }}
  .footer-meta {{ font-size: 12px; color: #A6AEC4; }}
  code {{ background: #ECEEF5; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
  @media (max-width: 760px) {{
    .hero-inner, .cover-grid, .setup-grid, .kpi-banner, .tiles {{ grid-template-columns: 1fr; }}
  }}
  @media print {{
    .fig-card, .setup-grid, .cover-grid {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>

<div class="hero">
  <div class="hero-inner">
    <div>
      <div class="kicker">Molecular Docking Report</div>
      <h1>{receptor_name} &middot; {ligand_name}</h1>
      <p class="subtitle">Automated analysis of binding energetics, conformational diversity, protein&ndash;ligand contacts and ligand pose geometry from molecular docking.</p>
    </div>
    <div>{silhouette_html}</div>
  </div>
</div>

<div class="kpi-banner">
  <div class="kpi">
    <div class="kpi-val">{best:.2f} kcal/mol</div>
    <div class="kpi-lbl">Top score (&Delta;E<sub>1&minus;2</sub> = {gap_1_2:.2f} kcal/mol)</div>
  </div>
  <div class="kpi">
    <div class="kpi-val">{kpi_contacts_val}</div>
    <div class="kpi-lbl">{kpi_contacts_lbl}</div>
  </div>
  <div class="kpi">
    <div class="kpi-val">{n_clusters} mode{cluster_s}</div>
    <div class="kpi-lbl">Mean pairwise RMSD {mean_rmsd:.2f} &Aring;</div>
  </div>
</div>

<div class="page">
  <div class="cover-grid">
    <div>
      <div class="quote-mark">&ldquo;</div>
      <p class="quote-text">{quote}</p>
      <div class="quote-by">&mdash; PandaDock summary</div>
      <div class="quote-sub">computed from the docked pose ensemble{pdf_badge}</div>
    </div>
    <div class="fact-card">
      {fact_rows_html}
    </div>
  </div>

  <div class="setup-grid">
    <div class="setup-card">
      <h3>Docking Setup</h3>
      {setup_html}
    </div>
    <div>
      <h2 style="margin-top:0;">Binding energetics and pose convergence</h2>
      <p>Across <strong>{n_poses}</strong> returned binding modes, the top-ranked pose scores <strong>{best:.2f} kcal/mol</strong> against a Boltzmann ensemble average of <strong>{ensemble:.2f} kcal/mol</strong> (mean {mean_score:.2f} &plusmn; {sd_score:.2f} kcal/mol). The margin to rank 2 is <strong>{gap_1_2:.2f} kcal/mol</strong>, placing <strong>{top1_boltz_pct:.1f}%</strong> Boltzmann weight at 298 K on the leading pose.</p>
      <p>Pairwise heavy-atom RMSD between modes averages <strong>{mean_rmsd:.2f} &Aring;</strong> (max {max_rmsd:.2f} &Aring;), forming <strong>{n_clusters}</strong> distinct conformational cluster(s) at a 2.0 &Aring; cutoff.</p>
      <div class="note">
        <strong>Scientific caveat:</strong> Docking scores are empirical ranking estimates in kcal/mol. They rank poses and congeneric compounds; they are not measured binding free energies and should not be converted to K<sub>d</sub> or IC<sub>50</sub> potency predictions.
      </div>
    </div>
  </div>

  <h3>Quality checks</h3>
  <ul class="checks">{checks_html}</ul>

  <h2>Top docked binding modes</h2>
  <table>
    <thead><tr><th>Rank</th><th>Score (kcal/mol)</th><th>&Delta;E</th><th>Boltzmann</th><th>RMSD<sub>1</sub> (&Aring;)</th><th>Mode</th><th>Strain</th><th>Contacts</th></tr></thead>
    <tbody>{pose_rows_html}</tbody>
  </table>

  {sections_html}

  {contacts_table_html}
  {hotspots_table_html}

  <h2>Analysis modules in this report</h2>
  <div class="tiles">
    <div class="tile"><h4>Search</h4><p>Flexible-ligand Monte Carlo with local L-BFGS optimisation over translation, SO(3) rotation and torsions.</p></div>
    <div class="tile"><h4>Scoring</h4><p>{scoring} empirical affinity grids with explicit favourable versus penalty energy decomposition.</p></div>
    <div class="tile"><h4>Interactions</h4><p>Geometric H-bond, hydrophobic, &pi;-stacking, ionic and metal-coordination occupancy profiling.</p></div>
    <div class="tile"><h4>Geometry</h4><p>Pairwise heavy-atom RMSD clustering, 2-D MDS landscape, atom RMSF and torsion distributions.</p></div>
  </div>

  <h2>Output files</h2>
  <ul>
    <li><code>report.pdf</code> &mdash; multi-page A4 vector publication report</li>
    <li><code>poses.sdf</code> &mdash; all docked poses with bond orders, scores and ranks</li>
    <li><code>pose{{N}}.pdb</code> / <code>complex{{N}}.pdb</code> &mdash; individual ligand poses and docked complexes</li>
    <li><code>interaction_analysis.json</code> &mdash; per-pose detected protein&ndash;ligand contacts</li>
  </ul>

  <div class="refs">
    <h3 style="margin-top:0;">References</h3>
    <ol>{refs_html}</ol>
  </div>
</div>

<div class="footer">
  <div class="footer-inner">
    <div class="brand">{logo_img_html}Panda<span>Dock</span></div>
    <div class="footer-meta">Molecular Docking Report &middot; v{version} &middot; generated {timestamp}</div>
  </div>
</div>

</body>
</html>
"""


def _load_ligand_mol_from_dir(output_dir: Path):
    """Best-effort load of top ligand RDKit Mol from poses.sdf for property/torsion plots."""
    poses_sdf = output_dir / "poses.sdf"
    if not poses_sdf.exists():
        return None
    try:
        from rdkit import Chem

        for mol in Chem.SDMolSupplier(str(poses_sdf), removeHs=False):
            if mol is not None:
                return mol
    except Exception:
        pass
    return None


def build_html_report(
    output_dir: Path,
    result,
    images: Dict[str, Path],
    ligand_name: str = "ligand",
    analytics: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Assemble the generated figures and analytics into an editorial HTML report."""
    try:
        data = analytics or compute_report_analytics(result)
        params = data["parameters"] or {}
        rec_info = data["receptor_info"]
        lig_info = data["ligand_info"]

        g_c = np.asarray(data["grid_center"], dtype=float)
        g_d = np.asarray(data["grid_dimensions"], dtype=float)

        rec_str = (
            f"{rec_info['n_residues']} residues, {rec_info['n_chains']} chain(s); {rec_info['n_atoms']:,} atoms"
            if rec_info["n_residues"] > 0
            else escape(Path(data["receptor_file"]).name if data["receptor_file"] else "Receptor PDB")
        )
        if rec_info["metals"]:
            rec_str += f" &middot; Metals: {escape(', '.join(rec_info['metals'][:3]))}"

        lig_str = f"{escape(ligand_name)}, {lig_info['n_atoms']} atoms ({lig_info['n_heavy']} heavy)"
        if lig_info["mw"] is not None:
            lig_str += f"; MW {lig_info['mw']:.1f} Da"

        fact_rows = [
            ("Receptor", rec_str),
            ("Ligand", lig_str),
            ("Engine", f"{escape(str(data['algorithm']))} &middot; {escape(str(data['scoring']))}"),
            ("Grid Box", f"({g_c[0]:.1f}, {g_c[1]:.1f}, {g_c[2]:.1f}) &Aring; | {g_d[0]:.1f} &times; {g_d[1]:.1f} &times; {g_d[2]:.1f} &Aring;"),
            ("Sampling", f"{data['n_poses']} poses in {data['runtime_seconds']:.1f} s (seed {escape(str(params.get('seed', 'random')))})"),
        ]
        fact_rows_html = "\n".join(
            f'<div class="fact-row"><div class="fact-k">{k}</div><div class="fact-v">{v}</div></div>'
            for k, v in fact_rows
        )

        setup_parts = [
            f"<p><strong>Receptor:</strong> {escape(data['receptor_name'])} ({rec_str})</p>",
            f"<p><strong>Ligand:</strong> {lig_str}"
            + (
                f", {lig_info['rot_bonds']} rotatable bond(s), cLogP {lig_info['logp']:.2f}, TPSA {lig_info['tpsa']:.1f} &Aring;&sup2;."
                if lig_info["mw"] is not None else "."
            )
            + "</p>",
            f"<p><strong>Box:</strong> centred at ({g_c[0]:.2f}, {g_c[1]:.2f}, {g_c[2]:.2f}) &Aring;, "
            f"dimensions {g_d[0]:.1f} &times; {g_d[1]:.1f} &times; {g_d[2]:.1f} &Aring;.</p>",
            f"<p><strong>Search:</strong> exhaustiveness {escape(str(params.get('exhaustiveness', 'default')))}, "
            f"{int(params.get('energy_evaluations', 0)):,} evaluations, runtime {data['runtime_seconds']:.1f} s.</p>",
        ]

        checks_html = "\n".join(
            f'<li><span class="badge {"badge-ok" if st == "ok" else "badge-warn"}">'
            f'{"&#10003;" if st == "ok" else "!"}</span><span>{escape(msg)}</span></li>'
            for st, msg in data["checks"]
        )

        pose_rows_html = "\n".join(
            f"<tr><td>#{r['rank']}</td><td>{r['score']:.3f}</td>"
            f"<td>{'+' if r['delta_e'] > 0 else ''}{r['delta_e']:.2f}</td>"
            f"<td>{r['boltz_pct']:.1f}%</td><td>{r['rmsd_top1']:.2f}</td>"
            f"<td>{escape(r['cluster'])}</td><td>{r['strain']:.2f}</td>"
            f"<td>HB:{r['hb']} HP:{r['hp']}"
            f"{(' &pi;:' + str(r['pi'])) if r['pi'] else ''}"
            f"{(' M:' + str(r['mt'])) if r['mt'] else ''}</td></tr>"
            for r in data["pose_rows"]
        )

        captions = [
            ("scores", "Binding energetics & scoring funnel",
             "Ranked docking scores (a), 298 K Boltzmann populations (b), and score-versus-RMSD binding funnel relative to rank 1 (c)."),
            ("components", "Energy term decomposition",
             "Per-pose energy contributions. Stabilising terms lie below zero; repulsive/strain/entropy penalties lie above zero."),
            ("diversity", "Pose diversity & conformational landscape",
             "Pairwise heavy-atom RMSD matrix (a) and 2-D classical MDS conformational embedding coloured by score (b)."),
            ("fingerprint", "Protein–ligand binding mode",
             "Stacked per-residue contact occupancy split by interaction type (a) and residue-by-pose contact heatmap (b)."),
            ("ligand_dynamics", "Ligand pose flexibility & torsion profiles",
             "Per-heavy-atom RMSF across the docked ensemble marking anchor vs mobile atoms (a) and rotatable-bond torsion histograms (b)."),
            ("hotspots", "Residue energy hotspots (empirical proxy)",
             "Top pocket residue contributions from distance-weighted contact proxy scores (qualitative ranking; not MM/PBSA or FEP)."),
            ("pandamap", "2D protein–ligand interaction map (PandaMap)",
             "Ligand-centric 2D interaction diagram for the top-ranked docked complex."),
        ]

        sections = []
        fig_num = 0
        for key, heading, caption in captions:
            path = images.get(key)
            if path is None or not Path(path).exists():
                continue
            fig_num += 1
            sections.append(
                f"<h2>{escape(heading)}</h2>\n"
                f'<div class="fig-card">'
                f'<img src="{escape(Path(path).name)}" alt="{escape(heading)}">'
                f'<p class="fig-cap"><strong>Figure {fig_num}:</strong> {escape(caption)}</p>'
                f"</div>"
            )

        contacts_table_html = ""
        if data["persistent_contacts"]:
            c_rows = []
            for c in data["persistent_contacts"][:10]:
                md_str = f"{c['mean_dist']:.2f}" if not math.isnan(c["mean_dist"]) else "&mdash;"
                c_rows.append(
                    f"<tr><td>{escape(c['residue'])}</td><td>{c['occupancy_pct']:.0f}%</td>"
                    f"<td>{escape(c['types'])}</td><td>{md_str}</td></tr>"
                )
            contacts_table_html = (
                "<h3>Persistent binding-site contacts</h3>"
                "<table><thead><tr><th>Residue</th><th>Occupancy</th><th>Interaction types</th><th>Mean dist. (&Aring;)</th></tr></thead>"
                f"<tbody>{''.join(c_rows)}</tbody></table>"
            )

        hotspots_table_html = ""
        if data["hotspots"]:
            h_rows = "\n".join(
                f"<tr><td>{escape(h['residue'])}</td><td>{h['proxy_score']:.2f} kcal/mol</td></tr>"
                for h in data["hotspots"][:8]
            )
            hotspots_table_html = (
                "<h3>Top residue hotspots (empirical proxy)</h3>"
                "<table style='max-width:460px;'><thead><tr><th>Residue</th><th>Proxy score</th></tr></thead>"
                f"<tbody>{h_rows}</tbody></table>"
            )

        sil_path = images.get("silhouette")
        silhouette_html = (
            f'<img class="silhouette" src="{escape(Path(sil_path).name)}" alt="Receptor and bound ligand silhouette">'
            if sil_path and Path(sil_path).exists()
            else ""
        )

        kpi_contacts_val = (
            f"{data['n_persistent_30']} residues"
            if data["persistent_contacts"]
            else f"{data['top1_boltz_pct']:.0f}% w₁"
        )
        kpi_contacts_lbl = (
            "Contact ligand in &ge;30% of poses"
            if data["persistent_contacts"]
            else f"Boltzmann weight across {data['n_poses']} modes"
        )

        pdf_badge = ' &middot; <a href="report.pdf" style="color:#2F4FB5;font-weight:600;">Download A4 PDF</a>' if (output_dir / "report.pdf").exists() else ""
        refs_html = "\n".join(f"<li>{ref.replace('**', '')}</li>" for ref in REFERENCES)

        logo_img_html = ""
        mark_path = Path(__file__).parent / "assets" / "pandadock_mark.png"
        if mark_path.exists():
            import base64
            b64 = base64.b64encode(mark_path.read_bytes()).decode("ascii")
            logo_img_html = f'<img src="data:image/png;base64,{b64}" alt="PandaDock Logo">'

        html = HTML_TEMPLATE.format(
            logo_img_html=logo_img_html,
            receptor_name=escape(data["receptor_name"]),
            ligand_name=escape(ligand_name),
            silhouette_html=silhouette_html,
            best=data["best_score"],
            gap_1_2=data["gap_1_2"],
            kpi_contacts_val=kpi_contacts_val,
            kpi_contacts_lbl=kpi_contacts_lbl,
            n_clusters=data["n_clusters"],
            cluster_s="" if data["n_clusters"] == 1 else "s",
            mean_rmsd=data["mean_rmsd"],
            max_rmsd=data["max_rmsd"],
            quote=escape(data["quote"]),
            pdf_badge=pdf_badge,
            fact_rows_html=fact_rows_html,
            setup_html="\n".join(setup_parts),
            n_poses=data["n_poses"],
            ensemble=data["ensemble_dg"],
            mean_score=data["mean_score"],
            sd_score=data["sd_score"],
            top1_boltz_pct=data["top1_boltz_pct"],
            checks_html=checks_html,
            pose_rows_html=pose_rows_html,
            sections_html="\n".join(sections),
            contacts_table_html=contacts_table_html,
            hotspots_table_html=hotspots_table_html,
            scoring=escape(str(data["scoring"])),
            refs_html=refs_html,
            version=escape(__version__),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        )
        path = output_dir / "report.html"
        path.write_text(html)
        return path
    except Exception as exc:
        logger.error("Could not build HTML report: %s", exc)
        return None


def generate_report(
    result,
    output_dir: Path,
    ligand_mol=None,
    receptor_file: Optional[str] = None,
    interaction_analyses: Optional[List[Dict]] = None,
    run_pandamap: bool = True,
) -> Dict[str, Path]:
    """
    Generate all publication-grade plots, the editorial HTML report, and the A4 vector PDF.

    Returns the mapping of generated artefacts. Failures are logged and skipped
    individually so one unavailable optional dependency cannot lose the rest.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images: Dict[str, Path] = {}

    if ligand_mol is None:
        ligand_mol = _load_ligand_mol_from_dir(output_dir)

    heavy = None
    if ligand_mol is not None:
        heavy = np.array(
            [a.GetIdx() for a in ligand_mol.GetAtoms() if a.GetAtomicNum() > 1],
            dtype=np.int64,
        )

    rec_path = receptor_file or getattr(result, "receptor_file", "")
    ligand_name = getattr(result, "ligand_name", "ligand")
    analytics = compute_report_analytics(
        result, ligand_mol=ligand_mol, receptor_file=rec_path,
        interaction_analyses=interaction_analyses,
    )

    top_coords = np.asarray(result.poses[0].coordinates, dtype=float) if result.poses else None
    tasks = [
        ("silhouette", lambda: plot_protein_silhouette(rec_path, top_coords, output_dir / "protein_silhouette.png")),
        ("scores", lambda: plot_pose_scores(result.poses, output_dir / "pose_scores.png",
                                            f"Pose scores — {ligand_name}", heavy_atoms=heavy)),
        ("components", lambda: plot_energy_components(result.poses, output_dir / "energy_components.png")),
        ("diversity", lambda: plot_pose_diversity(result.poses, output_dir / "pose_diversity.png", heavy)),
        ("ligand_dynamics", lambda: plot_ligand_dynamics(result.poses, ligand_mol, output_dir / "ligand_dynamics.png")),
    ]
    if interaction_analyses:
        tasks.append(("fingerprint", lambda: plot_interaction_fingerprint(
            interaction_analyses, output_dir / "interaction_fingerprint.png")))
        tasks.append(("hotspots", lambda: plot_residue_hotspots(
            interaction_analyses, output_dir / "residue_hotspots.png")))

    for key, task in tasks:
        try:
            path = task()
            if path is not None:
                images[key] = path
        except Exception as exc:
            logger.warning("Could not generate %s plot: %s", key, exc)

    if run_pandamap:
        try:
            path = _run_pandamap(output_dir)
            if path is not None:
                images["pandamap"] = path
        except Exception as exc:
            logger.warning("PandaMap visualisation unavailable: %s", exc)

    # Generate A4 brochure PDF first so report.html can link directly to report.pdf
    pdf_path = build_pdf_report(
        output_dir / "report.pdf",
        result,
        ligand_mol=ligand_mol,
        receptor_file=rec_path,
        interaction_analyses=interaction_analyses,
        pandamap_image=images.get("pandamap"),
        analytics=analytics,
    )
    if pdf_path is not None:
        images["pdf"] = pdf_path

    build_html_report(output_dir, result, images, ligand_name, analytics=analytics)
    return images


def _run_pandamap(output_dir: Path) -> Optional[Path]:
    """Generate a 2D interaction diagram for the top complex, if PandaMap is present."""
    candidates = (
        sorted(output_dir.glob("complex1.pdb"))
        or sorted(output_dir.glob("complex_1.pdb"))
        or sorted(output_dir.glob("covalent_complex1.pdb"))
    )
    if not candidates:
        return None

    from ..visualization.pandamap.pandamap_integration import PandaMapIntegration

    integration = PandaMapIntegration(str(output_dir))
    if not integration._check_pandamap_availability():
        logger.info("PandaMap is not installed; skipping the interaction diagram. "
                    "Install it with: pip install pandamap")
        return None

    if candidates[0].name.startswith("covalent_"):
        integration.create_interaction_map_for_pose(
            str(candidates[0]), "complex_1", binding_affinity=0.0, generate_3d=False
        )
    else:
        integration.analyze_complex_pdbs(str(output_dir), top_n=1, generate_3d=False)
    for candidate in sorted(output_dir.glob("pandamap_2d_*.png")):
        if not candidate.name.endswith("_classic.png") and not candidate.name.endswith("_3d.png"):
            return candidate
    for candidate in sorted(output_dir.glob("pandamap_2d_*.png")):
        return candidate
    return None


def load_result_from_dir(results_dir: Path):
    """
    Reconstruct a docking result from the JSON a run writes.

    Returns None when the directory holds no docking output, so callers can fall
    back to another loader rather than treating absence as failure.
    """
    from ..docking.core import DockingResult, Pose

    results_dir = Path(results_dir)
    summaries = sorted(results_dir.glob("*_summary.json"))
    poses_files = sorted(results_dir.glob("*_poses.json"))
    if not summaries or not poses_files:
        return None

    try:
        summary = json.loads(summaries[0].read_text())
        poses_blob = json.loads(poses_files[0].read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read docking output in %s: %s", results_dir, exc)
        return None

    raw_poses = poses_blob.get("poses", poses_blob) if isinstance(poses_blob, dict) else poses_blob
    if not raw_poses:
        return None

    poses = []
    for entry in raw_poses:
        try:
            poses.append(
                Pose(
                    coordinates=np.asarray(entry["coordinates"], dtype=float),
                    center=np.asarray(entry.get("center", [0, 0, 0]), dtype=float),
                    rotation=np.asarray(entry.get("rotation", [0, 0, 0, 1]), dtype=float),
                    conformer_id=int(entry.get("conformer_id", 0)),
                    energy=float(entry.get("energy", 0.0)),
                    energy_components=entry.get("energy_components") or {},
                    internal_strain=float(entry.get("internal_strain", 0.0)),
                    confidence=float(entry.get("confidence", 0.0)),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("Skipping malformed pose entry: %s", exc)

    if not poses:
        return None

    result = DockingResult(
        ligand_name=summary.get("ligand_name", "ligand"),
        receptor_file=summary.get("receptor_file", ""),
        grid_center=np.asarray(summary.get("grid_center", [0, 0, 0]), dtype=float),
        grid_dimensions=np.asarray(summary.get("grid_dimensions", [0, 0, 0]), dtype=float),
        algorithm_used=summary.get("algorithm_used", "pandadock"),
        scoring_function=summary.get("scoring_function", "vina"),
        poses=poses,
        runtime_seconds=float(summary.get("runtime_seconds", 0.0)),
        parameters=summary.get("parameters", {}),
    )
    result.ensemble_binding_energy = float(summary.get("ensemble_binding_energy", 0.0))
    result.ensemble_confidence = float(summary.get("ensemble_confidence", 0.0))
    return result


def load_interaction_analyses(results_dir: Path) -> List[Dict]:
    """Read per-pose interaction analyses if a run recorded them."""
    path = Path(results_dir) / "interaction_analysis.json"
    if not path.exists():
        return []
    try:
        blob = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(blob, dict) and "all_poses" in blob:
        return blob["all_poses"]
    return [blob] if isinstance(blob, dict) else []
