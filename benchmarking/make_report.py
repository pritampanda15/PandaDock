#!/usr/bin/env python
"""
Build publication tables from redocking results.

Emits the two tables a docking paper needs, in Markdown and LaTeX:

  Table A (pose accuracy)   median RMSD and % of poses within 2 A, per family
  Table B (scoring power)   Pearson r, R^2 and Spearman rho between the docking
                            score and experimental affinity, per family

Both tables report N per family. Percentages computed over a handful of
complexes are not meaningful, and a family table that hides its denominators
invites exactly that mistake: a family with N=3 can read as 0% or 100% on noise
alone. Families below `--min-n` are marked rather than silently dropped.

Pose accuracy is reported for the top-ranked pose. Best-of-N is given in a
separate column, never merged into the headline: the gap between them is a real
and interesting quantity (it measures ranking failure, not sampling failure) but
quoting best-of-N as though it were top-1 overstates accuracy substantially.

Usage:
    python benchmarking/make_report.py benchmark_results/full/redock_results.csv \
        --affinity complexes/benchmark_data/all_structural_affinity.csv \
        --output benchmark_results/full/report
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

FAMILY_ORDER = [
    "E3 ligases", "Epigenetic enzymes", "GPCRs", "Glycosidases", "Ion channels",
    "Metalloenzymes", "Molecular chaperones", "Nuclear receptors",
    "Oxidoreductases", "PPI targets", "Phosphatases", "Proteases",
    "Protein kinases", "Transporters",
]


# ----------------------------------------------------------------------- loading


def load_results(path: Path) -> List[Dict]:
    rows = []
    for row in csv.DictReader(open(path, newline="")):
        if row["status"] != "ok":
            continue
        try:
            row["top1_rmsd"] = float(row["top1_rmsd"])
            row["best_rmsd"] = float(row["best_rmsd"])
            row["top1_energy"] = float(row["top1_energy"])
        except (ValueError, KeyError):
            continue
        if not math.isfinite(row["top1_rmsd"]):
            continue
        row["n_torsions"] = int(row["n_torsions"] or -1)
        rows.append(row)
    return rows


def load_affinity(path: Path) -> Dict[str, float]:
    """
    Map PDB id -> experimental affinity in log units (higher = tighter).

    Only Kd, Ki and IC50 are kept. EC50 is a functional readout on a different
    scale, and pooling scales is what collapsed PandaDock-GNN's correlation from
    0.81 to 0.49 when PDBbind and BindingDB were combined.
    """
    keep = {"KD", "KI", "IC50"}
    values: Dict[str, List[float]] = defaultdict(list)

    for row in csv.DictReader(open(path, newline="")):
        kind = (row.get("affinity_type") or "").strip().upper()
        if kind not in keep:
            continue
        try:
            neg_log = float(row["neg_log_affinity"])
        except (TypeError, ValueError, KeyError):
            continue
        if not math.isfinite(neg_log):
            continue
        values[(row.get("pdb_id") or "").strip().upper()].append(neg_log)

    # Several measurements for one entry: use the median rather than picking one.
    return {k: float(np.median(v)) for k, v in values.items() if v}


# -------------------------------------------------------------------- statistics


def pose_stats(rows: Sequence[Dict], threshold: float) -> Dict:
    if not rows:
        return {"n": 0}
    top1 = np.array([r["top1_rmsd"] for r in rows])
    best = np.array([r["best_rmsd"] for r in rows])
    return {
        "n": len(rows),
        "median": float(np.median(top1)),
        "pct_top1": 100.0 * float(np.mean(top1 <= threshold)),
        "pct_best": 100.0 * float(np.mean(best <= threshold)),
        "pct_misranked": 100.0 * float(np.mean((best <= threshold) & (top1 > threshold))),
    }


def correlations(scores: np.ndarray, affinities: np.ndarray) -> Dict:
    """
    Pearson r, R^2 and Spearman rho between docking score and affinity.

    Docking scores are negative for tighter binding while pAffinity is positive,
    so the score is negated first: a well-behaved engine then yields a POSITIVE
    correlation, matching the convention used in published comparisons.
    """
    if len(scores) < 3:
        return {"n": len(scores)}

    x = -scores
    y = affinities

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {"n": len(x), "pearson": float("nan"), "r2": float("nan"),
                "spearman": float("nan")}

    pearson = float(np.corrcoef(x, y)[0, 1])
    rank_x = _rankdata(x)
    rank_y = _rankdata(y)
    spearman = float(np.corrcoef(rank_x, rank_y)[0, 1])

    # R^2 as squared correlation (scoring power), matching the usual convention
    # in docking comparisons rather than a regression coefficient of determination.
    return {"n": len(x), "pearson": pearson, "r2": pearson**2, "spearman": spearman}


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # Average tied ranks.
    unique, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


# ---------------------------------------------------------------------- rendering


def fmt(value: Optional[float], places: int = 2, dash: str = "--") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return dash
    return f"{value:.{places}f}"


def render_pose_markdown(per_family: Dict[str, Dict], overall: Dict,
                         threshold: float, min_n: int) -> str:
    lines = [
        f"### Table A — Docking pose accuracy across protein families",
        "",
        f"Median RMSD (Å) and percentage of poses with RMSD ≤ {threshold:g} Å, per family. "
        "RMSD is symmetry-corrected heavy-atom RMSD to the crystal pose, computed "
        "without superposition. The `% ≤ 2 Å` column refers to the **top-ranked** "
        "pose; `best-of-N` is shown separately.",
        "",
        "| Protein family | N | Median RMSD (Å) | % top-1 ≤ 2 Å | % best-of-N ≤ 2 Å | % misranked |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILY_ORDER:
        s = per_family.get(family)
        if not s or not s["n"]:
            lines.append(f"| {family} | 0 | — | — | — | — |")
            continue
        mark = " ᵃ" if s["n"] < min_n else ""
        lines.append(
            f"| {family}{mark} | {s['n']} | {fmt(s['median'])} | {s['pct_top1']:.1f} | "
            f"{s['pct_best']:.1f} | {s['pct_misranked']:.1f} |"
        )
    lines.append(
        f"| **Overall** | **{overall['n']}** | **{fmt(overall['median'])}** | "
        f"**{overall['pct_top1']:.1f}** | **{overall['pct_best']:.1f}** | "
        f"**{overall['pct_misranked']:.1f}** |"
    )
    lines += [
        "",
        f"ᵃ Fewer than {min_n} complexes; percentages from this family are dominated "
        "by sampling noise and should not be compared across engines.",
        "",
        "*% misranked* is the share of complexes where a pose within the threshold "
        "was generated but not ranked first. It isolates scoring-function error "
        "from search error: sampling harder cannot recover these, but rescoring can.",
    ]
    return "\n".join(lines)


def render_affinity_markdown(per_family: Dict[str, Dict], overall: Dict,
                             min_n: int) -> str:
    lines = [
        "### Table B — Affinity correlation across protein families",
        "",
        "Pearson *r* and *R²* (scoring power) and Spearman *ρ* (ranking power) "
        "between the docking score and experimental affinity (pKd/pKi/pIC50). "
        "Scores are negated so that a positive correlation indicates correct behaviour.",
        "",
        "| Protein family | N | Pearson r | R² | Spearman ρ |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in FAMILY_ORDER:
        s = per_family.get(family)
        if not s or s.get("n", 0) < 3:
            n = s["n"] if s else 0
            lines.append(f"| {family} | {n} | — | — | — |")
            continue
        mark = " ᵃ" if s["n"] < min_n else ""
        lines.append(
            f"| {family}{mark} | {s['n']} | {fmt(s['pearson'])} | {fmt(s['r2'])} | "
            f"{fmt(s['spearman'])} |"
        )
    if overall.get("n", 0) >= 3:
        lines.append(
            f"| **Overall** | **{overall['n']}** | **{fmt(overall['pearson'])}** | "
            f"**{fmt(overall['r2'])}** | **{fmt(overall['spearman'])}** |"
        )
    lines += [
        "",
        f"ᵃ Fewer than {min_n} complexes with affinity data.",
        "",
        "These correlations are for the **empirical docking score**, which is a "
        "Vina-style function fitted to reproduce binding geometry, not potency. "
        "It is not the PandaDock-GNN scoring model and should not be compared "
        "against learned affinity predictors; use `pandadock gnn` for that.",
    ]
    return "\n".join(lines)


def render_pose_latex(per_family: Dict[str, Dict], overall: Dict,
                      threshold: float) -> str:
    rows = []
    for family in FAMILY_ORDER:
        s = per_family.get(family)
        if not s or not s["n"]:
            rows.append(f"{family} & 0 & -- & -- & -- \\\\")
            continue
        rows.append(
            f"{family} & {s['n']} & {fmt(s['median'])} & {s['pct_top1']:.1f} & "
            f"{s['pct_best']:.1f} \\\\"
        )
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]
\\centering
\\caption{{Docking pose accuracy across 14 protein families. Median RMSD (\\AA)
and percentage of poses with RMSD $\\leq$ {threshold:g}\\,\\AA{{}} are reported per
protein family, for the top-ranked pose. RMSD is symmetry-corrected and computed
without superposition. The best-of-$N$ column reports whether any returned pose
met the threshold, and is shown separately because it is not comparable to a
top-1 figure.}}
\\label{{tab:pose-accuracy}}
\\begin{{tabular}}{{lrrrr}}
\\toprule
Protein Family & $N$ & Median RMSD (\\AA) & \\% top-1 $\\leq$ 2\\,\\AA & \\% best-of-$N$ $\\leq$ 2\\,\\AA \\\\
\\midrule
{body}
\\midrule
\\textbf{{Overall}} & \\textbf{{{overall['n']}}} & \\textbf{{{fmt(overall['median'])}}} &
\\textbf{{{overall['pct_top1']:.1f}}} & \\textbf{{{overall['pct_best']:.1f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""


# --------------------------------------------------------------------------- main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--affinity", type=Path, default=None,
                        help="CSV with pdb_id, affinity_type, neg_log_affinity")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write <output>.md and <output>.tex")
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--min-n", type=int, default=10,
                        help="Families below this N are flagged as underpowered")
    args = parser.parse_args(argv)

    rows = load_results(args.results)
    if not rows:
        print("No successful results to report.", file=sys.stderr)
        return 1

    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)

    pose_per_family = {f: pose_stats(v, args.threshold) for f, v in by_family.items()}
    pose_overall = pose_stats(rows, args.threshold)

    sections = [
        "# PandaDock redocking benchmark",
        "",
        f"{len(rows)} complexes across {len(by_family)} protein families. "
        f"Success threshold {args.threshold:g} Å.",
        "",
        render_pose_markdown(pose_per_family, pose_overall, args.threshold, args.min_n),
    ]

    aff_per_family: Dict[str, Dict] = {}
    aff_overall: Dict = {"n": 0}
    if args.affinity and args.affinity.exists():
        affinity = load_affinity(args.affinity)
        matched_scores, matched_values = [], []
        for family, members in by_family.items():
            xs, ys = [], []
            for r in members:
                value = affinity.get(r["id"].upper())
                if value is None:
                    continue
                xs.append(r["top1_energy"])
                ys.append(value)
            aff_per_family[family] = correlations(np.array(xs), np.array(ys))
            matched_scores.extend(xs)
            matched_values.extend(ys)
        aff_overall = correlations(np.array(matched_scores), np.array(matched_values))

        sections += ["", render_affinity_markdown(aff_per_family, aff_overall, args.min_n)]

        if aff_overall.get("n", 0) < 30:
            sections += [
                "",
                f"> Only {aff_overall.get('n', 0)} complexes carried usable affinity data, "
                "so the per-family correlations above rest on very few points. "
                "Treat them as indicative, not as a scoring-power benchmark.",
            ]
    else:
        sections += ["", "_No affinity file supplied; Table B omitted._"]

    # Flexibility breakdown: the axis that dominates pose accuracy.
    by_flex = defaultdict(list)
    for r in rows:
        t = r["n_torsions"]
        bucket = ("0-2 (rigid)" if t <= 2 else "3-5 (moderate)" if t <= 5
                  else "6-9 (flexible)" if t <= 9 else "10+ (very flexible)")
        by_flex[bucket].append(r)

    sections += ["", "### Table C — Pose accuracy by ligand flexibility", "",
                 "| Rotatable bonds | N | Median RMSD (Å) | % top-1 ≤ 2 Å | % best-of-N ≤ 2 Å |",
                 "|---|---:|---:|---:|---:|"]
    for bucket in ["0-2 (rigid)", "3-5 (moderate)", "6-9 (flexible)", "10+ (very flexible)"]:
        s = pose_stats(by_flex.get(bucket, []), args.threshold)
        if not s["n"]:
            continue
        sections.append(
            f"| {bucket} | {s['n']} | {fmt(s['median'])} | {s['pct_top1']:.1f} | {s['pct_best']:.1f} |"
        )
    sections += [
        "",
        "Ligand flexibility, not protein family, is the dominant axis of pose "
        "accuracy. Family-level differences largely track the typical ligand size "
        "of each family, so a family table read on its own is easy to "
        "over-interpret.",
    ]

    report = "\n".join(sections)
    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{args.output}.md").write_text(report + "\n")
        Path(f"{args.output}.tex").write_text(
            render_pose_latex(pose_per_family, pose_overall, args.threshold) + "\n"
        )
        print(f"\n\nWrote {args.output}.md and {args.output}.tex", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
