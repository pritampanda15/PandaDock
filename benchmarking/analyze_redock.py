#!/usr/bin/env python
"""
Analyse redocking results, separating search failures from ranking failures.

The headline "% within 2 A" conflates two very different problems, and they have
different fixes:

- If no pose in the returned set is close to the crystal pose, the search never
  visited the right basin. Fix by sampling harder (higher exhaustiveness) or by
  improving the search.
- If a good pose IS in the set but is not ranked first, the search worked and the
  scoring function ordered the poses wrongly. Sampling harder will not help; a
  better rescorer will.

The gap between top-1 and best-of-N accuracy measures exactly that second failure,
and is the quantity that tells you whether a learned rescorer is worth applying.

Usage:
    python benchmarking/analyze_redock.py benchmark_results/pilot/redock_results.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

# Nucleotides, redox and methyl-donor cofactors. These are highly charged and very
# flexible; empirical scoring functions without explicit electrostatics or
# desolvation handle them poorly, so their results are worth reading separately
# rather than letting them move the headline number.
COFACTOR_CODES = {
    "ATP", "ADP", "AMP", "ANP", "ACP", "AGS", "ATR", "GTP", "GDP", "GNP", "GSP",
    "NAD", "NAI", "NAP", "NDP", "FAD", "FMN", "SAM", "SAH", "COA", "ACO", "UDP",
    "UTP", "UD1", "UMP", "IHP", "IP9", "HEM", "TPP", "PLP", "BTN", "MTA", "5GP",
    "CTP", "CDP", "H4B", "THG", "FDA", "NDG",
}

THRESHOLD = 2.0


def load(path: Path) -> List[Dict]:
    rows = []
    for row in csv.DictReader(open(path, newline="")):
        if row["status"] != "ok":
            rows.append(row)
            continue
        for key in ("top1_rmsd", "best_rmsd", "top1_energy", "runtime_s"):
            row[key] = float(row[key]) if row[key] else float("nan")
        for key in ("n_torsions", "n_heavy_atoms", "n_poses"):
            row[key] = int(row[key]) if row[key] else -1
        rows.append(row)
    return rows


def stats(subset: Sequence[Dict]) -> Dict:
    ok = [r for r in subset if r["status"] == "ok" and np.isfinite(r["top1_rmsd"])]
    if not ok:
        return {"n": 0}
    top1 = np.array([r["top1_rmsd"] for r in ok])
    best = np.array([r["best_rmsd"] for r in ok])
    return {
        "n": len(ok),
        "median_top1": float(np.median(top1)),
        "pct_top1": 100.0 * float(np.mean(top1 <= THRESHOLD)),
        "pct_best": 100.0 * float(np.mean(best <= THRESHOLD)),
        # Complexes where a correct pose was found but ranked below a wrong one.
        "pct_misranked": 100.0 * float(np.mean((best <= THRESHOLD) & (top1 > THRESHOLD))),
        "median_runtime": float(np.median([r["runtime_s"] for r in ok])),
    }


def table(title: str, groups: Dict[str, List[Dict]], label: str = "Group") -> None:
    print(f"\n{title}")
    print("-" * 84)
    print(f"{label:<26}{'N':>5}{'Median':>9}{'top1<=2A':>10}{'best<=2A':>10}"
          f"{'misranked':>11}{'sec':>8}")
    print("-" * 84)
    for name in sorted(groups):
        s = stats(groups[name])
        if not s["n"]:
            continue
        print(f"{name:<26}{s['n']:>5}{s['median_top1']:>9.2f}{s['pct_top1']:>9.0f}%"
              f"{s['pct_best']:>9.0f}%{s['pct_misranked']:>10.0f}%{s['median_runtime']:>8.0f}")
    print("-" * 84)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--threshold", type=float, default=2.0)
    args = parser.parse_args(argv)

    global THRESHOLD
    THRESHOLD = args.threshold

    rows = load(args.results)
    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]

    print(f"Results: {args.results}")
    print(f"Docked {len(ok)}/{len(rows)} complexes "
          f"(success threshold {THRESHOLD} A, symmetry-corrected, no superposition)")

    if failed:
        reasons = defaultdict(int)
        for r in failed:
            reasons[r["status"]] += 1
        print("Not docked: " + ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))

    if not ok:
        return 1

    overall = stats(ok)
    print(f"\nOVERALL  median top-1 RMSD {overall['median_top1']:.2f} A | "
          f"top-1 <=2A {overall['pct_top1']:.0f}% | "
          f"best-of-N <=2A {overall['pct_best']:.0f}%")
    print(f"         of the {overall['pct_best']:.0f}% found, "
          f"{overall['pct_misranked']:.0f} points are found but MISRANKED "
          f"-- a scoring problem, not a sampling one")

    by_family = defaultdict(list)
    for r in ok:
        by_family[r["family"]].append(r)
    table("BY PROTEIN FAMILY", by_family, "Protein family")

    by_flex = defaultdict(list)
    for r in ok:
        t = r["n_torsions"]
        bucket = ("0-2 rigid" if t <= 2 else "3-5 moderate" if t <= 5
                  else "6-9 flexible" if t <= 9 else "10+ very flexible")
        by_flex[bucket].append(r)
    table("BY LIGAND FLEXIBILITY", by_flex, "Rotatable bonds")

    by_class = defaultdict(list)
    for r in ok:
        by_class["cofactor / nucleotide" if r["ligand_code"] in COFACTOR_CODES
                 else "drug-like ligand"].append(r)
    table("BY LIGAND CLASS", by_class, "Ligand class")

    print("\nWorst top-1 outcomes where the correct pose WAS sampled:")
    misranked = [r for r in ok if r["best_rmsd"] <= THRESHOLD and r["top1_rmsd"] > THRESHOLD]
    misranked.sort(key=lambda r: -r["top1_rmsd"])
    for r in misranked[:12]:
        print(f"  {r['id']:<8} {r['family']:<22} {r['ligand_code']:<5} "
              f"top1 {r['top1_rmsd']:6.2f}  best {r['best_rmsd']:5.2f}  "
              f"({r['n_torsions']} torsions)")
    if not misranked:
        print("  none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
