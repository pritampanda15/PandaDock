#!/usr/bin/env python
"""
Compare two runs on per-target correlation, paired by target.

Comparing medians across runs invites reading noise as improvement: the
within-target correlations have an interquartile range near 0.35, so a shift of
0.03 in the median is well inside the spread of the underlying distribution.

Both runs score the same targets, so the comparison can be paired -- asking
whether each target individually improved, which is far more sensitive than
comparing two summary statistics. Reports the paired difference, a Wilcoxon
signed-rank test, and a bootstrap interval on the median change.

Usage:
    python benchmarking/compare_runs.py \
        --baseline sair_test_predictions.csv \
        --candidate sair_ranked_predictions.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.sair_evaluate import pearson, spearman  # noqa: E402


def per_target_r(path: str, min_ligands: int, rank: bool):
    """Correlation per target, keyed by target id."""
    groups = defaultdict(lambda: ([], []))
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            target = int(row["target_id"])
            groups[target][0].append(float(row["predicted"]))
            groups[target][1].append(float(row["actual"]))

    correlate = spearman if rank else pearson
    out = {}
    for target, (predicted, actual) in groups.items():
        if len(predicted) < min_ligands:
            continue
        value = correlate(np.asarray(predicted), np.asarray(actual))
        if value is not None:
            out[target] = value
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--min-ligands", type=int, default=20)
    parser.add_argument("--spearman", action="store_true",
                        help="Compare rank correlation instead of Pearson")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    base = per_target_r(args.baseline, args.min_ligands, args.spearman)
    cand = per_target_r(args.candidate, args.min_ligands, args.spearman)

    shared = sorted(set(base) & set(cand))
    if not shared:
        print("No target qualified in both runs.")
        return 1

    a = np.array([base[t] for t in shared])
    b = np.array([cand[t] for t in shared])
    delta = b - a

    label = "spearman rho" if args.spearman else "pearson r"
    print(f"{len(shared):,} targets scored by both runs ({label})")
    print("=" * 62)
    print(f"  baseline   median {np.median(a):+.4f}   mean {a.mean():+.4f}")
    print(f"  candidate  median {np.median(b):+.4f}   mean {b.mean():+.4f}")
    print(f"  difference median {np.median(delta):+.4f}   mean {delta.mean():+.4f}")

    improved = int((delta > 0).sum())
    print(f"\n  improved on {improved:,}/{len(shared):,} targets "
          f"({100 * improved / len(shared):.1f}%)")

    # Bootstrap over targets: the spread of the median change under resampling
    # says whether the observed shift is distinguishable from zero.
    rng = np.random.default_rng(args.seed)
    medians = np.array([
        np.median(rng.choice(delta, delta.size, replace=True))
        for _ in range(args.bootstrap)
    ])
    low, high = np.percentile(medians, [2.5, 97.5])
    print(f"  95% CI on the median change: [{low:+.4f}, {high:+.4f}]")

    crosses_zero = low <= 0 <= high

    # The signed-rank test is the decision. It uses every paired difference,
    # whereas the bootstrap interval describes the median alone -- a noisier
    # statistic that can straddle zero while the paired differences are
    # consistently one-sided. Requiring both to agree, as this once did, calls a
    # real effect noise: p = 0.004 with a median CI of [-0.002, +0.098] is a
    # reliable difference whose *size* is uncertain, not an absent one.
    pvalue = None
    try:
        from scipy import stats

        pvalue = float(stats.wilcoxon(b, a).pvalue)
        print(f"\n  Wilcoxon signed-rank p = {pvalue:.4g}")
    except ImportError:
        print("\n  (scipy not installed; falling back to the bootstrap interval)")

    significant = (pvalue < 0.05) if pvalue is not None else not crosses_zero

    print("\n" + "-" * 62)
    if not significant:
        print("  The change is not distinguishable from noise. Whatever the")
        print("  summary medians suggest, this run did not measurably improve")
        print("  per-target ranking -- report the runs as equivalent.")
    else:
        direction = "improves on" if np.median(delta) > 0 else "degrades"
        print(f"  The candidate {direction} the baseline, consistently enough")
        print("  across targets to be distinguishable from noise.")
        print(f"  Median change {np.median(delta):+.4f}, mean {delta.mean():+.4f}.")
        if crosses_zero:
            print("\n  The interval on the median still spans zero, so the")
            print("  difference is reliable in direction but small and imprecise")
            print("  in size. Report the effect, not just the p-value.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
