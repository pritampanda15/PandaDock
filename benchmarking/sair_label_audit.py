#!/usr/bin/env python
"""
Audit the SAIR labels before trusting a correlation computed against them.

A test Pearson r is only as meaningful as the label distribution underneath it.
Two failure modes matter here, and neither shows up in the training log:

1. Degeneracy. ULVSH turned out to have 89 of 95 compounds sharing pIC50 = 4.0,
   so a model predicting the constant 4.0 scored well on metrics that looked
   like generalisation. Censored measurements ("> 10 uM") collapse to the same
   value and produce exactly this.

2. Target imbalance. If a handful of proteins supply most of the test set, the
   reported r describes those few targets rather than the model's behaviour
   across chemistry, and a per-target breakdown will disagree sharply with the
   pooled number.

Reports both, per split, using the same target-disjoint split the trainer uses.

Usage:
    python benchmarking/sair_label_audit.py --cache shard_cache/
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def describe(name: str, values: np.ndarray, targets: list) -> None:
    print(f"\n{name}  ({len(values):,} complexes, {len(set(targets)):,} targets)")
    print("-" * 62)

    if len(values) == 0:
        print("  empty")
        return

    print(f"  mean {values.mean():.3f}   sd {values.std():.3f}   "
          f"min {values.min():.2f}   max {values.max():.2f}")
    for q in (1, 25, 50, 75, 99):
        print(f"  p{q:<3} {np.percentile(values, q):.3f}")

    # Degeneracy: how much of the set sits on a single value, and how well a
    # constant predictor would do. A model cannot beat a constant on a set that
    # is mostly one number, so a high share here caps what r means.
    counts = Counter(np.round(values, 3).tolist())
    print("\n  most common labels:")
    for value, count in counts.most_common(5):
        print(f"    {value:6.3f}  {count:>7,}  ({100 * count / len(values):5.2f}%)")

    top_share = 100 * counts.most_common(1)[0][1] / len(values)
    distinct = len(counts)
    print(f"\n  distinct values: {distinct:,}"
          f"   largest single value: {top_share:.2f}%")

    if top_share > 20:
        print("  WARNING: one value dominates. A correlation against this "
              "distribution is not\n           evidence of a working model -- "
              "check the censoring rule.")

    # Target concentration: does the pooled r describe the whole split?
    per_target = Counter(targets)
    ranked = per_target.most_common()
    for k in (1, 10):
        if len(ranked) >= k:
            share = 100 * sum(c for _, c in ranked[:k]) / len(values)
            print(f"  top {k:>2} target(s) supply {share:5.2f}% of complexes")

    # Only meaningful when the split has appreciably more than ten targets;
    # otherwise the top ten trivially account for everything.
    if len(ranked) > 20:
        top10 = 100 * sum(c for _, c in ranked[:10]) / len(values)
        if top10 > 50:
            print(f"  WARNING: 10 of {len(ranked):,} targets supply "
                  f"{top10:.1f}% of this split. Report a\n           per-target "
                  "r alongside the pooled value.")


def variance_decomposition(name: str, values, targets) -> None:
    """
    Split label variance into between-target and within-target parts.

    This bounds what a model can score without ranking ligands at all. A
    predictor that outputs each target's mean affinity -- ignoring which ligand
    is bound -- achieves r = sqrt(between-target fraction). If a model's pooled
    r is at or below that, the pooled number is evidence it has learned to read
    the protein, not to discriminate between ligands against it, and the
    interesting quantity is the within-target correlation instead.

    Test targets are disjoint from training here, so a model cannot have
    memorised these means; it would have to infer them from the binding site.
    That is a legitimate capability, but it is a different claim from affinity
    prediction, and pooled r does not separate the two.
    """
    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets)

    print(f"\nvariance decomposition ({name})")
    print("-" * 62)

    grand_mean = values.mean()
    total = ((values - grand_mean) ** 2).sum()

    between = 0.0
    sizes = []
    for target in np.unique(targets):
        mask = targets == target
        group = values[mask]
        sizes.append(group.size)
        between += group.size * (group.mean() - grand_mean) ** 2

    fraction = between / total if total else 0.0
    sizes = np.asarray(sizes)

    print(f"  between-target variance: {100 * fraction:5.2f}%")
    print(f"  within-target variance:  {100 * (1 - fraction):5.2f}%")
    print(f"  ligands per target: median {int(np.median(sizes))}, "
          f"min {sizes.min()}, max {sizes.max()}")
    print(f"\n  a target-mean predictor -- one that ignores the ligand entirely --")
    print(f"  would score r = {np.sqrt(fraction):.3f} on this split.")
    print("\n  Compare your pooled test r against that number. At or below it,")
    print("  report the median within-target r as the headline instead: that is")
    print("  the quantity that says whether the model ranks ligands.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the training seed for the split to line up")
    args = parser.parse_args(argv)

    from pandadock.gnn.data.sair_dataset import build_index, load_shard

    index = build_index(args.cache)
    print(f"{len(index['entries']):,} complexes, "
          f"{index['n_sequences']:,} targets, {index['n_shards']:,} shards")

    # Reproduce the trainer's split exactly.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(index["n_sequences"])
    n_train = int(index["n_sequences"] * 0.8)
    n_val = int(index["n_sequences"] * 0.1)
    assignment = {}
    for seq_id in order[:n_train]:
        assignment[int(seq_id)] = "train"
    for seq_id in order[n_train:n_train + n_val]:
        assignment[int(seq_id)] = "val"
    for seq_id in order[n_train + n_val:]:
        assignment[int(seq_id)] = "test"

    by_entry = {e: (s, q) for s, e, q in index["entries"]}
    by_shard: dict = {}
    for shard_id, entry_id, _ in index["entries"]:
        by_shard.setdefault(shard_id, []).append(entry_id)

    splits: dict = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for n, shard_id in enumerate(sorted(by_shard), 1):
        records = load_shard(args.cache, shard_id)
        for entry_id in by_shard[shard_id]:
            record = records.get(entry_id)
            if record is None:
                continue
            seq_id = by_entry[entry_id][1]
            values, targets = splits[assignment[seq_id]]
            values.append(record["pic50"])
            targets.append(seq_id)
        if n % 100 == 0:
            print(f"  read {n}/{len(by_shard)} shards", flush=True)

    for name in ("train", "val", "test"):
        values, targets = splits[name]
        describe(name, np.asarray(values, dtype=np.float64), targets)

    variance_decomposition("test", *splits["test"])

    print("\n" + "=" * 62)
    print("A constant predictor scores r = 0 by definition, so a non-zero r is")
    print("not automatically meaningful -- but a label set concentrated on a")
    print("few values makes RMSE and MAE look better than the model deserves.")
    print("Read the RMSE against the spread (sd) printed above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
