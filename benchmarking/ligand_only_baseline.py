#!/usr/bin/env python
"""
Ligand-only baseline: how much of the affinity signal needs no structure?

The GNN ranks ligands against a fixed target at a median within-target r of
about 0.24. That number is only evidence of structural understanding if a model
that never sees the protein does worse. Simple ligand descriptors -- size,
composition, aromaticity -- are known to correlate with potency on their own,
and a graph network reading a co-folded pose could be recovering little more
than that.

This fits a linear model on descriptors computed from the ligand atoms alone,
trained on the training split and evaluated on the test split exactly as the GNN
is. No protein information enters at any point: the site coordinates in the cache
are ignored.

Interpretation:

  * baseline ~= GNN  -> the network is not extracting structural signal, and
    SAIR's co-folded poses are the thing to question next.
  * baseline << GNN  -> the structure is contributing, and architecture work is
    worth the GPU time.

Writes predictions in the same format as sair_evaluate.py, so the two can be
compared target-by-target with compare_runs.py.

Usage:
    python benchmarking/ligand_only_baseline.py \
        --cache shard_cache/ --out ligand_baseline_predictions.csv
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

# Enough to weigh a molecule to within a fraction of a percent; anything not
# listed is rare enough in drug-like ligands to fold into the default.
ATOMIC_WEIGHT = {
    "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06, "P": 30.974,
    "F": 18.998, "CL": 35.45, "BR": 79.904, "I": 126.90, "B": 10.81,
    "SE": 78.97, "SI": 28.085,
}
HALOGENS = {"F", "CL", "BR", "I"}

FEATURE_NAMES = [
    "n_heavy", "mol_weight", "n_carbon", "n_nitrogen", "n_oxygen", "n_sulfur",
    "n_halogen", "n_aromatic", "n_sp3", "n_sp2", "frac_aromatic", "frac_sp3",
    "n_polar", "radius_gyration", "max_extent", "log_n_heavy",
]


def descriptors(record: dict) -> np.ndarray:
    """
    Ligand-only features. The protein half of the record is never touched.

    Deliberately the kind of thing a medicinal chemist could compute by hand --
    the point is to establish what requires no structural modelling at all, so a
    richer featurisation would defeat the purpose.
    """
    types = record["lig_types"]
    xyz = np.asarray(record["lig_xyz"], dtype=np.float64)
    n = len(types)
    if n == 0:
        return None

    elements = [t.split(".")[0].upper() for t in types]

    weight = sum(ATOMIC_WEIGHT.get(e, 12.011) for e in elements)
    n_aromatic = sum(1 for t in types if "ar" in t.lower())
    n_sp3 = sum(1 for t in types if t.endswith(".3"))
    n_sp2 = sum(1 for t in types if t.endswith(".2"))
    n_polar = sum(1 for e in elements if e in ("N", "O"))

    centroid = xyz.mean(axis=0)
    offsets = xyz - centroid
    gyration = float(np.sqrt((offsets ** 2).sum(axis=1).mean()))

    # Largest pairwise distance. Ligands run to a few dozen atoms, so the
    # quadratic cost is irrelevant.
    if n > 1:
        diff = xyz[:, None, :] - xyz[None, :, :]
        extent = float(np.sqrt((diff ** 2).sum(axis=-1)).max())
    else:
        extent = 0.0

    return np.array([
        n,
        weight,
        elements.count("C"),
        elements.count("N"),
        elements.count("O"),
        elements.count("S"),
        sum(1 for e in elements if e in HALOGENS),
        n_aromatic,
        n_sp3,
        n_sp2,
        n_aromatic / n,
        n_sp3 / n,
        n_polar,
        gyration,
        extent,
        float(np.log1p(n)),
    ], dtype=np.float64)


def collect(cache, index, assignment, wanted: str):
    """Descriptors, labels, target ids and entry ids for one split."""
    from pandadock.gnn.data.sair_dataset import load_shard

    by_shard = defaultdict(list)
    for shard_id, entry_id, seq_id in index["entries"]:
        if assignment.get(seq_id) == wanted:
            by_shard[shard_id].append(entry_id)

    features, labels, targets, ids = [], [], [], []
    seq_of = {e: s for _, e, s in index["entries"]}

    for n, shard_id in enumerate(sorted(by_shard), 1):
        records = load_shard(cache, shard_id)
        for entry_id in by_shard[shard_id]:
            record = records.get(entry_id)
            if record is None:
                continue
            row = descriptors(record)
            if row is None:
                continue
            features.append(row)
            labels.append(float(record["pic50"]))
            targets.append(seq_of[entry_id])
            ids.append(entry_id)
        if n % 200 == 0:
            print(f"  {wanted}: {n}/{len(by_shard)} shards", flush=True)

    return (np.asarray(features), np.asarray(labels),
            np.asarray(targets), ids)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", default="ligand_baseline_predictions.csv")
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the training seed for the split to line up")
    parser.add_argument("--ridge", type=float, default=1.0,
                        help="L2 penalty; the features are collinear by construction")
    args = parser.parse_args(argv)

    from pandadock.gnn.data.sair_dataset import build_index

    index = build_index(args.cache)

    # Reproduce the trainer's target-disjoint split.
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

    print("Reading train split...")
    x_train, y_train, _, _ = collect(args.cache, index, assignment, "train")
    print("Reading test split...")
    x_test, y_test, t_test, id_test = collect(args.cache, index, assignment, "test")

    print(f"\ntrain {x_train.shape[0]:,} x {x_train.shape[1]} features"
          f"   test {x_test.shape[0]:,}")

    # Standardise on training statistics only, then ridge-regress. Closed form:
    # the design matrix is 16 columns, so there is nothing to iterate.
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1.0

    def design(x):
        z = (x - mean) / scale
        return np.hstack([z, np.ones((z.shape[0], 1))])

    a = design(x_train)
    penalty = args.ridge * np.eye(a.shape[1])
    penalty[-1, -1] = 0.0  # never penalise the intercept
    weights = np.linalg.solve(a.T @ a + penalty, a.T @ y_train)

    predicted = design(x_test) @ weights

    with open(args.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["entry_id", "target_id", "predicted", "actual"])
        for e, t, p, y in zip(id_test, t_test, predicted, y_test):
            writer.writerow([e, int(t), f"{p:.4f}", f"{y:.4f}"])

    from benchmarking.sair_evaluate import pearson

    print("\n" + "=" * 62)
    print("LIGAND-ONLY BASELINE (no protein information)")
    print("=" * 62)
    print(f"  pooled pearson r  {pearson(predicted, y_test):+.4f}")
    rmse = float(np.sqrt(((predicted - y_test) ** 2).mean()))
    print(f"  rmse              {rmse:.4f}   "
          f"(constant predictor: {y_test.std():.4f})")

    groups = defaultdict(lambda: ([], []))
    for p, y, t in zip(predicted, y_test, t_test):
        groups[t][0].append(p)
        groups[t][1].append(y)

    within = []
    for values in groups.values():
        if len(values[0]) < 20:
            continue
        r = pearson(np.asarray(values[0]), np.asarray(values[1]))
        if r is not None:
            within.append(r)

    if within:
        within = np.asarray(within)
        print(f"\n  within-target r over {len(within):,} targets:")
        print(f"    median {np.median(within):+.4f}   mean {within.mean():+.4f}"
              f"   IQR [{np.percentile(within, 25):+.4f}, "
              f"{np.percentile(within, 75):+.4f}]")

    print("\n  Largest descriptor weights (standardised):")
    for name, weight in sorted(
        zip(FEATURE_NAMES, weights[:-1]), key=lambda kv: -abs(kv[1])
    )[:6]:
        print(f"    {name:<18} {weight:+.4f}")
    print("    NOTE: these features are strongly collinear -- heavy-atom count,")
    print("    molecular weight and carbon count measure nearly the same thing.")
    print("    Ridge splits a shared effect across them with arbitrary signs, so")
    print("    read this as which *group* matters, not which single descriptor.")

    print(f"\nPredictions written to {args.out}")
    print("\nCompare against the GNN, paired by target:")
    print(f"  python benchmarking/compare_runs.py \\\n"
          f"      --baseline {args.out} --candidate <gnn_predictions.csv>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
