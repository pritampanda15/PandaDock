#!/usr/bin/env python
"""
Evaluate a trained SAIR model, separating ligand ranking from target ranking.

A pooled Pearson r across a multi-target test set conflates two abilities:

  * telling one protein from another ("this pocket binds tightly")
  * telling one ligand from another against the same protein

Only the second is affinity prediction in the sense that matters for lead
optimisation, and it is the harder one. On SAIR's test split, 29% of label
variance lies between targets, so a predictor emitting each target's mean and
ignoring the ligand entirely scores r = 0.541 -- above what the trained model
scores pooled. Pooled r therefore cannot demonstrate ligand discrimination here.

This script reports both, plus the decomposition of the model's own predictions,
so the two claims can be stated separately and honestly.

Usage:
    python benchmarking/sair_evaluate.py \
        --cache shard_cache/ --checkpoint sair_model/best_model.pt
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MIN_LIGANDS = 20


def pearson(a: np.ndarray, b: np.ndarray):
    """Pearson r, or None when a constant input makes it undefined."""
    if a.size < 3:
        return None
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return None
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def spearman(a: np.ndarray, b: np.ndarray):
    if a.size < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return pearson(ra, rb)


def fmt(value) -> str:
    """Correlations are undefined for constant or tiny inputs; say so."""
    return "undefined" if value is None else f"{value:+.4f}"


def summarise(name: str, values: list) -> None:
    if not values:
        print(f"  {name}: no target had enough ligands")
        return
    arr = np.asarray(values, dtype=np.float64)
    print(f"  {name}")
    print(f"    median {np.median(arr):+.3f}   mean {arr.mean():+.3f}   "
          f"IQR [{np.percentile(arr, 25):+.3f}, {np.percentile(arr, 75):+.3f}]")
    print(f"    targets with r > 0.3: {100 * (arr > 0.3).mean():.1f}%"
          f"    r < 0: {100 * (arr < 0).mean():.1f}%")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the training seed for the split to line up")
    parser.add_argument("--min-ligands", type=int, default=MIN_LIGANDS,
                        help="Targets with fewer ligands are skipped for the "
                             "within-target statistic, where small samples make "
                             "the correlation too noisy to mean anything")
    parser.add_argument("--csv", default=None, help="Write per-complex predictions here")
    args = parser.parse_args(argv)

    import torch
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from pandadock.gnn.data.sair_dataset import SAIRCachedDataset, build_index
    from pandadock.gnn.models.pandadock_gnn import PandaDockGNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    index = build_index(args.cache)
    dataset = SAIRCachedDataset(
        args.cache, split=args.split, seed=args.seed, max_open_shards=18,
    )
    print(f"{args.split}: {len(dataset):,} complexes")

    # entry_id -> sequence id, so predictions can be grouped by target.
    target_of = {entry_id: seq_id for _, entry_id, seq_id in index["entries"]}

    model = PandaDockGNN.load(args.checkpoint, map_location=str(device))
    model.to(device).eval()

    loader = PyGDataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    predictions, truths, entry_ids = [], [], []
    with torch.no_grad():
        for n, batch in enumerate(loader, 1):
            batch = batch.to(device)
            output = model(batch)
            value = output["affinity"] if isinstance(output, dict) else output
            predictions.append(value.detach().float().view(-1).cpu().numpy())
            truths.append(batch.y_affinity.detach().float().view(-1).cpu().numpy())
            # entry_id is collated into a tensor and moved to the device with
            # the rest of the batch, so it has to come back before numpy sees it.
            ids = batch.entry_id
            if torch.is_tensor(ids):
                ids = ids.detach().cpu().numpy()
            entry_ids.extend(int(e) for e in np.atleast_1d(np.asarray(ids)))
            if n % 50 == 0:
                print(f"  {n * args.batch_size:,} scored", flush=True)

    predicted = np.concatenate(predictions)
    actual = np.concatenate(truths)
    targets = np.array([target_of[e] for e in entry_ids])

    if predicted.size != actual.size or predicted.size != targets.size:
        raise RuntimeError(
            f"size mismatch: {predicted.size} predictions, {actual.size} labels, "
            f"{targets.size} targets"
        )

    print("\n" + "=" * 66)
    print(f"POOLED ({args.split}, {predicted.size:,} complexes)")
    print("=" * 66)
    pooled_r = pearson(predicted, actual)
    print(f"  pearson r     {fmt(pooled_r)}")
    print(f"  spearman rho  {fmt(spearman(predicted, actual))}")
    rmse = float(np.sqrt(((predicted - actual) ** 2).mean()))
    print(f"  rmse          {rmse:.4f}   (constant predictor: {actual.std():.4f})")
    if rmse >= actual.std():
        print("                NOTE: no better than predicting the mean.")
    print(f"  mae           {float(np.abs(predicted - actual).mean()):.4f}")

    # How much of the pooled score is target ranking? Collapse both sides to
    # per-target means and correlate those: that is the part of the pooled
    # number obtainable without looking at ligands at all.
    by_target = defaultdict(lambda: ([], []))
    for p, a, t in zip(predicted, actual, targets):
        by_target[t][0].append(p)
        by_target[t][1].append(a)

    target_pred = np.array([np.mean(v[0]) for v in by_target.values()])
    target_true = np.array([np.mean(v[1]) for v in by_target.values()])

    print("\n" + "=" * 66)
    print(f"TARGET RANKING ({len(by_target):,} targets)")
    print("=" * 66)
    print("  Correlation between per-target mean prediction and per-target mean")
    print("  truth. This is the model's ability to tell proteins apart.")
    target_r = pearson(target_pred, target_true)
    print(f"\n  pearson r     {fmt(target_r)}")
    if target_r is None:
        print("  (too few targets in this split to correlate)")

    # The ceiling for a pure target-ranker on this split.
    grand = actual.mean()
    total = ((actual - grand) ** 2).sum()
    between = sum(
        len(v[1]) * (np.mean(v[1]) - grand) ** 2 for v in by_target.values()
    )
    oracle = float(np.sqrt(between / total))
    print("\n  A perfect target-mean predictor that ignores the ligand entirely")
    print(f"  would score a POOLED r of {oracle:+.4f} on this split.")
    if pooled_r is not None and pooled_r <= oracle:
        print(f"\n  Your pooled r ({pooled_r:+.4f}) is at or below that. The pooled")
        print("  number is therefore not evidence of ligand discrimination --")
        print("  report the within-target figure below instead.")

    print("\n" + "=" * 66)
    print(f"LIGAND RANKING (within target, >= {args.min_ligands} ligands)")
    print("=" * 66)
    print("  Correlation computed separately per target, then summarised. This")
    print("  is the quantity that says whether the model ranks ligands against")
    print("  a fixed protein -- the ability lead optimisation actually needs.")
    print()

    within_r, within_rho, sizes = [], [], []
    for values in by_target.values():
        p = np.asarray(values[0])
        a = np.asarray(values[1])
        if p.size < args.min_ligands:
            continue
        r = pearson(p, a)
        rho = spearman(p, a)
        if r is not None:
            within_r.append(r)
            sizes.append(p.size)
        if rho is not None:
            within_rho.append(rho)

    summarise("pearson r", within_r)
    print()
    summarise("spearman rho", within_rho)
    if sizes:
        print(f"\n  {len(sizes):,} of {len(by_target):,} targets qualified "
              f"({sum(sizes):,} complexes)")

    if within_r:
        median = float(np.median(within_r))
        print("\n" + "-" * 66)
        if median < 0.1:
            print("  The model does not rank ligands against a fixed target.")
            print("  Whatever the pooled r reflects, it is not affinity")
            print("  discrimination, and it should not be reported as such.")
        elif median < 0.3:
            print("  Weak ligand ranking. Report this number, not the pooled r,")
            print("  and state the target-ranking figure separately.")
        else:
            print("  The model ranks ligands within a target. Report this")
            print("  alongside the pooled r, not instead of it.")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["entry_id", "target_id", "predicted", "actual"])
            for e, t, p, a in zip(entry_ids, targets, predicted, actual):
                writer.writerow([e, int(t), f"{p:.4f}", f"{a:.4f}"])
        print(f"\nPer-complex predictions written to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
