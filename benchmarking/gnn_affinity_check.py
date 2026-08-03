#!/usr/bin/env python
"""
Check a SAIR-trained model against experimental affinities and against pose.

Two questions the SAIR test split cannot answer:

1. Does it transfer? The model was trained on co-folded predicted structures.
   These complexes are experimental crystal structures with measured Ki, Kd,
   IC50 or EC50. If the correlation collapses here, the model learned something
   specific to the co-folding artefacts rather than to binding.

2. Does the pose matter? A scoring function that returns the same number for a
   native pose and one displaced several angstroms is not reading interactions,
   and cannot be used to rank docked poses -- which is the only reason to put a
   structure-based model in a docking pipeline at all. Displacing the ligand and
   rescoring answers this directly, and needs no docking run.

Both use the same site extraction and hydrogen handling as inference, so what is
measured is what a user would get.

Usage:
    python benchmarking/gnn_affinity_check.py \
        --model sair_ranked/best_model.pt \
        --prepared benchmark_prepared/ \
        --affinity complexes/benchmark_data/all_structural_affinity.csv \
        --strip-hydrogens
"""

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarking.sair_evaluate import pearson, spearman  # noqa: E402


def find_complexes(prepared: str):
    """{pdb_id: (receptor, ligand, family)} for every prepared pair."""
    found = {}
    for family in sorted(os.listdir(prepared)):
        directory = os.path.join(prepared, family)
        if not os.path.isdir(directory):
            continue
        for receptor in sorted(glob.glob(os.path.join(directory, "*_receptor.pdb"))):
            ligand = receptor.replace("_receptor.pdb", "_ligand.sdf")
            if os.path.exists(ligand):
                pdb_id = os.path.basename(receptor).split("_")[0].lower()
                found[pdb_id] = (receptor, ligand, family)
    return found


def load_affinities(path: str):
    """{pdb_id: (pAffinity, type, family)}, keeping the first row per structure."""
    out = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            pdb_id = row["pdb_id"].lower()
            if pdb_id in out:
                continue
            try:
                out[pdb_id] = (
                    float(row["neg_log_affinity"]), row["affinity_type"], row["family"]
                )
            except (KeyError, ValueError):
                continue
    return out


def displace(molecule, offset):
    """Copy of the ligand translated by `offset`, leaving the original intact."""
    from dataclasses import replace

    from pandadock.gnn.data.mol2_parser import ParsedMolecule

    moved = [
        replace(a, x=a.x + offset[0], y=a.y + offset[1], z=a.z + offset[2])
        for a in molecule.atoms
    ]
    out = ParsedMolecule(name=molecule.name, atoms=moved, bonds=[])
    out.num_atoms = len(moved)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prepared", default="benchmark_prepared")
    parser.add_argument("--affinity",
                        default="complexes/benchmark_data/all_structural_affinity.csv")
    parser.add_argument("--site-radius", type=float, default=10.0)
    parser.add_argument("--strip-hydrogens", action="store_true",
                        help="Required for SAIR-trained models")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--displacements", default="0,1,2,3,5",
                        help="Angstrom offsets for the pose-sensitivity test")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import torch

    from pandadock.gnn.data.graph_builder import (
        GraphConfig, HeterogeneousGraphBuilder, drop_hydrogens,
        extract_binding_site, parse_molecule_file,
    )
    from pandadock.gnn.models.pandadock_gnn import PandaDockGNN

    structures = find_complexes(args.prepared)
    affinities = load_affinities(args.affinity)
    shared = sorted(set(structures) & set(affinities))
    if args.limit:
        shared = shared[: args.limit]

    print(f"{len(structures):,} prepared complexes, {len(affinities):,} with "
          f"affinity, {len(shared):,} usable")
    if not shared:
        print("Nothing to score.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PandaDockGNN.load(args.model, map_location=str(device))
    model.to(device).eval()
    builder = HeterogeneousGraphBuilder(GraphConfig())
    offsets = [float(v) for v in args.displacements.split(",")]
    rng = np.random.default_rng(args.seed)

    def score(site, ligand):
        graph = builder.build_graph(site, ligand).to(device)
        with torch.no_grad():
            output = model(graph)
        value = output["affinity"] if isinstance(output, dict) else output
        return float(value.view(-1)[0])

    rows = []
    shifts = defaultdict(list)
    failures = defaultdict(int)

    for n, pdb_id in enumerate(shared, 1):
        receptor_path, ligand_path, family = structures[pdb_id]
        measured, kind, _ = affinities[pdb_id]
        try:
            receptor = parse_molecule_file(receptor_path)
            ligand = parse_molecule_file(ligand_path)
            if args.strip_hydrogens:
                receptor = drop_hydrogens(receptor)
                ligand = drop_hydrogens(ligand)
            if not ligand.atoms or not receptor.atoms:
                failures["empty"] += 1
                continue

            centroid = np.array(
                [[a.x, a.y, a.z] for a in ligand.atoms], dtype=float
            ).mean(axis=0)
            site = extract_binding_site(receptor, centroid, radius=args.site_radius)

            native = score(site, ligand)
            rows.append({
                "pdb_id": pdb_id, "family": family, "type": kind,
                "measured": measured, "predicted": native,
                "site_atoms": len(site.atoms), "ligand_atoms": len(ligand.atoms),
            })

            # Pose sensitivity: same complex, ligand pushed off its crystal pose.
            for distance in offsets:
                if distance == 0:
                    shifts[distance].append(0.0)
                    continue
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                moved = displace(ligand, direction * distance)
                shifts[distance].append(abs(score(site, moved) - native))
        except Exception as exc:
            failures[type(exc).__name__] += 1
        if n % 25 == 0:
            print(f"  {n}/{len(shared)}", flush=True)

    if not rows:
        print("Every complex failed.")
        for name, count in failures.items():
            print(f"  {count:>4} {name}")
        return 1

    predicted = np.array([r["predicted"] for r in rows])
    measured = np.array([r["measured"] for r in rows])

    print("\n" + "=" * 66)
    print(f"EXPERIMENTAL AFFINITY ({len(rows)} crystal complexes)")
    print("=" * 66)
    print(f"  pearson r     {pearson(predicted, measured):+.4f}")
    print(f"  spearman rho  {spearman(predicted, measured):+.4f}")
    rmse = float(np.sqrt(((predicted - measured) ** 2).mean()))
    print(f"  rmse          {rmse:.4f}   (constant predictor: {measured.std():.4f})")
    print(f"  mae           {float(np.abs(predicted - measured).mean()):.4f}")
    print(f"  predicted range {predicted.min():.2f} to {predicted.max():.2f}"
          f"   measured {measured.min():.2f} to {measured.max():.2f}")

    if rmse >= measured.std():
        print("\n  No better than predicting the mean of this set.")

    print("\n  By family:")
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append((row["predicted"], row["measured"]))
    for family, pairs in sorted(by_family.items()):
        if len(pairs) < 5:
            continue
        p = np.array([v[0] for v in pairs])
        m = np.array([v[1] for v in pairs])
        r = pearson(p, m)
        print(f"    {family:<22} n={len(pairs):>3}  "
              f"r={'undefined' if r is None else f'{r:+.3f}'}")

    print("\n" + "=" * 66)
    print("POSE SENSITIVITY")
    print("=" * 66)
    print("  Ligand displaced from its crystal pose, complex rescored.")
    print("  A model reading interactions should change its answer.\n")
    for distance in offsets:
        values = np.array(shifts[distance])
        if values.size == 0:
            continue
        print(f"    {distance:>4.1f} A   mean |delta pAffinity| {values.mean():.4f}"
              f"   median {np.median(values):.4f}   max {values.max():.4f}")

    largest = np.array(shifts[offsets[-1]]) if offsets else np.array([])
    if largest.size and largest.mean() < 0.1:
        print(f"\n  Displacing the ligand {offsets[-1]:.0f} A changes the score by")
        print(f"  {largest.mean():.3f} on average. The model is close to ignoring")
        print("  the pose, so it cannot rank docked poses and its output is")
        print("  effectively a function of composition, not of the complex.")

    if failures:
        print("\n  Failures:")
        for name, count in sorted(failures.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>4} {name}")

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-complex results written to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
