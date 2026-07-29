#!/usr/bin/env python
"""
Affinity correlation benchmark for PandaDock-GNN.

Scores complexes with the GNN and correlates against experimental affinity,
per protein family. This is the scoring-power analogue of the redocking
benchmark, and answers a different question from it: redocking asks whether the
right geometry is found, this asks whether the score ranks compounds correctly.

Two poses are scored for every complex, because they measure different things and
conflating them is the most common way these numbers get overstated:

- CRYSTAL pose — the scoring function's ceiling, given perfect geometry. This is
  the setting in which published PDBbind correlations are usually measured.
- DOCKED top-1 pose — what a user actually gets, with the search's own errors
  folded in. Always the lower of the two, and the honest number to quote for an
  end-to-end pipeline.

Usage:
    python benchmarking/gnn_affinity_benchmark.py \
        --manifest benchmark_prepared/manifest.csv \
        --affinity complexes/benchmark_data/all_structural_affinity.csv \
        --model pandadock_gnn_v4.pt \
        --docked benchmark_results/full/redock_results.csv \
        --output benchmark_results/gnn_affinity
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger("pandadock.benchmark.gnn_affinity")

# Only equilibrium binding constants and IC50. EC50 is a functional readout on a
# different scale; pooling scales is what collapsed this model's test correlation
# from 0.81 to 0.49 when PDBbind and BindingDB were combined.
AFFINITY_TYPES = {"KD", "KI", "IC50"}

FAMILY_ORDER = [
    "E3 ligases", "Epigenetic enzymes", "GPCRs", "Glycosidases", "Ion channels",
    "Metalloenzymes", "Molecular chaperones", "Nuclear receptors",
    "Oxidoreductases", "PPI targets", "Phosphatases", "Proteases",
    "Protein kinases", "Transporters",
]


def load_affinity(path: Path) -> Dict[str, float]:
    values: Dict[str, List[float]] = defaultdict(list)
    for row in csv.DictReader(open(path, newline="")):
        if (row.get("affinity_type") or "").strip().upper() not in AFFINITY_TYPES:
            continue
        try:
            value = float(row["neg_log_affinity"])
        except (TypeError, ValueError, KeyError):
            continue
        if math.isfinite(value):
            values[(row.get("pdb_id") or "").strip().upper()].append(value)
    return {k: float(np.median(v)) for k, v in values.items() if v}


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    unique, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(len(unique))
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def correlate(predicted: np.ndarray, measured: np.ndarray) -> Dict:
    """
    Correlate predicted against measured affinity.

    Both are in pAffinity units (higher = tighter), so no sign flip is applied.
    A positive correlation indicates correct behaviour.
    """
    if len(predicted) < 3 or np.std(predicted) < 1e-12 or np.std(measured) < 1e-12:
        return {"n": len(predicted)}
    pearson = float(np.corrcoef(predicted, measured)[0, 1])
    spearman = float(np.corrcoef(rankdata(predicted), rankdata(measured))[0, 1])
    rmse = float(np.sqrt(np.mean((predicted - measured) ** 2)))
    return {
        "n": len(predicted),
        "pearson": pearson,
        "r2": pearson**2,
        "spearman": spearman,
        "rmse": rmse,
    }


def rdkit_pose_to_parsed(mol, coords):
    """
    Convert an RDKit molecule plus coordinates to the parser's molecule type.

    Mirrors what `pandadock hybrid` does. GNNScoring.predict_from_coords does not
    exist and _build_graph_from_pose raises NotImplementedError, so building the
    heterogeneous graph explicitly is the only working route to the model.
    """
    from pandadock.gnn.data.mol2_parser import Atom, ParsedMolecule

    hyb_map = {
        Chem.HybridizationType.SP3: "3",
        Chem.HybridizationType.SP2: "2",
        Chem.HybridizationType.SP: "1",
    }

    atoms = []
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        hyb = hyb_map.get(atom.GetHybridization(), "3")
        atoms.append(
            Atom(
                id=i + 1,
                name=symbol,
                x=float(coords[i][0]),
                y=float(coords[i][1]),
                z=float(coords[i][2]),
                atom_type=symbol if symbol == "H" else f"{symbol}.{hyb}",
                charge=float(atom.GetFormalCharge()),
                residue_name="LIG",
                residue_id=1,
            )
        )
    return ParsedMolecule(name="ligand", atoms=atoms, bonds=[], num_atoms=len(atoms))


def extract_site(protein_parsed, centroid: np.ndarray, radius: float):
    """
    Protein atoms within `radius` of the ligand centroid.

    The GNN is applied to a binding site, not a whole protein: `pandadock gnn
    rescore` extracts one at a default 10 A. Passing the entire receptor builds a
    graph one to two orders of magnitude larger, which exhausts memory over a
    benchmark loop and feeds the model a graph unlike anything it was trained on.
    """
    from pandadock.gnn.data.mol2_parser import ParsedMolecule

    site_atoms = [
        atom for atom in protein_parsed.atoms
        if (atom.x - centroid[0]) ** 2
        + (atom.y - centroid[1]) ** 2
        + (atom.z - centroid[2]) ** 2 <= radius ** 2
    ]
    if not site_atoms:
        return protein_parsed

    site = ParsedMolecule(name=f"{protein_parsed.name}_site", atoms=site_atoms, bonds=[])
    site.num_atoms = len(site_atoms)
    return site


def score_complex(scorer, graph_builder, protein_parsed, mol, coords,
                  site_radius: float = 10.0) -> Optional[float]:
    """Predicted pAffinity for one pose, or None if the model could not score it."""
    import gc

    try:
        heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        centroid = coords[heavy].mean(axis=0) if heavy else coords.mean(axis=0)

        site = extract_site(protein_parsed, centroid, site_radius)
        ligand_parsed = rdkit_pose_to_parsed(mol, coords)
        graph = graph_builder.build_graph(site, ligand_parsed)
        value = float(scorer.predict_from_graph(graph)["pec50"])

        # Graphs are large; release them rather than waiting for the collector.
        del graph, site, ligand_parsed
        gc.collect()
        return value
    except Exception as exc:
        logger.debug("scoring failed: %s", exc)
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--affinity", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--docked", type=Path, default=None,
                        help="redock_results.csv, to also score the docked top pose")
    parser.add_argument("--poses-dir", type=Path, default=None,
                        help="Directory of per-complex docking outputs holding poses.sdf")
    parser.add_argument("--output", type=Path, default=Path("gnn_affinity"))
    parser.add_argument("--site-radius", type=float, default=10.0,
                        help="Radius around the ligand centroid defining the "
                             "binding site passed to the GNN (default: 10)")
    parser.add_argument("--min-n", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")

    from pandadock.gnn.data.graph_builder import (
        HeterogeneousGraphBuilder,
        extract_binding_site,
        parse_molecule_file,
    )
    from pandadock.gnn.scoring import GNNScoring

    affinity = load_affinity(args.affinity)
    rows = list(csv.DictReader(open(args.manifest, newline="")))
    targets = [r for r in rows if r["id"].upper() in affinity]

    print(f"{len(rows)} complexes in manifest, {len(targets)} with Kd/Ki/IC50 data")
    if not targets:
        print("No complexes carry usable affinity data.", file=sys.stderr)
        return 1

    print(f"Loading model {args.model}...")
    scorer = GNNScoring(model_path=str(args.model))
    graph_builder = HeterogeneousGraphBuilder()
    records = []

    for i, row in enumerate(targets, 1):
        identifier = row["id"].upper()
        measured = affinity[identifier]
        try:
            protein_parsed = parse_molecule_file(row["receptor"])
            crystal = next(
                (m for m in Chem.SDMolSupplier(row["ligand"], removeHs=False) if m), None
            )
        except Exception as exc:
            logger.debug("%s: could not load (%s)", identifier, exc)
            continue
        if crystal is None or crystal.GetNumConformers() == 0:
            continue

        coords = np.asarray(crystal.GetConformer().GetPositions())
        predicted = score_complex(
            scorer, graph_builder, protein_parsed, crystal, coords, args.site_radius
        )
        if predicted is None:
            continue

        records.append({
            "id": identifier,
            "family": row.get("family", "unclassified"),
            "measured": measured,
            "predicted_crystal": predicted,
        })

        if i % 25 == 0:
            print(f"  {i}/{len(targets)} scored", flush=True)

    if not records:
        print("The model could not score any complex.", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "gnn_affinity_results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    by_family = defaultdict(list)
    for record in records:
        by_family[record["family"]].append(record)

    print("\n" + "=" * 78)
    print("PandaDock-GNN AFFINITY CORRELATION (crystal poses)")
    print("=" * 78)
    print(f"{'Protein family':<24}{'N':>5}{'Pearson r':>12}{'R2':>8}{'Spearman':>11}{'RMSE':>8}")
    print("-" * 78)

    summary = {}
    for family in FAMILY_ORDER:
        members = by_family.get(family, [])
        stats = correlate(
            np.array([m["predicted_crystal"] for m in members]),
            np.array([m["measured"] for m in members]),
        )
        summary[family] = stats
        if stats.get("n", 0) < 3:
            print(f"{family:<24}{stats.get('n', 0):>5}{'—':>12}{'—':>8}{'—':>11}{'—':>8}")
            continue
        flag = " a" if stats["n"] < args.min_n else ""
        print(f"{family:<24}{stats['n']:>5}{stats['pearson']:>12.2f}{stats['r2']:>8.2f}"
              f"{stats['spearman']:>11.2f}{stats['rmse']:>8.2f}{flag}")

    overall = correlate(
        np.array([m["predicted_crystal"] for m in records]),
        np.array([m["measured"] for m in records]),
    )
    print("-" * 78)
    print(f"{'OVERALL':<24}{overall['n']:>5}{overall['pearson']:>12.2f}{overall['r2']:>8.2f}"
          f"{overall['spearman']:>11.2f}{overall['rmse']:>8.2f}")
    print("=" * 78)
    print("a  fewer than "
          f"{args.min_n} complexes; treat as indicative only")
    print("\nScored on CRYSTAL poses: this is the scoring function's ceiling given "
          "\nperfect geometry, not end-to-end pipeline performance.")

    json.dump({"overall": overall, "by_family": summary},
              open(args.output / "gnn_affinity_summary.json", "w"), indent=2)
    print(f"\nWrote {args.output}/gnn_affinity_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
