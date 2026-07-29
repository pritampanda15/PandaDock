#!/usr/bin/env python
"""
Prepare downloaded PDB entries for redocking.

Takes a directory of whole PDB entries organised by protein family, selects the
dockable ligand in each, and writes receptor/ligand pairs plus a manifest CSV that
`redock_benchmark.py` consumes.

Entries are rejected -- with a recorded reason -- when they are NMR ensembles
(no crystallographic pose to reproduce), apo structures, or contain nothing that
looks like a bound ligand. The rejection reasons are written out alongside the
manifest so the excluded fraction is visible rather than silently absorbed into
the accuracy figures.

Usage:
    python benchmarking/prepare_complexes.py \
        --input complexes/benchmark_data --output benchmark_prepared
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from pandadock.preprocessing.complex_splitter import (  # noqa: E402
    ComponentCache,
    collect_het_codes,
    split_complex,
)

logger = logging.getLogger("pandadock.benchmark.prepare")

FAMILY_LABELS = {
    "kinases": "Protein kinases",
    "gpcrs": "GPCRs",
    "nuclear_receptors": "Nuclear receptors",
    "proteases": "Proteases",
    "ion_channels": "Ion channels",
    "transporters": "Transporters",
    "epigenetic": "Epigenetic enzymes",
    "phosphatases": "Phosphatases",
    "e3_ligases": "E3 ligases",
    "chaperones": "Molecular chaperones",
    "ppi": "PPI targets",
    "metalloenzymes": "Metalloenzymes",
    "oxidoreductases": "Oxidoreductases",
    "glycosidases": "Glycosidases",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory of <family>/<pdbid>.pdb files")
    parser.add_argument("--output", type=Path, default=Path("benchmark_prepared"))
    parser.add_argument("--min-heavy-atoms", type=int, default=6)
    parser.add_argument("--max-heavy-atoms", type=int, default=120)
    parser.add_argument("--keep-nmr", action="store_true",
                        help="Keep model 1 of NMR ensembles (not recommended: an "
                             "NMR model is not a crystallographic pose)")
    parser.add_argument("--offline", action="store_true",
                        help="Do not contact RCSB; rely on the existing cache")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")

    family_dirs = sorted(
        p for p in args.input.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not family_dirs:
        print(f"No family directories under {args.input}", file=sys.stderr)
        return 1

    all_pdbs = [(d, p) for d in family_dirs for p in sorted(d.glob("*.pdb"))]
    print(f"Found {len(all_pdbs)} entries across {len(family_dirs)} families")

    args.output.mkdir(parents=True, exist_ok=True)
    cache = ComponentCache(args.output / "component_cache.json")

    if not args.offline:
        codes = collect_het_codes(p for _, p in all_pdbs)
        print(f"Resolving {len(codes)} distinct chemical components against the CCD...")
        cache.fetch_missing(codes)

    rows: List[Dict] = []
    skipped: List[Dict] = []
    reasons: Counter = Counter()

    for index, (family_dir, pdb_path) in enumerate(all_pdbs, start=1):
        family = FAMILY_LABELS.get(family_dir.name, family_dir.name)
        out_dir = args.output / family_dir.name

        try:
            receptor, ligand, reason, meta = split_complex(
                pdb_path, out_dir, cache,
                min_heavy_atoms=args.min_heavy_atoms,
                max_heavy_atoms=args.max_heavy_atoms,
                reject_nmr=not args.keep_nmr,
            )
        except Exception as exc:
            receptor, ligand, reason, meta = None, None, f"{type(exc).__name__}: {exc}", {}

        if receptor is None:
            reasons[reason.split("(")[0].strip()] += 1
            skipped.append({"id": pdb_path.stem, "family": family, "reason": reason})
        else:
            rows.append({
                "id": pdb_path.stem,
                "receptor": str(receptor.resolve()),
                "ligand": str(ligand.resolve()),
                "family": family,
                "ligand_code": meta.get("ligand_code", ""),
                "n_heavy_atoms": meta.get("n_heavy_atoms", ""),
                "contacts": meta.get("contacts", ""),
                "ligand_name": meta.get("ligand_name", "")[:60],
            })

        if index % 100 == 0:
            print(f"  {index}/{len(all_pdbs)} processed, {len(rows)} prepared", flush=True)

    manifest = args.output / "manifest.csv"
    with open(manifest, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "receptor", "ligand", "family", "ligand_code",
                        "n_heavy_atoms", "contacts", "ligand_name"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(args.output / "skipped.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "family", "reason"])
        writer.writeheader()
        writer.writerows(skipped)

    per_family = defaultdict(lambda: [0, 0])
    for _, pdb_path in all_pdbs:
        pass
    for family_dir, _ in all_pdbs:
        per_family[FAMILY_LABELS.get(family_dir.name, family_dir.name)][0] += 1
    for row in rows:
        per_family[row["family"]][1] += 1

    print("\n" + "=" * 62)
    print(f"{'Family':<24}{'Entries':>9}{'Prepared':>10}{'Kept %':>9}")
    print("-" * 62)
    for family in sorted(per_family):
        total, kept = per_family[family]
        print(f"{family:<24}{total:>9}{kept:>10}{100 * kept / max(total, 1):>8.0f}%")
    print("-" * 62)
    total = len(all_pdbs)
    print(f"{'TOTAL':<24}{total:>9}{len(rows):>10}{100 * len(rows) / max(total, 1):>8.0f}%")
    print("=" * 62)

    print("\nWhy entries were excluded:")
    for reason, count in reasons.most_common():
        print(f"  {count:>5}  {reason}")

    ligand_codes = Counter(r["ligand_code"] for r in rows)
    print(f"\nMost frequent selected ligands (of {len(ligand_codes)} distinct):")
    for code, count in ligand_codes.most_common(12):
        print(f"  {code:<6} {count:>4}")

    json.dump(
        {"prepared": len(rows), "attempted": total,
         "per_family": {k: v for k, v in per_family.items()},
         "exclusion_reasons": dict(reasons)},
        open(args.output / "prepare_summary.json", "w"), indent=2,
    )

    print(f"\nManifest: {manifest}  ({len(rows)} complexes)")
    print(f"Excluded: {args.output / 'skipped.csv'}  ({len(skipped)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
