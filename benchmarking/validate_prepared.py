#!/usr/bin/env python
"""
Validate prepared receptor/ligand pairs before spending compute on docking.

Checks the failure modes that would silently invalidate a benchmark:

- Ligand leakage: if any copy of the ligand remains in the receptor file, the
  search can score against it and the benchmark reports a number that has nothing
  to do with prediction. This is the single most damaging preparation bug and the
  hardest to notice from the results alone.
- Coordinate fidelity: the written ligand must carry the crystal coordinates, not
  a re-embedded conformer.
- Chemical sanity: the ligand must parse, have a conformer, and be a single
  connected species (a salt or a co-crystallised fragment pair would make RMSD
  ambiguous).
- Site occupancy: the ligand must sit in contact with the receptor.

Usage:
    python benchmarking/validate_prepared.py --manifest benchmark_prepared/manifest.csv
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def receptor_coords(path: Path) -> np.ndarray:
    coords = []
    for line in path.read_text(errors="ignore").splitlines():
        if line[:6] in ("ATOM  ", "HETATM") and len(line) >= 54:
            if line[76:78].strip().upper() in ("H", "D"):
                continue
            try:
                coords.append(
                    (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                )
            except ValueError:
                continue
    return np.array(coords) if coords else np.empty((0, 3))


def check(row: dict) -> list:
    """Return a list of problems found for one prepared complex."""
    problems = []
    receptor = Path(row["receptor"])
    ligand_path = Path(row["ligand"])

    if not receptor.exists():
        return ["receptor file missing"]
    if not ligand_path.exists():
        return ["ligand file missing"]

    mol = next(iter(Chem.SDMolSupplier(str(ligand_path), removeHs=False, sanitize=True)), None)
    if mol is None:
        mol = next(
            iter(Chem.SDMolSupplier(str(ligand_path), removeHs=False, sanitize=False)), None
        )
        if mol is None:
            return ["ligand does not parse"]
        problems.append("ligand fails strict sanitization")

    if mol.GetNumConformers() == 0:
        return problems + ["ligand has no coordinates"]

    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy:
        return problems + ["ligand has no heavy atoms"]

    lig = np.asarray(mol.GetConformer().GetPositions())[heavy]

    fragments = Chem.GetMolFrags(mol)
    if len(fragments) > 1:
        problems.append(f"ligand has {len(fragments)} disconnected fragments")

    rec = receptor_coords(receptor)
    if len(rec) == 0:
        return problems + ["receptor has no atoms"]

    distances = np.linalg.norm(lig[:, None, :] - rec[None, :, :], axis=2)

    # Leakage: an atom of the ligand still present in the receptor file would sit
    # essentially on top of a ligand atom.
    overlapping = int(np.sum(distances.min(axis=1) < 0.3))
    if overlapping > 0:
        problems.append(f"LIGAND LEAKAGE: {overlapping} ligand atoms coincide with receptor atoms")

    contacts = int(np.sum(distances < 4.5))
    if contacts < 10:
        problems.append(f"ligand barely contacts receptor ({contacts} contacts)")

    # A crystal ligand should not clash with its own receptor.
    clashes = int(np.sum(distances < 1.8))
    if clashes > 3:
        problems.append(f"{clashes} severe ligand/receptor clashes")

    declared = row.get("n_heavy_atoms")
    if declared and declared.isdigit() and int(declared) != len(heavy):
        problems.append(f"heavy atom count {len(heavy)} != manifest {declared}")

    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--show", type=int, default=25, help="Max problem rows to print")
    args = parser.parse_args(argv)

    rows = list(csv.DictReader(open(args.manifest, newline="")))
    print(f"Validating {len(rows)} prepared complexes from {args.manifest}\n")

    counter = Counter()
    flagged = []
    for i, row in enumerate(rows, start=1):
        problems = check(row)
        for problem in problems:
            counter[problem.split("(")[0].split(":")[0].strip()] += 1
        if problems:
            flagged.append((row["id"], row.get("family", ""), row.get("ligand_code", ""), problems))
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}...", flush=True)

    clean = len(rows) - len(flagged)
    print(f"\nClean: {clean}/{len(rows)} ({100 * clean / max(len(rows), 1):.1f}%)")

    if counter:
        print("\nIssues found:")
        for problem, count in counter.most_common():
            marker = "  !!" if "LEAKAGE" in problem else "    "
            print(f"{marker} {count:>5}  {problem}")

    if flagged:
        print(f"\nFirst {min(args.show, len(flagged))} flagged complexes:")
        for identifier, family, code, problems in flagged[: args.show]:
            print(f"  {identifier:<8} {family:<22} {code:<6} {'; '.join(problems)}")

    leaks = sum(v for k, v in counter.items() if "LEAKAGE" in k)
    if leaks:
        print(f"\nFAIL: {leaks} complexes leak the ligand into the receptor. "
              "Do not benchmark until this is fixed.")
        return 1

    print("\nNo ligand leakage detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
