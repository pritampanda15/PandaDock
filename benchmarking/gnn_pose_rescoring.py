#!/usr/bin/env python
"""
Does GNN rescoring pick better poses than the scoring function that made them?

Displacing a crystal ligand showed the model prefers native geometry, but rigid
translation is a soft decoy: it reduces contact count in a way that is easy to
notice. Real docking failures are in-pocket flips and rotations that keep the
contact count roughly intact. This runs PandaCore for real, takes the poses it
produces, and asks whether the GNN can identify the good ones.

Three selections are compared per complex:

  Vina    the pose PandaCore itself ranks first -- the current behaviour
  GNN     the pose the GNN scores highest, rescoring the same set
  oracle  the lowest-RMSD pose available, which bounds what any rescorer
          could achieve on this pose set

Rescoring is only worth adding to the pipeline if GNN beats Vina. If both sit
far below oracle, the poses are there and the scoring is what fails; if oracle
itself is poor, the search never found the right pose and no rescorer can help.

The binding site is cut once per complex around the crystal ligand and reused
for every pose, so the comparison isolates ligand geometry rather than also
varying which protein atoms are visible.

Usage:
    python benchmarking/gnn_pose_rescoring.py \
        --model sair_ranked/best_model.pt \
        --prepared benchmark_prepared/ --limit 50 --strip-hydrogens
"""

import argparse
import csv
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from benchmarking.gnn_affinity_check import find_complexes  # noqa: E402
from benchmarking.redock_benchmark import (  # noqa: E402
    box_from_ligand, load_ligand, mol_with_coords, rebuild_from_topology,
)
from benchmarking.sair_evaluate import pearson, spearman  # noqa: E402
from pandadock.analysis.rmsd import symmetry_corrected_rmsd  # noqa: E402


def dock_one(task):
    """
    Dock one complex and score its poses against the crystal geometry.

    Runs in a worker process and deliberately imports no torch: docking
    dominates the runtime, and loading a copy of the model per worker would
    cost memory for nothing. Only coordinates come back; the GNN scores them
    in the parent.
    """
    pdb_id, receptor_path, ligand_path, family, options = task

    from pandadock.docking.algorithms import PandaCoreDocker
    from pandadock.docking.scoring.vina_scoring import VinaScoring

    try:
        crystal = load_ligand(Path(ligand_path))
        if crystal is None:
            return pdb_id, None, "unreadable_ligand"
        start_mol = rebuild_from_topology(crystal)
        if start_mol is None:
            return pdb_id, None, "topology_rebuild"

        center, dimensions = box_from_ligand(crystal, options["padding"])
        docker = PandaCoreDocker()
        docker.set_scoring_function(VinaScoring())
        result = docker.dock(
            receptor_file=receptor_path,
            ligand_mol=start_mol,
            grid_center=center,
            grid_dimensions=dimensions,
            num_poses=options["num_poses"],
            exhaustiveness=options["exhaustiveness"],
            seed=options["seed"],
        )
        if not result.poses:
            return pdb_id, None, "no_poses"

        rmsds, coords, vina = [], [], []
        for pose in result.poses:
            pose_mol = mol_with_coords(start_mol, pose.coordinates)
            try:
                rmsds.append(symmetry_corrected_rmsd(pose_mol, crystal))
            except Exception:
                rmsds.append(float("nan"))
            coords.append(np.asarray(pose.coordinates, dtype=float))
            # Pose exposes `energy`, not `score`.
            vina.append(float(pose.energy))

        heavy = np.array([
            list(crystal.GetConformer().GetAtomPosition(i))
            for i in range(crystal.GetNumAtoms())
            if crystal.GetAtomWithIdx(i).GetAtomicNum() > 1
        ])
        return pdb_id, {
            "family": family, "receptor": receptor_path, "ligand": ligand_path,
            "rmsds": rmsds, "coords": coords, "vina": vina,
            "centroid": heavy.mean(axis=0),
        }, None
    except Exception as exc:
        # Message included: a bare type name cost a full docking round to
        # diagnose an attribute that simply had a different name.
        return pdb_id, None, f"{type(exc).__name__}: {exc}"


def summarise(name: str, rmsds, threshold: float) -> None:
    values = np.asarray(rmsds, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        print(f"  {name:<8} no result")
        return
    print(f"  {name:<8} median {np.median(values):5.2f}   mean {values.mean():5.2f}"
          f"   <= {threshold:.0f} A: {100 * (values <= threshold).mean():5.1f}%"
          f"   <= 1 A: {100 * (values <= 1.0).mean():5.1f}%")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prepared", default="benchmark_prepared")
    parser.add_argument("--limit", type=int, default=50,
                        help="Docking dominates the runtime; start small")
    parser.add_argument("--num-poses", type=int, default=20)
    parser.add_argument("--exhaustiveness", type=int, default=None)
    parser.add_argument("--padding", type=float, default=5.0)
    parser.add_argument("--site-radius", type=float, default=10.0)
    parser.add_argument("--strip-hydrogens", action="store_true",
                        help="Required for SAIR-trained models")
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel docking processes. Docking dominates the "
                             "runtime and the GNN scores in the parent, so this "
                             "scales close to linearly")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args(argv)

    if not os.path.exists(args.prepared):
        print(f"--prepared not found: {args.prepared}")
        print("See gnn_affinity_check.py --pack to move the benchmark subset.")
        return 1

    import torch

    from pandadock.docking.algorithms import PandaCoreDocker
    from pandadock.docking.scoring.vina_scoring import VinaScoring
    from pandadock.gnn.data.graph_builder import (
        GraphConfig, HeterogeneousGraphBuilder, drop_hydrogens,
        extract_binding_site, parse_molecule_file,
    )
    from pandadock.gnn.models.pandadock_gnn import PandaDockGNN

    structures = find_complexes(args.prepared)
    # Sampled, not taken alphabetically. The first identifiers happen to include
    # some of the largest ligands in the set -- one took eight minutes to dock on
    # its own -- so a prefix is both slow and unrepresentative.
    order = np.random.default_rng(args.seed).permutation(sorted(structures))
    chosen = sorted(order[: args.limit].tolist())
    print(f"{len(structures):,} prepared complexes, docking a sample of "
          f"{len(chosen)} (seed {args.seed})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PandaDockGNN.load(args.model, map_location=str(device))
    model.to(device).eval()
    builder = HeterogeneousGraphBuilder(GraphConfig())
    print(f"Device: {device}")

    def gnn_score(site, ligand):
        graph = builder.build_graph(site, ligand).to(device)
        with torch.no_grad():
            output = model(graph)
        value = output["affinity"] if isinstance(output, dict) else output
        return float(value.view(-1)[0])

    rows = []
    selections = defaultdict(list)
    per_complex_rho = []
    failures = defaultdict(int)
    started = time.time()

    options = {
        "padding": args.padding, "num_poses": args.num_poses,
        "exhaustiveness": args.exhaustiveness, "seed": args.seed,
    }
    tasks = [
        (pdb_id, structures[pdb_id][0], structures[pdb_id][1],
         structures[pdb_id][2], options)
        for pdb_id in chosen
    ]

    docked = {}
    print(f"\nDocking with {args.workers} worker process(es)...")
    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(dock_one, task) for task in tasks]
            for n, future in enumerate(as_completed(futures), 1):
                pdb_id, payload, error = future.result()
                if payload is None:
                    failures[error or "unknown"] += 1
                else:
                    docked[pdb_id] = payload
                rate = (time.time() - started) / n
                print(f"  docked {n}/{len(tasks)}  ({rate:.0f} s/complex, "
                      f"eta {rate * (len(tasks) - n) / 60:.0f} min)", flush=True)
    else:
        for n, task in enumerate(tasks, 1):
            pdb_id, payload, error = dock_one(task)
            if payload is None:
                failures[error or "unknown"] += 1
            else:
                docked[pdb_id] = payload
            rate = (time.time() - started) / n
            print(f"  docked {n}/{len(tasks)}  ({rate:.0f} s/complex, "
                  f"eta {rate * (len(tasks) - n) / 60:.0f} min)", flush=True)

    print(f"\nScoring {len(docked)} complexes with the GNN...")
    with tempfile.TemporaryDirectory() as tmp:
        for pdb_id, payload in sorted(docked.items()):
            try:
                crystal = load_ligand(Path(payload["ligand"]))
                start_mol = rebuild_from_topology(crystal)
                if start_mol is None:
                    failures["topology_rebuild"] += 1
                    continue

                receptor = parse_molecule_file(payload["receptor"])
                if args.strip_hydrogens:
                    receptor = drop_hydrogens(receptor)
                site = extract_binding_site(
                    receptor, payload["centroid"], radius=args.site_radius
                )

                gnn_values = []
                for index, coords in enumerate(payload["coords"]):
                    pose_mol = mol_with_coords(start_mol, coords)
                    # Routed through an SDF so the ligand reaches the model by
                    # the same parser inference uses.
                    path = os.path.join(tmp, f"{pdb_id}_{index}.sdf")
                    writer = Chem.SDWriter(path)
                    writer.write(pose_mol)
                    writer.close()
                    ligand = parse_molecule_file(path)
                    if args.strip_hydrogens:
                        ligand = drop_hydrogens(ligand)
                    gnn_values.append(gnn_score(site, ligand))
                    os.remove(path)

                rmsds = np.array(payload["rmsds"], dtype=float)
                gnn_values = np.array(gnn_values)
                vina_values = np.array(payload["vina"])
                usable = ~np.isnan(rmsds)
                if usable.sum() < 2:
                    failures["rmsd_failed"] += 1
                    continue

                # Higher GNN score should mean lower RMSD, so a working
                # rescorer gives a negative rank correlation.
                rho = spearman(gnn_values[usable], rmsds[usable])
                if rho is not None:
                    per_complex_rho.append(rho)

                # Vina scores are energies: lower is better.
                vina_pick = int(np.argmin(vina_values))
                gnn_pick = int(np.argmax(gnn_values))
                oracle = int(np.nanargmin(rmsds))

                selections["vina"].append(rmsds[vina_pick])
                selections["gnn"].append(rmsds[gnn_pick])
                selections["oracle"].append(rmsds[oracle])

                rows.append({
                    "pdb_id": pdb_id, "family": payload["family"],
                    "n_poses": len(rmsds),
                    "rmsd_vina": rmsds[vina_pick], "rmsd_gnn": rmsds[gnn_pick],
                    "rmsd_oracle": rmsds[oracle],
                    "spearman_gnn_rmsd": "" if rho is None else f"{rho:.4f}",
                })
            except Exception as exc:
                failures[f"{type(exc).__name__}: {exc}"] += 1

    if not rows:
        print("\nNothing scored.")
        for name, count in failures.items():
            print(f"  {count:>4} {name}")
        return 1

    print("\n" + "=" * 68)
    print(f"POSE SELECTION ({len(rows)} complexes, {args.num_poses} poses each)")
    print("=" * 68)
    summarise("vina", selections["vina"], args.threshold)
    summarise("gnn", selections["gnn"], args.threshold)
    summarise("oracle", selections["oracle"], args.threshold)

    print("\n" + "=" * 68)
    print("DOES THE GNN SCORE TRACK POSE QUALITY?")
    print("=" * 68)
    print("  Rank correlation between GNN score and RMSD, per complex.")
    print("  Negative is correct: a better pose should score higher.\n")
    if per_complex_rho:
        rho = np.array(per_complex_rho)
        print(f"    median {np.median(rho):+.4f}   mean {rho.mean():+.4f}"
              f"   IQR [{np.percentile(rho, 25):+.4f}, {np.percentile(rho, 75):+.4f}]")
        print(f"    correct direction on {100 * (rho < 0).mean():.1f}% of complexes")

    vina = np.array(selections["vina"], dtype=float)
    gnn = np.array(selections["gnn"], dtype=float)
    both = ~(np.isnan(vina) | np.isnan(gnn))
    print("\n" + "=" * 68)
    print("RESCORING vs THE EXISTING SCORING FUNCTION")
    print("=" * 68)
    if both.sum() >= 5:
        delta = gnn[both] - vina[both]
        print(f"  GNN picks a better pose on {100 * (delta < 0).mean():.1f}% of "
              f"complexes, worse on {100 * (delta > 0).mean():.1f}%")
        print(f"  median RMSD change {np.median(delta):+.3f} A")
        try:
            from scipy import stats

            pvalue = float(stats.wilcoxon(gnn[both], vina[both]).pvalue)
            print(f"  Wilcoxon signed-rank p = {pvalue:.4g}")
        except (ImportError, ValueError):
            pvalue = None

        print()
        if pvalue is not None and pvalue < 0.05 and np.median(delta) < 0:
            print("  Rescoring with the GNN improves pose selection. Worth wiring")
            print("  into the pipeline behind a flag and benchmarking in full.")
        elif pvalue is not None and pvalue < 0.05:
            print("  Rescoring makes pose selection reliably WORSE. Keep the")
            print("  existing scoring function for pose ranking.")
        else:
            print("  No reliable difference. The GNN is not better than the")
            print("  scoring function that generated the poses, so rescoring")
            print("  adds cost without adding accuracy.")

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
