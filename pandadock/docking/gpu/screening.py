"""
Virtual screening over a ligand library, batched across ligands.

Stage five made several ligands share one batch; this is the entry point that
makes that reachable. A screen is where the batching actually pays: one ligand
never fills a device, and a library is many such runs.

Two decisions here come straight from what stage five measured.

Ligands are bucketed by size before packing. Everything in a batch is padded to
the batch maximum in atoms and torsions, so mixing a 9-atom fragment with a
48-atom ligand wastes most of the work on padding. Size-matched batches measured
roughly twice the throughput of mixed ones, and bucketing costs a sort.

Grids are built once per signature and shared. That is not this module's doing
-- `GridCache` already keys on (radius, hydrophobic, donor, acceptor) rather
than on ligand identity -- but a screen is the workload it exists for, and
passing one cache through the whole library is what turns grid construction from
a per-ligand cost into a per-campaign one.

The output is deliberately thin: a ranked table and the best pose per ligand.
Writing a full report per ligand would dominate the runtime of the thing this
module exists to speed up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("pandadock.docking.gpu.screening")


@dataclass
class ScreenResult:
    """One ligand's outcome, in the form a screen is read in."""

    name: str
    energy: float
    dof: np.ndarray
    coords: np.ndarray
    n_torsions: int


def bucket_by_size(
    trees: Sequence, max_batch: int = 64
) -> List[List[int]]:
    """
    Group ligand indices so that each batch pads as little as possible.

    Sorted by (torsion count, atom count) and cut into runs of at most
    `max_batch`. Sorting rather than exact-matching keeps every ligand in some
    batch: exact buckets would strand odd sizes in batches of one, which is the
    case this module exists to avoid.

    Torsion count leads because a surplus torsion slot costs a full rotation of
    every atom in the batch, while a surplus atom costs one row.
    """
    order = sorted(
        range(len(trees)),
        key=lambda i: (trees[i].n_torsions, trees[i].base_coords.shape[0]),
    )
    return [order[i : i + max_batch] for i in range(0, len(order), max_batch)]


def screen_library(
    receptor_structure,
    ligands: Sequence,
    grid_center,
    grid_dimensions,
    names: Optional[Sequence[str]] = None,
    device=None,
    n_chains: int = 128,
    n_steps: int = 8,
    max_batch: int = 64,
    grid_spacing: float = 0.375,
    max_torsions: int = 32,
    rigid_ligand: bool = False,
    seed: Optional[int] = None,
    max_local_iter: int = 60,
    progress=None,
) -> List[ScreenResult]:
    """
    Dock a library into one site, batching ligands onto the device.

    Args:
        receptor_structure: parsed receptor, shared by every ligand.
        ligands: RDKit molecules, each with a conformer.
        names: labels for the output; defaults to the molecules' _Name.
        device: `cuda`, `mps`, or None to pick the best available.
        n_chains: chains per ligand. The batch holds n_chains * len(bucket).
        max_batch: ligands per batch, bounding device memory.
        progress: optional callable taking (done, total) for reporting.

    Returns results sorted by energy, best first.
    """
    from ..scoring.vina_scoring import VinaScoring
    from ..search import AffinityGrids, DockingObjective, TorsionTree
    from ..search.grid_maps import GridCache
    from .multi import build_multi_ligand_search
    from .optimize import LBFGSConfig
    from .rigid_search import RigidSearchConfig

    grid_center = np.asarray(grid_center, dtype=np.float64)
    grid_dimensions = np.asarray(grid_dimensions, dtype=np.float64)
    box_min = grid_center - grid_dimensions / 2.0
    box_max = grid_center + grid_dimensions / 2.0

    if names is None:
        names = [
            m.GetProp("_Name") if m.HasProp("_Name") else f"ligand_{i}"
            for i, m in enumerate(ligands)
        ]

    scoring = VinaScoring()
    # One cache for the whole campaign: this is the difference between building
    # grids once per signature and once per ligand.
    cache = GridCache()

    trees, grids, objectives = [], [], []
    for mol in ligands:
        tree = TorsionTree(
            mol, conf_id=0, rigid=rigid_ligand, max_torsions=max_torsions
        )
        g = AffinityGrids.build(
            receptor_structure,
            mol,
            grid_center,
            grid_dimensions,
            spacing=grid_spacing,
            scoring=scoring,
            cache=cache,
        )
        trees.append(tree)
        grids.append(g)
        objectives.append(DockingObjective(tree, g))

    buckets = bucket_by_size(trees, max_batch=max_batch)
    logger.info(
        "Screening %d ligands in %d batches of at most %d",
        len(ligands), len(buckets), max_batch,
    )

    results: List[ScreenResult] = []
    done = 0
    for bucket in buckets:
        search = build_multi_ligand_search(
            [grids[i] for i in bucket],
            [trees[i] for i in bucket],
            [objectives[i] for i in bucket],
            config=RigidSearchConfig(n_chains=n_chains, n_steps=n_steps, seed=seed),
            device=device,
            names=[names[i] for i in bucket],
        )
        best_x, best_energy = search.run_basin_hopping(
            box_min, box_max,
            LBFGSConfig(max_iter=max_local_iter, max_line_search=12),
        )

        # Rescored on the CPU in float64, for the reason handoff.py gives: a
        # float32 energy beside a float64 pose is a discrepancy between what a
        # run reports and what its own coordinates score.
        chain_energy = best_energy.detach().cpu().double().numpy()
        chain_dof = best_x.detach().cpu().double().numpy()

        for row, ligand_index in enumerate(bucket):
            tree, objective = trees[ligand_index], objectives[ligand_index]
            winner = int(np.argmin(chain_energy[row]))
            dof = chain_dof[row, winner, : tree.n_dof].copy()

            from ..search.rotations import wrap_rotvec

            dof[3:6] = wrap_rotvec(dof[3:6])
            results.append(
                ScreenResult(
                    name=names[ligand_index],
                    energy=float(objective.energy(dof)),
                    dof=dof,
                    coords=objective.coords(dof),
                    n_torsions=tree.n_torsions,
                )
            )
        done += len(bucket)
        if progress is not None:
            progress(done, len(ligands))

    results.sort(key=lambda r: r.energy)
    return results
