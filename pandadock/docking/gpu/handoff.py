"""
Returning batched minima to the CPU pipeline.

The batched search ends holding a tensor of DOF vectors. Everything downstream
of that -- symmetry-corrected clustering, pose construction, interaction
analysis, output -- already exists on the CPU and is what the manuscript
validated, so the GPU path stops here and hands over rather than reimplementing
any of it.

The handoff is deliberately the only place the two paths meet. A GPU run
produces exactly the `SearchResult` list that `MonteCarloSearch.run` produces,
so `cluster_poses` and `_build_poses` cannot tell which search produced their
input, and a `--device` flag changes how minima are found without changing what
is done with them.

One correctness point governs the design: the energies are recomputed on the CPU
rather than carried over from the device. On MPS the search runs in float32, and
a float32 energy reported alongside a float64 pose would be a small, permanent,
untraceable discrepancy between what a run reports and what its own coordinates
score. Recomputing costs one evaluation per returned minimum, which is nothing
against the search that produced them.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def gpu_minima_to_search_results(
    dof: "np.ndarray",
    tree,
    objective,
) -> List:
    """
    Convert a batch of DOF vectors into CPU `SearchResult` objects.

    Args:
        dof: (B, n_dof) parameters from a batched search, already on the host.
        tree: the `TorsionTree` the DOF vectors were generated against.
        objective: the CPU `DockingObjective`, used to rescore in float64.

    Returns the minima sorted by energy, matching `MonteCarloSearch.run`.
    """
    from ..search.monte_carlo import SearchResult
    from ..search.rotations import wrap_rotvec

    results = []
    for row in np.asarray(dof, dtype=np.float64):
        parameters = row[: tree.n_dof].copy()
        # The CPU wraps after every local optimisation; a pose arriving from the
        # device may not have been wrapped since its last move.
        parameters[3:6] = wrap_rotvec(parameters[3:6])

        energy = float(objective.energy(parameters))
        results.append(
            SearchResult(
                dof=parameters,
                energy=energy,
                coords=objective.coords(parameters),
                run=0,
            )
        )

    results.sort(key=lambda r: r.energy)
    return results


def run_batched_search(
    grids,
    tree,
    objective,
    box_min,
    box_max,
    device=None,
    n_chains: int = 512,
    n_steps: int = 8,
    seed: Optional[int] = None,
    max_local_iter: int = 60,
) -> List:
    """
    Run the batched search and return CPU minima ready for clustering.

    Chooses the rigid or flexible implementation from the tree, so a caller does
    not have to. Both reduce to the same `SearchResult` list.
    """
    import torch

    from .flexible_search import build_flexible_search
    from .optimize import LBFGSConfig
    from .rigid_search import RigidSearchConfig

    config = RigidSearchConfig(n_chains=n_chains, n_steps=n_steps, seed=seed)
    search = build_flexible_search(
        grids, tree, objective, config=config, device=device
    )
    best, _ = search.run_basin_hopping(
        np.asarray(box_min),
        np.asarray(box_max),
        LBFGSConfig(max_iter=max_local_iter, max_line_search=12),
    )
    return gpu_minima_to_search_results(
        best.detach().cpu().double().numpy(), tree, objective
    )
