"""
Monte Carlo search with local optimization.

Implements the iterated-local-search scheme that underpins modern docking
engines: a number of independent runs, each of which repeatedly perturbs the
current pose, relaxes the perturbed pose to a local minimum with a quasi-Newton
optimizer, and accepts or rejects the result by the Metropolis criterion.

Every run starts from a uniformly sampled position in the box, a uniformly
sampled orientation over SO(3), and uniformly sampled torsion angles. No part of
the search space is privileged by the input conformer's coordinates or by any
reference structure.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

from ...analysis.rmsd import min_rmsd_over_permutations
from .objective import DockingObjective
from .rotations import compose_rotvecs, random_rotvec, wrap_rotvec
from .torsion_tree import TorsionTree

logger = logging.getLogger("pandadock.docking.search.monte_carlo")


@dataclass
class SearchResult:
    """A local minimum found by the search."""

    dof: np.ndarray
    energy: float
    coords: np.ndarray
    run: int = -1


@dataclass
class SearchConfig:
    """
    Search control parameters.

    Args:
        exhaustiveness: Number of independent Monte Carlo runs. Higher values
            trade runtime for a lower chance of missing the global minimum.
        n_steps: Monte Carlo steps per run. When None, scaled from the number of
            degrees of freedom, since larger search spaces need more sampling.
        temperature: Metropolis temperature in kcal/mol.
        max_local_iter: Cap on quasi-Newton iterations per local optimization.
        translation_amplitude: Standard deviation (A) of trial translations.
        rotation_amplitude: Standard deviation (radians) of trial rotations.
        torsion_amplitude: Standard deviation (radians) of trial torsion changes.
        seed: Random seed. None draws a fresh, non-deterministic seed.

    Runs are independent and could be parallelised, but the useful granularity for
    a benchmark is one process per complex; see `benchmarking/redock_benchmark.py
    -j`. Parallelising within a single dock is left out rather than shipped
    untested.
    """

    exhaustiveness: int = 8
    n_steps: Optional[int] = None
    temperature: float = 1.2
    max_local_iter: int = 60
    translation_amplitude: float = 1.5
    rotation_amplitude: float = 0.5
    torsion_amplitude: float = 0.7
    seed: Optional[int] = None

    def steps_for(self, n_dof: int) -> int:
        if self.n_steps is not None:
            return int(self.n_steps)
        # Roughly linear in the dimensionality of the search space.
        return int(max(60, 25 * n_dof))


class MonteCarloSearch:
    """Iterated local search over the ligand's degrees of freedom."""

    def __init__(self, objective: DockingObjective, config: Optional[SearchConfig] = None):
        self.objective = objective
        self.config = config or SearchConfig()
        self.tree: TorsionTree = objective.tree

    # ------------------------------------------------------------------ local opt

    def local_optimize(self, dof: np.ndarray) -> SearchResult:
        """Relax a pose to the nearest local minimum with L-BFGS-B."""
        result = minimize(
            self.objective.scipy_objective,
            dof,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.config.max_local_iter, "maxcor": 10},
        )
        opt_dof = np.asarray(result.x, dtype=np.float64)
        opt_dof[3:6] = wrap_rotvec(opt_dof[3:6])
        energy = float(result.fun)
        return SearchResult(dof=opt_dof, energy=energy, coords=self.objective.coords(opt_dof))

    # ----------------------------------------------------------------- mutation

    def _mutate(self, dof: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Propose a trial pose.

        Occasionally the orientation is resampled uniformly rather than perturbed,
        which lets a run escape a basin whose orientation is qualitatively wrong
        instead of only refining it.
        """
        trial = dof.copy()
        cfg = self.config

        trial[:3] += rng.normal(0.0, cfg.translation_amplitude, 3)

        if rng.random() < 0.1:
            trial[3:6] = random_rotvec(rng)
        else:
            trial[3:6] = compose_rotvecs(
                rng.normal(0.0, cfg.rotation_amplitude, 3), trial[3:6]
            )

        n_tors = self.tree.n_torsions
        if n_tors:
            if rng.random() < 0.25:
                # Resample one torsion outright; small steps alone rarely cross the
                # barriers between rotamer wells.
                k = int(rng.integers(0, n_tors))
                trial[6 + k] = rng.uniform(-np.pi, np.pi)
            else:
                trial[6:] += rng.normal(0.0, cfg.torsion_amplitude, n_tors)

        return trial

    # --------------------------------------------------------------------- runs

    def _single_run(
        self, run_index: int, seed: int, box_min: np.ndarray, box_max: np.ndarray
    ) -> List[SearchResult]:
        rng = np.random.default_rng(seed)
        n_steps = self.config.steps_for(self.tree.n_dof)
        temperature = max(self.config.temperature, 1e-6)

        current = self.local_optimize(self.tree.random_dof(rng, box_min, box_max))
        minima: List[SearchResult] = [current]
        best = current

        for _ in range(n_steps):
            trial_dof = self._mutate(current.dof, rng)
            trial = self.local_optimize(trial_dof)
            minima.append(trial)

            delta = trial.energy - current.energy
            if delta <= 0.0 or rng.random() < np.exp(-delta / temperature):
                current = trial
            if trial.energy < best.energy:
                best = trial

        for m in minima:
            m.run = run_index

        logger.debug(
            "Run %d: best energy %.3f kcal/mol after %d steps", run_index, best.energy, n_steps
        )
        return minima

    def run(self, box_min: np.ndarray, box_max: np.ndarray) -> List[SearchResult]:
        """
        Execute all Monte Carlo runs and return every local minimum found.

        Results are sorted by energy but not yet clustered; call `cluster_poses`
        to reduce them to distinct binding modes.
        """
        cfg = self.config
        base_seed = cfg.seed
        if base_seed is None:
            base_seed = int(np.random.SeedSequence().entropy % (2**31 - 1))

        seeds = np.random.SeedSequence(base_seed).generate_state(cfg.exhaustiveness)

        all_minima: List[SearchResult] = []
        for i in range(cfg.exhaustiveness):
            all_minima.extend(self._single_run(i, int(seeds[i]), box_min, box_max))

        all_minima.sort(key=lambda r: r.energy)
        logger.info(
            "Search complete: %d runs, %d local minima, %d energy evaluations, "
            "best %.3f kcal/mol",
            cfg.exhaustiveness,
            len(all_minima),
            self.objective.n_evaluations,
            all_minima[0].energy if all_minima else float("nan"),
        )
        return all_minima


def cluster_poses(
    results: Sequence[SearchResult],
    heavy_atoms: np.ndarray,
    rmsd_cutoff: float = 2.0,
    max_poses: int = 20,
    automorphisms: Optional[np.ndarray] = None,
) -> List[SearchResult]:
    """
    Reduce local minima to distinct binding modes by greedy RMSD clustering.

    Walks the minima in energy order and keeps each one that is at least
    `rmsd_cutoff` heavy-atom RMSD away from every pose already kept, so the
    returned list represents genuinely different binding modes rather than many
    near-identical copies of the same one.

    `automorphisms`, from `pandadock.analysis.rmsd.heavy_atom_automorphisms`,
    makes the comparison symmetry-corrected. Without it, two poses related by a
    symmetry of the ligand -- a flipped phenyl, a swapped carboxylate -- compare
    as far apart under fixed atom indices despite being the same structure, and
    both are kept, spending the `max_poses` budget on duplicates. It is optional
    only so that callers without the molecule can still cluster; passing it is
    the correct behaviour.
    """
    kept: List[SearchResult] = []
    for candidate in results:
        c_heavy = candidate.coords[heavy_atoms]
        distinct = True
        for existing in kept:
            e_heavy = existing.coords[heavy_atoms]
            rmsd = min_rmsd_over_permutations(c_heavy, e_heavy, automorphisms)
            if rmsd < rmsd_cutoff:
                distinct = False
                break
        if distinct:
            kept.append(candidate)
        if len(kept) >= max_poses:
            break
    return kept
