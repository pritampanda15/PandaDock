"""
Batched rigid-ligand Monte Carlo.

Stage two of the GPU search. The ligand is held rigid so that this stage
isolates the parts that are easy to get subtly wrong -- pose construction,
batched scoring, the Metropolis criterion -- from torsional sampling, which
arrives in stage three. A rigid run is also a real workflow in its own right:
it is what `--rigid-ligand` already does on the CPU.

Relationship to the CPU search. This is not bit-identical to
`MonteCarloSearch`, and cannot be: the CPU runs `exhaustiveness` independent
chains in a Python loop, each drawing from its own NumPy generator, while this
advances thousands of chains in lockstep from a torch generator. The stream of
random numbers differs, so individual trajectories differ. What must agree is
everything that is not the random stream:

* a pose built from a given DOF vector must have the same coordinates
* that pose must receive the same energy
* the Metropolis rule must accept with the same probability

The tests assert each of those against the CPU implementation directly.

The loop deliberately never calls `.item()`, `.cpu()`, or `float()` on a tensor.
Each of those forces a device synchronisation, and at a few microseconds of
kernel time per step a single sync per iteration costs more than the arithmetic
it is waiting for. Acceptance is therefore resolved with a tensor mask rather
than a Python `if`, and nothing returns to the host until the run is over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .grids import TorchAffinityGrids, dtype_for_device, resolve_device
from .rotations import compose_rotvecs, random_rotvec, wrap_rotvec

try:
    import torch

    torch_available = True
except ImportError:  # pragma: no cover - exercised only in a base install
    torch = None
    torch_available = False


@dataclass
class RigidSearchConfig:
    """
    Search controls, named to match `SearchConfig` on the CPU.

    `n_chains` replaces `exhaustiveness`: the CPU runs that many chains one after
    another, and this runs them at once. The default is far larger because the
    whole point of the batch is that additional chains are close to free until
    the device is saturated.
    """

    n_chains: int = 1024
    n_steps: int = 200
    temperature: float = 1.0
    translation_amplitude: float = 2.0
    rotation_amplitude: float = 0.35
    reorient_probability: float = 0.1
    seed: Optional[int] = None


class RigidBatchedSearch:
    """
    Monte Carlo over rigid-body pose, with every chain advanced in parallel.

    Args:
        grids: receptor grids already resident on the target device.
        base_coords: (N, 3) heavy-atom reference geometry, centred as the CPU
            torsion tree centres it (on the root atom).
        type_ids: (N,) ligand atom types indexing the grid maps.
    """

    def __init__(
        self,
        grids: TorchAffinityGrids,
        base_coords: np.ndarray,
        type_ids: np.ndarray,
        config: Optional[RigidSearchConfig] = None,
    ):
        if not torch_available:
            raise ImportError(
                "The GPU search path needs the optional [gnn] extra "
                "(pip install pandadock[gnn]) for torch."
            )
        self.grids = grids
        self.config = config or RigidSearchConfig()
        self.device = grids.device
        self.dtype = grids.dtype

        self.base_coords = torch.as_tensor(
            base_coords, dtype=self.dtype, device=self.device
        )
        self.type_ids = torch.as_tensor(
            np.asarray(type_ids), dtype=torch.long, device=self.device
        )

    # ------------------------------------------------------------------ poses

    def build_coords(
        self, translation: "torch.Tensor", rotvec: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        (B, 3), (B, 3) -> (B, N, 3), matching `TorsionTree.build_coords`.

        The CPU computes `coords @ R.T + t`; the einsum below is the same
        contraction written for a batch of rotations against one geometry.
        """
        from .rotations import rodrigues_matrix

        rotation = rodrigues_matrix(rotvec)
        rotated = torch.einsum("bij,nj->bni", rotation, self.base_coords)
        return rotated + translation.unsqueeze(1)

    def score(self, translation, rotvec, need_gradient=False):
        """Energy, and optionally d(energy)/d(coord), for a batch of poses."""
        coords = self.build_coords(translation, rotvec)
        return self.grids.score_and_gradient(
            coords, self.type_ids, need_gradient=need_gradient
        )

    # ----------------------------------------------------------------- search

    def _generator(self):
        gen = torch.Generator(device="cpu")
        if self.config.seed is not None:
            gen.manual_seed(int(self.config.seed))
        return gen

    def _to_device(self, tensor):
        return tensor.to(device=self.device, dtype=self.dtype)

    def initial_state(self, box_min, box_max, gen):
        """
        Uniform over the box and over SO(3), as `TorsionTree.random_dof` is.

        Sampling on the CPU generator and moving the result keeps runs
        reproducible across devices: MPS and CUDA generators do not produce the
        same stream, so seeding a device generator would make the seed mean
        something different on each machine.
        """
        cfg = self.config
        lo = torch.as_tensor(np.asarray(box_min), dtype=torch.float64)
        hi = torch.as_tensor(np.asarray(box_max), dtype=torch.float64)
        u = torch.rand(cfg.n_chains, 3, generator=gen, dtype=torch.float64)
        translation = lo + u * (hi - lo)
        rotvec = random_rotvec(cfg.n_chains, generator=gen, dtype=torch.float64)
        return self._to_device(translation), self._to_device(rotvec)

    def propose(self, translation, rotvec, gen):
        """
        One trial move per chain, mirroring `MonteCarloSearch._mutate`.

        As on the CPU, the orientation is occasionally resampled outright rather
        than perturbed, so a chain can leave a basin whose orientation is
        qualitatively wrong instead of only refining it.
        """
        cfg = self.config
        n = translation.shape[0]

        step = torch.randn(n, 3, generator=gen, dtype=torch.float64)
        new_translation = translation + self._to_device(step) * cfg.translation_amplitude

        perturb = torch.randn(n, 3, generator=gen, dtype=torch.float64)
        perturbed = compose_rotvecs(
            self._to_device(perturb) * cfg.rotation_amplitude, rotvec
        )
        fresh = self._to_device(random_rotvec(n, generator=gen, dtype=torch.float64))

        reorient = torch.rand(n, 1, generator=gen, dtype=torch.float64)
        reorient = self._to_device(reorient) < cfg.reorient_probability
        new_rotvec = torch.where(reorient, fresh, perturbed)

        return new_translation, wrap_rotvec(new_rotvec)

    def run(self, box_min, box_max):
        """
        Advance every chain for `n_steps` and return the best pose of each.

        Returns:
            translation: (n_chains, 3) best translation seen by each chain
            rotvec: (n_chains, 3) matching orientation
            energy: (n_chains,) best energy seen by each chain

        Chains are returned unsorted and unclustered. Clustering stays on the
        CPU, where the symmetry-corrected comparison already lives.
        """
        cfg = self.config
        gen = self._generator()

        translation, rotvec = self.initial_state(box_min, box_max, gen)
        energy, _ = self.score(translation, rotvec)

        best_translation = translation.clone()
        best_rotvec = rotvec.clone()
        best_energy = energy.clone()

        temperature = max(cfg.temperature, 1e-6)

        for _ in range(cfg.n_steps):
            trial_t, trial_r = self.propose(translation, rotvec, gen)
            trial_e, _ = self.score(trial_t, trial_r)

            delta = trial_e - energy
            # Metropolis, as a mask. `delta <= 0` is folded in by the exponential
            # exceeding 1 there, which matches the CPU's `delta <= 0 or rand < exp`
            # without needing a second comparison.
            u = self._to_device(
                torch.rand(cfg.n_chains, generator=gen, dtype=torch.float64)
            )
            accept = u < torch.exp(-delta / temperature)

            keep = accept.unsqueeze(-1)
            translation = torch.where(keep, trial_t, translation)
            rotvec = torch.where(keep, trial_r, rotvec)
            energy = torch.where(accept, trial_e, energy)

            improved = trial_e < best_energy
            keep_best = improved.unsqueeze(-1)
            best_translation = torch.where(keep_best, trial_t, best_translation)
            best_rotvec = torch.where(keep_best, trial_r, best_rotvec)
            best_energy = torch.where(improved, trial_e, best_energy)

        return best_translation, best_rotvec, best_energy


def build_rigid_search(
    cpu_grids,
    tree,
    config: Optional[RigidSearchConfig] = None,
    device=None,
) -> RigidBatchedSearch:
    """
    Construct a batched search from the CPU objects the docking engine already has.

    Takes the same `AffinityGrids` and `TorsionTree` the CPU search uses, so the
    two paths cannot disagree about geometry or atom typing through a
    transcription mistake.
    """
    device = resolve_device(device)
    grids = TorchAffinityGrids.from_cpu(cpu_grids, device=device)
    return RigidBatchedSearch(
        grids=grids,
        base_coords=tree.base_coords[tree.heavy_atoms],
        type_ids=cpu_grids.typing.type_ids,
        config=config,
    )
