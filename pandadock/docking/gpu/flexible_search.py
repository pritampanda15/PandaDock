"""
Batched flexible-ligand Monte Carlo.

Stage three, built on stages one, two and four. The pose is the CPU's full DOF
vector -- [tx, ty, tz, rx, ry, rz, theta_1 .. theta_k] -- and the batched L-BFGS
from stage four is reused unchanged, because it was written against a flat
(B, D) parameter vector precisely so that adding torsions would not touch it.

The energy now has two parts, and they reach the optimiser by different routes:

* The receptor interaction keeps the analytic grid gradient from stage one,
  injected as a vector-Jacobian product so that autograd never has to traverse
  the interpolation's `floor` and integer indexing.
* The intramolecular clash term is differentiated by autograd directly, since it
  is smooth in the coordinates and has nothing to hide.

Both are combined into a single backward pass over the torsion chain, which is
the part that would be laborious and error-prone to differentiate by hand: each
torsion rotates about an axis its ancestors have already moved.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .flexible import IntramolecularClash, TorsionApplier
from .grids import TorchAffinityGrids, resolve_device
from .optimize import LBFGSConfig, batched_lbfgs
from .rigid_search import RigidSearchConfig
from .rotations import compose_rotvecs, random_rotvec, rodrigues_matrix, wrap_rotvec

try:
    import torch

    torch_available = True
except ImportError:  # pragma: no cover
    torch = None
    torch_available = False


class FlexibleBatchedSearch:
    """
    Monte Carlo over translation, orientation and torsions, batched over chains.

    Construct with `build_flexible_search`, which wires it to the same
    `AffinityGrids`, `TorsionTree` and `DockingObjective` the CPU search uses.
    """

    def __init__(
        self,
        grids: TorchAffinityGrids,
        torsions: TorsionApplier,
        clash: IntramolecularClash,
        type_ids: np.ndarray,
        config: Optional[RigidSearchConfig] = None,
        torsion_amplitude: float = 0.5,
        torsion_resample_probability: float = 0.25,
    ):
        if not torch_available:  # pragma: no cover
            raise ImportError("the GPU search path needs torch")

        self.grids = grids
        self.torsions = torsions
        self.clash = clash
        self.config = config or RigidSearchConfig()
        self.device = grids.device
        self.dtype = grids.dtype

        self.n_torsions = torsions.n_torsions
        self.n_dof = 6 + self.n_torsions
        self.torsion_amplitude = torsion_amplitude
        self.torsion_resample_probability = torsion_resample_probability

        self.type_ids = torch.as_tensor(
            np.asarray(type_ids), dtype=torch.long, device=self.device
        )

    # ------------------------------------------------------------------ poses

    def build_coords(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        (B, 6 + k) DOF -> (B, n_atoms, 3), matching `TorsionTree.build_coords`.

        Torsions first against the reference geometry, then the rigid-body
        rotation, then the translation -- the same order as the CPU, which
        matters because the operations do not commute.
        """
        translation = x[:, :3]
        rotvec = x[:, 3:6]
        angles = x[:, 6:]

        coords = self.torsions.apply(angles)
        rotation = rodrigues_matrix(rotvec)
        coords = torch.einsum("bij,bnj->bni", rotation, coords)
        return coords + translation.unsqueeze(1)

    def energy(self, x: "torch.Tensor") -> "torch.Tensor":
        """Total energy for a batch of DOF vectors, no gradient."""
        coords = self.build_coords(x)
        heavy = coords[:, self.torsions.heavy_atoms, :]
        inter, _ = self.grids.score_and_gradient(
            heavy, self.type_ids, need_gradient=False
        )
        return inter + self.clash.energy(coords)

    def energy_and_dof_gradient(self, x: "torch.Tensor"):
        """
        Energy and d(energy)/d(DOF), combining an analytic and an autograd path.

        The grid contribution is folded in as `(coords * dE/dcoords).sum()` with
        the Cartesian gradient detached. That surrogate has no meaningful value,
        but its derivative with respect to the DOF is exactly the chain rule for
        the grid energy -- which is what lets the validated analytic gradient be
        reused while autograd supplies the torsion chain it multiplies through.
        """
        x = x.detach().requires_grad_(True)

        coords = self.build_coords(x)
        heavy = coords[:, self.torsions.heavy_atoms, :]

        inter, grad_heavy = self.grids.score_and_gradient(
            heavy.detach(), self.type_ids, need_gradient=True
        )
        clash = self.clash.energy(coords)

        surrogate = (heavy * grad_heavy.detach()).sum() + clash.sum()
        (grad,) = torch.autograd.grad(surrogate, x)

        return inter + clash.detach(), grad

    def local_optimize(self, x: "torch.Tensor", config: Optional[LBFGSConfig] = None):
        """
        Relax a batch of poses, then re-canonicalise the orientation.

        `wrap_rotvec` is applied for the same reason the CPU applies it: the
        optimiser moves the rotation parameters continuously and can drift many
        turns from the origin, which costs precision in the rotation gradient
        without changing the pose. Torsions are deliberately not wrapped, since
        the CPU does not wrap them either.
        """
        x, energy, _ = batched_lbfgs(
            x, self.energy_and_dof_gradient, config or LBFGSConfig()
        )
        x = torch.cat([x[:, :3], wrap_rotvec(x[:, 3:6]), x[:, 6:]], dim=-1)
        return x, energy

    # ----------------------------------------------------------------- search

    def _generator(self):
        gen = torch.Generator(device="cpu")
        if self.config.seed is not None:
            gen.manual_seed(int(self.config.seed))
        return gen

    def _to_device(self, tensor):
        return tensor.to(device=self.device, dtype=self.dtype)

    def initial_state(self, box_min, box_max, gen):
        """Uniform over the box, over SO(3), and over torsion angles."""
        cfg = self.config
        n = cfg.n_chains

        lo = torch.as_tensor(np.asarray(box_min), dtype=torch.float64)
        hi = torch.as_tensor(np.asarray(box_max), dtype=torch.float64)
        u = torch.rand(n, 3, generator=gen, dtype=torch.float64)
        translation = lo + u * (hi - lo)
        rotvec = random_rotvec(n, generator=gen, dtype=torch.float64)

        parts = [translation, rotvec]
        if self.n_torsions:
            angles = (
                torch.rand(n, self.n_torsions, generator=gen, dtype=torch.float64)
                * 2.0
                * np.pi
                - np.pi
            )
            parts.append(angles)
        return self._to_device(torch.cat(parts, dim=-1))

    def propose(self, x: "torch.Tensor", gen) -> "torch.Tensor":
        """
        One trial move per chain, mirroring `MonteCarloSearch._mutate`.

        Including its two escape mechanisms: the orientation is occasionally
        resampled outright, and one torsion is occasionally resampled over its
        whole range, because small steps alone rarely cross the barriers between
        rotamer wells.
        """
        cfg = self.config
        n = x.shape[0]

        step = torch.randn(n, 3, generator=gen, dtype=torch.float64)
        translation = x[:, :3] + self._to_device(step) * cfg.translation_amplitude

        perturb = torch.randn(n, 3, generator=gen, dtype=torch.float64)
        perturbed = compose_rotvecs(
            self._to_device(perturb) * cfg.rotation_amplitude, x[:, 3:6]
        )
        fresh = self._to_device(random_rotvec(n, generator=gen, dtype=torch.float64))
        reorient = self._to_device(
            torch.rand(n, 1, generator=gen, dtype=torch.float64)
        ) < cfg.reorient_probability
        rotvec = wrap_rotvec(torch.where(reorient, fresh, perturbed))

        if not self.n_torsions:
            return torch.cat([translation, rotvec], dim=-1)

        angles = x[:, 6:]
        jitter = torch.randn(n, self.n_torsions, generator=gen, dtype=torch.float64)
        jittered = angles + self._to_device(jitter) * self.torsion_amplitude

        # Resample a single torsion, chosen per chain, rather than all of them.
        which = torch.randint(
            0, self.n_torsions, (n, 1), generator=gen, dtype=torch.long
        ).to(self.device)
        fresh_angle = self._to_device(
            torch.rand(n, 1, generator=gen, dtype=torch.float64) * 2.0 * np.pi - np.pi
        )
        one_hot = torch.zeros_like(angles, dtype=torch.bool)
        one_hot.scatter_(1, which, True)
        resampled = torch.where(one_hot, fresh_angle.expand_as(angles), angles)

        do_resample = self._to_device(
            torch.rand(n, 1, generator=gen, dtype=torch.float64)
        ) < self.torsion_resample_probability
        angles = torch.where(do_resample, resampled, jittered)

        return torch.cat([translation, rotvec, angles], dim=-1)

    def run_basin_hopping(self, box_min, box_max, lbfgs_config=None):
        """
        Monte Carlo over local minima, as `MonteCarloSearch._single_run` does.

        Returns:
            x: (n_chains, 6 + k) best DOF vector per chain
            energy: (n_chains,) energy at those parameters
        """
        cfg = self.config
        gen = self._generator()

        x = self.initial_state(box_min, box_max, gen)
        x, energy = self.local_optimize(x, lbfgs_config)

        best_x = x.clone()
        best_energy = energy.clone()
        temperature = max(cfg.temperature, 1e-6)

        for _ in range(cfg.n_steps):
            trial = self.propose(x, gen)
            trial, trial_energy = self.local_optimize(trial, lbfgs_config)

            delta = trial_energy - energy
            u = self._to_device(
                torch.rand(cfg.n_chains, generator=gen, dtype=torch.float64)
            )
            accept = u < torch.exp(-delta / temperature)

            x = torch.where(accept.unsqueeze(-1), trial, x)
            energy = torch.where(accept, trial_energy, energy)

            improved = trial_energy < best_energy
            best_x = torch.where(improved.unsqueeze(-1), trial, best_x)
            best_energy = torch.where(improved, trial_energy, best_energy)

        return best_x, best_energy


def build_flexible_search(
    cpu_grids,
    tree,
    objective,
    config: Optional[RigidSearchConfig] = None,
    device=None,
) -> FlexibleBatchedSearch:
    """
    Construct from the CPU objects the docking engine already holds.

    Takes the live `DockingObjective` so the clash term reads its pair list and
    radii rather than rebuilding them, which is what keeps the two paths from
    disagreeing about which contacts are penalised.
    """
    device = resolve_device(device)
    grids = TorchAffinityGrids.from_cpu(cpu_grids, device=device)
    return FlexibleBatchedSearch(
        grids=grids,
        torsions=TorsionApplier(tree, grids.dtype, device),
        clash=IntramolecularClash(objective, grids.dtype, device),
        type_ids=cpu_grids.typing.type_ids,
        config=config,
    )
