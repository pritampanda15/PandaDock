"""
Batched flexible-ligand Monte Carlo.

Stage three, built on stages one, two and four. The pose is the CPU's full DOF
vector -- [tx, ty, tz, rx, ry, rz, theta_1 .. theta_k] -- and the batched L-BFGS
from stage four is reused unchanged, because it was written against a flat
(B, D) parameter vector precisely so that adding torsions would not touch it.

There are two gradient implementations here, and both are kept on purpose.

`energy_and_dof_gradient` is the autograd route: the analytic grid gradient
enters as a vector-Jacobian product, the clash term is differentiated directly,
and a single backward pass covers the torsion chain. It was written first
because it is obviously correct -- autograd cannot get the chain rule wrong.

`analytic_energy_and_dof_gradient` is the closed form, and is what the search
uses. Profiling the first version put roughly two thirds of its cost in that
backward pass, and the CPU objective never needed one because it derives every
block analytically. Porting that derivation is worth 1.5x at three torsions and
1.9x at eleven, the gain growing with exactly the flexibility that made it slow.

Keeping both is the point rather than duplication: the tests assert they agree
to round-off, so the fast path is checked against an implementation that cannot
have made an algebra mistake, not only against the CPU.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .flexible import IntramolecularClash, TorsionApplier
from .grids import TorchAffinityGrids, resolve_device
from .optimize import LBFGSConfig, batched_lbfgs
from .rigid_search import RigidSearchConfig
from .rotations import (
    compose_rotvecs,
    random_rotvec,
    rodrigues_matrix,
    rotvec_gradient,
    wrap_rotvec,
)

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

    def analytic_energy_and_dof_gradient(self, x: "torch.Tensor"):
        """
        The same energy and gradient as `energy_and_dof_gradient`, with no autograd.

        Profiling the autograd version showed roughly two thirds of its cost in
        the backward pass through the sequential torsion chain: 12.9 ms of which
        ~8 ms was backward, at 256 chains and 11 torsions. The CPU objective
        never needed that pass because it derives every block in closed form, and
        this is that derivation, batched.

        Each block mirrors its CPU counterpart:

        * translation -- the pose shifts rigidly, so the gradient is the atom sum
        * orientation -- torque about the translation anchor, mapped through the
          derivative of the SO(3) exponential map
        * torsions -- rotating about axis u through pivot p moves atom m with
          velocity u x (c_m - p), so dE/dtheta is that velocity projected onto
          the Cartesian gradient, evaluated in the final rotated frame

        The autograd version is kept as the reference the tests compare against,
        which is what makes replacing it safe rather than a leap of faith.
        """
        translation = x[:, :3]
        rotvec = x[:, 3:6]
        angles = x[:, 6:]

        torsioned = self.torsions.apply(angles)
        rotation = rodrigues_matrix(rotvec)
        rotated = torch.einsum("bij,bnj->bni", rotation, torsioned)
        coords = rotated + translation.unsqueeze(1)

        heavy_idx = self.torsions.heavy_atoms
        heavy = coords[:, heavy_idx, :]

        inter, grad_heavy = self.grids.score_and_gradient(
            heavy, self.type_ids, need_gradient=True
        )
        clash, grad_atoms = self.clash.energy_and_gradient(coords)

        # Hydrogens score nothing on the grid but still move, so they simply
        # carry no interaction gradient -- as on the CPU.
        grad_atoms = grad_atoms.index_add(1, heavy_idx, grad_heavy)

        grad = torch.empty_like(x)
        grad[:, :3] = grad_atoms.sum(dim=1)

        torque = torch.cross(
            coords - translation.unsqueeze(1), grad_atoms, dim=-1
        ).sum(dim=1)
        grad[:, 3:6] = rotvec_gradient(rotvec, rotation, torque)

        if self.n_torsions:
            for k in range(self.n_torsions):
                origin = coords[:, self.torsions._origin[k], :]
                pivot = coords[:, self.torsions._axis[k], :]
                axis = pivot - origin

                norm = axis.norm(dim=-1, keepdim=True)
                fallback = torch.zeros_like(axis)
                fallback[:, 0] = 1.0
                unit = torch.where(
                    norm > 1e-8, axis / norm.clamp_min(1e-12), fallback
                )

                moving = self.torsions._moving[k]
                offset = coords[:, moving, :] - pivot.unsqueeze(1)
                velocity = torch.cross(
                    unit.unsqueeze(1).expand_as(offset), offset, dim=-1
                )
                grad[:, 6 + k] = (grad_atoms[:, moving, :] * velocity).sum((1, 2))

        return inter + clash, grad

    def local_optimize(
        self,
        x: "torch.Tensor",
        config: Optional[LBFGSConfig] = None,
        use_autograd: bool = False,
    ):
        """
        Relax a batch of poses, then re-canonicalise the orientation.

        Uses the closed-form gradient by default. `use_autograd=True` selects the
        autograd path instead, which is retained as the reference the analytic
        one is tested against rather than as a fallback -- if the two ever
        disagree, that is a bug to find, not a knob to flip.

        `wrap_rotvec` is applied for the same reason the CPU applies it: the
        optimiser moves the rotation parameters continuously and can drift many
        turns from the origin, which costs precision in the rotation gradient
        without changing the pose. Torsions are deliberately not wrapped, since
        the CPU does not wrap them either.
        """
        gradient_fn = (
            self.energy_and_dof_gradient
            if use_autograd
            else self.analytic_energy_and_dof_gradient
        )
        x, energy, _ = batched_lbfgs(x, gradient_fn, config or LBFGSConfig())
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
