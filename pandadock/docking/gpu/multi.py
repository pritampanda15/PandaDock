"""
Several ligands docked against one receptor in a single batch.

Stage five. The receptor grids are the largest tensor in the calculation and are
already shared across ligands by the CPU's signature-keyed cache; the point here
is to stop paying per-ligand launch overhead as well, by giving the device one
large batch instead of a sequence of small ones. That matters most for exactly
the case the earlier stages measured as weakest: a few hundred chains of a small
ligand does not fill a GPU, and screening a library is many such runs.

Ligands vary in every dimension that matters -- atom count, torsion count, the
size of each torsion's moving set, the intramolecular pair list -- so they are
padded to the batch maximum and masked. Two consequences are worth stating
because they shape the whole module:

* Torsions are applied by rotating every atom and selecting with a boolean mask
  rather than by indexing a moving set. Ragged per-ligand index arrays cannot be
  batched; a mask can. It rotates atoms that will be discarded, which is wasted
  arithmetic, but it is uniform across the batch and the alternative is a Python
  loop over ligands. A ligand with fewer torsions than the batch maximum simply
  has all-False masks for the surplus, which makes those rotations the identity.

* A padded atom is not free. It has coordinates, it lands somewhere in the grid,
  and without a mask it would contribute an energy and a boundary penalty for an
  atom that does not exist. Every reduction here is therefore masked, and the
  tests check a padded batch against the same ligands run alone.

Grouping ligands by size before packing would cut the padding waste, and callers
are free to do that -- `pack_ligands` imposes no ordering. It is left out here
because the right bucketing depends on the library, and a correct padded path is
the prerequisite for any of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

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


@dataclass
class PackedLigands:
    """
    Several ligands' geometry and topology in uniformly shaped tensors.

    Every field is padded to the batch maximum in atoms (A), torsions (T) and
    intramolecular pairs (P), with a mask saying which entries are real.
    """

    base_coords: "torch.Tensor"  # (L, A, 3)
    atom_mask: "torch.Tensor"  # (L, A) bool
    heavy_mask: "torch.Tensor"  # (L, A) bool
    type_ids: "torch.Tensor"  # (L, A) long, 0 where padded
    torsion_origin: "torch.Tensor"  # (L, T) long
    torsion_axis: "torch.Tensor"  # (L, T) long
    torsion_moving: "torch.Tensor"  # (L, T, A) bool
    pair_a: "torch.Tensor"  # (L, P) long
    pair_b: "torch.Tensor"  # (L, P) long
    pair_mask: "torch.Tensor"  # (L, P) bool
    pair_radii: "torch.Tensor"  # (L, P)
    intra_weight: "torch.Tensor"  # (L,)
    n_torsions: "torch.Tensor"  # (L,) long, real count per ligand
    names: List[str]

    @property
    def n_ligands(self) -> int:
        return self.base_coords.shape[0]

    @property
    def max_torsions(self) -> int:
        return self.torsion_moving.shape[1]

    @property
    def n_dof(self) -> int:
        return 6 + self.max_torsions


def pack_ligands(trees, objectives, type_ids, dtype, device, names=None) -> PackedLigands:
    """
    Pad a list of CPU torsion trees and objectives into batched tensors.

    Args:
        trees: one `TorsionTree` per ligand.
        objectives: the matching `DockingObjective`, read for its pair list and
            radii so the clash term cannot drift from the CPU's.
        type_ids: per-ligand heavy-atom type arrays, as `AffinityGrids` produces.
    """
    if not torch_available:  # pragma: no cover
        raise ImportError("the GPU search path needs torch")

    n_ligands = len(trees)
    max_atoms = max(t.base_coords.shape[0] for t in trees)
    max_tors = max(t.n_torsions for t in trees) or 0
    max_pairs = max(len(o.pair_a) for o in objectives) or 0

    base = np.zeros((n_ligands, max_atoms, 3))
    atom_mask = np.zeros((n_ligands, max_atoms), dtype=bool)
    heavy_mask = np.zeros((n_ligands, max_atoms), dtype=bool)
    types = np.zeros((n_ligands, max_atoms), dtype=np.int64)
    t_origin = np.zeros((n_ligands, max(max_tors, 1)), dtype=np.int64)
    t_axis = np.zeros((n_ligands, max(max_tors, 1)), dtype=np.int64)
    t_moving = np.zeros((n_ligands, max(max_tors, 1), max_atoms), dtype=bool)
    p_a = np.zeros((n_ligands, max(max_pairs, 1)), dtype=np.int64)
    p_b = np.zeros((n_ligands, max(max_pairs, 1)), dtype=np.int64)
    p_mask = np.zeros((n_ligands, max(max_pairs, 1)), dtype=bool)
    p_radii = np.zeros((n_ligands, max(max_pairs, 1)))
    weights = np.zeros(n_ligands)
    counts = np.zeros(n_ligands, dtype=np.int64)

    for i, (tree, objective, tids) in enumerate(zip(trees, objectives, type_ids)):
        n_atoms = tree.base_coords.shape[0]
        base[i, :n_atoms] = tree.base_coords
        atom_mask[i, :n_atoms] = True

        heavy = np.asarray(tree.heavy_atoms)
        heavy_mask[i, heavy] = True
        # `type_ids` is indexed by heavy atom, not by atom; scatter it back onto
        # the full atom axis so one padded array serves both.
        types[i, heavy] = np.asarray(tids)

        counts[i] = tree.n_torsions
        for k, torsion in enumerate(tree.torsions):
            t_origin[i, k] = int(torsion.origin_atom)
            t_axis[i, k] = int(torsion.axis_atom)
            t_moving[i, k, np.asarray(torsion.moving)] = True

        n_pairs = len(objective.pair_a)
        if n_pairs:
            p_a[i, :n_pairs] = np.asarray(objective.pair_a)
            p_b[i, :n_pairs] = np.asarray(objective.pair_b)
            p_radii[i, :n_pairs] = np.asarray(objective.pair_radii_sum)
            p_mask[i, :n_pairs] = True
        weights[i] = float(objective.intra_weight)

    def t(array, kind=None):
        return torch.as_tensor(array, dtype=kind or dtype, device=device)

    return PackedLigands(
        base_coords=t(base),
        atom_mask=t(atom_mask, torch.bool),
        heavy_mask=t(heavy_mask, torch.bool),
        type_ids=t(types, torch.long),
        torsion_origin=t(t_origin, torch.long),
        torsion_axis=t(t_axis, torch.long),
        torsion_moving=t(t_moving, torch.bool),
        pair_a=t(p_a, torch.long),
        pair_b=t(p_b, torch.long),
        pair_mask=t(p_mask, torch.bool),
        pair_radii=t(p_radii),
        intra_weight=t(weights),
        n_torsions=t(counts, torch.long),
        names=list(names) if names is not None else [f"ligand_{i}" for i in range(n_ligands)],
    )


class MultiLigandSearch:
    """
    Batched basin hopping over several ligands and many chains each.

    The batch is shaped (L, C) -- every ligand gets the same number of chains --
    and flattened to (L*C, D) only where the optimiser needs a matrix. Keeping
    the ligand axis explicit is what lets each ligand's own padding masks
    broadcast over its chains without a gather.
    """

    def __init__(
        self,
        grids: TorchAffinityGrids,
        packed: PackedLigands,
        vina,
        config: Optional[RigidSearchConfig] = None,
        torsion_amplitude: float = 0.5,
        torsion_resample_probability: float = 0.25,
    ):
        if not torch_available:  # pragma: no cover
            raise ImportError("the GPU search path needs torch")

        self.grids = grids
        self.packed = packed
        self.config = config or RigidSearchConfig()
        self.device = grids.device
        self.dtype = grids.dtype
        self.torsion_amplitude = torsion_amplitude
        self.torsion_resample_probability = torsion_resample_probability

        self.cutoff = float(vina.cutoff)
        self.g1_offset, self.g1_width = float(vina.gauss1_offset), float(vina.gauss1_width)
        self.g2_offset, self.g2_width = float(vina.gauss2_offset), float(vina.gauss2_width)
        self.rep_cutoff = float(vina.repulsion_cutoff)
        self.w_g1 = float(vina.weights["gauss1"])
        self.w_g2 = float(vina.weights["gauss2"])
        self.w_rep = float(vina.weights["repulsion"])

    # ------------------------------------------------------------------ poses

    def build_coords(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        (L, C, 6 + T) DOF -> (L, C, A, 3).

        Torsions, then the rigid-body rotation, then the translation, in the
        order `TorsionTree.build_coords` uses.
        """
        p = self.packed
        n_lig, n_chain = x.shape[0], x.shape[1]

        coords = p.base_coords.unsqueeze(1).expand(n_lig, n_chain, -1, -1).clone()

        for k in range(p.max_torsions):
            origin = torch.gather(
                coords, 2,
                p.torsion_origin[:, k].view(n_lig, 1, 1, 1).expand(n_lig, n_chain, 1, 3),
            )
            pivot = torch.gather(
                coords, 2,
                p.torsion_axis[:, k].view(n_lig, 1, 1, 1).expand(n_lig, n_chain, 1, 3),
            )
            axis = pivot - origin
            norm = axis.norm(dim=-1, keepdim=True)
            fallback = torch.zeros_like(axis)
            fallback[..., 0] = 1.0
            unit = torch.where(norm > 1e-8, axis / norm.clamp_min(1e-12), fallback)

            v = coords - pivot
            angle = x[:, :, 6 + k].view(n_lig, n_chain, 1, 1)
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            rotated = (
                v * cos_a
                + torch.cross(unit.expand_as(v), v, dim=-1) * sin_a
                + unit * (v * unit).sum(-1, keepdim=True) * (1.0 - cos_a)
            ) + pivot

            # The mask, not an index list, is what makes ligands with different
            # moving sets and different torsion counts share one loop.
            moving = p.torsion_moving[:, k].view(n_lig, 1, -1, 1)
            coords = torch.where(moving, rotated, coords)

        flat = x.reshape(n_lig * n_chain, -1)
        rotation = rodrigues_matrix(flat[:, 3:6]).view(n_lig, n_chain, 3, 3)
        coords = torch.einsum("lcij,lcnj->lcni", rotation, coords)
        return coords + x[:, :, :3].unsqueeze(2)

    def _clash(self, coords, need_gradient):
        """Masked intramolecular term over each ligand's own pair list."""
        p = self.packed
        n_lig, n_chain = coords.shape[0], coords.shape[1]

        idx_a = p.pair_a.view(n_lig, 1, -1, 1).expand(n_lig, n_chain, -1, 3)
        idx_b = p.pair_b.view(n_lig, 1, -1, 1).expand(n_lig, n_chain, -1, 3)
        delta = torch.gather(coords, 2, idx_a) - torch.gather(coords, 2, idx_b)

        dist = delta.pow(2).sum(-1).clamp_min(1e-12).sqrt()
        surf = dist - p.pair_radii.unsqueeze(1)
        mask = p.pair_mask.unsqueeze(1) & (surf < self.cutoff)

        g1 = torch.exp(-(((surf - self.g1_offset) / self.g1_width) ** 2))
        g2 = torch.exp(-(((surf - self.g2_offset) / self.g2_width) ** 2))
        rep = torch.where(surf < self.rep_cutoff, surf**2, torch.zeros_like(surf))

        per_pair = self.w_g1 * g1 + self.w_g2 * g2 + self.w_rep * rep
        per_pair = torch.where(mask, per_pair, torch.zeros_like(per_pair))
        weight = p.intra_weight.view(n_lig, 1)
        energy = weight * per_pair.sum(-1)

        if not need_gradient:
            return energy, None

        d_g1 = g1 * (-2.0 * (surf - self.g1_offset) / self.g1_width**2)
        d_g2 = g2 * (-2.0 * (surf - self.g2_offset) / self.g2_width**2)
        d_rep = torch.where(surf < self.rep_cutoff, 2.0 * surf, torch.zeros_like(surf))
        d_dsurf = self.w_g1 * d_g1 + self.w_g2 * d_g2 + self.w_rep * d_rep
        d_dsurf = torch.where(mask, d_dsurf, torch.zeros_like(d_dsurf))
        d_dsurf = d_dsurf * weight.unsqueeze(-1)

        contrib = d_dsurf.unsqueeze(-1) * (delta / dist.unsqueeze(-1))
        grad = torch.zeros_like(coords)
        grad.scatter_add_(2, idx_a, contrib)
        grad.scatter_add_(2, idx_b, -contrib)
        return energy, grad

    def energy_and_dof_gradient(self, x: "torch.Tensor"):
        """
        Energy and d(energy)/d(DOF) for (L, C, 6 + T), fully closed form.

        The same three blocks as the single-ligand path, with every reduction
        masked so padded atoms, padded pairs and surplus torsions contribute
        nothing at all rather than merely something small.
        """
        p = self.packed
        n_lig, n_chain = x.shape[0], x.shape[1]
        n_atoms = p.base_coords.shape[1]

        coords = self.build_coords(x)
        flat_coords = coords.reshape(n_lig * n_chain, n_atoms, 3)

        heavy_mask = p.heavy_mask.unsqueeze(1).expand(n_lig, n_chain, -1)
        types = p.type_ids.unsqueeze(1).expand(n_lig, n_chain, -1)

        inter, grad_inter = self.grids.score_and_gradient(
            flat_coords,
            types.reshape(n_lig * n_chain, -1),
            need_gradient=True,
            atom_mask=heavy_mask.reshape(n_lig * n_chain, -1),
        )
        inter = inter.view(n_lig, n_chain)
        grad_atoms = grad_inter.view(n_lig, n_chain, n_atoms, 3)

        clash, grad_clash = self._clash(coords, need_gradient=True)
        grad_atoms = grad_atoms + grad_clash

        grad = torch.empty_like(x)
        grad[:, :, :3] = grad_atoms.sum(dim=2)

        offset = coords - x[:, :, :3].unsqueeze(2)
        torque = torch.cross(offset, grad_atoms, dim=-1).sum(dim=2)
        flat = x.reshape(n_lig * n_chain, -1)
        rotation = rodrigues_matrix(flat[:, 3:6])
        grad[:, :, 3:6] = rotvec_gradient(
            flat[:, 3:6], rotation, torque.reshape(n_lig * n_chain, 3)
        ).view(n_lig, n_chain, 3)

        for k in range(p.max_torsions):
            origin = torch.gather(
                coords, 2,
                p.torsion_origin[:, k].view(n_lig, 1, 1, 1).expand(n_lig, n_chain, 1, 3),
            )
            pivot = torch.gather(
                coords, 2,
                p.torsion_axis[:, k].view(n_lig, 1, 1, 1).expand(n_lig, n_chain, 1, 3),
            )
            axis = pivot - origin
            norm = axis.norm(dim=-1, keepdim=True)
            fallback = torch.zeros_like(axis)
            fallback[..., 0] = 1.0
            unit = torch.where(norm > 1e-8, axis / norm.clamp_min(1e-12), fallback)

            velocity = torch.cross((coords - pivot), unit.expand_as(coords), dim=-1)
            # cross(u, c - p) is the velocity; the arguments above are reversed,
            # so negate rather than reorder a broadcast expand.
            velocity = -velocity

            moving = p.torsion_moving[:, k].view(n_lig, 1, -1, 1)
            contribution = torch.where(
                moving, grad_atoms * velocity, torch.zeros_like(velocity)
            )
            grad[:, :, 6 + k] = contribution.sum((2, 3))

        return inter + clash, grad

    # ----------------------------------------------------------------- search

    def _generator(self):
        gen = torch.Generator(device="cpu")
        if self.config.seed is not None:
            gen.manual_seed(int(self.config.seed))
        return gen

    def _to_device(self, tensor):
        return tensor.to(device=self.device, dtype=self.dtype)

    def initial_state(self, box_min, box_max, gen):
        p = self.packed
        n_lig, n_chain = p.n_ligands, self.config.n_chains
        n = n_lig * n_chain

        lo = torch.as_tensor(np.asarray(box_min), dtype=torch.float64)
        hi = torch.as_tensor(np.asarray(box_max), dtype=torch.float64)
        translation = lo + torch.rand(n, 3, generator=gen, dtype=torch.float64) * (hi - lo)
        rotvec = random_rotvec(n, generator=gen, dtype=torch.float64)

        parts = [translation, rotvec]
        if p.max_torsions:
            angles = (
                torch.rand(n, p.max_torsions, generator=gen, dtype=torch.float64)
                * 2.0 * np.pi - np.pi
            )
            parts.append(angles)
        x = self._to_device(torch.cat(parts, dim=-1))
        return x.view(n_lig, n_chain, -1)

    def propose(self, x, gen):
        p = self.packed
        n_lig, n_chain = x.shape[0], x.shape[1]
        n = n_lig * n_chain
        flat = x.reshape(n, -1)
        cfg = self.config

        step = self._to_device(torch.randn(n, 3, generator=gen, dtype=torch.float64))
        translation = flat[:, :3] + step * cfg.translation_amplitude

        perturb = self._to_device(torch.randn(n, 3, generator=gen, dtype=torch.float64))
        perturbed = compose_rotvecs(perturb * cfg.rotation_amplitude, flat[:, 3:6])
        fresh = self._to_device(random_rotvec(n, generator=gen, dtype=torch.float64))
        reorient = self._to_device(
            torch.rand(n, 1, generator=gen, dtype=torch.float64)
        ) < cfg.reorient_probability
        rotvec = wrap_rotvec(torch.where(reorient, fresh, perturbed))

        if not p.max_torsions:
            return torch.cat([translation, rotvec], -1).view(n_lig, n_chain, -1)

        angles = flat[:, 6:]
        jitter = self._to_device(
            torch.randn(n, p.max_torsions, generator=gen, dtype=torch.float64)
        )
        jittered = angles + jitter * self.torsion_amplitude

        which = torch.randint(
            0, p.max_torsions, (n, 1), generator=gen, dtype=torch.long
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

        return torch.cat([translation, rotvec, angles], -1).view(n_lig, n_chain, -1)

    def local_optimize(self, x, config: Optional[LBFGSConfig] = None):
        """Relax every chain of every ligand in one call."""
        n_lig, n_chain, n_dof = x.shape

        def closure(flat):
            energy, grad = self.energy_and_dof_gradient(flat.view(n_lig, n_chain, n_dof))
            return energy.reshape(-1), grad.reshape(-1, n_dof)

        flat, energy, _ = batched_lbfgs(
            x.reshape(-1, n_dof), closure, config or LBFGSConfig()
        )
        flat = torch.cat([flat[:, :3], wrap_rotvec(flat[:, 3:6]), flat[:, 6:]], dim=-1)
        return flat.view(n_lig, n_chain, n_dof), energy.view(n_lig, n_chain)

    def run_basin_hopping(self, box_min, box_max, lbfgs_config=None):
        """
        Returns:
            x: (L, C, 6 + T) best DOF per chain
            energy: (L, C) energy at those parameters

        Take `energy.min(dim=1)` for each ligand's best pose.
        """
        cfg = self.config
        gen = self._generator()

        x = self.initial_state(box_min, box_max, gen)
        x, energy = self.local_optimize(x, lbfgs_config)
        best_x, best_energy = x.clone(), energy.clone()
        temperature = max(cfg.temperature, 1e-6)

        for _ in range(cfg.n_steps):
            trial = self.propose(x, gen)
            trial, trial_energy = self.local_optimize(trial, lbfgs_config)

            delta = trial_energy - energy
            u = self._to_device(
                torch.rand(
                    energy.shape[0], energy.shape[1], generator=gen, dtype=torch.float64
                )
            )
            accept = u < torch.exp(-delta / temperature)

            x = torch.where(accept.unsqueeze(-1), trial, x)
            energy = torch.where(accept, trial_energy, energy)

            improved = trial_energy < best_energy
            best_x = torch.where(improved.unsqueeze(-1), trial, best_x)
            best_energy = torch.where(improved, trial_energy, best_energy)

        return best_x, best_energy


def union_grids_and_type_ids(cpu_grids_per_ligand: Sequence):
    """
    One map stack covering every ligand's atom types, with remapped type ids.

    This is not optional bookkeeping. `LigandTyping` numbers types by order of
    first appearance *within one ligand*, so ligand A's type 0 and ligand B's
    type 0 are generally different physical types. Handing several ligands to a
    single grid stack without remapping would score most of them against the
    wrong maps -- quietly, since every index is in range and every energy looks
    plausible.

    A map is determined entirely by its signature (radius, hydrophobic, donor,
    acceptor), so the union stack takes each distinct signature from whichever
    ligand first carries it, and each ligand's ids are rewritten to point into
    it. Grids for identical signatures are identical by construction, which is
    the same fact the CPU's signature-keyed cache relies on.

    Returns:
        maps: (n_union_types, nx, ny, nz)
        type_ids: list of per-ligand heavy-atom id arrays into that stack
    """
    reference = cpu_grids_per_ligand[0]
    union_index = {}
    union_maps = []

    for grids in cpu_grids_per_ligand:
        if grids.maps.shape[1:] != reference.maps.shape[1:] or not np.allclose(
            grids.origin, reference.origin
        ):
            raise ValueError(
                "every ligand in a batch must share one receptor site: got grids "
                f"of shape {grids.maps.shape[1:]} at origin {grids.origin} against "
                f"{reference.maps.shape[1:]} at {reference.origin}"
            )
        for local_id, signature in enumerate(grids.typing.signatures):
            if signature not in union_index:
                union_index[signature] = len(union_maps)
                union_maps.append(grids.maps[local_id])

    remapped = []
    for grids in cpu_grids_per_ligand:
        lookup = np.array(
            [union_index[s] for s in grids.typing.signatures], dtype=np.int64
        )
        remapped.append(lookup[grids.typing.type_ids])

    return np.stack(union_maps), remapped


def build_multi_ligand_search(
    cpu_grids_per_ligand: Sequence,
    trees: Sequence,
    objectives: Sequence,
    config: Optional[RigidSearchConfig] = None,
    device=None,
    names=None,
) -> MultiLigandSearch:
    """
    Pack several ligands against one receptor site into a single batch.

    All ligands must share a receptor and site, the same condition the CPU's
    grid cache requires and the reason batching them is worth doing. Their atom
    types are unified first; see `union_grids_and_type_ids` for why that step
    cannot be skipped.
    """
    from ..scoring.vina_scoring import VinaScoring

    device = resolve_device(device)
    maps, type_ids = union_grids_and_type_ids(cpu_grids_per_ligand)
    reference = cpu_grids_per_ligand[0]

    grids = TorchAffinityGrids(
        origin=reference.origin,
        spacing=reference.spacing,
        maps=maps,
        shape=reference.shape,
        out_of_box_penalty=reference.out_of_box_penalty,
        device=device,
    )
    packed = pack_ligands(
        trees, objectives, type_ids, grids.dtype, device, names=names
    )
    return MultiLigandSearch(grids, packed, VinaScoring(), config=config)
