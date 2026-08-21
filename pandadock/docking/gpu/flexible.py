"""
Batched flexible-ligand pose construction.

Stage three. Adds torsional degrees of freedom to the batched search, so a pose
is now (translation, rotation, theta_1 .. theta_k) exactly as on the CPU.

Two things arrive together here, and both are required for parity:

* Torsion application. Rotations are applied root-outward against the running
  coordinates, so each torsion turns about an axis its ancestors have already
  moved. That is sequential in the torsions -- and must stay sequential, since
  applying them in parallel against the reference geometry silently produces a
  different molecule -- but it is batched across poses, which is where the work
  is. Ligands have at most 32 torsions and typically fewer than eight, so the
  serial loop is short.

* The intramolecular clash term. For a rigid ligand this is exactly zero: no
  torsion can change any distance, so `intramolecular_pairs` returns nothing and
  stages one, two and four could compare against the grid energy alone. As soon
  as torsions move, pairs can be driven into each other, and the CPU objective
  charges for it. Omitting the term here would leave the GPU scoring folded
  conformations the CPU rejects.

The clash term is written in plain torch and left to autograd rather than having
its analytic gradient ported. It is a smooth function of coordinates, so
autograd is exact for it, and there is no non-differentiable indexing to hide
from as there is in the grid interpolation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import torch

    torch_available = True
except ImportError:  # pragma: no cover - exercised only in a base install
    torch = None
    torch_available = False


class TorsionApplier:
    """
    Applies a batch of torsion angles to one ligand's reference geometry.

    The per-torsion index arrays are uploaded once at construction. They are
    fixed properties of the molecular graph, so rebuilding them per call would
    be pure overhead in the inner loop of the search.
    """

    def __init__(self, tree, dtype, device):
        if not torch_available:  # pragma: no cover
            raise ImportError("the GPU search path needs torch")

        self.dtype = dtype
        self.device = device
        self.n_torsions = tree.n_torsions

        self.base_coords = torch.as_tensor(
            tree.base_coords, dtype=dtype, device=device
        )
        self.heavy_atoms = torch.as_tensor(
            np.asarray(tree.heavy_atoms), dtype=torch.long, device=device
        )

        self._origin: List[int] = []
        self._axis: List[int] = []
        self._moving: List["torch.Tensor"] = []
        for torsion in tree.torsions:
            self._origin.append(int(torsion.origin_atom))
            self._axis.append(int(torsion.axis_atom))
            self._moving.append(
                torch.as_tensor(
                    np.asarray(torsion.moving), dtype=torch.long, device=device
                )
            )

    def apply(self, angles: "torch.Tensor") -> "torch.Tensor":
        """
        (B, k) torsion angles -> (B, n_atoms, 3) coordinates.

        Mirrors `TorsionTree.apply_torsions`. The CPU skips a torsion whose angle
        is exactly zero; that is an optimisation for a scalar, and skipping here
        would need the angle to be zero for every pose in the batch at once, so
        the rotation is simply always applied. It is the identity at zero, so the
        result is the same.
        """
        batch = angles.shape[0]
        coords = self.base_coords.unsqueeze(0).expand(batch, -1, -1)
        if self.n_torsions == 0:
            return coords

        # `expand` gives a read-only view; the first write needs real storage.
        coords = coords.clone()

        for i in range(self.n_torsions):
            origin = coords[:, self._origin[i], :]
            pivot = coords[:, self._axis[i], :]
            axis = pivot - origin

            norm = axis.norm(dim=-1, keepdim=True)
            # Matches the CPU's degenerate-axis fallback: a zero-length bond
            # vector has no meaningful rotation axis, so it uses +x and the
            # angle then rotates nothing of consequence.
            fallback = torch.zeros_like(axis)
            fallback[:, 0] = 1.0
            k = torch.where(norm > 1e-8, axis / norm.clamp_min(1e-12), fallback)

            moving = self._moving[i]
            points = coords[:, moving, :]
            v = points - pivot.unsqueeze(1)

            angle = angles[:, i].reshape(batch, 1, 1)
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            k_exp = k.unsqueeze(1)

            # v cos + (k x v) sin + k (k . v)(1 - cos), as on the CPU.
            rotated = (
                v * cos_a
                + torch.cross(k_exp.expand_as(v), v, dim=-1) * sin_a
                + k_exp * (v * k_exp).sum(-1, keepdim=True) * (1.0 - cos_a)
            )
            rotated = rotated + pivot.unsqueeze(1)

            # index_copy along the atom axis; out-of-place so autograd can
            # differentiate through the sequence of torsions.
            coords = coords.index_copy(1, moving, rotated)

        return coords


class IntramolecularClash:
    """
    Vina gauss/repulsion penalty over pairs that torsions can bring together.

    Uses the same pair list, radii and weights the CPU objective builds, read
    from it directly rather than recomputed, so the two cannot drift apart
    through a transcription error.
    """

    def __init__(self, objective, dtype, device):
        if not torch_available:  # pragma: no cover
            raise ImportError("the GPU search path needs torch")

        self.dtype = dtype
        self.device = device
        self.n_pairs = len(objective.pair_a)
        self.weight = float(objective.intra_weight)

        if self.n_pairs == 0:
            return

        self.pair_a = torch.as_tensor(
            np.asarray(objective.pair_a), dtype=torch.long, device=device
        )
        self.pair_b = torch.as_tensor(
            np.asarray(objective.pair_b), dtype=torch.long, device=device
        )
        self.radii_sum = torch.as_tensor(
            np.asarray(objective.pair_radii_sum), dtype=dtype, device=device
        )

        from ..scoring.vina_scoring import VinaScoring

        s = objective._scoring or VinaScoring()
        self.cutoff = float(s.cutoff)
        self.g1_offset = float(s.gauss1_offset)
        self.g1_width = float(s.gauss1_width)
        self.g2_offset = float(s.gauss2_offset)
        self.g2_width = float(s.gauss2_width)
        self.rep_cutoff = float(s.repulsion_cutoff)
        self.w_g1 = float(s.weights["gauss1"])
        self.w_g2 = float(s.weights["gauss2"])
        self.w_rep = float(s.weights["repulsion"])

    def energy(self, coords: "torch.Tensor") -> "torch.Tensor":
        """(B, n_atoms, 3) -> (B,), differentiable in `coords`."""
        if self.n_pairs == 0:
            return torch.zeros(
                coords.shape[0], dtype=coords.dtype, device=coords.device
            )

        delta = coords[:, self.pair_a, :] - coords[:, self.pair_b, :]
        # clamp_min before the sqrt: at coincident atoms the norm's derivative is
        # undefined, and a NaN there would propagate to every torsion upstream.
        dist = delta.pow(2).sum(-1).clamp_min(1e-12).sqrt()
        surf = dist - self.radii_sum

        g1 = torch.exp(-(((surf - self.g1_offset) / self.g1_width) ** 2))
        g2 = torch.exp(-(((surf - self.g2_offset) / self.g2_width) ** 2))
        rep = torch.where(surf < self.rep_cutoff, surf**2, torch.zeros_like(surf))

        per_pair = self.w_g1 * g1 + self.w_g2 * g2 + self.w_rep * rep
        per_pair = torch.where(surf < self.cutoff, per_pair, torch.zeros_like(per_pair))
        return self.weight * per_pair.sum(dim=-1)

    def energy_and_gradient(self, coords: "torch.Tensor"):
        """
        Energy and d(energy)/d(coord) without autograd.

        Mirrors the CPU's `_intramolecular(..., need_gradient=True)`. `surf`
        depends only on the pair distance, so the Cartesian derivative follows
        the unit separation vector and each pair contributes equally and
        oppositely to its two atoms.
        """
        n_atoms = coords.shape[1]
        if self.n_pairs == 0:
            zeros = torch.zeros_like(coords)
            return (
                torch.zeros(coords.shape[0], dtype=coords.dtype, device=coords.device),
                zeros,
            )

        delta = coords[:, self.pair_a, :] - coords[:, self.pair_b, :]
        dist = delta.pow(2).sum(-1).clamp_min(1e-12).sqrt()
        surf = dist - self.radii_sum
        active = surf < self.cutoff

        g1 = torch.exp(-(((surf - self.g1_offset) / self.g1_width) ** 2))
        g2 = torch.exp(-(((surf - self.g2_offset) / self.g2_width) ** 2))
        rep = torch.where(surf < self.rep_cutoff, surf**2, torch.zeros_like(surf))

        per_pair = self.w_g1 * g1 + self.w_g2 * g2 + self.w_rep * rep
        per_pair = torch.where(active, per_pair, torch.zeros_like(per_pair))
        energy = self.weight * per_pair.sum(dim=-1)

        d_g1 = g1 * (-2.0 * (surf - self.g1_offset) / self.g1_width**2)
        d_g2 = g2 * (-2.0 * (surf - self.g2_offset) / self.g2_width**2)
        d_rep = torch.where(
            surf < self.rep_cutoff, 2.0 * surf, torch.zeros_like(surf)
        )

        d_dsurf = self.w_g1 * d_g1 + self.w_g2 * d_g2 + self.w_rep * d_rep
        d_dsurf = torch.where(active, d_dsurf, torch.zeros_like(d_dsurf)) * self.weight

        unit = delta / dist.unsqueeze(-1)
        contrib = d_dsurf.unsqueeze(-1) * unit

        grad = torch.zeros_like(coords)
        grad.index_add_(1, self.pair_a, contrib)
        grad.index_add_(1, self.pair_b, -contrib)
        return energy, grad
