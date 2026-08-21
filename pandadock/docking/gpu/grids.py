"""
Batched trilinear grid scoring on the GPU.

This is a direct port of `AffinityGrids.score_and_gradient`, kept deliberately
line-comparable to the NumPy original so that the two can be read side by side
when a parity test fails. The only intentional differences are that this version
scores a batch of poses at once and returns tensors on the grid's device.

Two properties are load-bearing and are asserted by the tests:

* The energy and gradient must match the CPU implementation. The CPU path is the
  one validated against 814 complexes in the manuscript; this path has no
  independent claim to correctness and is only useful insofar as it agrees.
* Atoms outside the grid must receive the same quadratic boundary penalty. It is
  what keeps the optimiser inside the box, and dropping it would let poses drift
  out of the pocket while scoring well.

Precision note: the CPU search runs its arithmetic in float64, but the maps
themselves are stored as float32. CUDA supports float64 (slowly); Apple's MPS
backend does not support it at all, so on MPS the grids fall back to float32 and
the parity tolerance is correspondingly looser. `dtype_for_device` encodes that
rule in one place rather than leaving each caller to discover it.

One deliberate numerical difference from the CPU. The x-derivative begins with
`c100 - c000`, and on the CPU both operands are raw float32 map values, so that
subtraction happens in float32 and loses roughly seven digits to cancellation
before anything is promoted. The y and z derivatives subtract intermediates that
numpy has already promoted to float64 and so are unaffected. Because this class
uploads the maps as float64, it performs that subtraction in float64 and is the
more accurate of the two -- agreement on the x-gradient is therefore about 1e-8
relative rather than round-off. Energies are unaffected: they never difference
two corners. Do not restore bit-identity by downcasting; see
tests/test_gpu_grids.py::test_x_gradient_differs_only_by_float32_cancellation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch

    torch_available = True
except ImportError:  # pragma: no cover - exercised only in a base install
    torch = None
    torch_available = False


def dtype_for_device(device) -> "torch.dtype":
    """
    Widest float the device actually supports.

    MPS silently has no float64 -- allocating one raises -- so asking for the
    CPU's precision there fails at construction rather than producing quietly
    wrong numbers later.
    """
    return torch.float32 if torch.device(device).type == "mps" else torch.float64


def resolve_device(device=None) -> "torch.device":
    """
    Pick a device, preferring an accelerator but never failing because of one.

    Explicit requests are honoured as given so that a test asking for CUDA fails
    loudly on a machine without it instead of silently measuring the CPU.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchAffinityGrids:
    """
    Receptor affinity grids resident on a torch device.

    Built once per receptor and reused across every pose and every ligand that
    shares the grids, which is what makes the batched search worthwhile: the maps
    are by far the largest tensor involved and must not be re-uploaded per pose.
    """

    def __init__(
        self,
        origin: np.ndarray,
        spacing: float,
        maps: np.ndarray,
        shape: np.ndarray,
        out_of_box_penalty: float = 10.0,
        device=None,
        dtype: Optional["torch.dtype"] = None,
    ):
        if not torch_available:
            raise ImportError(
                "The GPU search path needs the optional [gnn] extra "
                "(pip install pandadock[gnn]) for torch."
            )
        self.device = resolve_device(device)
        self.dtype = dtype if dtype is not None else dtype_for_device(self.device)

        self.maps = torch.as_tensor(maps, dtype=self.dtype, device=self.device)
        self.origin = torch.as_tensor(origin, dtype=self.dtype, device=self.device)
        self.spacing = float(spacing)
        self.out_of_box_penalty = float(out_of_box_penalty)

        # Kept as a tensor for the clamp bounds and as ints for indexing maths.
        self.shape = torch.as_tensor(
            np.asarray(shape, dtype=np.int64), dtype=torch.long, device=self.device
        )

        # The corner lookup is a single flat gather rather than four-dimensional
        # advanced indexing: one contiguous read per corner is far friendlier to
        # a GPU's memory system, and it avoids materialising broadcast index
        # tensors for each of the four dimensions.
        _, nx, ny, nz = self.maps.shape
        self._flat_maps = self.maps.contiguous().reshape(-1)
        # Row-major strides of (n_types, nx, ny, nz); the trailing z stride of 1
        # is applied directly in _gather_corners.
        self._stride = torch.as_tensor(
            [nx * ny * nz, ny * nz, nz], dtype=torch.long, device=self.device
        )

    @classmethod
    def from_cpu(cls, grids, device=None, dtype: Optional["torch.dtype"] = None):
        """Upload an existing CPU `AffinityGrids`, preserving every parameter."""
        return cls(
            origin=grids.origin,
            spacing=grids.spacing,
            maps=grids.maps,
            shape=grids.shape,
            out_of_box_penalty=grids.out_of_box_penalty,
            device=device,
            dtype=dtype,
        )

    def _gather_corners(self, type_ids, x, y, z):
        """One flat index per atom per corner, then a single gather."""
        flat = (
            type_ids * self._stride[0]
            + x * self._stride[1]
            + y * self._stride[2]
            + z
        )
        return self._flat_maps[flat.reshape(-1)].reshape(flat.shape)

    def score_and_gradient(
        self,
        coords: "torch.Tensor",
        type_ids: "torch.Tensor",
        need_gradient: bool = True,
        atom_mask: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Trilinear interpolation with analytic gradient, over a batch of poses.

        Args:
            coords: (B, N, 3) heavy-atom coordinates.
            type_ids: (N,) atom types, or (B, N) when poses carry different
                ligands, as they do once several are packed into one batch.
            need_gradient: skip the derivative when only the energy is wanted.
            atom_mask: (B, N) bool, False for padding. Required when ligands of
                different sizes share a batch: a padded slot still lands
                somewhere in the grid and would otherwise contribute both an
                energy and a boundary penalty for an atom that does not exist.

        Returns:
            energy: (B,) total interaction energy including the boundary penalty.
            gradient: (B, N, 3) d(energy)/d(coord), or None. Padded slots carry
                zero, so they cannot move anything downstream.
        """
        if coords.dim() != 3 or coords.shape[-1] != 3:
            raise ValueError(f"coords must be (B, N, 3), got {tuple(coords.shape)}")

        coords = coords.to(device=self.device, dtype=self.dtype)
        type_ids = type_ids.to(device=self.device, dtype=torch.long)

        frac = (coords - self.origin) / self.spacing

        # Clamp to the last cell that still has a +1 neighbour, and keep the part
        # that was clipped away: that overflow is what the penalty charges for,
        # and it is exactly zero for any atom inside the grid.
        hi = (self.shape - 1).to(self.dtype)
        clamped = torch.minimum(
            torch.clamp(frac, min=0.0), hi - 1e-6
        )
        overflow = frac - clamped

        i0 = torch.minimum(clamped.floor().long(), self.shape - 2)
        t = clamped - i0.to(self.dtype)
        u, v, w = t[..., 0], t[..., 1], t[..., 2]

        x0, y0, z0 = i0[..., 0], i0[..., 1], i0[..., 2]
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

        # type_ids is (N,) for one ligand or (B, N) for a mixed batch; either
        # broadcasts against the (B, N) coordinate indices.
        tid = type_ids.expand_as(x0) if type_ids.dim() == 1 else type_ids

        c000 = self._gather_corners(tid, x0, y0, z0)
        c100 = self._gather_corners(tid, x1, y0, z0)
        c010 = self._gather_corners(tid, x0, y1, z0)
        c110 = self._gather_corners(tid, x1, y1, z0)
        c001 = self._gather_corners(tid, x0, y0, z1)
        c101 = self._gather_corners(tid, x1, y0, z1)
        c011 = self._gather_corners(tid, x0, y1, z1)
        c111 = self._gather_corners(tid, x1, y1, z1)

        one_u, one_v, one_w = 1.0 - u, 1.0 - v, 1.0 - w

        c00 = c000 * one_u + c100 * u
        c10 = c010 * one_u + c110 * u
        c01 = c001 * one_u + c101 * u
        c11 = c011 * one_u + c111 * u

        c0 = c00 * one_v + c10 * v
        c1 = c01 * one_v + c11 * v

        values = c0 * one_w + c1 * w

        overflow_sq = (overflow**2).sum(dim=-1)
        if atom_mask is not None:
            zero = torch.zeros_like(values)
            values = torch.where(atom_mask, values, zero)
            overflow_sq = torch.where(atom_mask, overflow_sq, torch.zeros_like(overflow_sq))

        penalty = self.out_of_box_penalty * overflow_sq.sum(dim=1)
        energy = values.sum(dim=1) + penalty

        if not need_gradient:
            return energy, None

        d_du = ((c100 - c000) * one_v + (c110 - c010) * v) * one_w + (
            (c101 - c001) * one_v + (c111 - c011) * v
        ) * w
        d_dv = (c10 - c00) * one_w + (c11 - c01) * w
        d_dw = c1 - c0

        grad = torch.stack([d_du, d_dv, d_dw], dim=-1) / self.spacing
        grad = grad + 2.0 * self.out_of_box_penalty * overflow / self.spacing
        if atom_mask is not None:
            grad = torch.where(atom_mask.unsqueeze(-1), grad, torch.zeros_like(grad))
        return energy, grad
