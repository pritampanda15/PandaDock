"""
Precomputed affinity grids for fast pose scoring.

A docking search evaluates the scoring function tens of thousands of times per
ligand. Recomputing every ligand-receptor atom pair each time is what makes naive
implementations too slow to run a real search. Instead the receptor contribution
is precomputed once per ligand atom type on a regular grid, and pose scoring
becomes a trilinear lookup per ligand atom.

The functional form and weights are taken from `VinaScoring`, so grid scores and
direct scores agree to within interpolation error, and fixing the atom typing in
one place fixes it for both.

Gradients are the analytic gradients of the trilinear interpolant itself, which
keeps the objective and its gradient exactly consistent for the local optimizer.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem

from ..scoring.vina_scoring import VinaScoring

logger = logging.getLogger("pandadock.docking.search.grid_maps")


class LigandTyping:
    """Vina-style types, radii and interaction flags for the ligand's heavy atoms."""

    def __init__(self, mol: Chem.Mol, scoring: VinaScoring):
        self.heavy_atoms = np.array(
            [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1], dtype=np.int64
        )

        all_types = scoring._get_ligand_atom_types(mol)
        hydrophobic = scoring._is_hydrophobic_ligand(mol)
        donors, acceptors = scoring._get_hbond_atoms_ligand(mol)
        donor_set, acceptor_set = set(donors), set(acceptors)

        self.radii = np.array(
            [scoring._get_vdw_radius(all_types[i]) for i in self.heavy_atoms], dtype=np.float64
        )
        self.is_hydrophobic = np.array(
            [bool(hydrophobic[i]) for i in self.heavy_atoms], dtype=bool
        )
        self.is_donor = np.array([int(i) in donor_set for i in self.heavy_atoms], dtype=bool)
        self.is_acceptor = np.array([int(i) in acceptor_set for i in self.heavy_atoms], dtype=bool)

        # Atoms sharing a (radius, hydrophobic, donor, acceptor) signature see an
        # identical receptor field, so they can share one grid.
        signatures = [
            (round(float(r), 3), bool(h), bool(d), bool(a))
            for r, h, d, a in zip(self.radii, self.is_hydrophobic, self.is_donor, self.is_acceptor)
        ]
        unique: List[Tuple] = []
        index: Dict[Tuple, int] = {}
        for sig in signatures:
            if sig not in index:
                index[sig] = len(unique)
                unique.append(sig)
        self.signatures = unique
        self.type_ids = np.array([index[s] for s in signatures], dtype=np.int64)

    @property
    def n_types(self) -> int:
        return len(self.signatures)

    @property
    def n_heavy(self) -> int:
        return len(self.heavy_atoms)


class AffinityGrids:
    """
    Per-ligand-atom-type receptor interaction grids over the docking box.

    Args:
        origin: Cartesian coordinate of grid point (0, 0, 0).
        spacing: Grid spacing in Angstrom.
        maps: Array of shape (n_types, nx, ny, nz) with interaction energies.
        typing: Ligand typing used to build the maps.
        box_min / box_max: The user's docking box (not the padded grid extent).
    """

    def __init__(
        self,
        origin: np.ndarray,
        spacing: float,
        maps: np.ndarray,
        typing: LigandTyping,
        box_min: np.ndarray,
        box_max: np.ndarray,
        out_of_box_penalty: float = 10.0,
    ):
        self.origin = origin
        self.spacing = float(spacing)
        self.maps = maps
        self.typing = typing
        self.box_min = box_min
        self.box_max = box_max
        self.out_of_box_penalty = out_of_box_penalty
        self.shape = np.array(maps.shape[1:], dtype=np.int64)

    # ---------------------------------------------------------------- construction

    @classmethod
    def build(
        cls,
        receptor_structure,
        ligand_mol: Chem.Mol,
        grid_center: np.ndarray,
        grid_dimensions: np.ndarray,
        spacing: float = 0.375,
        margin: float = 4.0,
        scoring: Optional[VinaScoring] = None,
        chunk_size: int = 4096,
    ) -> "AffinityGrids":
        """
        Precompute interaction grids for `ligand_mol` against `receptor_structure`.

        `margin` pads the grid beyond the docking box so that atoms of a ligand
        whose centre sits at the box edge still land on valid grid points.
        """
        scoring = scoring or VinaScoring()
        typing = LigandTyping(ligand_mol, scoring)

        grid_center = np.asarray(grid_center, dtype=np.float64)
        grid_dimensions = np.asarray(grid_dimensions, dtype=np.float64)
        box_min = grid_center - grid_dimensions / 2.0
        box_max = grid_center + grid_dimensions / 2.0

        origin = box_min - margin
        extent = grid_dimensions + 2.0 * margin
        shape = np.ceil(extent / spacing).astype(int) + 1

        n_points = int(np.prod(shape))
        logger.info(
            "Building %d affinity grid(s) at %.3f A spacing: %d x %d x %d = %d points",
            typing.n_types,
            spacing,
            shape[0],
            shape[1],
            shape[2],
            n_points,
        )

        rec_coords, rec_radii, rec_hydrophobic, rec_donor, rec_acceptor = cls._prepare_receptor(
            receptor_structure, scoring, origin, origin + (shape - 1) * spacing, scoring.cutoff
        )

        if len(rec_coords) == 0:
            logger.warning(
                "No receptor atoms within the docking box; grids will be empty. "
                "Check that the box centre and dimensions match the receptor."
            )
            maps = np.zeros((typing.n_types,) + tuple(shape), dtype=np.float32)
            return cls(origin, spacing, maps, typing, box_min, box_max)

        # Flattened grid point coordinates, generated per chunk to bound memory.
        gx = origin[0] + np.arange(shape[0]) * spacing
        gy = origin[1] + np.arange(shape[1]) * spacing
        gz = origin[2] + np.arange(shape[2]) * spacing

        maps = np.zeros((typing.n_types, n_points), dtype=np.float32)

        w = scoring.weights
        cutoff = scoring.cutoff

        for start in range(0, n_points, chunk_size):
            stop = min(start + chunk_size, n_points)
            flat = np.arange(start, stop)
            ix, rem = np.divmod(flat, shape[1] * shape[2])
            iy, iz = np.divmod(rem, shape[2])
            pts = np.stack([gx[ix], gy[iy], gz[iz]], axis=1)

            # Pairwise distances for this chunk, shared across all ligand types.
            d = np.linalg.norm(pts[:, None, :] - rec_coords[None, :, :], axis=2)
            within = d < (cutoff + rec_radii.max() + 2.5)
            if not np.any(within):
                continue

            for t, (radius, hydrophobic, donor, acceptor) in enumerate(typing.signatures):
                surf = d - radius - rec_radii[None, :]
                active = surf < cutoff

                energy = np.zeros_like(surf)

                g1 = np.exp(-((surf - scoring.gauss1_offset) / scoring.gauss1_width) ** 2)
                g2 = np.exp(-((surf - scoring.gauss2_offset) / scoring.gauss2_width) ** 2)
                energy += w["gauss1"] * g1 + w["gauss2"] * g2

                rep = np.where(surf < scoring.repulsion_cutoff, surf**2, 0.0)
                energy += w["repulsion"] * rep

                if hydrophobic:
                    h = np.clip(
                        (scoring.hydrophobic_bad - surf)
                        / (scoring.hydrophobic_bad - scoring.hydrophobic_good),
                        0.0,
                        1.0,
                    )
                    energy += w["hydrophobic"] * (h * rec_hydrophobic[None, :])

                hb_partner = np.zeros(len(rec_coords), dtype=np.float64)
                if donor:
                    hb_partner += rec_acceptor
                if acceptor:
                    hb_partner += rec_donor
                if np.any(hb_partner > 0):
                    hb = np.clip(
                        (scoring.hbond_bad - surf) / (scoring.hbond_bad - scoring.hbond_good),
                        0.0,
                        1.0,
                    )
                    energy += w["hydrogen"] * (hb * np.minimum(hb_partner, 1.0)[None, :])

                maps[t, start:stop] = np.sum(np.where(active, energy, 0.0), axis=1)

        maps = maps.reshape((typing.n_types,) + tuple(shape))
        logger.info("Affinity grids built (%.1f MB)", maps.nbytes / 1e6)
        return cls(origin, spacing, maps, typing, box_min, box_max)

    @staticmethod
    def _prepare_receptor(
        receptor_structure,
        scoring: VinaScoring,
        lo: np.ndarray,
        hi: np.ndarray,
        cutoff: float,
    ):
        """Type the receptor and keep only atoms that can influence the grid."""
        atoms = [a for a in receptor_structure.get_atoms() if a.element.strip() != "H"]
        if not atoms:
            atoms = list(receptor_structure.get_atoms())

        coords = np.array([a.get_coord() for a in atoms], dtype=np.float64)
        keep = np.all((coords >= lo - cutoff) & (coords <= hi + cutoff), axis=1)

        kept_atoms = [a for a, k in zip(atoms, keep) if k]
        if not kept_atoms:
            empty = np.zeros((0, 3))
            return empty, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)

        types = scoring._get_receptor_atom_types(kept_atoms)
        radii = np.array([scoring._get_vdw_radius(t) for t in types], dtype=np.float64)
        hydrophobic = np.array(
            scoring._is_hydrophobic_receptor(kept_atoms), dtype=np.float64
        )
        donors, acceptors = scoring._get_hbond_atoms_receptor(kept_atoms)
        is_donor = np.zeros(len(kept_atoms), dtype=np.float64)
        is_acceptor = np.zeros(len(kept_atoms), dtype=np.float64)
        is_donor[list(donors)] = 1.0
        is_acceptor[list(acceptors)] = 1.0

        logger.debug("Receptor atoms contributing to grid: %d", len(kept_atoms))
        return coords[keep], radii, hydrophobic, is_donor, is_acceptor

    # -------------------------------------------------------------------- scoring

    def score(self, heavy_coords: np.ndarray) -> float:
        """Interpolated receptor interaction energy for the ligand's heavy atoms."""
        energy, _ = self.score_and_gradient(heavy_coords, need_gradient=False)
        return energy

    def score_and_gradient(
        self, heavy_coords: np.ndarray, need_gradient: bool = True
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Trilinear interpolation of the affinity grids, with analytic gradient.

        Atoms outside the grid receive a quadratic penalty that grows with the
        distance past the boundary, which keeps the optimizer's search confined to
        the box without introducing a discontinuity at the edge.
        """
        type_ids = self.typing.type_ids
        frac = (heavy_coords - self.origin) / self.spacing

        hi = self.shape - 1
        clamped = np.clip(frac, 0.0, hi.astype(np.float64) - 1e-6)
        overflow = frac - clamped  # zero inside the grid

        i0 = np.floor(clamped).astype(np.int64)
        i0 = np.minimum(i0, hi - 1)
        t = clamped - i0
        u, v, w = t[:, 0], t[:, 1], t[:, 2]

        x0, y0, z0 = i0[:, 0], i0[:, 1], i0[:, 2]
        x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

        m = self.maps
        c000 = m[type_ids, x0, y0, z0]
        c100 = m[type_ids, x1, y0, z0]
        c010 = m[type_ids, x0, y1, z0]
        c110 = m[type_ids, x1, y1, z0]
        c001 = m[type_ids, x0, y0, z1]
        c101 = m[type_ids, x1, y0, z1]
        c011 = m[type_ids, x0, y1, z1]
        c111 = m[type_ids, x1, y1, z1]

        one_u, one_v, one_w = 1.0 - u, 1.0 - v, 1.0 - w

        c00 = c000 * one_u + c100 * u
        c10 = c010 * one_u + c110 * u
        c01 = c001 * one_u + c101 * u
        c11 = c011 * one_u + c111 * u

        c0 = c00 * one_v + c10 * v
        c1 = c01 * one_v + c11 * v

        values = c0 * one_w + c1 * w

        penalty = self.out_of_box_penalty * np.sum(overflow**2)
        energy = float(np.sum(values)) + float(penalty)

        if not need_gradient:
            return energy, None

        d_du = ((c100 - c000) * one_v + (c110 - c010) * v) * one_w + (
            (c101 - c001) * one_v + (c111 - c011) * v
        ) * w
        d_dv = (c10 - c00) * one_w + (c11 - c01) * w
        d_dw = c1 - c0

        grad = np.stack([d_du, d_dv, d_dw], axis=1) / self.spacing
        grad += 2.0 * self.out_of_box_penalty * overflow / self.spacing
        return energy, grad
