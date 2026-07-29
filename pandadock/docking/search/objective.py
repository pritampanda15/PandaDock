"""
Docking objective function over the ligand's degrees of freedom.

Maps a DOF vector (translation, orientation, torsions) to an energy and its
gradient. The receptor term comes from precomputed affinity grids; the
intramolecular term prevents the torsional search from folding the ligand
through itself.

All gradients are analytic. Translation is the atom-gradient sum, torsions come
from the rotational velocity of each moving subtree, and orientation is obtained
by mapping the accumulated torque through the derivative of the SO(3)
exponential map. No finite differences are involved, so the objective and its
gradient are exactly consistent for the quasi-Newton optimizer.
"""

import logging
from typing import Optional, Tuple

import numpy as np

from .grid_maps import AffinityGrids
from .rotations import cross_arrays, cross_vec_array, rodrigues_matrix, rotvec_gradient
from .torsion_tree import TorsionTree

logger = logging.getLogger("pandadock.docking.search.objective")


class DockingObjective:
    """
    Energy and gradient as a function of the ligand DOF vector.

    Args:
        tree: Articulated ligand model.
        grids: Precomputed receptor affinity grids.
        intra_weight: Scale on the intramolecular clash term.
    """

    def __init__(
        self,
        tree: TorsionTree,
        grids: AffinityGrids,
        intra_weight: float = 1.0,
    ):
        self.tree = tree
        self.grids = grids
        self.intra_weight = intra_weight

        self.heavy_atoms = tree.heavy_atoms
        self.n_atoms = tree.n_atoms
        self.n_dof = tree.n_dof

        # Intramolecular pair list and the Vina radii that go with it.
        pair_a, pair_b = tree.intramolecular_pairs()
        self.pair_a = pair_a
        self.pair_b = pair_b

        radii_by_atom = np.zeros(self.n_atoms, dtype=np.float64)
        radii_by_atom[grids.typing.heavy_atoms] = grids.typing.radii
        self.pair_radii_sum = (
            radii_by_atom[pair_a] + radii_by_atom[pair_b] if len(pair_a) else np.zeros(0)
        )

        self._scoring = None  # populated lazily for Vina intra parameters
        self.n_evaluations = 0

        # Cache: torsion angles -> torsioned coordinates, so the six orientation
        # finite-difference evaluations do not repeat the torsion chain.
        self._cached_angles: Optional[np.ndarray] = None
        self._cached_torsioned: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ geometry

    def _torsioned(self, angles: np.ndarray) -> np.ndarray:
        if (
            self._cached_torsioned is not None
            and self._cached_angles is not None
            and len(angles) == len(self._cached_angles)
            and np.array_equal(angles, self._cached_angles)
        ):
            return self._cached_torsioned
        coords = self.tree.apply_torsions(angles)
        self._cached_angles = angles.copy()
        self._cached_torsioned = coords
        return coords

    def coords(self, dof: np.ndarray) -> np.ndarray:
        """Full-molecule coordinates for a DOF vector."""
        torsioned = self._torsioned(dof[6:])
        return torsioned @ rodrigues_matrix(dof[3:6]).T + dof[:3]

    # -------------------------------------------------------------- energy terms

    def _intramolecular(
        self, coords: np.ndarray, need_gradient: bool
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Repulsive intramolecular term over pairs that torsions can bring together.

        Uses Vina's gauss/repulsion forms so that self-contact is penalised on the
        same scale as receptor contact.
        """
        if len(self.pair_a) == 0:
            return 0.0, (np.zeros((self.n_atoms, 3)) if need_gradient else None)

        if self._scoring is None:
            from ..scoring.vina_scoring import VinaScoring

            self._scoring = VinaScoring()
        s = self._scoring
        w = s.weights

        delta = coords[self.pair_a] - coords[self.pair_b]
        dist = np.linalg.norm(delta, axis=1)
        dist = np.maximum(dist, 1e-6)
        surf = dist - self.pair_radii_sum

        active = surf < s.cutoff

        g1 = np.exp(-((surf - s.gauss1_offset) / s.gauss1_width) ** 2)
        g2 = np.exp(-((surf - s.gauss2_offset) / s.gauss2_width) ** 2)
        rep = np.where(surf < s.repulsion_cutoff, surf**2, 0.0)

        per_pair = w["gauss1"] * g1 + w["gauss2"] * g2 + w["repulsion"] * rep
        per_pair = np.where(active, per_pair, 0.0)
        energy = self.intra_weight * float(np.sum(per_pair))

        if not need_gradient:
            return energy, None

        d_g1 = g1 * (-2.0 * (surf - s.gauss1_offset) / s.gauss1_width**2)
        d_g2 = g2 * (-2.0 * (surf - s.gauss2_offset) / s.gauss2_width**2)
        d_rep = np.where(surf < s.repulsion_cutoff, 2.0 * surf, 0.0)

        d_dsurf = w["gauss1"] * d_g1 + w["gauss2"] * d_g2 + w["repulsion"] * d_rep
        d_dsurf = np.where(active, d_dsurf, 0.0) * self.intra_weight

        # surf depends on distance only, so d/d(coords) follows the unit separation.
        unit = delta / dist[:, None]
        contrib = d_dsurf[:, None] * unit

        grad = np.zeros((self.n_atoms, 3), dtype=np.float64)
        np.add.at(grad, self.pair_a, contrib)
        np.add.at(grad, self.pair_b, -contrib)
        return energy, grad

    # ---------------------------------------------------------------- public API

    def energy(self, dof: np.ndarray) -> float:
        self.n_evaluations += 1
        coords = self.coords(dof)
        inter = self.grids.score(coords[self.heavy_atoms])
        intra, _ = self._intramolecular(coords, need_gradient=False)
        return inter + intra

    def energy_and_gradient(self, dof: np.ndarray) -> Tuple[float, np.ndarray]:
        self.n_evaluations += 1

        torsioned = self._torsioned(dof[6:])
        rot = rodrigues_matrix(dof[3:6])
        coords = torsioned @ rot.T + dof[:3]

        inter, grad_heavy = self.grids.score_and_gradient(coords[self.heavy_atoms])
        intra, grad_intra = self._intramolecular(coords, need_gradient=True)
        energy = inter + intra

        # Per-atom Cartesian gradient (hydrogens contribute nothing to the score,
        # but they still move, so they simply carry zero gradient).
        grad_atoms = grad_intra
        grad_atoms[self.heavy_atoms] += grad_heavy

        grad = np.zeros(self.n_dof, dtype=np.float64)

        # Translation: the pose shifts rigidly, so the gradient is the atom sum.
        grad[:3] = np.sum(grad_atoms, axis=0)

        # Orientation: accumulate the torque about the translation anchor, then map
        # it through the derivative of the SO(3) exponential map. Exact, and needs
        # no extra energy evaluations.
        # (The intramolecular term is rotation invariant, so it contributes zero
        # torque and is correctly included as part of grad_atoms.)
        torque = np.sum(cross_arrays(coords - dof[:3], grad_atoms), axis=0)
        grad[3:6] = rotvec_gradient(dof[3:6], rot, torque)

        # Torsions: rotating about axis u through pivot p moves atom m with
        # velocity u x (c_m - p), so dE/dtheta is that velocity projected on the
        # Cartesian gradient. Exact, and evaluated in the final (rotated) frame.
        if self.tree.n_torsions:
            axes = self.tree.torsion_axes(coords)
            for k, (torsion, (axis, pivot)) in enumerate(zip(self.tree.torsions, axes)):
                moving = torsion.moving
                velocity = cross_vec_array(axis, coords[moving] - pivot)
                grad[6 + k] = float(np.sum(grad_atoms[moving] * velocity))

        return energy, grad

    def scipy_objective(self, dof: np.ndarray) -> Tuple[float, np.ndarray]:
        """Adapter for `scipy.optimize.minimize(..., jac=True)`."""
        return self.energy_and_gradient(np.asarray(dof, dtype=np.float64))
