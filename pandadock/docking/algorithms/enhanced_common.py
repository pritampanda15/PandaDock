"""
Shared pose-handling utilities for docking algorithms.

Provides unbiased pose generation, clash detection, and diversity filtering.

Note on history: earlier versions of this module contained "crystal-guided"
sampling, which placed the ligand within ~1 A of a reference point using a
rotation drawn from a narrow Gaussian about the input conformer's orientation,
and then added an energy bonus for staying near that point. That is not a search:
it never explored orientation or conformation, and the bonus dominated the
interaction energy, so poses were effectively ranked by distance from the box
centre. It also carried a hardcoded ligand coordinate that silently captured any
docking box placed near it. All of that has been removed. Pose generation here is
uniform over position and orientation, and ranking uses the scoring function
alone.

For production docking use `PandaCoreDocker`, which searches translation,
orientation and ligand torsions against precomputed affinity grids.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from rdkit import Chem

from ..core import Pose
from ..search.rotations import quaternion_from_rotvec, random_rotvec, rodrigues_matrix

logger = logging.getLogger("pandadock.docking.algorithms.common")


class EnhancedDockingMixin:
    """Pose generation, clash detection and diversity filtering for docking algorithms."""

    def _generate_random_pose(
        self,
        mol: Chem.Mol,
        conf_id: int,
        grid_center: np.ndarray,
        grid_dimensions: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> Pose:
        """
        Generate a pose uniformly over the box and over SO(3).

        Both the position and the orientation are drawn without reference to the
        input conformer's placement, so no region of the search space is favoured
        by how the ligand file happened to be written.
        """
        rng = rng if rng is not None else np.random.default_rng()

        translation = rng.uniform(
            grid_center - grid_dimensions / 2.0,
            grid_center + grid_dimensions / 2.0,
        )
        rotvec = random_rotvec(rng)

        conf = mol.GetConformer(conf_id)
        ligand_coords = np.asarray(conf.GetPositions())
        centered = ligand_coords - ligand_coords.mean(axis=0)
        coords = centered @ rodrigues_matrix(rotvec).T + translation

        return Pose(
            coordinates=coords,
            center=translation,
            rotation=quaternion_from_rotvec(rotvec),
            conformer_id=conf_id,
        )

    def _score_pose(self, pose: Pose, receptor_structure, ligand_mol: Chem.Mol) -> float:
        """
        Evaluate a pose with the configured scoring function.

        `ligand_mol` must be passed through: the Vina-style scoring function needs
        it for atom typing and returns 0.0 without it, which previously caused
        every pose in this pipeline to score identically.
        """
        scoring = getattr(self, "_scoring_function", None)
        if scoring is None:
            return self._contact_energy(pose.coordinates, receptor_structure)
        return scoring.calculate_binding_energy(
            pose.coordinates, receptor_structure, ligand_mol
        )

    def _has_severe_clashes(
        self,
        pose: Pose,
        receptor_structure,
        clash_threshold: float = 1.6,
        max_clash_fraction: float = 0.5,
        max_severe_clashes: int = 5,
    ) -> bool:
        """
        Check whether a pose overlaps the receptor badly enough to discard.

        Args:
            clash_threshold: Heavy-atom separation below which a contact counts
                as a clash.
            max_clash_fraction: Fraction of ligand atoms allowed to clash.
            max_severe_clashes: Number of unphysical (<1.3 A) overlaps allowed.
        """
        receptor_coords = self._receptor_coords(receptor_structure)
        if len(receptor_coords) == 0:
            return False

        distances = np.linalg.norm(
            pose.coordinates[:, np.newaxis] - receptor_coords[np.newaxis, :], axis=2
        )
        min_distances = np.min(distances, axis=1)

        clashes = int(np.sum(min_distances < clash_threshold))
        unphysical = int(np.sum(min_distances < 1.3))
        clash_fraction = clashes / len(pose.coordinates)

        if unphysical > max_severe_clashes:
            return True
        if clash_fraction > max_clash_fraction and unphysical > 2:
            return True
        return False

    def _contact_energy(self, ligand_coords: np.ndarray, receptor_structure) -> float:
        """
        Distance-based fallback used only when no scoring function is configured.

        This is a crude shape term, not a binding energy; it exists so that
        algorithms remain runnable without a scoring function, and its values are
        not comparable to kcal/mol.
        """
        receptor_coords = self._receptor_coords(receptor_structure)
        if len(receptor_coords) == 0:
            return 0.0

        distances = np.linalg.norm(
            ligand_coords[:, np.newaxis] - receptor_coords[np.newaxis, :], axis=2
        )

        energy = 0.0
        energy += float(np.sum(distances < 1.5)) * 150.0
        energy += float(np.sum((distances >= 1.5) & (distances < 2.0))) * 75.0

        close = distances[(distances >= 2.0) & (distances < 2.5)]
        if close.size:
            energy += float(np.sum((2.5 - close) * 15.0))

        favorable = distances[(distances >= 2.5) & (distances < 6.0)]
        if favorable.size:
            energy += float(np.sum(-3.0 * np.exp(-((favorable - 3.8) ** 2) / 1.5)))

        return energy

    @staticmethod
    def _receptor_coords(receptor_structure) -> np.ndarray:
        """Cached heavy-atom coordinates of the receptor."""
        cached = getattr(receptor_structure, "_pandadock_coords", None)
        if cached is not None:
            return cached
        atoms = [a for a in receptor_structure.get_atoms() if a.element.strip() != "H"]
        if not atoms:
            atoms = list(receptor_structure.get_atoms())
        coords = np.array([a.get_coord() for a in atoms], dtype=np.float64)
        try:
            receptor_structure._pandadock_coords = coords
        except AttributeError:
            pass
        return coords

    def _filter_and_rank_poses(
        self,
        poses: List[Pose],
        receptor_structure,
        ligand_mol: Chem.Mol,
        max_poses: int,
        relaxed: bool = False,
    ) -> List[Pose]:
        """
        Discard clashing poses, rank the rest by score, and keep distinct modes.

        Ranking uses the scoring function alone. No proximity or reference bonus is
        applied, and scores are not clamped, so the returned ordering reflects the
        interaction energy and remains comparable across ligands.
        """
        clash_kwargs: Dict[str, float] = (
            {"clash_threshold": 1.4, "max_clash_fraction": 0.7, "max_severe_clashes": 8}
            if relaxed
            else {}
        )

        valid: List[Pose] = []
        for pose in poses:
            if self._has_severe_clashes(pose, receptor_structure, **clash_kwargs):
                continue
            pose.energy = self._score_pose(pose, receptor_structure, ligand_mol)
            valid.append(pose)

        valid.sort(key=lambda p: p.energy)
        return self._select_diverse_poses(valid, max_poses)

    def _select_diverse_poses(
        self, poses: List[Pose], max_poses: int, diversity_threshold: float = 2.0
    ) -> List[Pose]:
        """
        Keep the best-scoring pose from each distinct binding mode.

        Poses are compared by heavy-atom RMSD rather than centroid separation:
        two poses can share a centroid while being flipped relative to one another,
        and reporting both as the same mode hides a real alternative.
        """
        if len(poses) <= max_poses:
            return poses

        selected: List[Pose] = []
        for pose in poses:
            distinct = True
            for chosen in selected:
                if pose.coordinates.shape != chosen.coordinates.shape:
                    continue
                rmsd = float(
                    np.sqrt(np.mean(np.sum((pose.coordinates - chosen.coordinates) ** 2, axis=1)))
                )
                if rmsd < diversity_threshold:
                    distinct = False
                    break
            if distinct:
                selected.append(pose)
            if len(selected) >= max_poses:
                break

        # If clustering was too aggressive to fill the request, top up by score.
        if len(selected) < max_poses:
            chosen_ids = {id(p) for p in selected}
            for pose in poses:
                if id(pose) not in chosen_ids:
                    selected.append(pose)
                if len(selected) >= max_poses:
                    break

        return selected[:max_poses]
