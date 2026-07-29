"""
Torsion tree construction for flexible-ligand docking.

Builds an articulated model of the ligand: a root rigid fragment plus a tree of
rotatable bonds. Each rotatable bond becomes an explicit search degree of freedom,
so the search can change the ligand's internal conformation rather than only
translating and rotating a fixed input conformer.

The tree is ordered root-outward so that applying rotations sequentially against
the running coordinates composes correctly (a parent rotation carries its child
subtrees, including their axes, along with it).
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from rdkit import Chem

from .rotations import random_rotvec, rodrigues_matrix, rotate_about_axis

logger = logging.getLogger("pandadock.docking.search.torsion_tree")

# Single, acyclic bond between two non-terminal heavy atoms, excluding triple bonds.
ROTATABLE_SMARTS = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"

# Bonds that are formally single but have partial double-bond character. Vina and
# Glide both treat these as rigid; leaving them free wastes search effort and
# produces non-physical cis-amides.
AMIDE_SMARTS = "[NX3][CX3](=[OX1])"
CONJUGATED_SMARTS = "[NX3][CX3]=[CX3,NX2]"


class Torsion:
    """A single rotatable degree of freedom."""

    __slots__ = ("origin_atom", "axis_atom", "moving", "depth")

    def __init__(self, origin_atom: int, axis_atom: int, moving: np.ndarray, depth: int):
        # Rotation is about the vector origin_atom -> axis_atom.
        # `moving` holds every atom strictly distal to axis_atom (axis_atom itself
        # lies on the axis and is unaffected).
        self.origin_atom = origin_atom
        self.axis_atom = axis_atom
        self.moving = moving
        self.depth = depth

    def __repr__(self) -> str:
        return (
            f"Torsion({self.origin_atom}->{self.axis_atom}, "
            f"{len(self.moving)} moving atoms, depth={self.depth})"
        )


class TorsionTree:
    """
    Articulated ligand model: rigid root fragment + tree of rotatable bonds.

    Args:
        mol: RDKit molecule with at least one conformer.
        conf_id: Conformer to use as the internal reference geometry.
        rigid: If True, expose zero torsional degrees of freedom.
        freeze_amides: Treat amide and conjugated N-C bonds as rigid.
        max_torsions: Cap the number of torsional DOF. When the ligand has more
            rotatable bonds than this, the ones nearest the root are kept, since
            they move the most atoms and matter most for placement.
    """

    def __init__(
        self,
        mol: Chem.Mol,
        conf_id: int = 0,
        rigid: bool = False,
        freeze_amides: bool = True,
        max_torsions: Optional[int] = 32,
    ):
        if mol.GetNumConformers() == 0:
            raise ValueError("TorsionTree requires a molecule with a 3D conformer")

        self.mol = mol
        self.n_atoms = mol.GetNumAtoms()
        self.conf_id = conf_id

        conf = mol.GetConformer(conf_id)
        self.ref_coords = np.asarray(conf.GetPositions(), dtype=np.float64)

        self.heavy_atoms = np.array(
            [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1], dtype=np.int64
        )
        if len(self.heavy_atoms) == 0:
            raise ValueError("Ligand has no heavy atoms")

        self._adjacency = self._build_adjacency()

        if rigid:
            rot_bonds: List[Tuple[int, int]] = []
        else:
            rot_bonds = self._find_rotatable_bonds(freeze_amides)

        self.root_atom, self.torsions = self._build_tree(rot_bonds, max_torsions)

        # Reference geometry anchored so the root atom sits at the origin. Poses are
        # then produced as R * (torsioned coords) + translation, which keeps the
        # translation DOF meaningful even as torsions change the molecule's extent.
        self.base_coords = self.ref_coords - self.ref_coords[self.root_atom]

        self.moving_masks = [self._as_mask(t.moving) for t in self.torsions]
        self._intra_pairs: Optional[Tuple[np.ndarray, np.ndarray]] = None

        logger.debug(
            "TorsionTree: %d atoms (%d heavy), root=%d, %d torsional DOF",
            self.n_atoms,
            len(self.heavy_atoms),
            self.root_atom,
            len(self.torsions),
        )

    # ------------------------------------------------------------------ topology

    @property
    def n_torsions(self) -> int:
        return len(self.torsions)

    @property
    def n_dof(self) -> int:
        """Translation (3) + orientation (3) + torsions."""
        return 6 + self.n_torsions

    def _build_adjacency(self) -> List[List[int]]:
        adj: List[List[int]] = [[] for _ in range(self.n_atoms)]
        for bond in self.mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj[i].append(j)
            adj[j].append(i)
        return adj

    def _find_rotatable_bonds(self, freeze_amides: bool) -> List[Tuple[int, int]]:
        patt = Chem.MolFromSmarts(ROTATABLE_SMARTS)
        matches = self.mol.GetSubstructMatches(patt)

        frozen: Set[frozenset] = set()
        if freeze_amides:
            for smarts in (AMIDE_SMARTS, CONJUGATED_SMARTS):
                q = Chem.MolFromSmarts(smarts)
                if q is None:
                    continue
                for match in self.mol.GetSubstructMatches(q):
                    # First two atoms of each pattern span the partial-double bond.
                    frozen.add(frozenset((match[0], match[1])))

        bonds: List[Tuple[int, int]] = []
        for i, j in matches:
            if frozenset((i, j)) in frozen:
                continue
            # A bond whose rotation only spins terminal hydrogens (e.g. a methyl or
            # hydroxyl) changes no heavy-atom position; it is not worth a DOF.
            if self._only_moves_hydrogens(i, j) or self._only_moves_hydrogens(j, i):
                continue
            bonds.append((i, j))
        return bonds

    def _only_moves_hydrogens(self, anchor: int, distal: int) -> bool:
        """True if every heavy atom beyond `distal` (excluding it) is a hydrogen."""
        subtree = self._reachable_from(distal, blocked_edge=(anchor, distal))
        for idx in subtree:
            if idx == distal:
                continue
            if self.mol.GetAtomWithIdx(int(idx)).GetAtomicNum() > 1:
                return False
        return True

    def _reachable_from(self, start: int, blocked_edge: Tuple[int, int]) -> List[int]:
        """BFS over the molecular graph from `start` without crossing `blocked_edge`."""
        a, b = blocked_edge
        blocked = frozenset((a, b))
        seen = {start}
        queue = deque([start])
        out = [start]
        while queue:
            cur = queue.popleft()
            for nxt in self._adjacency[cur]:
                if nxt in seen:
                    continue
                if frozenset((cur, nxt)) == blocked:
                    continue
                seen.add(nxt)
                out.append(nxt)
                queue.append(nxt)
        return out

    def _build_tree(
        self, rot_bonds: Sequence[Tuple[int, int]], max_torsions: Optional[int]
    ) -> Tuple[int, List[Torsion]]:
        """
        Choose a root fragment and order the rotatable bonds root-outward.

        The root is the largest rigid fragment (the fragment containing the most
        heavy atoms once every rotatable bond is cut), matching AutoDock's
        convention. Anchoring the search on the largest rigid piece keeps the
        best-determined part of the ligand under direct rigid-body control.
        """
        if not rot_bonds:
            return int(self.heavy_atoms[0]), []

        rot_set = {frozenset(b) for b in rot_bonds}

        # Connected components of the graph with rotatable bonds removed.
        frag_id = np.full(self.n_atoms, -1, dtype=np.int64)
        fragments: List[List[int]] = []
        for start in range(self.n_atoms):
            if frag_id[start] >= 0:
                continue
            comp: List[int] = []
            queue = deque([start])
            frag_id[start] = len(fragments)
            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for nxt in self._adjacency[cur]:
                    if frag_id[nxt] >= 0:
                        continue
                    if frozenset((cur, nxt)) in rot_set:
                        continue
                    frag_id[nxt] = len(fragments)
                    queue.append(nxt)
            fragments.append(comp)

        def heavy_count(frag: Sequence[int]) -> int:
            return sum(1 for a in frag if self.mol.GetAtomWithIdx(int(a)).GetAtomicNum() > 1)

        root_frag = int(np.argmax([heavy_count(f) for f in fragments]))

        # Fragment-level adjacency, so we can walk outward from the root fragment.
        frag_edges: Dict[int, List[Tuple[int, int, int]]] = {i: [] for i in range(len(fragments))}
        for i, j in rot_bonds:
            fi, fj = int(frag_id[i]), int(frag_id[j])
            frag_edges[fi].append((fj, i, j))
            frag_edges[fj].append((fi, j, i))

        torsions: List[Torsion] = []
        visited = {root_frag}
        queue = deque([(root_frag, 0)])
        while queue:
            frag, depth = queue.popleft()
            for nbr_frag, near_atom, far_atom in frag_edges[frag]:
                if nbr_frag in visited:
                    continue
                visited.add(nbr_frag)
                moving = self._reachable_from(far_atom, blocked_edge=(near_atom, far_atom))
                # far_atom lies on the rotation axis, so it does not move.
                moving = np.array([m for m in moving if m != far_atom], dtype=np.int64)
                if len(moving) > 0:
                    torsions.append(Torsion(near_atom, far_atom, moving, depth))
                queue.append((nbr_frag, depth + 1))

        if max_torsions is not None and len(torsions) > max_torsions:
            logger.warning(
                "Ligand has %d rotatable bonds; capping torsional DOF at %d "
                "(keeping those closest to the root fragment)",
                len(torsions),
                max_torsions,
            )
            torsions.sort(key=lambda t: (t.depth, -len(t.moving)))
            torsions = torsions[:max_torsions]
            torsions.sort(key=lambda t: t.depth)

        root_atom = next(
            (a for a in fragments[root_frag] if self.mol.GetAtomWithIdx(int(a)).GetAtomicNum() > 1),
            fragments[root_frag][0],
        )
        return int(root_atom), torsions

    def _as_mask(self, indices: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.n_atoms, dtype=bool)
        mask[indices] = True
        return mask

    # ------------------------------------------------------------- pose building

    def apply_torsions(self, angles: Sequence[float]) -> np.ndarray:
        """
        Apply torsion angles (radians) to the reference geometry.

        Torsions are applied root-outward against the running coordinates, so each
        rotation uses the axis position produced by its ancestors.
        """
        coords = self.base_coords.copy()
        if not self.torsions:
            return coords

        for torsion, angle in zip(self.torsions, angles):
            if angle == 0.0:
                continue
            axis = coords[torsion.axis_atom] - coords[torsion.origin_atom]
            pivot = coords[torsion.axis_atom]
            coords[torsion.moving] = rotate_about_axis(
                coords[torsion.moving], axis, pivot, float(angle)
            )
        return coords

    def build_coords(self, dof: np.ndarray) -> np.ndarray:
        """
        Map a DOF vector to full-molecule Cartesian coordinates.

        dof layout: [tx, ty, tz, rx, ry, rz, theta_1 ... theta_k]
        where (rx, ry, rz) is a rotation vector and theta_i are torsion angles.
        """
        translation = dof[:3]
        rotvec = dof[3:6]
        angles = dof[6:]

        coords = self.apply_torsions(angles)
        coords = coords @ rodrigues_matrix(rotvec).T
        coords += translation
        return coords

    def torsion_axes(self, coords: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Unit axis and pivot point for each torsion, in the frame of `coords`."""
        out = []
        for torsion in self.torsions:
            axis = coords[torsion.axis_atom] - coords[torsion.origin_atom]
            norm = float(np.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2))
            axis = axis / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0])
            out.append((axis, coords[torsion.axis_atom]))
        return out

    # ------------------------------------------------- intramolecular interactions

    def intramolecular_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Heavy-atom pairs whose separation can change as torsions vary.

        A pair qualifies when at least one torsion moves exactly one of the two
        atoms, and the atoms are more than three bonds apart. Pairs inside a single
        rigid fragment are excluded: their distance is fixed, so including them
        would add a constant offset that varies between ligands and corrupts
        score comparability.
        """
        if self._intra_pairs is not None:
            return self._intra_pairs

        heavy = self.heavy_atoms
        if len(self.torsions) == 0 or len(heavy) < 2:
            empty = np.array([], dtype=np.int64)
            self._intra_pairs = (empty, empty)
            return self._intra_pairs

        dmat = Chem.GetDistanceMatrix(self.mol)
        masks = np.stack(self.moving_masks)  # (n_torsions, n_atoms)

        idx_i, idx_j = np.triu_indices(len(heavy), k=1)
        a = heavy[idx_i]
        b = heavy[idx_j]

        # Separated by more than three bonds (1-2, 1-3 and 1-4 are handled by the
        # ligand's own force field / fixed geometry, as in Vina).
        far_enough = dmat[a, b] > 3.0

        # Relative motion: some torsion moves exactly one of the pair.
        separable = np.any(masks[:, a] != masks[:, b], axis=0)

        keep = far_enough & separable
        self._intra_pairs = (a[keep], b[keep])
        return self._intra_pairs

    # -------------------------------------------------------------------- helpers

    def to_mol(self, coords: np.ndarray, name: Optional[str] = None) -> Chem.Mol:
        """Return a copy of the ligand carrying `coords` as its single conformer."""
        out = Chem.Mol(self.mol)
        out.RemoveAllConformers()
        conf = Chem.Conformer(self.n_atoms)
        for i in range(self.n_atoms):
            conf.SetAtomPosition(i, coords[i].tolist())
        out.AddConformer(conf, assignId=True)
        if name is not None:
            out.SetProp("_Name", name)
        return out

    def random_dof(
        self,
        rng: np.random.Generator,
        box_min: np.ndarray,
        box_max: np.ndarray,
    ) -> np.ndarray:
        """
        Draw a starting point uniformly over the search space.

        Orientation is sampled uniformly over SO(3) and torsions uniformly over
        [-pi, pi), so no part of conformational space is privileged by the choice
        of input conformer.
        """
        dof = np.empty(self.n_dof, dtype=np.float64)
        dof[:3] = rng.uniform(box_min, box_max)
        dof[3:6] = random_rotvec(rng)
        if self.n_torsions:
            dof[6:] = rng.uniform(-np.pi, np.pi, self.n_torsions)
        return dof
