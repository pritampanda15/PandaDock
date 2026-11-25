"""
PandaCore docking algorithm.

Flexible-ligand docking by Monte Carlo search with local optimization over
translation, orientation and ligand torsions, scored against precomputed
receptor affinity grids.

Replaces the earlier "crystal-guided" algorithms, which perturbed the input
conformer by a few degrees about the box centre and therefore never searched
orientation or conformation at all.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem
from ..search.rotations import quaternion_from_rotvec

from ..core import BaseDockingAlgorithm, DockingResult, Pose
from ..scoring.vina_scoring import VinaScoring
from ...analysis.rmsd import heavy_atom_automorphisms
from ..search import (
    AffinityGrids,
    DockingObjective,
    MonteCarloSearch,
    SearchConfig,
    TorsionTree,
    cluster_poses,
)


class PandaCoreDocker(BaseDockingAlgorithm):
    """Flexible-ligand Monte Carlo docking with grid-accelerated scoring."""

    def __init__(self, name: str = "pandacore", grid_cache=None):
        super().__init__(name, supports_gpu=True)
        self.logger = logging.getLogger(f"pandadock.docking.{name}")
        # Off by default: a single dock() gains nothing from it, and holding
        # grids for a receptor that is never revisited only costs memory. Pass a
        # GridCache when docking many ligands into one receptor.
        self.grid_cache = grid_cache

    # ------------------------------------------------------------------ docking

    def dock(
        self,
        receptor_file: str,
        ligand_mol: Chem.Mol,
        grid_center: np.ndarray,
        grid_dimensions: np.ndarray,
        **kwargs,
    ) -> DockingResult:
        """
        Dock a ligand into the specified box.

        Parameters (via kwargs):
            num_poses: Number of distinct binding modes to return (default 9)
            exhaustiveness: Independent Monte Carlo runs (default 8)
            n_steps: Monte Carlo steps per run (default scales with ligand DOF)
            rigid_ligand: Disable torsional search (default False)
            grid_spacing: Affinity grid spacing in Angstrom (default 0.375)
            seed: Random seed for reproducible runs (default None)
            rmsd_cutoff: Clustering threshold in Angstrom (default 2.0)
            max_torsions: Cap on torsional degrees of freedom (default 32)

        Set `grid_cache` on the docker to reuse affinity grids across ligands
        docked into the same receptor and box. Grids depend on the receptor and
        on ligand atom types, not on ligand identity, so a screening run pays
        for them once rather than once per ligand.
        """
        self._validate_inputs(receptor_file, ligand_mol, grid_center, grid_dimensions)
        start_time = time.time()

        grid_center = np.asarray(grid_center, dtype=np.float64)
        grid_dimensions = np.asarray(grid_dimensions, dtype=np.float64)

        num_poses = int(kwargs.get("num_poses", 9))
        rigid_ligand = bool(kwargs.get("rigid_ligand", False))
        grid_spacing = float(kwargs.get("grid_spacing", 0.375))
        rmsd_cutoff = float(kwargs.get("rmsd_cutoff", 2.0))
        max_torsions = kwargs.get("max_torsions", 32)

        mol = self._prepare_ligand(ligand_mol)

        tree = TorsionTree(mol, conf_id=0, rigid=rigid_ligand, max_torsions=max_torsions)

        # Sampling has to scale with ligand flexibility. A fixed budget that is
        # ample for a rigid fragment leaves a highly rotatable ligand's search
        # space badly under-explored, and the failure is silent: the run returns
        # a confident-looking pose from a local minimum far above the global one.
        exhaustiveness = kwargs.get("exhaustiveness")
        if exhaustiveness is None:
            exhaustiveness = self._default_exhaustiveness(tree.n_torsions)
            self.logger.info(
                "Exhaustiveness not specified; using %d for %d torsional DOF",
                exhaustiveness,
                tree.n_torsions,
            )

        config = SearchConfig(
            exhaustiveness=int(exhaustiveness),
            n_steps=kwargs.get("n_steps"),
            temperature=float(kwargs.get("temperature", 1.2)),
            max_local_iter=int(kwargs.get("max_local_iter", 60)),
            seed=kwargs.get("seed"),
            n_workers=int(kwargs.get("n_workers", 1)),
        )
        self.logger.info(
            "Ligand: %d atoms (%d heavy), %d torsional DOF, %d total DOF",
            tree.n_atoms,
            len(tree.heavy_atoms),
            tree.n_torsions,
            tree.n_dof,
        )

        receptor_structure = self._load_receptor(receptor_file)

        scoring = self._scoring_function if isinstance(self._scoring_function, VinaScoring) else None
        grid_scoring = scoring or VinaScoring()
        weights_override = kwargs.get("weights_override")
        if isinstance(weights_override, dict):
            w_params = {}
            for k in ("gauss1", "gauss2", "repulsion", "hydrophobic", "hydrogen"):
                if k in weights_override:
                    w_params[f"{k}_weight"] = float(weights_override[k])
            if "n_rot_weight" in weights_override:
                w_params["n_rot_weight"] = float(weights_override["n_rot_weight"])
            grid_scoring = VinaScoring(preset=grid_scoring.preset, **w_params)
        if self._scoring_function is not None and scoring is None:
            self.logger.info(
                "Search is driven by the grid-accelerated Vina function; "
                "'%s' will be applied as a rescoring pass over the final poses.",
                type(self._scoring_function).__name__,
            )

        grid_build_start = time.time()
        grids = AffinityGrids.build(
            receptor_structure=receptor_structure,
            ligand_mol=mol,
            grid_center=grid_center,
            grid_dimensions=grid_dimensions,
            spacing=grid_spacing,
            scoring=grid_scoring,
            cache=self.grid_cache,
        )
        self.logger.info("Grid construction took %.2f s", time.time() - grid_build_start)

        # Optional ring-conformer enumeration for ligands with flexible
        # non-aromatic rings (>= 5 atoms), which the torsion tree holds rigid.
        sample_rings = bool(kwargs.get("sample_rings", False))
        trees_to_search = [tree]
        if sample_rings and not rigid_ligand:
            trees_to_search = self._ring_conformer_trees(
                mol,
                rigid=rigid_ligand,
                max_torsions=max_torsions,
                max_ring_confs=int(kwargs.get("max_ring_confs", 4)),
                seed=config.seed,
            )

        box_min = grid_center - grid_dimensions / 2.0
        box_max = grid_center + grid_dimensions / 2.0

        restraints = kwargs.get("restraints")
        device = kwargs.get("device")
        all_minima = []
        total_evals = 0
        primary_objective = None
        for idx, conf_tree in enumerate(trees_to_search):
            objective = DockingObjective(
                conf_tree,
                grids,
                tether_center=kwargs.get("tether_center"),
                tether_radius=float(kwargs.get("tether_radius", 0.0)),
                tether_force=float(kwargs.get("tether_force", 10.0)),
                restraints=restraints,
            )
            if primary_objective is None:
                primary_objective = objective

            conf_seed = None if config.seed is None else int(config.seed) + idx * 1009
            if device in (None, "cpu"):
                conf_cfg = SearchConfig(
                    exhaustiveness=config.exhaustiveness,
                    n_steps=config.n_steps,
                    temperature=config.temperature,
                    max_local_iter=config.max_local_iter,
                    seed=conf_seed,
                    n_workers=config.n_workers,
                )
                minima = MonteCarloSearch(objective, conf_cfg).run(box_min, box_max)
            else:
                from ..gpu.handoff import run_batched_search

                self.logger.info("Running the batched search on %s", device)
                gpu_steps = kwargs.get("gpu_steps")
                minima = run_batched_search(
                    grids,
                    conf_tree,
                    objective,
                    box_min,
                    box_max,
                    device=device,
                    n_chains=int(kwargs.get("n_chains", 512)),
                    n_steps=int(gpu_steps) if gpu_steps is not None else None,
                    seed=conf_seed,
                    max_local_iter=config.max_local_iter,
                )
            total_evals += objective.n_evaluations
            all_minima.extend(minima)

        all_minima.sort(key=lambda r: r.energy)
        objective = primary_objective
        if not all_minima:
            self.logger.warning("Search produced no poses")
            return self._empty_result(
                mol, receptor_file, grid_center, grid_dimensions, config
            )

        clustered = cluster_poses(
            all_minima,
            tree.heavy_atoms,
            rmsd_cutoff=rmsd_cutoff,
            max_poses=num_poses,
            automorphisms=heavy_atom_automorphisms(mol),
        )

        poses = self._build_poses(
            clustered, tree, objective, mol, receptor_structure, grid_scoring
        )

        restraint_summary = None
        if restraints and poses:
            for pose in poses:
                r_e, _ = restraints.energy_and_gradient(pose.coordinates, need_gradient=False)
                pose.energy_components["restraint_penalty"] = float(r_e)
            poses.sort(key=lambda p: p.energy + p.energy_components.get("restraint_penalty", 0.0))
            best_eff = poses[0].energy + poses[0].energy_components.get("restraint_penalty", 0.0)
            for pose in poses:
                eff = pose.energy + pose.energy_components.get("restraint_penalty", 0.0)
                pose.confidence = float(np.exp(-(eff - best_eff) / 1.2))
            restraint_summary = restraints.evaluate_summary(poses[0].coordinates)

        result = DockingResult(
            ligand_name=mol.GetProp("_Name") if mol.HasProp("_Name") else "unknown",
            receptor_file=receptor_file,
            grid_center=grid_center,
            grid_dimensions=grid_dimensions,
            algorithm_used=self.name,
            scoring_function=grid_scoring.preset + "_grid",
            poses=poses,
            runtime_seconds=time.time() - start_time,
            parameters={
                "exhaustiveness": config.exhaustiveness,
                "n_steps": config.steps_for(tree.n_dof),
                "n_torsions": tree.n_torsions,
                "n_dof": tree.n_dof,
                "grid_spacing": grid_spacing,
                "rigid_ligand": rigid_ligand,
                "rmsd_cutoff": rmsd_cutoff,
                "seed": config.seed,
                "n_workers": config.n_workers,
                "sample_rings": sample_rings,
                "ring_conformers": len(trees_to_search),
                "energy_evaluations": total_evals,
                "restraints": restraint_summary,
                "calibrated_weights": bool(weights_override),
            },
        )

        self.logger.info(
            "Docking finished in %.2f s: %d binding modes, best %.3f kcal/mol",
            result.runtime_seconds,
            len(poses),
            poses[0].energy if poses else float("nan"),
        )
        return result

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _default_exhaustiveness(n_torsions: int) -> int:
        """
        Number of independent runs to use when the caller does not specify one.

        Grows with torsional flexibility. Measured on a 13-DOF ligand, 8 runs
        converged to a local minimum 11 kcal/mol above the global one (12.7 A from
        the reference pose) while 24 runs found the global minimum (1.0 A), so a
        flat default is not defensible across ligand sizes.
        """
        return int(np.clip(8 + 2 * n_torsions, 8, 32))

    def _prepare_ligand(self, ligand_mol: Chem.Mol) -> Chem.Mol:
        """
        Ensure the ligand carries a usable 3D conformer.

        Only one conformer is needed: the torsional search generates conformational
        diversity itself, so an ensemble of input conformers would duplicate work
        the search already does.
        """
        mol = Chem.Mol(ligand_mol)

        if mol.GetNumConformers() == 0:
            self.logger.info("Ligand has no 3D conformer; embedding one")
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 0xF00D
            if AllChem.EmbedMolecule(mol, params) != 0:
                # Fall back to a less strict embedding rather than failing outright.
                if AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=0xF00D) != 0:
                    raise ValueError("Failed to generate a 3D conformer for the ligand")
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            except Exception as exc:  # geometry is usable even if minimisation fails
                self.logger.debug("MMFF optimisation skipped: %s", exc)

        return mol

    def _load_receptor(self, receptor_file: str):
        parser = PDBParser(QUIET=True)
        return parser.get_structure("receptor", receptor_file)

    def _ring_conformer_trees(
        self,
        mol: Chem.Mol,
        rigid: bool = False,
        max_torsions: int = 32,
        max_ring_confs: int = 4,
        seed: Optional[int] = None,
    ) -> List[TorsionTree]:
        """Generate TorsionTrees across distinct ring puckers for flexible rings.

        Non-aromatic rings of size >= 5 (e.g., cyclohexanes, piperidines,
        macrocycles) are held rigid by a single TorsionTree. When the ligand
        contains such rings, embed a small conformer pool with ETKDGv3, retain
        conformers whose ring-atom geometry differs by >= 0.4 A RMSD from the
        input conformer, and return one TorsionTree per distinct ring state.
        """
        ring_info = mol.GetRingInfo()
        flex_ring_atoms = set()
        for ring in ring_info.AtomRings():
            if len(ring) >= 5 and not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                flex_ring_atoms.update(ring)

        base_tree = TorsionTree(mol, conf_id=0, rigid=rigid, max_torsions=max_torsions)
        if not flex_ring_atoms or max_ring_confs <= 1:
            return [base_tree]

        ring_idx = np.array(sorted(flex_ring_atoms), dtype=np.int64)
        work = Chem.Mol(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed) if seed is not None else 0xF00D
        params.pruneRmsThresh = 0.35
        n_embed = max(8, max_ring_confs * 4)
        cids = list(AllChem.EmbedMultipleConfs(work, numConfs=n_embed, params=params))
        if not cids:
            return [base_tree]

        # Keep conformer 0 (input geometry) plus conformers with distinct ring puckers
        def ring_coords(m, cid):
            pos = np.asarray(m.GetConformer(cid).GetPositions(), dtype=np.float64)[ring_idx]
            return pos - pos.mean(axis=0)

        kept_ring_coords = [ring_coords(mol, 0)]
        trees = [base_tree]
        for cid in cids:
            if len(trees) >= max_ring_confs:
                break
            try:
                AllChem.MMFFOptimizeMolecule(work, confId=cid, maxIters=200)
            except Exception:
                pass
            rc = ring_coords(work, cid)
            # Kabsch-aligned RMSD over ring atoms
            distinct = True
            for ref in kept_ring_coords:
                cov = rc.T @ ref
                u, _, vt = np.linalg.svd(cov)
                d = np.linalg.det(vt.T @ u.T)
                rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
                diff = rc @ rot.T - ref
                rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
                if rmsd < 0.4:
                    distinct = False
                    break
            if distinct:
                kept_ring_coords.append(rc)
                trees.append(
                    TorsionTree(work, conf_id=int(cid), rigid=rigid, max_torsions=max_torsions)
                )

        if len(trees) > 1:
            self.logger.info(
                "Sampled %d distinct ring conformers across %d flexible ring atoms",
                len(trees),
                len(ring_idx),
            )
        return trees

    def _build_poses(
        self,
        results,
        tree: TorsionTree,
        objective: DockingObjective,
        mol: Chem.Mol,
        receptor_structure,
        grid_scoring: Optional[VinaScoring] = None,
    ) -> List[Pose]:
        """Convert search results to Pose objects with per-term energy breakdowns.

        Applies two corrections over the raw search objective:
        1. Only repulsive intramolecular clashes (`max(0.0, intra)`) are added
           to the reported docking score, so folded conformations never receive
           an artificial attractive intramolecular bonus over receptor contacts.
        2. Applies Vina's torsional entropy denominator `1 + w_rot * n_rot` so
           scores sit on the standard Vina kcal/mol scale.
        3. When a non-Vina scoring function (e.g. `PhysicsBasedScoring`) is
           attached, runs it as a real rescoring pass and re-ranks the poses.
        """
        scoring = grid_scoring or VinaScoring()
        custom_rescorer = (
            self._scoring_function
            if self._scoring_function is not None
            and not isinstance(self._scoring_function, VinaScoring)
            else None
        )
        poses: List[Pose] = []

        for res in results:
            coords = res.coords
            heavy = coords[tree.heavy_atoms]

            inter = float(objective.grids.score(heavy))
            intra, _ = objective._intramolecular(coords, need_gradient=False)
            intra = float(intra)
            intra_penalty = max(0.0, intra)
            vina_score = scoring.normalize_energy(inter + intra_penalty, tree.n_torsions)

            components = {
                "intermolecular": inter,
                "intramolecular": intra,
                "intramolecular_penalty": intra_penalty,
                "flexibility_factor": 1.0 + scoring.n_rot_weight * tree.n_torsions,
                "vina_grid": vina_score,
            }

            if custom_rescorer is not None:
                try:
                    rescored = float(
                        custom_rescorer.calculate_binding_energy(
                            coords, receptor_structure, mol
                        )
                    )
                except TypeError:
                    rescored = float(
                        custom_rescorer.calculate_binding_energy(
                            coords, receptor_structure
                        )
                    )
                final_energy = rescored
                components["rescored"] = rescored
            else:
                final_energy = vina_score

            components["total"] = final_energy

            pose = Pose(
                coordinates=coords,
                center=np.mean(heavy, axis=0),
                rotation=quaternion_from_rotvec(res.dof[3:6]),
                conformer_id=0,
                energy=final_energy,
                internal_strain=intra_penalty,
                energy_components=components,
            )
            pose.torsion_angles = res.dof[6:].copy()
            poses.append(pose)

        poses.sort(key=lambda p: p.energy)
        best_energy = poses[0].energy if poses else 0.0
        for pose in poses:
            # Rank-relative confidence: how close this mode is to the best one found.
            pose.confidence = float(np.exp(-(pose.energy - best_energy) / 1.2))

        return poses

    def _empty_result(
        self,
        mol: Chem.Mol,
        receptor_file: str,
        grid_center: np.ndarray,
        grid_dimensions: np.ndarray,
        config: SearchConfig,
    ) -> DockingResult:
        return DockingResult(
            ligand_name=mol.GetProp("_Name") if mol.HasProp("_Name") else "unknown",
            receptor_file=receptor_file,
            grid_center=grid_center,
            grid_dimensions=grid_dimensions,
            algorithm_used=self.name,
            scoring_function="vina_grid",
            poses=[],
            parameters={"exhaustiveness": config.exhaustiveness},
        )
