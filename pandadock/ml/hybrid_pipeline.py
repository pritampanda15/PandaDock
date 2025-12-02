"""
Hybrid ML+Physics Docking Pipeline

Combines machine learning pose generation with physics-based refinement
for enhanced accuracy and novel pose discovery.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import logging
import time
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

# Import PandaDock core components
from ..docking.core import DockingEngine, DockingResult, Pose, BaseDockingAlgorithm
from ..docking.scoring.physics_based import PhysicsBasedScoring
from ..docking.scoring.precision_score import PrecisionScoring

# Import ML components
from .models.diffusion import DiffusionDockingModel
from .models.energy_prediction import EnergyPredictionModel
from .models.pose_ranking import PoseRankingModel
from .feature_extraction import ComplexFeatureExtractor


class HybridMLPhysicsPipeline:
    """Hybrid pipeline combining ML pose generation with physics refinement"""

    def __init__(self,
                 ml_model: Optional[Any] = None,
                 model_path: Optional[str] = None,
                 physics_refinement: str = 'enhanced_hierarchical_cpu',
                 hybrid_scoring: bool = True):
        """
        Initialize hybrid pipeline

        Args:
            ml_model: Pre-initialized ML model
            model_path: Path to trained ML model weights
            physics_refinement: Physics algorithm for refinement
            hybrid_scoring: Use combined ML+Physics scoring
        """
        self.logger = logging.getLogger(__name__)

        # Initialize ML model
        if ml_model is not None:
            self.ml_model = ml_model
        else:
            # Default to diffusion model
            self.ml_model = DiffusionDockingModel()
            if model_path:
                self.ml_model.load_weights(model_path)

        # Initialize physics components
        self.physics_engine = DockingEngine()
        self._register_physics_algorithms()

        self.physics_refinement = physics_refinement
        self.hybrid_scoring = hybrid_scoring

        # Scoring functions
        self.physics_scoring = PhysicsBasedScoring()
        self.precision_scoring = PrecisionScoring()

        # Feature extractor for ML scoring
        self.feature_extractor = ComplexFeatureExtractor()

    def _register_physics_algorithms(self):
        """Register physics-based algorithms"""
        try:
            from ..docking.algorithms.monte_carlo_cpu import MonteCarloDocker as MCDocker
            from ..docking.algorithms.genetic_algorithm_cpu import GeneticAlgorithmDocker
            from ..docking.algorithms.hierarchical_cpu import HierarchicalDocker
            from ..docking.algorithms.enhanced_hierarchical_cpu import EnhancedHierarchicalDocker

            self.physics_engine.register_algorithm(MCDocker())
            self.physics_engine.register_algorithm(GeneticAlgorithmDocker())
            self.physics_engine.register_algorithm(HierarchicalDocker())
            self.physics_engine.register_algorithm(EnhancedHierarchicalDocker())

            # Register scoring functions
            self.physics_engine.register_scoring_function('physics_based', self.physics_scoring)
            self.physics_engine.register_scoring_function('precision_score', self.precision_scoring)

        except ImportError as e:
            self.logger.warning(f"Some physics algorithms not available: {e}")

    def dock(self,
             receptor_file: str,
             ligand_mol: Chem.Mol,
             grid_center: np.ndarray,
             grid_dimensions: np.ndarray,
             num_proposals: int = 50,
             temperature: float = 1.0,
             confidence_threshold: float = 0.7,
             **kwargs) -> DockingResult:
        """
        Perform ML-enhanced docking

        Args:
            receptor_file: Path to receptor PDB file
            ligand_mol: RDKit molecule object
            grid_center: Center of docking grid
            grid_dimensions: Dimensions of docking grid
            num_proposals: Number of ML-generated pose proposals
            temperature: Sampling temperature for ML models
            confidence_threshold: Minimum confidence for poses

        Returns:
            DockingResult with ML-enhanced poses
        """
        start_time = time.time()

        # Set ligand name
        ligand_name = ligand_mol.GetProp('_Name') if ligand_mol.HasProp('_Name') else 'ligand'

        self.logger.info(f"Starting ML-enhanced docking for {ligand_name}")

        # Step 1: Generate initial poses with ML
        self.logger.info("Generating poses with ML model...")
        ml_start = time.time()

        ml_poses = self._generate_ml_poses(
            receptor_file, ligand_mol, grid_center, grid_dimensions,
            num_proposals, temperature
        )

        ml_time = time.time() - ml_start
        self.logger.info(f"ML pose generation completed in {ml_time:.2f}s")

        # Step 2: Filter poses by confidence
        filtered_poses = [pose for pose in ml_poses if pose.confidence >= confidence_threshold]
        self.logger.info(f"Filtered {len(filtered_poses)}/{len(ml_poses)} poses by confidence")

        if not filtered_poses:
            self.logger.warning("No poses passed confidence threshold, using all poses")
            filtered_poses = ml_poses

        # Step 3: Hybrid scoring
        if self.hybrid_scoring:
            self.logger.info("Applying hybrid ML+Physics scoring...")
            scoring_start = time.time()

            scored_poses = self._apply_hybrid_scoring(
                filtered_poses, receptor_file, ligand_mol
            )

            scoring_time = time.time() - scoring_start
            self.logger.info(f"Hybrid scoring completed in {scoring_time:.2f}s")
        else:
            scored_poses = filtered_poses
            scoring_time = 0.0

        # Create result
        result = DockingResult(
            ligand_name=ligand_name,
            receptor_file=receptor_file,
            grid_center=grid_center,
            grid_dimensions=grid_dimensions,
            algorithm_used=f"ml_enhanced_{type(self.ml_model).__name__.lower()}",
            scoring_function="hybrid_ml_physics" if self.hybrid_scoring else "ml_only",
            poses=scored_poses
        )

        # Add ML-specific metadata
        result.ml_generation_time = ml_time
        result.hybrid_scoring_time = scoring_time
        result.runtime_seconds = time.time() - start_time

        self.logger.info(f"ML-enhanced docking completed: {len(scored_poses)} final poses")
        return result

    def hybrid_dock(self,
                   receptor_file: str,
                   ligand_mol: Chem.Mol,
                   grid_center: np.ndarray,
                   grid_dimensions: np.ndarray,
                   ml_proposals: int = 100,
                   top_k_refinement: int = 20,
                   **kwargs) -> DockingResult:
        """
        Perform full hybrid ML+Physics docking with refinement

        Args:
            receptor_file: Path to receptor PDB file
            ligand_mol: RDKit molecule object
            grid_center: Center of docking grid
            grid_dimensions: Dimensions of docking grid
            ml_proposals: Number of ML-generated proposals
            top_k_refinement: Number of top poses to refine with physics

        Returns:
            DockingResult with hybrid ML+Physics poses
        """
        start_time = time.time()

        ligand_name = ligand_mol.GetProp('_Name') if ligand_mol.HasProp('_Name') else 'ligand'
        self.logger.info(f"Starting hybrid ML+Physics docking for {ligand_name}")

        # Step 1: Generate ML proposals
        self.logger.info("Generating ML proposals...")
        ml_start = time.time()

        ml_poses = self._generate_ml_poses(
            receptor_file, ligand_mol, grid_center, grid_dimensions,
            ml_proposals, temperature=1.0
        )

        ml_time = time.time() - ml_start

        # Step 2: Rank and select top poses for refinement
        ranked_poses = sorted(ml_poses, key=lambda x: x.confidence, reverse=True)
        top_poses = ranked_poses[:top_k_refinement]

        self.logger.info(f"Selected top {len(top_poses)} poses for physics refinement")

        # Step 3: Physics-based refinement
        self.logger.info("Performing physics-based refinement...")
        refinement_start = time.time()

        refined_poses = self._refine_poses_with_physics(
            top_poses, receptor_file, ligand_mol, grid_center, grid_dimensions
        )

        refinement_time = time.time() - refinement_start

        # Step 4: Final hybrid scoring
        self.logger.info("Applying final hybrid scoring...")
        scoring_start = time.time()

        final_poses = self._apply_hybrid_scoring(
            refined_poses, receptor_file, ligand_mol
        )

        scoring_time = time.time() - scoring_start

        # Create result
        result = DockingResult(
            ligand_name=ligand_name,
            receptor_file=receptor_file,
            grid_center=grid_center,
            grid_dimensions=grid_dimensions,
            algorithm_used=f"hybrid_ml_physics_{self.physics_refinement}",
            scoring_function="hybrid_ml_physics",
            poses=final_poses
        )

        # Add detailed timing metadata
        result.ml_generation_time = ml_time
        result.physics_refinement_time = refinement_time
        result.hybrid_scoring_time = scoring_time
        result.runtime_seconds = time.time() - start_time

        self.logger.info(f"Hybrid docking completed: {len(final_poses)} final poses")
        return result

    def _generate_ml_poses(self,
                          receptor_file: str,
                          ligand_mol: Chem.Mol,
                          grid_center: np.ndarray,
                          grid_dimensions: np.ndarray,
                          num_poses: int,
                          temperature: float) -> List[Pose]:
        """Generate poses using ML model"""
        try:
            # Extract features for ML model
            features = self.feature_extractor.extract_from_complex(receptor_file)

            # Generate poses with ML model
            poses = self.ml_model.generate_poses(
                receptor_features=features['protein'],
                ligand_mol=ligand_mol,
                grid_center=grid_center,
                grid_dimensions=grid_dimensions,
                num_poses=num_poses,
                temperature=temperature
            )

            return poses

        except Exception as e:
            self.logger.error(f"ML pose generation failed: {e}")
            # Fallback: return random poses
            return self._generate_fallback_poses(ligand_mol, grid_center, num_poses)

    def _generate_fallback_poses(self, ligand_mol: Chem.Mol, center: np.ndarray, num_poses: int) -> List[Pose]:
        """Generate fallback poses when ML fails"""
        poses = []

        # Generate conformers
        mol_copy = Chem.Mol(ligand_mol)
        mol_copy = Chem.AddHs(mol_copy)

        try:
            cids = AllChem.EmbedMultipleConfs(mol_copy, numConfs=num_poses, randomSeed=42)

            for i, cid in enumerate(cids):
                conf = mol_copy.GetConformer(cid)
                coords = []

                for atom_idx in range(mol_copy.GetNumAtoms()):
                    pos = conf.GetAtomPosition(atom_idx)
                    coords.append([pos.x, pos.y, pos.z])

                coords = np.array(coords)

                # Create pose
                pose = Pose(
                    coordinates=coords,
                    center=np.mean(coords, axis=0),
                    rotation=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
                    conformer_id=i,
                    energy=0.0,
                    confidence=0.5  # Default confidence
                )

                poses.append(pose)

        except Exception as e:
            self.logger.error(f"Fallback pose generation failed: {e}")

        return poses

    def _refine_poses_with_physics(self,
                                  poses: List[Pose],
                                  receptor_file: str,
                                  ligand_mol: Chem.Mol,
                                  grid_center: np.ndarray,
                                  grid_dimensions: np.ndarray) -> List[Pose]:
        """Refine poses using physics-based algorithms"""
        refined_poses = []

        try:
            # Use physics engine for refinement
            for i, pose in enumerate(poses):
                self.logger.debug(f"Refining pose {i+1}/{len(poses)}")

                # Create a temporary ligand with this pose's coordinates
                temp_mol = self._create_mol_with_coordinates(ligand_mol, pose.coordinates)

                # Run physics refinement (limited iterations for speed)
                try:
                    result = self.physics_engine.dock_ligand(
                        receptor_file=receptor_file,
                        ligand_mol=temp_mol,
                        grid_center=grid_center,
                        grid_dimensions=grid_dimensions,
                        algorithm=self.physics_refinement,
                        scoring_function='physics_based',
                        num_poses=1,  # Only refine this pose
                        max_attempts=50,  # Limited for speed
                        refinement_steps=10
                    )

                    if result.poses:
                        refined_pose = result.poses[0]
                        # Preserve ML confidence
                        refined_pose.ml_confidence = pose.confidence
                        refined_poses.append(refined_pose)
                    else:
                        # Keep original if refinement fails
                        refined_poses.append(pose)

                except Exception as e:
                    self.logger.warning(f"Physics refinement failed for pose {i}: {e}")
                    refined_poses.append(pose)

        except Exception as e:
            self.logger.error(f"Physics refinement pipeline failed: {e}")
            return poses  # Return original poses if refinement fails

        return refined_poses

    def _create_mol_with_coordinates(self, mol: Chem.Mol, coordinates: np.ndarray) -> Chem.Mol:
        """Create RDKit molecule with specific coordinates"""
        mol_copy = Chem.Mol(mol)

        # Add a conformer with the specified coordinates
        conf = Chem.Conformer(mol_copy.GetNumAtoms())

        for i, coord in enumerate(coordinates):
            conf.SetAtomPosition(i, (float(coord[0]), float(coord[1]), float(coord[2])))

        mol_copy.AddConformer(conf)
        return mol_copy

    def _apply_hybrid_scoring(self,
                            poses: List[Pose],
                            receptor_file: str,
                            ligand_mol: Chem.Mol) -> List[Pose]:
        """Apply hybrid ML+Physics scoring to poses"""
        scored_poses = []

        for pose in poses:
            try:
                # Physics-based score
                temp_mol = self._create_mol_with_coordinates(ligand_mol, pose.coordinates)
                physics_score = self.physics_scoring.score_pose(pose, receptor_file, temp_mol)

                # ML-based score (if model supports it)
                ml_score = getattr(pose, 'ml_confidence', 0.5)

                # Combine scores (weighted average)
                # This is a simple combination - could be learned
                combined_score = 0.7 * physics_score + 0.3 * (1.0 - ml_score)  # Lower is better

                # Update pose
                pose.energy = combined_score
                pose.physics_score = physics_score
                pose.ml_score = ml_score

                scored_poses.append(pose)

            except Exception as e:
                self.logger.warning(f"Scoring failed for pose: {e}")
                # Keep pose with original score
                scored_poses.append(pose)

        return scored_poses