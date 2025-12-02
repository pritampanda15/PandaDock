"""
Metal Docking Core Engine

Core classes for metal-aware molecular docking following Glide and AutoDock protocols.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

from ..docking.core import DockingEngine, DockingResult, Pose
from .metal_parameters import MetalParameterManager
from .metal_preparation import MetalLigandPreparator, MetalloproteinPreparator
from .metal_scoring import MetalAwareScoring, MetalConstrainedScoring

class MetalCoordinationGeometry:
    """Handles metal coordination geometry calculations"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Standard coordination geometries
        self.geometries = {
            2: 'linear',
            3: 'trigonal_planar',
            4: 'tetrahedral',
            5: 'trigonal_bipyramidal',
            6: 'octahedral',
            8: 'square_antiprismatic'
        }

    def calculate_coordination_geometry_score(self, metal_coords: np.ndarray,
                                           coordinating_coords: List[np.ndarray],
                                           expected_geometry: str = 'tetrahedral') -> float:
        """
        Calculate geometry score based on deviation from ideal geometry

        Args:
            metal_coords: Metal center coordinates
            coordinating_coords: List of coordinating atom coordinates
            expected_geometry: Expected coordination geometry

        Returns:
            Geometry score (lower is better)
        """
        n_coords = len(coordinating_coords)

        if n_coords < 2:
            return 0.0

        # Calculate all pairwise angles
        angles = []
        for i in range(n_coords):
            for j in range(i + 1, n_coords):
                vec1 = coordinating_coords[i] - metal_coords
                vec2 = coordinating_coords[j] - metal_coords

                vec1_norm = vec1 / np.linalg.norm(vec1)
                vec2_norm = vec2 / np.linalg.norm(vec2)

                angle = np.arccos(np.clip(np.dot(vec1_norm, vec2_norm), -1.0, 1.0))
                angles.append(np.degrees(angle))

        # Expected angles for different geometries
        ideal_angles = {
            'linear': [180.0],
            'trigonal_planar': [120.0],
            'tetrahedral': [109.47],
            'square_planar': [90.0],
            'trigonal_bipyramidal': [90.0, 120.0],
            'octahedral': [90.0],
            'square_antiprismatic': [90.0]
        }

        if expected_geometry not in ideal_angles:
            expected_geometry = 'tetrahedral'  # Default

        target_angles = ideal_angles[expected_geometry]

        # Calculate RMSD from ideal geometry
        rmsd = 0.0
        for angle in angles:
            min_dev = min(abs(angle - target) for target in target_angles)
            rmsd += min_dev ** 2

        rmsd = np.sqrt(rmsd / len(angles)) if angles else 0.0
        return rmsd

class MetalDockingResult(DockingResult):
    """Extended docking result for metal-aware docking"""

    def __init__(self):
        super().__init__()
        self.metal_interactions = []
        self.coordination_scores = []
        self.metal_binding_scores = []
        self.constraint_violations = []

    def add_metal_pose(self, pose: Pose, metal_interactions: Dict,
                      coordination_score: float, metal_binding_score: float,
                      constraint_violations: List[str]):
        """Add pose with metal-specific information"""
        self.add_pose(pose)
        self.metal_interactions.append(metal_interactions)
        self.coordination_scores.append(coordination_score)
        self.metal_binding_scores.append(metal_binding_score)
        self.constraint_violations.append(constraint_violations)

    def get_metal_summary(self) -> Dict:
        """Get summary of metal interactions"""
        if not self.poses:
            return {}

        return {
            'n_poses': len(self.poses),
            'avg_coordination_score': np.mean(self.coordination_scores),
            'avg_metal_binding_score': np.mean(self.metal_binding_scores),
            'best_coordination_score': min(self.coordination_scores),
            'poses_with_violations': sum(1 for v in self.constraint_violations if v),
            'violation_rate': sum(1 for v in self.constraint_violations if v) / len(self.poses)
        }

class MetalDockingEngine:
    """Main engine for metal-aware molecular docking"""

    def __init__(self, parameter_file: Optional[str] = None):
        """
        Initialize metal docking engine

        Args:
            parameter_file: Path to metal parameters file
        """
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.param_manager = MetalParameterManager(parameter_file)
        self.ligand_prep = MetalLigandPreparator()
        self.protein_prep = MetalloproteinPreparator()
        self.geometry_calc = MetalCoordinationGeometry()

        # Initialize base docking engine
        self.base_engine = DockingEngine()

    def prepare_metal_ligand(self, ligand_file: str, output_file: Optional[str] = None) -> Chem.Mol:
        """
        Prepare metal-containing ligand

        Args:
            ligand_file: Input ligand file path
            output_file: Output file path (optional)

        Returns:
            Prepared RDKit molecule
        """
        # Load ligand
        if ligand_file.endswith('.sdf'):
            mol = Chem.SDMolSupplier(ligand_file)[0]
        elif ligand_file.endswith('.mol2'):
            mol = Chem.MolFromMol2File(ligand_file)
        elif ligand_file.endswith('.pdb'):
            mol = Chem.MolFromPDBFile(ligand_file)
        else:
            mol = Chem.MolFromMolFile(ligand_file)

        if mol is None:
            raise ValueError(f"Could not load ligand from {ligand_file}")

        # Prepare metal ligand
        prepared_mol = self.ligand_prep.prepare_metal_ligand(mol)

        # Save if requested
        if output_file:
            if output_file.endswith('.sdf'):
                writer = Chem.SDWriter(output_file)
                writer.write(prepared_mol)
                writer.close()
            elif output_file.endswith('.mol2'):
                Chem.MolToMol2File(prepared_mol, output_file)

        return prepared_mol

    def dock_with_metal_awareness(self,
                                receptor_file: str,
                                ligand_mol: Chem.Mol,
                                grid_center: np.ndarray,
                                grid_dimensions: np.ndarray,
                                algorithm: str = 'enhanced_hierarchical_cpu',
                                scoring_function: str = 'metal_aware',
                                num_poses: int = 20,
                                use_constraints: bool = True,
                                constraint_weight: float = 10.0,
                                **kwargs) -> MetalDockingResult:
        """
        Perform metal-aware molecular docking

        Args:
            receptor_file: Receptor PDB file
            ligand_mol: Prepared ligand molecule
            grid_center: Grid center coordinates
            grid_dimensions: Grid box dimensions
            algorithm: Docking algorithm
            scoring_function: Scoring function type
            num_poses: Number of poses to generate
            use_constraints: Whether to use geometric constraints
            constraint_weight: Weight for constraint violations
            **kwargs: Additional parameters

        Returns:
            Metal docking results
        """
        self.logger.info("Starting metal-aware docking...")

        # Prepare metalloprotein
        protein_info = self.protein_prep.prepare_metalloprotein(receptor_file)
        metal_sites = protein_info['metal_sites']

        if not metal_sites:
            self.logger.warning("No metal sites found in receptor - using standard docking")
            return self._dock_without_metals(
                receptor_file, ligand_mol, grid_center, grid_dimensions,
                algorithm, num_poses, **kwargs
            )

        self.logger.info(f"Found {len(metal_sites)} metal sites in receptor")

        # Check if ligand contains metals
        ligand_metals = self.ligand_prep.identify_metal_atoms(ligand_mol)
        if ligand_metals:
            self.logger.info(f"Ligand contains {len(ligand_metals)} metal atoms")

        # Initialize scoring function
        if use_constraints:
            scorer = MetalConstrainedScoring(self.param_manager, constraint_weight)
        else:
            scorer = MetalAwareScoring(self.param_manager)

        # Perform docking with multiple starting poses
        result = MetalDockingResult()

        # Generate initial poses using base engine
        base_result = self.base_engine.dock(
            receptor_file=receptor_file,
            ligand_mol=ligand_mol,
            algorithm=algorithm,
            scoring_function='physics_based',  # Use physics-based for initial poses
            grid_center=grid_center,
            grid_dimensions=grid_dimensions,
            num_poses=num_poses * 2,  # Generate more poses for filtering
            **kwargs
        )

        if not base_result.poses:
            self.logger.error("Base docking failed to generate poses")
            return result

        # Score poses with metal-aware scoring
        self.logger.info(f"Rescoring {len(base_result.poses)} poses with metal-aware scoring")

        scored_poses = []
        for pose in base_result.poses:
            # Get pose coordinates (this needs to be implemented based on your Pose class)
            ligand_coords = pose.coordinates  # Assuming this exists

            # Score with metal awareness
            if use_constraints:
                scores = scorer.score_constrained_pose(
                    ligand_coords, ligand_mol, np.array([]), [],  # Receptor coords not needed here
                    metal_sites
                )
            else:
                scores = scorer.score_metal_pose(
                    ligand_coords, ligand_mol, np.array([]), [],
                    metal_sites
                )

            # Create new pose with metal-aware score
            metal_pose = Pose(
                coordinates=pose.coordinates,
                energy=scores['total_score'],
                confidence=pose.confidence
            )

            # Store additional metal information
            metal_interactions = {
                'coordination_score': scores.get('metal_coordination', 0.0),
                'binding_score': scores.get('metal_binding', 0.0),
                'base_score': scores.get('base_score', 0.0)
            }

            constraint_violations = []
            if 'constraint_penalty' in scores and scores['constraint_penalty'] > 0:
                constraint_violations.append(f"Geometry constraint violation: {scores['constraint_penalty']:.2f}")

            result.add_metal_pose(
                metal_pose, metal_interactions,
                scores.get('metal_coordination', 0.0),
                scores.get('metal_binding', 0.0),
                constraint_violations
            )

            scored_poses.append((metal_pose, scores['total_score']))

        # Sort poses by metal-aware score and keep best ones
        scored_poses.sort(key=lambda x: x[1])
        final_poses = [pose for pose, score in scored_poses[:num_poses]]

        # Update result with final poses
        result.poses = final_poses
        result.best_pose = final_poses[0] if final_poses else None

        self.logger.info(f"Metal-aware docking completed with {len(final_poses)} poses")

        return result

    def _dock_without_metals(self, receptor_file: str, ligand_mol: Chem.Mol,
                           grid_center: np.ndarray, grid_dimensions: np.ndarray,
                           algorithm: str, num_poses: int, **kwargs) -> MetalDockingResult:
        """Fallback to standard docking when no metals are present"""

        base_result = self.base_engine.dock(
            receptor_file=receptor_file,
            ligand_mol=ligand_mol,
            algorithm=algorithm,
            scoring_function='physics_based',
            grid_center=grid_center,
            grid_dimensions=grid_dimensions,
            num_poses=num_poses,
            **kwargs
        )

        # Convert to MetalDockingResult
        result = MetalDockingResult()
        result.poses = base_result.poses
        result.best_pose = base_result.best_pose

        # Fill metal-specific fields with empty/zero values
        for _ in result.poses:
            result.metal_interactions.append({})
            result.coordination_scores.append(0.0)
            result.metal_binding_scores.append(0.0)
            result.constraint_violations.append([])

        return result

    def validate_metal_preparation(self, ligand_mol: Chem.Mol) -> Dict[str, any]:
        """
        Validate metal ligand preparation

        Args:
            ligand_mol: Prepared ligand molecule

        Returns:
            Validation report
        """
        report = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'metal_atoms': [],
            'coordination_info': []
        }

        # Check for metal atoms
        metal_indices = self.ligand_prep.identify_metal_atoms(ligand_mol)

        if not metal_indices:
            report['warnings'].append("No metal atoms found in ligand")
            return report

        report['metal_atoms'] = metal_indices

        # Validate each metal center
        for metal_idx in metal_indices:
            metal_atom = ligand_mol.GetAtomWithIdx(metal_idx)
            metal_symbol = metal_atom.GetSymbol()

            # Check if metal is supported
            if not self.param_manager.is_metal(metal_symbol):
                report['errors'].append(f"Unsupported metal: {metal_symbol}")
                report['is_valid'] = False
                continue

            # Find coordinating atoms
            coordinating_atoms = self.ligand_prep.identify_coordinating_atoms(ligand_mol, metal_idx)

            if not coordinating_atoms:
                report['warnings'].append(f"No coordinating atoms found for metal {metal_symbol}")

            coord_info = {
                'metal_index': metal_idx,
                'metal_symbol': metal_symbol,
                'coordination_number': len(coordinating_atoms),
                'coordinating_atoms': coordinating_atoms
            }

            # Check formal charges
            if metal_atom.GetFormalCharge() == 0:
                report['warnings'].append(f"Metal {metal_symbol} has zero formal charge")

            # Validate geometry
            if not self.ligand_prep.validate_coordination_geometry(ligand_mol, metal_idx, coordinating_atoms):
                report['warnings'].append(f"Questionable coordination geometry for metal {metal_symbol}")

            report['coordination_info'].append(coord_info)

        return report

    def get_supported_metals(self) -> List[str]:
        """Get list of supported metal types"""
        return self.param_manager.get_supported_metals()

    def get_metal_info(self, metal_type: str) -> str:
        """Get information about a specific metal type"""
        return self.param_manager.get_metal_info(metal_type)