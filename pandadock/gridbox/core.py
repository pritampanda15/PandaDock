import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from Bio.PDB import PDBParser, Selection
import json


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class GridBox:
    """Grid box definition for molecular docking"""
    center: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]
    spacing: float = 0.375
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    receptor_file: Optional[str] = None
    method_used: str = "manual"
    confidence_score: float = 1.0
    binding_site_info: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'center': list(self.center),
            'dimensions': list(self.dimensions),
            'spacing': self.spacing,
            'rotation': list(self.rotation),
            'receptor_file': self.receptor_file,
            'method_used': self.method_used,
            'confidence_score': self.confidence_score,
            'binding_site_info': self.binding_site_info or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GridBox':
        """Create from dictionary"""
        return cls(
            center=tuple(data['center']),
            dimensions=tuple(data['dimensions']),
            spacing=data.get('spacing', 0.375),
            rotation=tuple(data.get('rotation', [0.0, 0.0, 0.0])),
            receptor_file=data.get('receptor_file'),
            method_used=data.get('method_used', 'manual'),
            confidence_score=data.get('confidence_score', 1.0),
            binding_site_info=data.get('binding_site_info', {})
        )

    def get_corners(self) -> List[Tuple[float, float, float]]:
        """Get all 8 corners of the grid box"""
        cx, cy, cz = self.center
        dx, dy, dz = [d/2 for d in self.dimensions]

        corners = []
        for x_sign in [-1, 1]:
            for y_sign in [-1, 1]:
                for z_sign in [-1, 1]:
                    corner = (
                        cx + x_sign * dx,
                        cy + y_sign * dy,
                        cz + z_sign * dz
                    )
                    corners.append(corner)
        return corners

    def contains_point(self, point: Tuple[float, float, float]) -> bool:
        """Check if a point is inside the grid box"""
        px, py, pz = point
        cx, cy, cz = self.center
        dx, dy, dz = [d/2 for d in self.dimensions]

        return (abs(px - cx) <= dx and
                abs(py - cy) <= dy and
                abs(pz - cz) <= dz)

    def volume(self) -> float:
        """Calculate grid box volume"""
        return self.dimensions[0] * self.dimensions[1] * self.dimensions[2]

    def save(self, filename: str):
        """Save grid box to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, cls=NumpyEncoder)

    @classmethod
    def load(cls, filename: str) -> 'GridBox':
        """Load grid box from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class GridBoxGenerator:
    """Base class for grid box generation methods"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate(self, receptor_file: str, **kwargs) -> List[GridBox]:
        """Generate grid boxes - to be implemented by subclasses"""
        raise NotImplementedError

    def _parse_receptor(self, receptor_file: str):
        """Parse receptor structure"""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('receptor', receptor_file)
        return structure

    def _get_residue_center(self, residues: List) -> Tuple[float, float, float]:
        """Calculate center of mass for a list of residues"""
        atoms = []
        for residue in residues:
            atoms.extend(list(residue.get_atoms()))

        if not atoms:
            raise ValueError("No atoms found in specified residues")

        coords = np.array([atom.get_coord() for atom in atoms])
        center = np.mean(coords, axis=0)
        return tuple(center)

    def _calculate_bounding_box(self, atoms: List, padding: float = 5.0) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Calculate bounding box for atoms with padding"""
        if not atoms:
            raise ValueError("No atoms provided")

        coords = np.array([atom.get_coord() for atom in atoms])
        min_coords = np.min(coords, axis=0) - padding
        max_coords = np.max(coords, axis=0) + padding

        center = (min_coords + max_coords) / 2
        dimensions = max_coords - min_coords

        return tuple(center), tuple(dimensions)


class ManualGridBoxGenerator(GridBoxGenerator):
    """Generate grid box from manual coordinates"""

    def generate(self, receptor_file: str, center: Tuple[float, float, float],
                 dimensions: Tuple[float, float, float], spacing: float = 0.375, **kwargs) -> List[GridBox]:
        """Generate grid box from manual coordinates"""

        grid_box = GridBox(
            center=center,
            dimensions=dimensions,
            spacing=spacing,
            receptor_file=receptor_file,
            method_used="manual_coordinates",
            confidence_score=1.0
        )

        return [grid_box]


class ResidueBasedGridBoxGenerator(GridBoxGenerator):
    """Generate grid box based on specified residues"""

    def generate(self, receptor_file: str, residues: List[str],
                 padding: float = 5.0, spacing: float = 0.375, **kwargs) -> List[GridBox]:
        """Generate grid box from residue specifications"""

        structure = self._parse_receptor(receptor_file)
        selected_residues = []

        for residue_spec in residues:
            # Parse residue specification (e.g., "A:123", "B:67")
            try:
                if ':' in residue_spec:
                    chain_id, res_num = residue_spec.split(':')
                    res_num = int(res_num)
                else:
                    # Assume single chain
                    chain_id = list(structure[0].get_chains())[0].id
                    res_num = int(residue_spec)

                chain = structure[0][chain_id]
                residue = chain[res_num]
                selected_residues.append(residue)

            except (ValueError, KeyError) as e:
                self.logger.warning(f"Could not find residue {residue_spec}: {e}")

        if not selected_residues:
            raise ValueError("No valid residues found")

        # Get all atoms from selected residues
        atoms = []
        for residue in selected_residues:
            atoms.extend(list(residue.get_atoms()))

        center, dimensions = self._calculate_bounding_box(atoms, padding)

        grid_box = GridBox(
            center=center,
            dimensions=dimensions,
            spacing=spacing,
            receptor_file=receptor_file,
            method_used="residue_based",
            confidence_score=0.9,
            binding_site_info={
                'selected_residues': residues,
                'num_atoms': len(atoms),
                'padding': padding
            }
        )

        return [grid_box]