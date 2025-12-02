"""
Metal Parameter Manager for PandaDock

Handles metal ion parameters similar to AutoDock and GOLD protocols.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

class MetalParameterManager:
    """Manages metal ion parameters for docking calculations"""

    def __init__(self, parameter_file: Optional[str] = None):
        """
        Initialize metal parameter manager

        Args:
            parameter_file: Path to metal parameters file
        """
        self.logger = logging.getLogger(__name__)

        # Default parameter file path
        if parameter_file is None:
            parameter_file = Path(__file__).parent.parent.parent / "Metal_ions_parameter.txt"

        self.parameter_file = parameter_file
        self.metal_params = {}
        self.coordination_geometries = {}

        # Load parameters
        self._load_metal_parameters()
        self._initialize_coordination_geometries()

    def _load_metal_parameters(self):
        """Load metal parameters from parameter file"""
        try:
            with open(self.parameter_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line.startswith('atom_par') and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 11:
                        atom_type = parts[1]

                        # Check if this is a metal (based on common metal symbols)
                        metal_symbols = [
                            'Li', 'Na', 'K', 'Mg', 'MG', 'Ca', 'CA', 'Ba',
                            'Al', 'Ti', 'V', 'Cr', 'Mn', 'MN', 'Fe', 'FE',
                            'Co', 'Ni', 'Cu', 'Zn', 'ZN', 'Mo', 'Pd', 'Ag',
                            'Cd', 'Pt', 'Au', 'Hg', 'Pb', 'La', 'Gd'
                        ]

                        if atom_type in metal_symbols:
                            self.metal_params[atom_type] = {
                                'rii': float(parts[2]),           # van der Waals radius
                                'epsii': float(parts[3]),         # well depth
                                'vol': float(parts[4]),           # solvation volume
                                'solpar': float(parts[5]),        # solvation parameter
                                'r_hbond': float(parts[6]),       # H-bond radius
                                'eps_hbond': float(parts[7]),     # H-bond energy
                                'hbond': int(parts[8]),           # H-bond type
                                'bond_index': int(parts[11])      # bond index
                            }

            self.logger.info(f"Loaded parameters for {len(self.metal_params)} metal types")

        except Exception as e:
            self.logger.error(f"Error loading metal parameters: {e}")
            raise

    def _initialize_coordination_geometries(self):
        """Initialize common coordination geometries for metals"""

        self.coordination_geometries = {
            # Coordination number 2
            2: {
                'linear': {
                    'angles': [180.0],
                    'description': 'Linear coordination',
                    'common_metals': ['Cu', 'Ag', 'Au']
                }
            },

            # Coordination number 3
            3: {
                'trigonal_planar': {
                    'angles': [120.0, 120.0, 120.0],
                    'description': 'Trigonal planar',
                    'common_metals': ['Al', 'Cu']
                }
            },

            # Coordination number 4
            4: {
                'tetrahedral': {
                    'angles': [109.47, 109.47, 109.47, 109.47, 109.47, 109.47],
                    'description': 'Tetrahedral coordination',
                    'common_metals': ['Zn', 'ZN', 'Cu', 'Fe', 'FE', 'Mn', 'MN']
                },
                'square_planar': {
                    'angles': [90.0, 90.0, 90.0, 90.0],
                    'description': 'Square planar coordination',
                    'common_metals': ['Pt', 'Pd', 'Cu']
                }
            },

            # Coordination number 5
            5: {
                'trigonal_bipyramidal': {
                    'angles': [120.0, 120.0, 120.0, 90.0, 90.0],
                    'description': 'Trigonal bipyramidal',
                    'common_metals': ['Fe', 'FE', 'Co', 'Ni']
                },
                'square_pyramidal': {
                    'angles': [90.0, 90.0, 90.0, 90.0],
                    'description': 'Square pyramidal',
                    'common_metals': ['Cu', 'Ni']
                }
            },

            # Coordination number 6
            6: {
                'octahedral': {
                    'angles': [90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0],
                    'description': 'Octahedral coordination',
                    'common_metals': ['Mg', 'MG', 'Ca', 'CA', 'Fe', 'FE', 'Co', 'Ni', 'Zn', 'ZN']
                }
            }
        }

    def get_metal_parameters(self, metal_type: str) -> Optional[Dict]:
        """
        Get parameters for a specific metal type

        Args:
            metal_type: Metal atom type (e.g., 'Zn', 'Fe', 'Cu')

        Returns:
            Dictionary of metal parameters or None if not found
        """
        return self.metal_params.get(metal_type)

    def is_metal(self, atom_type: str) -> bool:
        """Check if an atom type is a metal"""
        return atom_type in self.metal_params

    def get_coordination_geometry(self, metal_type: str, coordination_number: int) -> Optional[Dict]:
        """
        Get preferred coordination geometry for a metal

        Args:
            metal_type: Metal atom type
            coordination_number: Number of coordinating atoms

        Returns:
            Dictionary with geometry information
        """
        if coordination_number not in self.coordination_geometries:
            return None

        geometries = self.coordination_geometries[coordination_number]

        # Find preferred geometry for this metal
        for geom_name, geom_data in geometries.items():
            if metal_type in geom_data['common_metals']:
                return {
                    'name': geom_name,
                    'angles': geom_data['angles'],
                    'description': geom_data['description']
                }

        # Return first available geometry if no specific preference
        first_geom = list(geometries.keys())[0]
        return {
            'name': first_geom,
            'angles': geometries[first_geom]['angles'],
            'description': geometries[first_geom]['description']
        }

    def calculate_metal_ligand_interaction(self, metal_type: str, distance: float,
                                         ligand_atom_type: str = 'OA') -> float:
        """
        Calculate metal-ligand interaction energy

        Args:
            metal_type: Metal atom type
            distance: Metal-ligand distance in Angstroms
            ligand_atom_type: Ligand atom type

        Returns:
            Interaction energy in kcal/mol
        """
        if not self.is_metal(metal_type):
            return 0.0

        metal_params = self.metal_params[metal_type]

        # Lennard-Jones 12-6 potential
        r_min = metal_params['rii']
        epsilon = metal_params['epsii']

        if distance < 0.1:  # Avoid division by zero
            return 1000.0  # Large repulsive energy

        # LJ potential: E = epsilon * [(r_min/r)^12 - 2*(r_min/r)^6]
        r_ratio = r_min / distance
        energy = epsilon * (r_ratio**12 - 2 * r_ratio**6)

        return energy

    def get_supported_metals(self) -> List[str]:
        """Get list of all supported metal types"""
        return list(self.metal_params.keys())

    def get_metal_info(self, metal_type: str) -> str:
        """Get detailed information about a metal type"""
        if not self.is_metal(metal_type):
            return f"Metal type '{metal_type}' not supported"

        params = self.metal_params[metal_type]
        info = f"""
Metal Type: {metal_type}
van der Waals radius: {params['rii']:.2f} Å
Well depth: {params['epsii']:.3f} kcal/mol
Solvation volume: {params['vol']:.1f} Å³
Solvation parameter: {params['solpar']:.5f}
"""
        return info.strip()