"""
Interaction Energy Analysis and Decomposition

Provides detailed analysis of protein-ligand interactions including:
1. Energy decomposition by interaction type
2. Per-residue interaction contributions
3. Interaction fingerprinting
4. Correlation analysis with experimental data
5. Visualization of interaction networks
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import logging
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio.PDB.Structure import Structure
import matplotlib.pyplot as plt
import seaborn as sns

from ..core import Pose, DockingResult


class InteractionAnalyzer:
    """Analyze and decompose protein-ligand interactions"""

    def __init__(self):
        self.logger = logging.getLogger("pandadock.analysis.interaction")

        # Interaction types
        self.interaction_types = [
            'hydrogen_bond', 'hydrophobic', 'electrostatic', 'van_der_waals',
            'pi_pi_stacking', 'pi_cation', 'metal_coordination', 'halogen_bond'
        ]

        # Distance cutoffs for different interactions (Å)
        self.cutoffs = {
            'hydrogen_bond': 3.5,
            'hydrophobic': 4.0,
            'electrostatic': 6.0,
            'van_der_waals': 4.5,
            'pi_pi_stacking': 5.0,
            'pi_cation': 6.0,
            'metal_coordination': 3.0,
            'halogen_bond': 4.0
        }

        # Amino acid classifications
        self.hydrophobic_residues = {'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TRP', 'PRO'}
        self.polar_residues = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'}
        self.charged_residues = {'ARG', 'LYS', 'ASP', 'GLU', 'HIS'}
        self.aromatic_residues = {'PHE', 'TRP', 'TYR', 'HIS'}

    def analyze_docking_result(self, result: DockingResult, ligand_mol: Chem.Mol,
                             receptor_structure: Structure, experimental_data: Optional[Dict] = None) -> Dict:
        """
        Comprehensive analysis of docking result

        Args:
            result: Docking result with poses
            ligand_mol: RDKit molecule object
            receptor_structure: BioPython structure
            experimental_data: Optional experimental binding data (IC50, Kd, etc.)

        Returns:
            Dictionary with comprehensive interaction analysis
        """
        analysis = {
            'pose_analysis': [],
            'interaction_summary': {},
            'residue_contributions': {},
            'interaction_fingerprint': {},
            'energy_correlations': {},
            'binding_site_analysis': {}
        }

        self.logger.info(f"Analyzing {len(result.poses)} poses for interaction patterns")

        # Analyze each pose
        for i, pose in enumerate(result.poses):
            pose_analysis = self.analyze_single_pose(
                pose, ligand_mol, receptor_structure, pose_id=i
            )
            analysis['pose_analysis'].append(pose_analysis)

        # Generate summary statistics
        analysis['interaction_summary'] = self._generate_interaction_summary(analysis['pose_analysis'])
        analysis['residue_contributions'] = self._calculate_residue_contributions(analysis['pose_analysis'])
        analysis['interaction_fingerprint'] = self._generate_interaction_fingerprint(analysis['pose_analysis'])
        analysis['binding_site_analysis'] = self._analyze_binding_site(analysis['pose_analysis'], receptor_structure)

        # Correlation analysis if experimental data provided
        if experimental_data:
            analysis['energy_correlations'] = self._analyze_energy_correlations(
                result, analysis, experimental_data
            )

        return analysis

    def analyze_pose_interactions(self, ligand_coords: np.ndarray,
                                 receptor_structure: Optional[Structure] = None,
                                 ligand_mol: Optional[Chem.Mol] = None) -> Dict:
        """
        Analyze interactions for a single pose given coordinates

        Args:
            ligand_coords: Ligand atom coordinates [N, 3]
            receptor_structure: Optional receptor structure
            ligand_mol: Optional ligand molecule

        Returns:
            Dictionary with interaction analysis results
        """
        if receptor_structure is None or ligand_mol is None:
            self.logger.warning("Missing receptor or ligand data for interaction analysis")
            return {
                'total_interactions': 0,
                'interaction_types': {},
                'binding_affinity_estimate': 0.0,
                'interaction_summary': "Analysis requires receptor structure and ligand molecule"
            }

        receptor_atoms = list(receptor_structure.get_atoms())
        if not receptor_atoms:
            return {
                'total_interactions': 0,
                'interaction_types': {},
                'interaction_summary': "Receptor structure contains no atoms",
            }

        # Use the chemistry-aware detectors rather than raw distance counting.
        #
        # This previously took each ligand atom's distance to its single nearest
        # receptor atom and thresholded it three times. That counted a carbon
        # near a carbon as a hydrogen bond, never checked donor/acceptor identity
        # or hydrophobicity, left electrostatics permanently at zero, and summed
        # three nested subsets of the same array into a "total" that
        # double-counted every close contact. The detectors below were already
        # present in this module and simply were not being called.
        detected = {
            'hydrogen_bonds': self._find_hydrogen_bonds(
                ligand_coords, receptor_atoms, ligand_mol),
            'hydrophobic_contacts': self._find_hydrophobic_contacts(
                ligand_coords, receptor_atoms, ligand_mol),
            'electrostatic_interactions': self._find_electrostatic_interactions(
                ligand_coords, receptor_atoms, ligand_mol),
            'van_der_waals_contacts': self._find_vdw_contacts(
                ligand_coords, receptor_atoms, ligand_mol),
            'pi_stacking': self._find_pi_stacking(
                ligand_coords, receptor_atoms, ligand_mol),
            'pi_cation': self._find_pi_cation_interactions(
                ligand_coords, receptor_atoms, ligand_mol),
            'metal_coordination': self._find_metal_coordination(
                ligand_coords, receptor_atoms, ligand_mol),
        }

        interactions = {}
        for name, found in detected.items():
            try:
                interactions[name] = int(len(found))
            except TypeError:
                interactions[name] = 0

        total_interactions = sum(interactions.values())

        # No fabricated affinity is reported here. The previous
        # 'binding_affinity_estimate' applied invented per-interaction weights to
        # the miscounted totals above and clamped the result to [-15, 5], so it
        # saturated at -15.0 for most ligands and sat next to the real docking
        # score in the same output directory, inviting confusion between the two.
        # Use the score in the poses/summary files for binding energy.
        return {
            'total_interactions': int(total_interactions),
            'interaction_types': interactions,
            'interaction_details': {
                name: found for name, found in detected.items() if found
            },
            'interaction_summary': (
                f"Found {total_interactions} interactions: "
                + ", ".join(f"{n.replace('_', ' ')}={c}"
                            for n, c in interactions.items() if c)
            ),
        }

    def analyze_single_pose(self, pose: Pose, ligand_mol: Chem.Mol,
                          receptor_structure: Structure, pose_id: int = 0) -> Dict:
        """Detailed analysis of a single pose"""

        # Get receptor atoms and residues
        receptor_atoms = list(receptor_structure.get_atoms())
        receptor_coords = np.array([atom.get_coord() for atom in receptor_atoms])

        # Initialize analysis
        pose_analysis = {
            'pose_id': pose_id,
            'binding_energy': pose.energy,
            'interaction_energy': getattr(pose, 'interaction_energy', 0.0),
            'interactions': {itype: [] for itype in self.interaction_types},
            'residue_contacts': defaultdict(list),
            'energy_decomposition': getattr(pose, 'mmgbsa_components', {}),
            'geometric_properties': {}
        }

        # Calculate geometric properties
        pose_analysis['geometric_properties'] = self._calculate_geometric_properties(
            pose.coordinates, ligand_mol
        )

        # Analyze interactions by type
        pose_analysis['interactions']['hydrogen_bond'] = self._find_hydrogen_bonds(
            pose.coordinates, receptor_atoms, ligand_mol
        )

        pose_analysis['interactions']['hydrophobic'] = self._find_hydrophobic_contacts(
            pose.coordinates, receptor_atoms, ligand_mol
        )

        pose_analysis['interactions']['electrostatic'] = self._find_electrostatic_interactions(
            pose.coordinates, receptor_atoms, ligand_mol
        )

        pose_analysis['interactions']['van_der_waals'] = self._find_vdw_contacts(
            pose.coordinates, receptor_atoms, ligand_mol
        )

        pose_analysis['interactions']['pi_pi_stacking'] = self._find_pi_stacking(
            pose.coordinates, receptor_atoms, ligand_mol
        )

        pose_analysis['interactions']['pi_cation'] = self._find_pi_cation_interactions(
            pose.coordinates, receptor_atoms, ligand_mol
        )

        pose_analysis['interactions']['metal_coordination'] = self._find_metal_coordination(
            pose.coordinates, receptor_atoms, ligand_mol
        )

        # Calculate per-residue contributions
        pose_analysis['residue_contacts'] = self._calculate_residue_contacts(
            pose.coordinates, receptor_atoms, ligand_mol
        )

        return pose_analysis

    def _estimate_binding_affinity(self, interactions: Dict[str, int], total_interactions: int) -> float:
        """Estimate binding affinity based on interaction counts"""
        # Simple scoring model based on interaction types
        h_bonds = interactions.get('hydrogen_bonds', 0)
        hydrophobic = interactions.get('hydrophobic_contacts', 0)
        vdw = interactions.get('van_der_waals_contacts', 0)
        electrostatic = interactions.get('electrostatic_interactions', 0)

        # Scoring weights (kcal/mol per interaction)
        affinity = 0.0
        affinity -= h_bonds * 1.0        # H-bonds are worth ~1 kcal/mol each
        affinity -= hydrophobic * 0.3    # Hydrophobic contacts ~0.3 kcal/mol
        affinity -= vdw * 0.1           # VdW contacts ~0.1 kcal/mol
        affinity -= electrostatic * 1.5  # Electrostatic ~1.5 kcal/mol

        # Add baseline penalty for ligand size (entropy loss)
        affinity += 2.0  # Typical entropy penalty

        # Clamp to reasonable binding energy range
        return max(-15.0, min(5.0, affinity))

    def _find_hydrogen_bonds(self, ligand_coords: np.ndarray, receptor_atoms: List,
                           ligand_mol: Chem.Mol) -> List[Dict]:
        """Find hydrogen bond interactions"""
        hbonds = []

        # Get H-bond donors and acceptors
        lig_hb_atoms = self._get_ligand_hb_atoms(ligand_mol)
        rec_hb_atoms = self._get_receptor_hb_atoms(receptor_atoms)

        for lig_idx, lig_type in lig_hb_atoms.items():
            lig_coord = ligand_coords[lig_idx]

            for rec_idx, rec_info in rec_hb_atoms.items():
                rec_coord = receptor_atoms[rec_idx].get_coord()
                distance = np.linalg.norm(lig_coord - rec_coord)

                if distance <= self.cutoffs['hydrogen_bond']:
                    # Check if donor-acceptor pair
                    if self._hb_pair_is_complementary(lig_type, rec_info['type']):

                        residue = receptor_atoms[rec_idx].get_parent()
                        hbond = {
                            'ligand_atom': lig_idx,
                            'receptor_atom': rec_idx,
                            'residue': f"{residue.get_resname()}_{residue.get_id()[1]}",
                            'distance': distance,
                            'strength': self._calculate_hbond_strength(distance),
                            'geometry': self._calculate_hbond_geometry(lig_coord, rec_coord, lig_type, rec_info)
                        }
                        hbonds.append(hbond)

        return hbonds

    def _find_hydrophobic_contacts(self, ligand_coords: np.ndarray, receptor_atoms: List,
                                 ligand_mol: Chem.Mol) -> List[Dict]:
        """Find hydrophobic contacts"""
        contacts = []

        # Find hydrophobic atoms
        lig_hydrophobic = self._get_hydrophobic_ligand_atoms(ligand_mol)

        for lig_idx in lig_hydrophobic:
            lig_coord = ligand_coords[lig_idx]

            for rec_idx, atom in enumerate(receptor_atoms):
                if atom.element == 'C':  # Carbon atoms
                    residue = atom.get_parent()
                    if residue.get_resname() in self.hydrophobic_residues:
                        rec_coord = atom.get_coord()
                        distance = np.linalg.norm(lig_coord - rec_coord)

                        if distance <= self.cutoffs['hydrophobic']:
                            contact = {
                                'ligand_atom': lig_idx,
                                'receptor_atom': rec_idx,
                                'residue': f"{residue.get_resname()}_{residue.get_id()[1]}",
                                'distance': distance,
                                'strength': self._calculate_hydrophobic_strength(distance)
                            }
                            contacts.append(contact)

        return contacts

    def _find_electrostatic_interactions(self, ligand_coords: np.ndarray, receptor_atoms: List,
                                       ligand_mol: Chem.Mol) -> List[Dict]:
        """Find electrostatic interactions"""
        interactions = []

        # Get charged atoms
        try:
            AllChem.ComputeGasteigerCharges(ligand_mol)
            ligand_charges = [float(atom.GetProp('_GasteigerCharge')) for atom in ligand_mol.GetAtoms()]
        except:
            ligand_charges = [0.0] * ligand_mol.GetNumAtoms()

        for lig_idx, lig_charge in enumerate(ligand_charges):
            if abs(lig_charge) > 0.1:  # Only consider significantly charged atoms
                lig_coord = ligand_coords[lig_idx]

                for rec_idx, atom in enumerate(receptor_atoms):
                    residue = atom.get_parent()
                    if residue.get_resname() in self.charged_residues:
                        rec_coord = atom.get_coord()
                        distance = np.linalg.norm(lig_coord - rec_coord)

                        if distance <= self.cutoffs['electrostatic']:
                            # Get receptor charge
                            rec_charge = self._get_atom_charge(atom)

                            if rec_charge != 0:
                                interaction = {
                                    'ligand_atom': lig_idx,
                                    'receptor_atom': rec_idx,
                                    'residue': f"{residue.get_resname()}_{residue.get_id()[1]}",
                                    'distance': distance,
                                    'ligand_charge': lig_charge,
                                    'receptor_charge': rec_charge,
                                    'strength': self._calculate_electrostatic_strength(lig_charge, rec_charge, distance)
                                }
                                interactions.append(interaction)

        return interactions

    def _find_vdw_contacts(self, ligand_coords: np.ndarray, receptor_atoms: List,
                         ligand_mol: Chem.Mol) -> List[Dict]:
        """Find van der Waals contacts"""
        contacts = []

        for lig_idx in range(len(ligand_coords)):
            lig_coord = ligand_coords[lig_idx]

            for rec_idx, atom in enumerate(receptor_atoms):
                rec_coord = atom.get_coord()
                distance = np.linalg.norm(lig_coord - rec_coord)

                if distance <= self.cutoffs['van_der_waals']:
                    residue = atom.get_parent()
                    contact = {
                        'ligand_atom': lig_idx,
                        'receptor_atom': rec_idx,
                        'residue': f"{residue.get_resname()}_{residue.get_id()[1]}",
                        'distance': distance,
                        'strength': self._calculate_vdw_strength(distance)
                    }
                    contacts.append(contact)

        return contacts

    def _find_pi_stacking(self, ligand_coords: np.ndarray, receptor_atoms: List,
                        ligand_mol: Chem.Mol) -> List[Dict]:
        """Find π-π stacking interactions"""
        stacking = []

        # Find aromatic rings in ligand
        lig_rings = self._get_aromatic_rings(ligand_mol)

        for ring_atoms in lig_rings:
            ring_center = np.mean(ligand_coords[ring_atoms], axis=0)

            # Find aromatic residues in receptor
            for rec_idx, atom in enumerate(receptor_atoms):
                residue = atom.get_parent()
                if (residue.get_resname() in self.aromatic_residues and
                    atom.element == 'C' and
                    atom.get_name() in ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'CH2']):

                    # Get aromatic ring center (simplified)
                    rec_coord = atom.get_coord()
                    distance = np.linalg.norm(ring_center - rec_coord)

                    if distance <= self.cutoffs['pi_pi_stacking']:
                        interaction = {
                            'ligand_ring': ring_atoms,
                            'receptor_atom': rec_idx,
                            'residue': f"{residue.get_resname()}_{residue.get_id()[1]}",
                            'distance': distance,
                            'strength': self._calculate_pi_stacking_strength(distance)
                        }
                        stacking.append(interaction)

        return stacking

    def _find_pi_cation_interactions(self, ligand_coords: np.ndarray, receptor_atoms: List,
                                   ligand_mol: Chem.Mol) -> List[Dict]:
        """Find π-cation interactions"""
        interactions = []

        # Find aromatic rings and cations
        lig_rings = self._get_aromatic_rings(ligand_mol)

        for ring_atoms in lig_rings:
            ring_center = np.mean(ligand_coords[ring_atoms], axis=0)

            # Find cationic residues
            for rec_idx, atom in enumerate(receptor_atoms):
                residue = atom.get_parent()
                if (residue.get_resname() in ['ARG', 'LYS'] and
                    atom.element == 'N'):

                    rec_coord = atom.get_coord()
                    distance = np.linalg.norm(ring_center - rec_coord)

                    if distance <= self.cutoffs['pi_cation']:
                        interaction = {
                            'ligand_ring': ring_atoms,
                            'receptor_atom': rec_idx,
                            'residue': f"{residue.get_resname()}_{residue.get_id()[1]}",
                            'distance': distance,
                            'strength': self._calculate_pi_cation_strength(distance)
                        }
                        interactions.append(interaction)

        return interactions

    def _find_metal_coordination(self, ligand_coords: np.ndarray, receptor_atoms: List,
                               ligand_mol: Chem.Mol) -> List[Dict]:
        """Find metal coordination interactions"""
        coordinations = []

        # Find metal atoms
        metal_elements = {'MG', 'CA', 'ZN', 'FE', 'MN', 'CO', 'NI', 'CU'}

        for lig_idx, atom in enumerate(ligand_mol.GetAtoms()):
            if atom.GetSymbol() in ['O', 'N', 'S']:  # Potential coordinating atoms
                lig_coord = ligand_coords[lig_idx]

                for rec_idx, rec_atom in enumerate(receptor_atoms):
                    if rec_atom.element in metal_elements:
                        rec_coord = rec_atom.get_coord()
                        distance = np.linalg.norm(lig_coord - rec_coord)

                        if distance <= self.cutoffs['metal_coordination']:
                            residue = rec_atom.get_parent()
                            coordination = {
                                'ligand_atom': lig_idx,
                                'metal_atom': rec_idx,
                                'metal_type': rec_atom.element,
                                'residue': f"{residue.get_resname()}_{residue.get_id()[1]}",
                                'distance': distance,
                                'strength': self._calculate_metal_coordination_strength(distance)
                            }
                            coordinations.append(coordination)

        return coordinations

    def _calculate_residue_contacts(self, ligand_coords: np.ndarray, receptor_atoms: List,
                                  ligand_mol: Chem.Mol) -> Dict:
        """Calculate contacts grouped by residue"""
        residue_contacts = defaultdict(lambda: {
            'contact_count': 0,
            'min_distance': float('inf'),
            'interaction_types': set(),
            'total_strength': 0.0
        })

        for rec_idx, atom in enumerate(receptor_atoms):
            residue = atom.get_parent()
            residue_key = f"{residue.get_resname()}_{residue.get_id()[1]}"

            rec_coord = atom.get_coord()

            for lig_idx in range(len(ligand_coords)):
                lig_coord = ligand_coords[lig_idx]
                distance = np.linalg.norm(lig_coord - rec_coord)

                if distance <= 5.0:  # Contact cutoff
                    residue_contacts[residue_key]['contact_count'] += 1
                    residue_contacts[residue_key]['min_distance'] = min(
                        residue_contacts[residue_key]['min_distance'], distance
                    )

                    # Classify interaction type
                    if distance <= 3.5:
                        residue_contacts[residue_key]['interaction_types'].add('close_contact')
                    if residue.get_resname() in self.hydrophobic_residues and distance <= 4.0:
                        residue_contacts[residue_key]['interaction_types'].add('hydrophobic')
                    if residue.get_resname() in self.charged_residues and distance <= 6.0:
                        residue_contacts[residue_key]['interaction_types'].add('electrostatic')

        return dict(residue_contacts)

    def _generate_interaction_summary(self, pose_analyses: List[Dict]) -> Dict:
        """Generate summary statistics across all poses"""
        summary = {
            'total_poses': len(pose_analyses),
            'interaction_counts': {itype: [] for itype in self.interaction_types},
            'average_interactions': {},
            'interaction_frequencies': {},
            'common_residues': defaultdict(int)
        }

        for pose in pose_analyses:
            for itype in self.interaction_types:
                count = len(pose['interactions'][itype])
                summary['interaction_counts'][itype].append(count)

            # Count residue occurrences
            for residue in pose['residue_contacts']:
                summary['common_residues'][residue] += 1

        # Calculate averages and frequencies
        for itype in self.interaction_types:
            counts = summary['interaction_counts'][itype]
            summary['average_interactions'][itype] = np.mean(counts) if counts else 0
            summary['interaction_frequencies'][itype] = len([c for c in counts if c > 0]) / len(pose_analyses)

        return summary

    def _calculate_residue_contributions(self, pose_analyses: List[Dict]) -> Dict:
        """Calculate per-residue energy contributions"""
        contributions = defaultdict(lambda: {
            'frequency': 0,
            'average_contacts': 0.0,
            'total_strength': 0.0,
            'interaction_types': set()
        })

        for pose in pose_analyses:
            for residue, contact_info in pose['residue_contacts'].items():
                contributions[residue]['frequency'] += 1
                contributions[residue]['average_contacts'] += contact_info['contact_count']
                contributions[residue]['total_strength'] += contact_info.get('total_strength', 0.0)
                contributions[residue]['interaction_types'].update(contact_info['interaction_types'])

        # Normalize by number of poses
        num_poses = len(pose_analyses)
        for residue in contributions:
            contributions[residue]['frequency'] /= num_poses
            contributions[residue]['average_contacts'] /= num_poses
            contributions[residue]['interaction_types'] = list(contributions[residue]['interaction_types'])

        return dict(contributions)

    def _generate_interaction_fingerprint(self, pose_analyses: List[Dict]) -> Dict:
        """Generate interaction fingerprint for poses"""
        fingerprint = {
            'residue_fingerprint': defaultdict(set),
            'interaction_pattern': defaultdict(list),
            'consensus_interactions': {}
        }

        for pose in pose_analyses:
            for itype, interactions in pose['interactions'].items():
                for interaction in interactions:
                    residue = interaction.get('residue', 'unknown')
                    fingerprint['residue_fingerprint'][itype].add(residue)
                    fingerprint['interaction_pattern'][residue].append(itype)

        # Convert sets to lists for JSON serialization
        fingerprint['residue_fingerprint'] = {
            k: list(v) for k, v in fingerprint['residue_fingerprint'].items()
        }

        return fingerprint

    def _analyze_binding_site(self, pose_analyses: List[Dict], receptor_structure: Structure) -> Dict:
        """Analyze binding site properties"""
        binding_site = {
            'involved_residues': set(),
            'residue_types': defaultdict(int),
            'binding_site_volume': 0.0,
            'hydrophobicity': 0.0,
            'electrostatic_potential': 0.0
        }

        # Collect all involved residues
        for pose in pose_analyses:
            for residue in pose['residue_contacts']:
                binding_site['involved_residues'].add(residue)
                res_name = residue.split('_')[0]
                binding_site['residue_types'][res_name] += 1

        binding_site['involved_residues'] = list(binding_site['involved_residues'])

        return binding_site

    def _analyze_energy_correlations(self, result: DockingResult, analysis: Dict,
                                   experimental_data: Dict) -> Dict:
        """Analyze correlations between computed and experimental data"""
        correlations = {
            'binding_affinity_correlation': 0.0,
            'interaction_energy_correlation': 0.0,
            'rmse': 0.0,
            'r_squared': 0.0
        }

        # Extract computed energies
        computed_energies = [pose.energy for pose in result.poses]
        interaction_energies = [getattr(pose, 'interaction_energy', 0.0) for pose in result.poses]

        # Experimental values
        experimental_values = experimental_data.get('binding_affinities', [])

        if len(experimental_values) == len(computed_energies):
            # Calculate correlations
            correlations['binding_affinity_correlation'] = np.corrcoef(
                computed_energies, experimental_values
            )[0, 1]

            if any(ie != 0 for ie in interaction_energies):
                correlations['interaction_energy_correlation'] = np.corrcoef(
                    interaction_energies, experimental_values
                )[0, 1]

            # Calculate RMSE
            correlations['rmse'] = np.sqrt(np.mean((np.array(computed_energies) - np.array(experimental_values))**2))

            # Calculate R²
            ss_res = np.sum((np.array(experimental_values) - np.array(computed_energies))**2)
            ss_tot = np.sum((np.array(experimental_values) - np.mean(experimental_values))**2)
            correlations['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return correlations

    # Helper methods for calculating interaction properties
    def _get_ligand_hb_atoms(self, mol: Chem.Mol) -> Dict[int, str]:
        """Get hydrogen bond donors and acceptors in ligand"""
        hb_atoms = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            symbol = atom.GetSymbol()

            if symbol in ['N', 'O']:
                if atom.GetTotalNumHs() > 0:
                    hb_atoms[idx] = 'donor'
                else:
                    hb_atoms[idx] = 'acceptor'
            elif symbol in ['S', 'F']:
                hb_atoms[idx] = 'acceptor'

        return hb_atoms

    # Residue-specific hydrogen bonding sidechain atoms.
    # 'both' covers hydroxyls and imidazole nitrogens, which donate and accept
    # depending on rotamer and tautomer; at the resolution of a distance-based
    # analysis it is wrong to commit them to one role.
    SIDECHAIN_HB_ATOMS = {
        ('SER', 'OG'): 'both',
        ('THR', 'OG1'): 'both',
        ('TYR', 'OH'): 'both',
        ('CYS', 'SG'): 'both',
        ('ASN', 'ND2'): 'donor',
        ('ASN', 'OD1'): 'acceptor',
        ('GLN', 'NE2'): 'donor',
        ('GLN', 'OE1'): 'acceptor',
        ('HIS', 'ND1'): 'both',
        ('HIS', 'NE2'): 'both',
        ('LYS', 'NZ'): 'donor',
        ('ARG', 'NE'): 'donor',
        ('ARG', 'NH1'): 'donor',
        ('ARG', 'NH2'): 'donor',
        ('TRP', 'NE1'): 'donor',
        ('ASP', 'OD1'): 'acceptor',
        ('ASP', 'OD2'): 'acceptor',
        ('GLU', 'OE1'): 'acceptor',
        ('GLU', 'OE2'): 'acceptor',
    }

    # Backbone amide NH donates; backbone carbonyl O accepts.
    BACKBONE_HB_ATOMS = {'N': 'donor', 'O': 'acceptor', 'OXT': 'acceptor'}

    def _get_receptor_hb_atoms(self, receptor_atoms: List) -> Dict[int, Dict]:
        """
        Hydrogen bond donors and acceptors in the receptor.

        Includes the backbone. The previous version matched only a short list of
        sidechain atom names and ignored backbone N and O entirely, so it could
        not detect main-chain hydrogen bonds at all. That is not a minor gap:
        kinase hinge binding is predominantly backbone-mediated, so a correctly
        docked kinase inhibitor reported zero hydrogen bonds.
        """
        hb_atoms = {}

        for i, atom in enumerate(receptor_atoms):
            element = (atom.element or '').strip().upper()
            if element not in ('N', 'O', 'S'):
                continue

            residue_name = atom.get_parent().get_resname()
            atom_name = atom.get_name().strip()

            role = self.SIDECHAIN_HB_ATOMS.get((residue_name, atom_name))
            if role is None:
                role = self.BACKBONE_HB_ATOMS.get(atom_name)
            if role is None:
                continue

            hb_atoms[i] = {'type': role, 'residue': residue_name}

        return hb_atoms

    @staticmethod
    def _hb_pair_is_complementary(ligand_role: str, receptor_role: str) -> bool:
        """A donor must meet an acceptor; 'both' satisfies either side."""
        if ligand_role == 'both' or receptor_role == 'both':
            return True
        return ligand_role != receptor_role

    def _get_hydrophobic_ligand_atoms(self, mol: Chem.Mol) -> List[int]:
        """Get hydrophobic atoms in ligand"""
        hydrophobic = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C' and atom.GetTotalNumHs() > 0:
                hydrophobic.append(atom.GetIdx())
        return hydrophobic

    def _get_aromatic_rings(self, mol: Chem.Mol) -> List[List[int]]:
        """Get aromatic rings in molecule"""
        rings = []
        for ring in mol.GetRingInfo().AtomRings():
            if len(ring) == 6:  # Assume 6-membered rings are aromatic
                rings.append(list(ring))
        return rings

    def _get_atom_charge(self, atom) -> float:
        """Get partial charge for receptor atom"""
        residue_name = atom.get_parent().get_resname()
        atom_name = atom.get_name()

        if residue_name == 'ARG' and atom_name in ['NH1', 'NH2']:
            return 0.8
        elif residue_name == 'LYS' and atom_name == 'NZ':
            return 1.0
        elif residue_name == 'ASP' and atom_name in ['OD1', 'OD2']:
            return -0.8
        elif residue_name == 'GLU' and atom_name in ['OE1', 'OE2']:
            return -0.8

        return 0.0

    def _calculate_geometric_properties(self, coords: np.ndarray, mol: Chem.Mol) -> Dict:
        """Calculate geometric properties of ligand pose"""
        properties = {}

        # Center of mass
        properties['center_of_mass'] = np.mean(coords, axis=0).tolist()

        # Principal moments of inertia (simplified)
        centered_coords = coords - properties['center_of_mass']
        properties['span'] = np.linalg.norm(np.max(centered_coords, axis=0) - np.min(centered_coords, axis=0))

        # Surface area approximation
        properties['molecular_volume'] = len(coords) * 20.0  # Rough approximation

        return properties

    # Strength calculation methods
    def _calculate_hbond_strength(self, distance: float) -> float:
        """Calculate hydrogen bond strength based on distance"""
        optimal_distance = 2.8
        if distance <= optimal_distance:
            return 1.0
        else:
            return np.exp(-(distance - optimal_distance) / 0.5)

    def _calculate_hydrophobic_strength(self, distance: float) -> float:
        """Calculate hydrophobic interaction strength"""
        return max(0, 1.0 - distance / 4.0)

    def _calculate_electrostatic_strength(self, q1: float, q2: float, distance: float) -> float:
        """Calculate electrostatic interaction strength"""
        return abs(q1 * q2) / distance

    def _calculate_vdw_strength(self, distance: float) -> float:
        """Calculate van der Waals interaction strength"""
        return max(0, 1.0 - distance / 4.5)

    def _calculate_pi_stacking_strength(self, distance: float) -> float:
        """Calculate π-stacking interaction strength"""
        return max(0, 1.0 - distance / 5.0)

    def _calculate_pi_cation_strength(self, distance: float) -> float:
        """Calculate π-cation interaction strength"""
        return max(0, 1.0 - distance / 6.0)

    def _calculate_metal_coordination_strength(self, distance: float) -> float:
        """Calculate metal coordination strength"""
        return max(0, 1.0 - distance / 3.0)

    def _calculate_hbond_geometry(self, lig_coord: np.ndarray, rec_coord: np.ndarray,
                                 lig_type: str, rec_info: Dict) -> Dict:
        """Calculate hydrogen bond geometry"""
        # Simplified geometry calculation
        return {
            'distance': np.linalg.norm(lig_coord - rec_coord),
            'angle': 180.0  # Simplified - would need proper angle calculation
        }

    def generate_interaction_report(self, analysis: Dict, output_file: Optional[str] = None) -> str:
        """Generate comprehensive interaction analysis report"""

        report_lines = [
            "=== PandaDock Interaction Analysis Report ===\n",
            f"Total poses analyzed: {analysis['interaction_summary']['total_poses']}\n",
            "\n=== Interaction Summary ===",
        ]

        # Interaction type summary
        for itype in self.interaction_types:
            avg_count = analysis['interaction_summary']['average_interactions'].get(itype, 0)
            frequency = analysis['interaction_summary']['interaction_frequencies'].get(itype, 0)
            report_lines.append(
                f"{itype.replace('_', ' ').title()}: "
                f"avg={avg_count:.1f}, frequency={frequency:.1%}"
            )

        report_lines.extend([
            "\n=== Key Residue Contributions ===",
        ])

        # Top contributing residues
        residue_contribs = analysis.get('residue_contributions', {})
        sorted_residues = sorted(residue_contribs.items(),
                               key=lambda x: x[1]['frequency'], reverse=True)

        for residue, contrib in sorted_residues[:10]:
            report_lines.append(
                f"{residue}: frequency={contrib['frequency']:.1%}, "
                f"avg_contacts={contrib['average_contacts']:.1f}"
            )

        # Energy correlations if available
        if 'energy_correlations' in analysis:
            corr = analysis['energy_correlations']
            report_lines.extend([
                "\n=== Correlation with Experimental Data ===",
                f"Binding Affinity Correlation: {corr['binding_affinity_correlation']:.3f}",
                f"Interaction Energy Correlation: {corr['interaction_energy_correlation']:.3f}",
                f"RMSE: {corr['rmse']:.3f}",
                f"R²: {corr['r_squared']:.3f}"
            ])

        report = "\n".join(report_lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            self.logger.info(f"Interaction analysis report saved to {output_file}")

        return report

    def create_interaction_plots(self, analysis: Dict, output_dir: str = ".") -> List[str]:
        """Create visualization plots for interaction analysis"""
        plot_files = []

        try:
            # 1. Interaction type distribution
            fig, ax = plt.subplots(figsize=(10, 6))

            interaction_counts = {}
            for itype in self.interaction_types:
                counts = analysis['interaction_summary']['interaction_counts'][itype]
                interaction_counts[itype.replace('_', ' ').title()] = np.mean(counts)

            bars = ax.bar(interaction_counts.keys(), interaction_counts.values())
            ax.set_title('Average Interaction Counts per Pose')
            ax.set_ylabel('Average Count')
            ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plot_file = f"{output_dir}/interaction_distribution.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plot_files.append(plot_file)
            plt.close()

            # 2. Residue contribution heatmap
            if analysis.get('residue_contributions'):
                residue_data = analysis['residue_contributions']

                # Prepare data for heatmap
                residues = list(residue_data.keys())[:20]  # Top 20 residues
                frequencies = [residue_data[r]['frequency'] for r in residues]
                contacts = [residue_data[r]['average_contacts'] for r in residues]

                fig, ax = plt.subplots(figsize=(12, 8))

                # Create heatmap data
                heatmap_data = np.array([frequencies, contacts])

                sns.heatmap(heatmap_data,
                           xticklabels=residues,
                           yticklabels=['Frequency', 'Avg Contacts'],
                           annot=True, fmt='.2f', cmap='viridis')

                ax.set_title('Residue Interaction Contributions')
                plt.tight_layout()

                plot_file = f"{output_dir}/residue_contributions.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plot_files.append(plot_file)
                plt.close()

        except Exception as e:
            self.logger.warning(f"Failed to create interaction plots: {e}")

        return plot_files