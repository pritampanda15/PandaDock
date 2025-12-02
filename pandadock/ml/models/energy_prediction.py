"""
Energy prediction model for ML-enhanced docking

Deep learning model for predicting binding energies and affinities.
"""

import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import logging
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Import PandaDock core components
from ...docking.core import Pose


class EnergyPredictionModel:
    """Deep learning model for binding energy prediction"""

    def __init__(self, model_size: str = 'medium', device: str = 'auto'):
        """
        Initialize energy prediction model

        Args:
            model_size: Size of the model ('small', 'medium', 'large')
            device: Device to run model on ('cpu', 'cuda', 'auto')
        """
        self.logger = logging.getLogger(__name__)
        self.model_size = model_size

        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Model parameters
        self.model_params = self._get_model_params(model_size)

        # Initialize model
        self._build_model()

        # Training state
        self.is_trained = False

        self.logger.info(f"Initialized EnergyPredictionModel ({model_size}) on {self.device}")

    def _get_model_params(self, size: str) -> Dict[str, Any]:
        """Get model parameters based on size"""
        params = {
            'small': {
                'hidden_dim': 128,
                'num_layers': 3,
                'dropout': 0.1,
                'max_atoms': 50
            },
            'medium': {
                'hidden_dim': 256,
                'num_layers': 4,
                'dropout': 0.2,
                'max_atoms': 100
            },
            'large': {
                'hidden_dim': 512,
                'num_layers': 6,
                'dropout': 0.3,
                'max_atoms': 150
            }
        }
        return params.get(size, params['medium'])

    def _build_model(self):
        """Build the energy prediction model"""
        self.energy_predictor = EnergyPredictorNetwork(
            hidden_dim=self.model_params['hidden_dim'],
            num_layers=self.model_params['num_layers'],
            dropout=self.model_params['dropout'],
            max_atoms=self.model_params['max_atoms']
        ).to(self.device)

    def generate_poses(self,
                      receptor_features: Dict[str, np.ndarray],
                      ligand_mol: Chem.Mol,
                      grid_center: np.ndarray,
                      grid_dimensions: np.ndarray,
                      num_poses: int = 20,
                      temperature: float = 1.0) -> List[Pose]:
        """
        Generate poses using energy-guided sampling

        Args:
            receptor_features: Extracted protein features
            ligand_mol: RDKit molecule object
            grid_center: Center of docking grid
            grid_dimensions: Dimensions of docking grid
            num_poses: Number of poses to generate
            temperature: Sampling temperature

        Returns:
            List of generated poses with predicted energies
        """
        self.logger.debug(f"Generating {num_poses} poses with energy prediction model")

        try:
            # Generate initial poses using simple sampling
            poses = self._generate_initial_poses(ligand_mol, grid_center, grid_dimensions, num_poses * 3)

            # Predict energies for all poses
            scored_poses = self._predict_pose_energies(poses, receptor_features, ligand_mol)

            # Select best poses based on predicted energies
            scored_poses.sort(key=lambda x: x.energy)
            selected_poses = scored_poses[:num_poses]

            # Set confidence based on energy ranking
            for i, pose in enumerate(selected_poses):
                pose.confidence = 1.0 - (i / len(selected_poses))
                pose.ml_confidence = pose.confidence
                pose.generation_method = 'energy_guided'

            self.logger.debug(f"Generated {len(selected_poses)} energy-guided poses")
            return selected_poses

        except Exception as e:
            self.logger.error(f"Energy-guided pose generation failed: {e}")
            return self._generate_fallback_poses(ligand_mol, grid_center, num_poses)

    def predict_binding_energy(self,
                             pose: Pose,
                             receptor_features: Dict[str, np.ndarray],
                             ligand_mol: Chem.Mol) -> float:
        """
        Predict binding energy for a single pose

        Args:
            pose: Ligand pose
            receptor_features: Protein features
            ligand_mol: RDKit molecule

        Returns:
            Predicted binding energy (kcal/mol)
        """
        try:
            with torch.no_grad():
                # Prepare features
                ligand_features = self._prepare_ligand_features(ligand_mol, pose.coordinates)
                receptor_tensor = self._prepare_receptor_features(receptor_features)

                # Predict energy
                energy = self.energy_predictor(
                    ligand_features.unsqueeze(0),
                    receptor_tensor.unsqueeze(0)
                ).item()

                return energy

        except Exception as e:
            self.logger.error(f"Energy prediction failed: {e}")
            return 0.0

    def _generate_initial_poses(self,
                               mol: Chem.Mol,
                               center: np.ndarray,
                               dimensions: np.ndarray,
                               num_poses: int) -> List[Pose]:
        """Generate initial poses for energy evaluation"""
        poses = []

        try:
            from rdkit.Chem import AllChem

            # Create molecule copy
            mol_copy = Chem.Mol(mol)
            mol_copy = Chem.AddHs(mol_copy)

            # Generate conformers
            cids = AllChem.EmbedMultipleConfs(
                mol_copy,
                numConfs=num_poses,
                randomSeed=42,
                useExpTorsionAnglePrefs=True
            )

            for i, cid in enumerate(cids):
                conf = mol_copy.GetConformer(cid)
                coords = []

                for atom_idx in range(mol_copy.GetNumAtoms()):
                    pos = conf.GetAtomPosition(atom_idx)
                    coords.append([pos.x, pos.y, pos.z])

                coords = np.array(coords)

                # Random translation within grid
                random_offset = np.random.uniform(-dimensions/4, dimensions/4, 3)
                coords = coords - np.mean(coords, axis=0) + center + random_offset

                # Random rotation
                rotation_matrix = self._random_rotation_matrix()
                coords_centered = coords - center
                coords = np.dot(coords_centered, rotation_matrix.T) + center

                pose = Pose(
                    coordinates=coords,
                    center=np.mean(coords, axis=0),
                    rotation=np.array([1.0, 0.0, 0.0, 0.0]),
                    conformer_id=i,
                    energy=0.0,
                    confidence=0.5
                )

                poses.append(pose)

        except Exception as e:
            self.logger.error(f"Initial pose generation failed: {e}")

        return poses

    def _random_rotation_matrix(self) -> np.ndarray:
        """Generate random rotation matrix"""
        # Generate random rotation using Rodrigues' formula
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.uniform(0, 2 * np.pi)

        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        cross_matrix = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        rotation_matrix = (cos_angle * np.eye(3) +
                          sin_angle * cross_matrix +
                          (1 - cos_angle) * np.outer(axis, axis))

        return rotation_matrix

    def _predict_pose_energies(self,
                              poses: List[Pose],
                              receptor_features: Dict[str, np.ndarray],
                              ligand_mol: Chem.Mol) -> List[Pose]:
        """Predict energies for multiple poses"""
        scored_poses = []

        # Prepare receptor features once
        receptor_tensor = self._prepare_receptor_features(receptor_features)

        with torch.no_grad():
            for pose in poses:
                try:
                    # Prepare ligand features for this pose
                    ligand_features = self._prepare_ligand_features(ligand_mol, pose.coordinates)

                    # Predict energy
                    energy = self.energy_predictor(
                        ligand_features.unsqueeze(0),
                        receptor_tensor.unsqueeze(0)
                    ).item()

                    pose.energy = energy
                    pose.ml_energy = energy
                    scored_poses.append(pose)

                except Exception as e:
                    self.logger.warning(f"Energy prediction failed for pose: {e}")
                    pose.energy = 999.0  # High energy for failed poses
                    scored_poses.append(pose)

        return scored_poses

    def _prepare_ligand_features(self, mol: Chem.Mol, coordinates: np.ndarray) -> torch.Tensor:
        """Prepare ligand features including 3D coordinates"""
        atom_features = []
        max_atoms = self.model_params['max_atoms']

        for i, atom in enumerate(mol.GetAtoms()):
            if i >= max_atoms:
                break

            # Atom type features
            atomic_num = atom.GetAtomicNum()
            atom_type = [0] * 10
            if atomic_num < 10:
                atom_type[atomic_num] = 1

            # Chemical features
            features = atom_type + [
                atom.GetDegree(),
                atom.GetTotalNumHs(),
                int(atom.GetIsAromatic()),
                atom.GetFormalCharge()
            ]

            # Add 3D coordinates
            if i < len(coordinates):
                features.extend(coordinates[i].tolist())
            else:
                features.extend([0.0, 0.0, 0.0])

            atom_features.append(features)

        # Pad to max_atoms
        while len(atom_features) < max_atoms:
            atom_features.append([0] * 17)  # 14 chemical + 3 coordinate features

        return torch.tensor(atom_features, dtype=torch.float32, device=self.device)

    def _prepare_receptor_features(self, features: Dict[str, np.ndarray]) -> torch.Tensor:
        """Prepare receptor features for model input"""
        # Combine different feature types
        feature_list = []

        # Pocket features
        pocket_features = features.get('pocket_features', np.zeros(100))
        if len(pocket_features) > 200:
            pocket_features = pocket_features[:200]
        else:
            padding = np.zeros(200 - len(pocket_features))
            pocket_features = np.concatenate([pocket_features, padding])
        feature_list.append(pocket_features)

        # Sequence features
        sequence_features = features.get('sequence_features', np.zeros(200))
        if len(sequence_features) > 200:
            sequence_features = sequence_features[:200]
        else:
            padding = np.zeros(200 - len(sequence_features))
            sequence_features = np.concatenate([sequence_features, padding])
        feature_list.append(sequence_features)

        # Combine all features
        combined_features = np.concatenate(feature_list)

        return torch.tensor(combined_features, dtype=torch.float32, device=self.device)

    def _generate_fallback_poses(self, mol: Chem.Mol, center: np.ndarray, num_poses: int) -> List[Pose]:
        """Generate fallback poses when model fails"""
        return self._generate_initial_poses(mol, center, np.array([20.0, 20.0, 20.0]), num_poses)

    def load_weights(self, model_path: str):
        """Load pre-trained model weights"""
        try:
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                self.energy_predictor.load_state_dict(checkpoint['energy_predictor'])
                self.is_trained = True
                self.logger.info(f"Loaded energy model weights from {model_path}")
            else:
                self.logger.warning(f"Energy model weights not found at {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load energy model weights: {e}")

    def save_weights(self, model_path: str):
        """Save model weights"""
        try:
            checkpoint = {
                'energy_predictor': self.energy_predictor.state_dict(),
                'model_params': self.model_params
            }
            torch.save(checkpoint, model_path)
            self.logger.info(f"Saved energy model weights to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save energy model weights: {e}")


class EnergyPredictorNetwork(nn.Module):
    """Neural network for predicting binding energies"""

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float, max_atoms: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_atoms = max_atoms

        # Ligand encoder (processes atoms with coordinates)
        self.ligand_encoder = nn.Sequential(
            nn.Linear(17, hidden_dim),  # 14 chemical + 3 coordinate features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Receptor encoder
        self.receptor_encoder = nn.Sequential(
            nn.Linear(400, hidden_dim),  # Combined receptor features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Interaction layers
        self.interaction_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.interaction_layers.append(
                InteractionLayer(hidden_dim, dropout)
            )

        # Energy prediction head
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, ligand_features, receptor_features):
        """
        Forward pass

        Args:
            ligand_features: (batch_size, max_atoms, 17)
            receptor_features: (batch_size, 400)

        Returns:
            energy: (batch_size, 1)
        """
        batch_size = ligand_features.shape[0]

        # Encode ligand
        ligand_emb = self.ligand_encoder(ligand_features)  # (batch_size, max_atoms, hidden_dim)

        # Encode receptor
        receptor_emb = self.receptor_encoder(receptor_features)  # (batch_size, hidden_dim)

        # Apply interaction layers
        for layer in self.interaction_layers:
            ligand_emb = layer(ligand_emb, receptor_emb)

        # Global pooling of ligand representation
        ligand_global = torch.mean(ligand_emb, dim=1)  # (batch_size, hidden_dim)

        # Combine ligand and receptor representations
        combined = torch.cat([ligand_global, receptor_emb], dim=1)  # (batch_size, hidden_dim * 2)

        # Predict energy
        energy = self.energy_head(combined)

        return energy


class InteractionLayer(nn.Module):
    """Layer for modeling protein-ligand interactions"""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Self-attention for ligand
        self.ligand_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )

        # Cross-attention between ligand and receptor
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, ligand_emb, receptor_emb):
        """
        Args:
            ligand_emb: (batch_size, max_atoms, hidden_dim)
            receptor_emb: (batch_size, hidden_dim)

        Returns:
            updated_ligand_emb: (batch_size, max_atoms, hidden_dim)
        """
        # Self-attention for ligand
        attn_out, _ = self.ligand_attention(ligand_emb, ligand_emb, ligand_emb)
        ligand_emb = self.norm1(ligand_emb + attn_out)

        # Cross-attention with receptor
        receptor_emb_expanded = receptor_emb.unsqueeze(1).expand(-1, ligand_emb.shape[1], -1)
        cross_attn_out, _ = self.cross_attention(ligand_emb, receptor_emb_expanded, receptor_emb_expanded)
        ligand_emb = self.norm2(ligand_emb + cross_attn_out)

        # Feed-forward
        ffn_out = self.ffn(ligand_emb)
        ligand_emb = ligand_emb + ffn_out

        return ligand_emb