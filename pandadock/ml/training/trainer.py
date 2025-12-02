"""
ML model trainer for PandaDock

Handles training of diffusion, energy prediction, and pose ranking models.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import h5py

# Import ML models
from ..models.diffusion import DiffusionDockingModel
from ..models.energy_prediction import EnergyPredictionModel
from ..models.pose_ranking import PoseRankingModel


class MLDockingTrainer:
    """Trainer for ML docking models"""

    def __init__(self, model_type: str, gpu: bool = False):
        """
        Initialize trainer

        Args:
            model_type: Type of model to train ('diffusion', 'energy_prediction', 'pose_ranking')
            gpu: Use GPU for training
        """
        self.logger = logging.getLogger(__name__)
        self.model_type = model_type
        self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

        # Initialize model
        self.model = self._create_model()

        # Training components
        self.optimizer = None
        self.criterion = None
        self.scheduler = None

        # Data
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        self.logger.info(f"Initialized {model_type} trainer on {self.device}")

    def _create_model(self):
        """Create model based on type"""
        if self.model_type == 'diffusion':
            return DiffusionDockingModel(device=self.device)
        elif self.model_type == 'energy_prediction':
            return EnergyPredictionModel(device=self.device)
        elif self.model_type == 'pose_ranking':
            return PoseRankingModel(device=self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def load_dataset(self, dataset_path: str, validation_split: float = 0.2):
        """
        Load training dataset

        Args:
            dataset_path: Path to dataset (JSON or HDF5)
            validation_split: Fraction of data for validation
        """
        self.logger.info(f"Loading dataset from {dataset_path}")

        try:
            if dataset_path.endswith('.h5') or dataset_path.endswith('.hdf5'):
                dataset = HDF5DockingDataset(dataset_path, self.model_type)
            elif dataset_path.endswith('.json'):
                dataset = JSONDockingDataset(dataset_path, self.model_type)
            else:
                raise ValueError("Dataset must be HDF5 (.h5/.hdf5) or JSON (.json)")

            # Split dataset
            total_size = len(dataset)
            val_size = int(total_size * validation_split)
            train_size = total_size - val_size

            self.train_dataset, self.val_dataset = random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            self.logger.info(f"Loaded dataset: {train_size} training, {val_size} validation samples")

        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

    def configure_training(self,
                          epochs: int = 100,
                          batch_size: int = 32,
                          learning_rate: float = 1e-4,
                          output_dir: Optional[Path] = None):
        """
        Configure training parameters

        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            output_dir: Directory to save models and logs
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_dir = output_dir or Path('ml_training_output')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4 if self.device.type == 'cuda' else 2
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4 if self.device.type == 'cuda' else 2
        )

        # Configure optimizer and loss
        self._configure_optimizer_and_loss()

        # Configure scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        self.logger.info(f"Configured training: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")

    def _configure_optimizer_and_loss(self):
        """Configure optimizer and loss function based on model type"""
        # Get model parameters
        if self.model_type == 'diffusion':
            model_params = list(self.model.position_predictor.parameters()) + \
                          list(self.model.confidence_predictor.parameters())
            self.criterion = nn.MSELoss()  # For coordinate prediction
        elif self.model_type == 'energy_prediction':
            model_params = self.model.energy_predictor.parameters()
            self.criterion = nn.MSELoss()  # For energy prediction
        elif self.model_type == 'pose_ranking':
            model_params = self.model.pose_ranker.parameters()
            self.criterion = nn.BCELoss()  # For ranking scores
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Adam optimizer with weight decay
        self.optimizer = optim.Adam(
            model_params,
            lr=self.learning_rate,
            weight_decay=1e-5
        )

    def train(self) -> Dict[str, List[float]]:
        """
        Train the model

        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting training for {self.epochs} epochs")
        start_time = time.time()

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Training phase
            train_loss = self._train_epoch()

            # Validation phase
            val_loss = self._validate_epoch()

            # Update scheduler
            self.scheduler.step(val_loss)

            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint('best_model.pth')

            # Logging
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs}: "
                f"Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')

        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")

        return self.training_history

    def _train_epoch(self) -> float:
        """Train for one epoch"""
        if self.model_type == 'diffusion':
            return self._train_diffusion_epoch()
        elif self.model_type == 'energy_prediction':
            return self._train_energy_epoch()
        elif self.model_type == 'pose_ranking':
            return self._train_ranking_epoch()

    def _train_diffusion_epoch(self) -> float:
        """Train diffusion model for one epoch"""
        self.model.position_predictor.train()
        self.model.confidence_predictor.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            # Unpack batch
            ligand_features = batch['ligand_features'].to(self.device)
            receptor_features = batch['receptor_features'].to(self.device)
            target_coords = batch['target_coordinates'].to(self.device)
            target_confidence = batch['target_confidence'].to(self.device)

            # Add noise to coordinates (diffusion training)
            noise = torch.randn_like(target_coords) * 0.5
            noisy_coords = target_coords + noise

            # Predict denoised coordinates
            pred_coords = self.model.position_predictor(
                noisy_coords, ligand_features, receptor_features
            )

            # Predict confidence
            pred_confidence = self.model.confidence_predictor(
                target_coords, ligand_features, receptor_features
            )

            # Calculate losses
            coord_loss = self.criterion(pred_coords, target_coords)
            confidence_loss = self.criterion(pred_confidence, target_confidence)
            total_batch_loss = coord_loss + 0.1 * confidence_loss

            total_batch_loss.backward()
            self.optimizer.step()

            total_loss += total_batch_loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _train_energy_epoch(self) -> float:
        """Train energy prediction model for one epoch"""
        self.model.energy_predictor.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            # Unpack batch
            ligand_features = batch['ligand_features'].to(self.device)
            receptor_features = batch['receptor_features'].to(self.device)
            target_energy = batch['target_energy'].to(self.device)

            # Predict energy
            pred_energy = self.model.energy_predictor(ligand_features, receptor_features)

            # Calculate loss
            loss = self.criterion(pred_energy, target_energy)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _train_ranking_epoch(self) -> float:
        """Train pose ranking model for one epoch"""
        self.model.pose_ranker.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            # Unpack batch
            ligand_features = batch['ligand_features'].to(self.device)
            receptor_features = batch['receptor_features'].to(self.device)
            target_score = batch['target_ranking_score'].to(self.device)

            # Predict ranking score
            pred_score = self.model.pose_ranker(ligand_features, receptor_features)

            # Calculate loss
            loss = self.criterion(pred_score, target_score)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate_epoch(self) -> float:
        """Validate for one epoch"""
        if self.model_type == 'diffusion':
            return self._validate_diffusion_epoch()
        elif self.model_type == 'energy_prediction':
            return self._validate_energy_epoch()
        elif self.model_type == 'pose_ranking':
            return self._validate_ranking_epoch()

    def _validate_diffusion_epoch(self) -> float:
        """Validate diffusion model"""
        self.model.position_predictor.eval()
        self.model.confidence_predictor.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                ligand_features = batch['ligand_features'].to(self.device)
                receptor_features = batch['receptor_features'].to(self.device)
                target_coords = batch['target_coordinates'].to(self.device)
                target_confidence = batch['target_confidence'].to(self.device)

                # Predict coordinates and confidence
                pred_coords = self.model.position_predictor(
                    target_coords, ligand_features, receptor_features
                )
                pred_confidence = self.model.confidence_predictor(
                    target_coords, ligand_features, receptor_features
                )

                # Calculate losses
                coord_loss = self.criterion(pred_coords, target_coords)
                confidence_loss = self.criterion(pred_confidence, target_confidence)
                total_batch_loss = coord_loss + 0.1 * confidence_loss

                total_loss += total_batch_loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _validate_energy_epoch(self) -> float:
        """Validate energy prediction model"""
        self.model.energy_predictor.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                ligand_features = batch['ligand_features'].to(self.device)
                receptor_features = batch['receptor_features'].to(self.device)
                target_energy = batch['target_energy'].to(self.device)

                pred_energy = self.model.energy_predictor(ligand_features, receptor_features)
                loss = self.criterion(pred_energy, target_energy)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def _validate_ranking_epoch(self) -> float:
        """Validate pose ranking model"""
        self.model.pose_ranker.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                ligand_features = batch['ligand_features'].to(self.device)
                receptor_features = batch['receptor_features'].to(self.device)
                target_score = batch['target_ranking_score'].to(self.device)

                pred_score = self.model.pose_ranker(ligand_features, receptor_features)
                loss = self.criterion(pred_score, target_score)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def load_pretrained_weights(self, weights_path: str):
        """Load pretrained weights for transfer learning"""
        try:
            self.model.load_weights(weights_path)
            self.logger.info(f"Loaded pretrained weights from {weights_path}")
        except Exception as e:
            self.logger.error(f"Failed to load pretrained weights: {e}")
            raise

    def save_model(self, model_path: Path):
        """Save trained model"""
        try:
            self.model.save_weights(str(model_path))
            self.logger.info(f"Saved model to {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise

    def save_training_history(self, history_path: Path):
        """Save training history"""
        try:
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            self.logger.info(f"Saved training history to {history_path}")
        except Exception as e:
            self.logger.error(f"Failed to save training history: {e}")

    def _save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint_path = self.output_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'model_type': self.model_type,
            'model_state_dict': self._get_model_state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }

        torch.save(checkpoint, checkpoint_path)

    def _get_model_state_dict(self) -> Dict:
        """Get model state dict based on model type"""
        if self.model_type == 'diffusion':
            return {
                'position_predictor': self.model.position_predictor.state_dict(),
                'confidence_predictor': self.model.confidence_predictor.state_dict()
            }
        elif self.model_type == 'energy_prediction':
            return {'energy_predictor': self.model.energy_predictor.state_dict()}
        elif self.model_type == 'pose_ranking':
            return {'pose_ranker': self.model.pose_ranker.state_dict()}


class HDF5DockingDataset(Dataset):
    """Dataset for loading docking data from HDF5 files"""

    def __init__(self, hdf5_path: str, model_type: str):
        self.hdf5_path = hdf5_path
        self.model_type = model_type

        # Open HDF5 file and get dataset info
        with h5py.File(hdf5_path, 'r') as f:
            self.complex_names = list(f.keys())

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded HDF5 dataset with {len(self.complex_names)} complexes")

    def __len__(self):
        return len(self.complex_names)

    def __getitem__(self, idx):
        complex_name = self.complex_names[idx]

        with h5py.File(self.hdf5_path, 'r') as f:
            complex_group = f[complex_name]

            # Load features based on model type
            sample = {
                'ligand_features': torch.tensor(complex_group['ligand_features'][...], dtype=torch.float32),
                'receptor_features': torch.tensor(complex_group['receptor_features'][...], dtype=torch.float32),
            }

            # Add model-specific targets
            if self.model_type == 'diffusion':
                sample['target_coordinates'] = torch.tensor(complex_group['coordinates'][...], dtype=torch.float32)
                sample['target_confidence'] = torch.tensor(complex_group['confidence'][...], dtype=torch.float32)
            elif self.model_type == 'energy_prediction':
                sample['target_energy'] = torch.tensor(complex_group['energy'][...], dtype=torch.float32)
            elif self.model_type == 'pose_ranking':
                sample['target_ranking_score'] = torch.tensor(complex_group['ranking_score'][...], dtype=torch.float32)

        return sample


class JSONDockingDataset(Dataset):
    """Dataset for loading docking data from JSON files"""

    def __init__(self, json_path: str, model_type: str):
        self.model_type = model_type

        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loaded JSON dataset with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        sample = {
            'ligand_features': torch.tensor(item['ligand_features'], dtype=torch.float32),
            'receptor_features': torch.tensor(item['receptor_features'], dtype=torch.float32),
        }

        # Add model-specific targets
        if self.model_type == 'diffusion':
            sample['target_coordinates'] = torch.tensor(item['coordinates'], dtype=torch.float32)
            sample['target_confidence'] = torch.tensor(item['confidence'], dtype=torch.float32)
        elif self.model_type == 'energy_prediction':
            sample['target_energy'] = torch.tensor(item['energy'], dtype=torch.float32)
        elif self.model_type == 'pose_ranking':
            sample['target_ranking_score'] = torch.tensor(item['ranking_score'], dtype=torch.float32)

        return sample