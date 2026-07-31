"""
Training pipeline for PandaDock-GNN.

Provides complete training loop with:
- Mixed precision training
- Learning rate scheduling
- Early stopping
- Checkpointing
- Logging and visualization
"""

import os
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    OneCycleLR
)

from .losses import MultiTaskLoss
from .metrics import compute_metrics, MetricTracker


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = 'adamw'  # 'adamw', 'adam', 'sgd'
    scheduler: str = 'cosine'  # 'cosine', 'plateau', 'onecycle', 'none'

    # Training
    epochs: int = 100
    batch_size: int = 32
    gradient_clip: float = 1.0

    # Loss weights
    w_affinity: float = 1.0
    w_activity: float = 0.5
    learnable_weights: bool = False

    # Early stopping
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    monitor: str = 'val_affinity_pearson_r'
    mode: str = 'max'  # 'max' or 'min'

    # Checkpointing
    save_best: bool = True
    save_last: bool = True
    checkpoint_dir: str = 'checkpoints'

    # Logging
    log_interval: int = 10
    eval_interval: int = 1

    # Mixed precision
    use_amp: bool = True

    # Device
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'


class EarlyStopping:
    """Early stopping handler."""

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class GNNTrainer:
    """
    Trainer for PandaDock-GNN model.

    Handles complete training loop with validation, checkpointing,
    and metric tracking.

    Example:
        model = PandaDockGNN()
        trainer = GNNTrainer(model, config)
        trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize trainer.

        Args:
            model: PandaDockGNN model
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.model = model

        # Setup device
        if self.config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config.device)

        self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup loss
        self.loss_fn = MultiTaskLoss(
            w_affinity=self.config.w_affinity,
            w_activity=self.config.w_activity,
            learnable_weights=self.config.learnable_weights
        )

        # Setup scheduler (will be initialized in train())
        self.scheduler = None

        # Early stopping
        if self.config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                mode=self.config.mode
            )
        else:
            self.early_stopping = None

        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if self.config.use_amp else None

        # Metric tracking
        self.metric_tracker = MetricTracker()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None

    def _create_optimizer(self):
        """Create optimizer based on config."""
        if self.config.optimizer == 'adamw':
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'adam':
            return Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == 'sgd':
            return SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""
        if self.config.scheduler == 'cosine':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2
            )
        elif self.config.scheduler == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.mode,
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        elif self.config.scheduler == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate * 10,
                total_steps=num_training_steps
            )
        else:
            return None

    def train(
        self,
        train_loader,
        val_loader,
        test_loader=None
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader

        Returns:
            Dict with training history and best metrics
        """
        # Setup checkpoint directory
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup scheduler
        num_training_steps = len(train_loader) * self.config.epochs
        self.scheduler = self._create_scheduler(num_training_steps)

        # Training loop
        history = defaultdict(list)
        start_time = time.time()

        print(f"Training on {self.device}", flush=True)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}", flush=True)

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_metrics = self._train_epoch(train_loader)
            for key, value in train_metrics.items():
                history[f'train_{key}'].append(value)

            # Validate
            if epoch % self.config.eval_interval == 0:
                val_metrics = self._validate(val_loader)
                for key, value in val_metrics.items():
                    history[f'val_{key}'].append(value)

                # Update metric tracker
                all_metrics = {**{f'train_{k}': v for k, v in train_metrics.items()},
                              **{f'val_{k}': v for k, v in val_metrics.items()}}
                self.metric_tracker.update(all_metrics, epoch)

                # Logging
                self._log_epoch(epoch, train_metrics, val_metrics)

                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        monitor_value = val_metrics.get(
                            self.config.monitor.replace('val_', ''),
                            val_metrics.get('affinity_pearson_r', 0)
                        )
                        self.scheduler.step(monitor_value)
                    elif not isinstance(self.scheduler, OneCycleLR):
                        self.scheduler.step()

                # Save best model
                monitor_value = val_metrics.get(
                    self.config.monitor.replace('val_', ''),
                    val_metrics.get('affinity_pearson_r', 0)
                )

                if self.config.save_best:
                    if self.best_metric is None:
                        self.best_metric = monitor_value
                        self._save_checkpoint(checkpoint_dir / 'best_model.pt')
                    elif (self.config.mode == 'max' and monitor_value > self.best_metric) or \
                         (self.config.mode == 'min' and monitor_value < self.best_metric):
                        self.best_metric = monitor_value
                        self._save_checkpoint(checkpoint_dir / 'best_model.pt')

                # Early stopping
                if self.early_stopping is not None:
                    if self.early_stopping(monitor_value):
                        print(f"Early stopping at epoch {epoch}")
                        break

        # Save final model
        if self.config.save_last:
            self._save_checkpoint(checkpoint_dir / 'last_model.pt')

        # Final evaluation on test set
        test_metrics = None
        if test_loader is not None:
            # Load best model
            best_path = checkpoint_dir / 'best_model.pt'
            if best_path.exists():
                self._load_checkpoint(best_path)

            test_metrics = self._validate(test_loader)
            print("\nTest metrics:")
            for key, value in test_metrics.items():
                print(f"  {key}: {value:.4f}")

        # Training summary
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time / 60:.1f} minutes")
        print(self.metric_tracker.summary())

        return {
            'history': dict(history),
            'best_metrics': self.metric_tracker.best,
            'test_metrics': test_metrics,
            'elapsed_time': elapsed_time
        }

    def _train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = defaultdict(list)
        all_predictions = defaultdict(list)
        all_targets = defaultdict(list)

        num_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{num_batches}", flush=True)
            batch = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.config.use_amp and self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    predictions = self.model(batch)
                    targets = self._get_targets(batch)
                    losses = self.loss_fn(predictions, targets)
                    loss = losses['total']

                # Backward pass with scaling
                self.scaler.scale(loss).backward()

                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(batch)
                targets = self._get_targets(batch)
                losses = self.loss_fn(predictions, targets)
                loss = losses['total']

                # Backward pass
                loss.backward()

                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )

                self.optimizer.step()

            # OneCycle scheduler step per batch
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            # Record losses
            for key, value in losses.items():
                epoch_losses[key].append(value.item())

            # Record predictions
            with torch.no_grad():
                for key in predictions:
                    all_predictions[key].extend(predictions[key].cpu().numpy())
                for key in targets:
                    all_targets[key].extend(targets[key].cpu().numpy())

            self.global_step += 1

        # Compute metrics
        metrics = {'loss': np.mean(epoch_losses['total'])}

        pred_np = {k: np.array(v) for k, v in all_predictions.items()}
        target_np = {k: np.array(v) for k, v in all_targets.items()}

        computed_metrics = compute_metrics(pred_np, target_np)
        metrics.update(computed_metrics)

        return metrics

    def _validate(self, val_loader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()

        all_losses = []
        all_predictions = defaultdict(list)
        all_targets = defaultdict(list)

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                predictions = self.model(batch)
                targets = self._get_targets(batch)
                losses = self.loss_fn(predictions, targets)

                all_losses.append(losses['total'].item())

                for key in predictions:
                    all_predictions[key].extend(predictions[key].cpu().numpy())
                for key in targets:
                    all_targets[key].extend(targets[key].cpu().numpy())

        # Compute metrics
        metrics = {'loss': np.mean(all_losses)}

        pred_np = {k: np.array(v) for k, v in all_predictions.items()}
        target_np = {k: np.array(v) for k, v in all_targets.items()}

        computed_metrics = compute_metrics(pred_np, target_np)
        metrics.update(computed_metrics)

        return metrics

    def _get_targets(self, batch) -> Dict[str, torch.Tensor]:
        """Extract targets from batch."""
        targets = {}

        if hasattr(batch, 'y_affinity'):
            # Flatten to 1D but keep at least 1 dimension
            targets['affinity'] = batch.y_affinity.view(-1)
        if hasattr(batch, 'y_active'):
            targets['activity'] = batch.y_active.view(-1)

        # A batch with no recognised target produces no loss terms, so the total
        # stays the float 0.0 it was initialised to. That surfaces much later as
        # "'float' object has no attribute 'backward'", or -- with AMP off and a
        # tensor total -- as an epoch of training on nothing. Name the cause here.
        if not targets:
            raise ValueError(
                "Batch carries no training target: expected y_affinity and/or "
                "y_active, found "
                f"{sorted(k for k in batch.keys() if k.startswith('y'))!r}. "
                "Datasets must set y_affinity; setting only `y` is not enough."
            )

        return targets

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """Log epoch results."""
        lr = self.optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:3d} | "
              f"LR: {lr:.2e} | "
              f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val R: {val_metrics.get('affinity_pearson_r', 0):.4f} | "
              f"Val AUC: {val_metrics.get('activity_auc_roc', 0):.4f}",
              flush=True)

    def _save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config,  # Save ModelConfig, not TrainingConfig
            'training_config': self.config,  # Also save training config for reference
            'best_metric': self.best_metric
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def _load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        # weights_only=False because the checkpoint embeds the ModelConfig
        # dataclass, which torch 2.6 refuses to unpickle under the new default.
        # This file was written by _save_checkpoint moments earlier in this same
        # run, so there is no untrusted input involved.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', None)

    def predict(self, data_loader) -> Dict[str, np.ndarray]:
        """
        Generate predictions for a dataset.

        Args:
            data_loader: Data loader

        Returns:
            Dict with predictions
        """
        self.model.eval()

        all_predictions = defaultdict(list)
        compound_ids = []

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                predictions = self.model(batch)

                for key in predictions:
                    all_predictions[key].extend(predictions[key].cpu().numpy())

                if hasattr(batch, 'compound_id'):
                    compound_ids.extend(batch.compound_id)

        result = {k: np.array(v) for k, v in all_predictions.items()}
        if compound_ids:
            result['compound_id'] = compound_ids

        return result


if __name__ == "__main__":
    print("GNNTrainer module loaded successfully")
    print("Use with: trainer = GNNTrainer(model, config)")
