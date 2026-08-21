"""
Loss functions for PandaDock-GNN training.

Implements multi-task loss combining:
- Binding affinity regression (MSE)
- Activity classification (BCE)
- Optional: Confidence/pose quality prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class AffinityLoss(nn.Module):
    """
    Loss for binding affinity prediction (pEC50).

    Uses MSE loss with optional Huber/smooth L1 for robustness.
    """

    def __init__(self, loss_type: str = 'mse', delta: float = 1.0):
        """
        Args:
            loss_type: 'mse', 'huber', or 'smooth_l1'
            delta: Delta parameter for Huber loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.delta = delta

        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'huber':
            self.loss_fn = nn.HuberLoss(delta=delta)
        elif loss_type == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute affinity loss.

        Args:
            pred: Predicted pEC50 (B,)
            target: True pEC50 (B,)

        Returns:
            Scalar loss
        """
        return self.loss_fn(pred, target)


class ActivityLoss(nn.Module):
    """
    Loss for activity classification (active/inactive).

    Uses BCE with logits, with optional class weighting.
    """

    def __init__(self, pos_weight: Optional[float] = None):
        """
        Args:
            pos_weight: Weight for positive class (to handle imbalance)
        """
        super().__init__()

        if pos_weight is not None:
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute activity loss.

        Args:
            pred: Predicted logits (B,)
            target: True labels (B,), 0 or 1

        Returns:
            Scalar loss
        """
        return self.loss_fn(pred, target)


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for PandaDock-GNN.

    L = w_affinity * L_affinity + w_activity * L_activity + w_confidence * L_confidence

    Supports uncertainty weighting (Kendall et al. 2018) for automatic weight learning.
    """

    def __init__(
        self,
        w_affinity: float = 1.0,
        w_activity: float = 0.5,
        w_confidence: float = 0.0,
        affinity_loss_type: str = 'mse',
        pos_weight: Optional[float] = None,
        learnable_weights: bool = False
    ):
        """
        Args:
            w_affinity: Weight for affinity loss
            w_activity: Weight for activity loss
            w_confidence: Weight for confidence loss
            affinity_loss_type: Type of affinity loss
            pos_weight: Positive class weight for activity
            learnable_weights: Learn task weights via uncertainty
        """
        super().__init__()

        self.w_affinity = w_affinity
        self.w_activity = w_activity
        self.w_confidence = w_confidence
        self.learnable_weights = learnable_weights

        # Individual losses
        self.affinity_loss = AffinityLoss(affinity_loss_type)
        self.activity_loss = ActivityLoss(pos_weight)
        self.confidence_loss = AffinityLoss('mse')

        # Learnable log-variances for uncertainty weighting
        if learnable_weights:
            self.log_var_affinity = nn.Parameter(torch.zeros(1))
            self.log_var_activity = nn.Parameter(torch.zeros(1))
            self.log_var_confidence = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.

        Args:
            predictions: Dict with 'affinity', 'activity', 'confidence' predictions
            targets: Dict with 'affinity', 'activity' targets

        Returns:
            Dict with 'total', 'affinity', 'activity' losses
        """
        losses = {}

        # Affinity loss
        if 'affinity' in predictions and 'affinity' in targets:
            losses['affinity'] = self.affinity_loss(
                predictions['affinity'],
                targets['affinity']
            )

        # Activity loss
        if 'activity' in predictions and 'activity' in targets:
            losses['activity'] = self.activity_loss(
                predictions['activity'],
                targets['activity']
            )

        # Confidence loss
        if 'confidence' in predictions and 'confidence' in targets:
            losses['confidence'] = self.confidence_loss(
                predictions['confidence'],
                targets['confidence']
            )

        # Combine losses
        if self.learnable_weights:
            # Uncertainty weighting: L = sum_i (L_i / (2 * sigma_i^2) + log(sigma_i))
            total = 0.0

            if 'affinity' in losses:
                precision = torch.exp(-self.log_var_affinity)
                total = total + precision * losses['affinity'] + self.log_var_affinity

            if 'activity' in losses:
                precision = torch.exp(-self.log_var_activity)
                total = total + precision * losses['activity'] + self.log_var_activity

            if 'confidence' in losses:
                precision = torch.exp(-self.log_var_confidence)
                total = total + precision * losses['confidence'] + self.log_var_confidence

            losses['total'] = total.squeeze()
        else:
            # Fixed weights
            total = 0.0

            if 'affinity' in losses:
                total = total + self.w_affinity * losses['affinity']

            if 'activity' in losses:
                total = total + self.w_activity * losses['activity']

            if 'confidence' in losses:
                total = total + self.w_confidence * losses['confidence']

            losses['total'] = total

        return losses

    def get_weights(self) -> Dict[str, float]:
        """Get current task weights."""
        if self.learnable_weights:
            return {
                'affinity': torch.exp(-self.log_var_affinity).item(),
                'activity': torch.exp(-self.log_var_activity).item(),
                'confidence': torch.exp(-self.log_var_confidence).item()
            }
        else:
            return {
                'affinity': self.w_affinity,
                'activity': self.w_activity,
                'confidence': self.w_confidence
            }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in activity prediction.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Reduces loss for well-classified examples, focusing on hard examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            pred: Predicted logits (B,)
            target: True labels (B,), 0 or 1

        Returns:
            Scalar loss
        """
        # Convert logits to probabilities
        prob = torch.sigmoid(pred)

        # Compute focal weights
        p_t = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Weighted loss
        loss = focal_weight * bce

        return loss.mean()


if __name__ == "__main__":
    # Test losses
    print("Testing loss functions...")

    batch_size = 16

    # Random predictions and targets
    pred_affinity = torch.randn(batch_size)
    target_affinity = torch.randn(batch_size) + 6  # pEC50 ~6

    pred_activity = torch.randn(batch_size)
    target_activity = (torch.rand(batch_size) > 0.5).float()

    # Test individual losses
    aff_loss = AffinityLoss()
    act_loss = ActivityLoss()

    print(f"Affinity loss: {aff_loss(pred_affinity, target_affinity):.4f}")
    print(f"Activity loss: {act_loss(pred_activity, target_activity):.4f}")

    # Test multi-task loss
    mt_loss = MultiTaskLoss(w_affinity=1.0, w_activity=0.5)

    predictions = {'affinity': pred_affinity, 'activity': pred_activity}
    targets = {'affinity': target_affinity, 'activity': target_activity}

    losses = mt_loss(predictions, targets)
    print(f"Multi-task losses: {losses}")

    # Test with learnable weights
    mt_loss_learnable = MultiTaskLoss(learnable_weights=True)
    losses = mt_loss_learnable(predictions, targets)
    print(f"Learnable weights: {mt_loss_learnable.get_weights()}")


def within_target_loss(predictions, targets, groups):
    """
    Squared error after removing each target's mean, within a batch.

    Plain MSE on absolute affinity is dominated by between-target variance --
    29% of the total on SAIR, and by far the easier part to fit. A model can
    drive that loss down by learning which protein binds tightly in general
    while barely discriminating between ligands against it, which is what
    PandaDock-GNN did: pooled r 0.43 against a median within-target r of 0.21.

    This term removes the per-target mean from both sides, so the only thing it
    can reward is ordering ligands against a fixed protein. Added alongside the
    absolute loss rather than replacing it, because a model trained on residuals
    alone cannot emit an absolute pIC50 for an unseen target: the target's mean
    is precisely what is unknown at inference.

    Means are taken over the batch, not the dataset, so no label outside the
    current batch is involved and there is nothing to leak. Targets represented
    by a single complex in a batch carry no ordering information and are
    dropped.

    Returns None when no target has two or more complexes in the batch, which
    the caller must treat as "no contribution" rather than as zero loss.
    """
    # Forced to float32: under autocast the predictions arrive as half while the
    # labels stay float, and the accumulations below require a single dtype.
    # Reductions in half would also lose precision -- these are sums over a
    # batch, and computing losses in float32 under AMP is the normal practice.
    # The cast is differentiable, so gradients still reach half-precision
    # parameters.
    predictions = predictions.view(-1).float()
    targets = targets.view(-1).float()
    groups = groups.view(-1)

    unique, inverse = torch.unique(groups, return_inverse=True)
    ones = torch.ones_like(predictions)

    counts = torch.zeros(unique.numel(), device=predictions.device, dtype=predictions.dtype)
    counts = counts.index_add(0, inverse, ones)

    pred_sum = torch.zeros_like(counts).index_add(0, inverse, predictions)
    true_sum = torch.zeros_like(counts).index_add(0, inverse, targets)

    usable = counts >= 2
    if not bool(usable.any()):
        return None

    keep = usable[inverse]
    centred_pred = predictions - (pred_sum / counts)[inverse]
    centred_true = targets - (true_sum / counts)[inverse]

    return ((centred_pred - centred_true)[keep] ** 2).mean()
