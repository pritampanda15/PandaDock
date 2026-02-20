"""
Evaluation metrics for PandaDock-GNN.

Metrics for binding affinity prediction:
- Pearson R (linear correlation)
- Spearman rho (rank correlation)
- RMSE, MAE

Metrics for activity classification:
- AUC-ROC, AUC-PR
- Accuracy, Precision, Recall, F1
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from sklearn import metrics as sklearn_metrics


def pearson_r(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute Pearson correlation coefficient.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Pearson R (-1 to 1)
    """
    if len(y_true) < 2:
        return 0.0

    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r) if not np.isnan(r) else 0.0


def spearman_rho(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute Spearman rank correlation coefficient.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Spearman rho (-1 to 1)
    """
    if len(y_true) < 2:
        return 0.0

    rho, _ = stats.spearmanr(y_true, y_pred)
    return float(rho) if not np.isnan(rho) else 0.0


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def auc_roc(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> float:
    """
    Compute Area Under ROC Curve.

    Args:
        y_true: True binary labels
        y_score: Predicted scores (probabilities or logits)

    Returns:
        AUC-ROC (0 to 1)
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    try:
        return float(sklearn_metrics.roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.5


def auc_pr(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> float:
    """
    Compute Area Under Precision-Recall Curve.

    Args:
        y_true: True binary labels
        y_score: Predicted scores

    Returns:
        AUC-PR (0 to 1)
    """
    if len(np.unique(y_true)) < 2:
        return 0.5

    try:
        precision, recall, _ = sklearn_metrics.precision_recall_curve(y_true, y_score)
        return float(sklearn_metrics.auc(recall, precision))
    except ValueError:
        return 0.5


def accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute classification accuracy.

    Args:
        y_true: True binary labels
        y_pred: Predicted scores
        threshold: Classification threshold

    Returns:
        Accuracy (0 to 1)
    """
    y_pred_binary = (y_pred > threshold).astype(int)
    return float(sklearn_metrics.accuracy_score(y_true, y_pred_binary))


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score.

    Args:
        y_true: True binary labels
        y_pred: Predicted scores
        threshold: Classification threshold

    Returns:
        Tuple of (precision, recall, f1)
    """
    y_pred_binary = (y_pred > threshold).astype(int)

    precision = sklearn_metrics.precision_score(y_true, y_pred_binary, zero_division=0)
    recall = sklearn_metrics.recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = sklearn_metrics.f1_score(y_true, y_pred_binary, zero_division=0)

    return float(precision), float(recall), float(f1)


def compute_affinity_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute all affinity prediction metrics.

    Args:
        y_true: True pEC50 values
        y_pred: Predicted pEC50 values

    Returns:
        Dict with pearson_r, spearman_rho, rmse, mae
    """
    return {
        'pearson_r': pearson_r(y_true, y_pred),
        'spearman_rho': spearman_rho(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred)
    }


def compute_activity_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute all activity classification metrics.

    Args:
        y_true: True binary labels (0/1)
        y_score: Predicted scores (probabilities)
        threshold: Classification threshold

    Returns:
        Dict with auc_roc, auc_pr, accuracy, precision, recall, f1
    """
    precision, recall, f1 = precision_recall_f1(y_true, y_score, threshold)

    return {
        'auc_roc': auc_roc(y_true, y_score),
        'auc_pr': auc_pr(y_true, y_score),
        'accuracy': accuracy(y_true, y_score, threshold),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_metrics(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """
    Compute all metrics for predictions.

    Args:
        predictions: Dict with 'affinity' and/or 'activity' predictions
        targets: Dict with corresponding targets

    Returns:
        Dict with all metrics
    """
    metrics = {}

    # Affinity metrics
    if 'affinity' in predictions and 'affinity' in targets:
        affinity_metrics = compute_affinity_metrics(
            targets['affinity'],
            predictions['affinity']
        )
        for key, value in affinity_metrics.items():
            metrics[f'affinity_{key}'] = value

    # Activity metrics
    if 'activity' in predictions and 'activity' in targets:
        activity_metrics = compute_activity_metrics(
            targets['activity'],
            predictions['activity']
        )
        for key, value in activity_metrics.items():
            metrics[f'activity_{key}'] = value

    return metrics


class MetricTracker:
    """
    Track metrics over training epochs.
    """

    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        self.best: Dict[str, float] = {}
        self.best_epoch: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], epoch: int) -> None:
        """Add metrics for an epoch."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
                self.best[key] = float('-inf') if self._higher_is_better(key) else float('inf')
                self.best_epoch[key] = 0

            self.history[key].append(value)

            # Update best
            if self._higher_is_better(key):
                if value > self.best[key]:
                    self.best[key] = value
                    self.best_epoch[key] = epoch
            else:
                if value < self.best[key]:
                    self.best[key] = value
                    self.best_epoch[key] = epoch

    def _higher_is_better(self, metric_name: str) -> bool:
        """Determine if higher values are better for a metric."""
        higher_better = ['pearson_r', 'spearman_rho', 'auc_roc', 'auc_pr',
                        'accuracy', 'precision', 'recall', 'f1']
        for name in higher_better:
            if name in metric_name:
                return True
        return False

    def get_best(self, metric: str) -> Tuple[float, int]:
        """Get best value and epoch for a metric."""
        return self.best.get(metric, 0.0), self.best_epoch.get(metric, 0)

    def summary(self) -> str:
        """Get summary string of best metrics."""
        lines = ["Best metrics:"]
        for key in sorted(self.best.keys()):
            lines.append(f"  {key}: {self.best[key]:.4f} (epoch {self.best_epoch[key]})")
        return "\n".join(lines)


class PearsonR:
    """Callable Pearson R metric for PyTorch."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return pearson_r(y_true, y_pred)


class SpearmanRho:
    """Callable Spearman rho metric for PyTorch."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return spearman_rho(y_true, y_pred)


if __name__ == "__main__":
    # Test metrics
    print("Testing metrics...")

    np.random.seed(42)

    # Affinity data
    y_true_aff = np.random.randn(100) + 6
    y_pred_aff = y_true_aff + np.random.randn(100) * 0.5

    print("Affinity metrics:")
    aff_metrics = compute_affinity_metrics(y_true_aff, y_pred_aff)
    for key, value in aff_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Activity data
    y_true_act = (np.random.rand(100) > 0.5).astype(int)
    y_pred_act = np.random.rand(100)

    print("\nActivity metrics:")
    act_metrics = compute_activity_metrics(y_true_act, y_pred_act)
    for key, value in act_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Test metric tracker
    print("\nTesting MetricTracker...")
    tracker = MetricTracker()

    for epoch in range(5):
        metrics = {
            'pearson_r': 0.5 + epoch * 0.05,
            'rmse': 1.0 - epoch * 0.1
        }
        tracker.update(metrics, epoch)

    print(tracker.summary())
