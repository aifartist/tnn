"""
Metrics tracking for classification experiments
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

class ClassificationMetrics:
    """Track and compute classification metrics"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.losses = []
        
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, loss: float):
        """
        Update metrics with batch results
        
        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth labels
            loss: Batch loss value
        """
        # Convert to numpy and store
        if predictions.dim() > 1:
            pred_classes = torch.argmax(predictions, dim=1)
        else:
            pred_classes = predictions
            
        self.predictions.extend(pred_classes.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.losses.append(loss)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        if not self.predictions:
            return {}
            
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # Accuracy
        accuracy = np.mean(predictions == targets)
        
        # Per-class accuracy
        per_class_acc = {}
        for class_id in range(self.num_classes):
            mask = targets == class_id
            if mask.sum() > 0:
                class_acc = np.mean(predictions[mask] == targets[mask])
                per_class_acc[f'class_{class_id}_acc'] = class_acc
        
        # Average loss
        avg_loss = np.mean(self.losses) if self.losses else 0.0
        
        # Confusion matrix statistics
        confusion_stats = self._compute_confusion_stats(predictions, targets)
        
        metrics = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'num_samples': len(predictions),
            **per_class_acc,
            **confusion_stats
        }
        
        return metrics
    
    def _compute_confusion_stats(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute precision, recall, F1 per class"""
        stats = {}
        
        for class_id in range(self.num_classes):
            # True positives, false positives, false negatives
            tp = np.sum((predictions == class_id) & (targets == class_id))
            fp = np.sum((predictions == class_id) & (targets != class_id))
            fn = np.sum((predictions != class_id) & (targets == class_id))
            
            # Precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            stats[f'class_{class_id}_precision'] = precision
            stats[f'class_{class_id}_recall'] = recall
            stats[f'class_{class_id}_f1'] = f1
        
        # Macro averages
        precisions = [stats[f'class_{i}_precision'] for i in range(self.num_classes)]
        recalls = [stats[f'class_{i}_recall'] for i in range(self.num_classes)]
        f1s = [stats[f'class_{i}_f1'] for i in range(self.num_classes)]
        
        stats['macro_precision'] = np.mean(precisions)
        stats['macro_recall'] = np.mean(recalls)
        stats['macro_f1'] = np.mean(f1s)
        
        return stats
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix"""
        if not self.predictions:
            return np.zeros((self.num_classes, self.num_classes))
            
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(self.targets, self.predictions, labels=range(self.num_classes))


class MetricsTracker:
    """Track metrics across multiple epochs"""
    
    def __init__(self):
        self.epoch_metrics = defaultdict(list)
        
    def add_epoch(self, epoch: int, train_metrics: Dict[str, float], 
                  val_metrics: Optional[Dict[str, float]] = None):
        """Add metrics for an epoch"""
        self.epoch_metrics['epoch'].append(epoch)
        
        # Add train metrics with prefix
        for key, value in train_metrics.items():
            self.epoch_metrics[f'train_{key}'].append(value)
            
        # Add validation metrics with prefix
        if val_metrics:
            for key, value in val_metrics.items():
                self.epoch_metrics[f'val_{key}'].append(value)
    
    def get_best_epoch(self, metric: str = 'val_accuracy', mode: str = 'max') -> Tuple[int, float]:
        """Get epoch with best performance"""
        if metric not in self.epoch_metrics:
            raise ValueError(f"Metric {metric} not found")
            
        values = self.epoch_metrics[metric]
        epochs = self.epoch_metrics['epoch']
        
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
            
        return epochs[best_idx], values[best_idx]
    
    def get_final_metrics(self) -> Dict[str, float]:
        """Get metrics from the final epoch"""
        if not self.epoch_metrics['epoch']:
            return {}
            
        final_metrics = {}
        for key, values in self.epoch_metrics.items():
            if key != 'epoch':
                final_metrics[key] = values[-1]
                
        return final_metrics
    
    def to_dict(self) -> Dict[str, List]:
        """Convert to dictionary for logging/saving"""
        return dict(self.epoch_metrics)
