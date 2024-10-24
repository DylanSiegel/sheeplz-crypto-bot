# File: training/metrics.py

import time
from typing import Dict, List, Optional
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

@dataclass
class BatchMetrics:
    """Metrics for a single training batch"""
    loss: float
    grad_norm: float
    forward_time: float
    backward_time: float
    total_time: float
    learning_rate: float

@dataclass
class EpochMetrics:
    """Aggregated metrics for a training epoch"""
    epoch: int
    avg_loss: float = 0.0
    avg_grad_norm: float = 0.0
    avg_forward_time: float = 0.0
    avg_backward_time: float = 0.0
    avg_total_time: float = 0.0
    val_loss: Optional[float] = None
    epoch_time: float = 0.0

class PerformanceMetrics:
    """Track and analyze model performance metrics"""
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {
            'loss': [],
            'grad_norm': [],
            'learning_rate': [],
            'forward_time': [],
            'backward_time': [],
            'total_time': [],
            'val_loss': []
        }
        self.epoch_metrics: List[EpochMetrics] = []
        self.current_batch_metrics: List[BatchMetrics] = []

        self.epoch_start_time: Optional[float] = None
        self.batch_start_time: Optional[float] = None

    def start_epoch(self):
        """Mark start of epoch"""
        self.epoch_start_time = time.time()
        self.current_batch_metrics = []

    def start_batch(self):
        """Mark start of batch"""
        self.batch_start_time = time.time()

    def record_batch(
        self,
        loss: float,
        grad_norm: float,
        learning_rate: float,
        forward_time: float,
        backward_time: float,
        total_time: float
    ):
        """Record metrics for a single batch"""
        metrics = BatchMetrics(
            loss=loss,
            grad_norm=grad_norm,
            learning_rate=learning_rate,
            forward_time=forward_time,
            backward_time=backward_time,
            total_time=total_time
        )

        self.current_batch_metrics.append(metrics)

        # Update running metrics
        self.metrics['loss'].append(loss)
        self.metrics['grad_norm'].append(grad_norm)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['forward_time'].append(forward_time)
        self.metrics['backward_time'].append(backward_time)
        self.metrics['total_time'].append(total_time)

    def get_epoch_stats(self) -> EpochMetrics:
        """Compute aggregated statistics for the epoch"""
        if not self.current_batch_metrics:
            return EpochMetrics(epoch=0)

        metrics = EpochMetrics(
            epoch=len(self.epoch_metrics) + 1,
            avg_loss=np.mean([m.loss for m in self.current_batch_metrics]),
            avg_grad_norm=np.mean([m.grad_norm for m in self.current_batch_metrics]),
            avg_forward_time=np.mean([m.forward_time for m in self.current_batch_metrics]),
            avg_backward_time=np.mean([m.backward_time for m in self.current_batch_metrics]),
            avg_total_time=np.mean([m.total_time for m in self.current_batch_metrics]),
            epoch_time=time.time() - self.epoch_start_time if self.epoch_start_time else 0.0
        )

        if any(m.val_loss for m in self.current_batch_metrics):
            metrics.val_loss = np.mean([m.val_loss for m in self.current_batch_metrics if m.val_loss is not None])

        self.epoch_metrics.append(metrics)
        return metrics

    def reset_epoch_stats(self):
        """Reset batch metrics for new epoch"""
        self.current_batch_metrics = []
        self.epoch_start_time = None

    def save_metrics(self, path: Path):
        """Save metrics to JSON file"""
        metrics_dict = {
            'batch_metrics': {
                key: self.metrics[key] for key in self.metrics
            },
            'epoch_metrics': [vars(m) for m in self.epoch_metrics]
        }

        with open(path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

    def plot_metrics(self, path: Path):
        """Create visualization of training metrics"""
        plt.figure(figsize=(15, 10))

        # Loss plot
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['loss'], label='Training Loss')
        if self.metrics['val_loss']:
            plt.plot(self.metrics['val_loss'], label='Validation Loss')
        plt.title('Loss History')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()

        # Gradient norm plot
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics['grad_norm'], label='Gradient Norm')
        plt.title('Gradient Norms')
        plt.xlabel('Batch')
        plt.ylabel('Norm')
        plt.legend()

        # Learning rate plot
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['learning_rate'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Batch')
        plt.ylabel('Learning Rate')
        plt.legend()

        # Timing plot
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics['forward_time'], label='Forward Time')
        plt.plot(self.metrics['backward_time'], label='Backward Time')
        plt.plot(self.metrics['total_time'], label='Total Time')
        plt.title('Computation Times')
        plt.xlabel('Batch')
        plt.ylabel('Seconds')
        plt.legend()

        plt.tight_layout()
        plt.savefig(path)
        plt.close()
