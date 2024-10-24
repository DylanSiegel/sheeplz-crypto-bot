import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import time
from pathlib import Path

from ..models.nlnn import NLNN
from ..config.config import NLNNConfig
from ..optimizers.modified_adam import ModifiedAdam
from .metrics import PerformanceMetrics
from ..utils.monitoring import ResourceMonitor
from ..utils.logging import get_logger

logger = get_logger(__name__)

class NLNNTrainer:
    """Trainer class for NLNN model"""
    def __init__(
        self,
        model: NLNN,
        config: NLNNConfig,
        optimizer: Optional[ModifiedAdam] = None
    ):
        self.model = model
        self.config = config
        self.optimizer = optimizer or ModifiedAdam(
            model.parameters(),
            lr=config.learning_rate,
            eps=config.epsilon
        )
        
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metrics = PerformanceMetrics()
        self.monitor = ResourceMonitor()
        
        self.device = model.device
        self.best_loss = float('inf')
        
        logger.info(f"Initialized trainer with device: {self.device}")
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Record timing
        start_time = time.time()
        forward_start = time.time()
        
        # Forward pass
        x, y = [b.to(self.device) for b in batch]
        with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
            outputs, _ = self.model(x)
            outputs = outputs.view(-1, outputs.size(-1))
            y = y.view(-1)
            loss = self.criterion(outputs, y)
        
        forward_time = time.time() - forward_start
        
        # Backward pass
        backward_start = time.time()
        self.model.scaler.scale(loss).backward()
        
        # Compute gradient norm
        self.model.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip
        )
        
        # Optimizer step
        self.model.scaler.step(self.optimizer)
        self.model.scaler.update()
        
        backward_time = time.time() - backward_start
        total_time = time.time() - start_time
        
        metrics = {
            'loss': loss.item(),
            'grad_norm': grad_norm.item(),
            'forward_time': forward_time,
            'backward_time': backward_time,
            'total_time': total_time
        }
        
        return metrics
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x, y = [b.to(self.device) for b in batch]
                with torch.cuda.amp.autocast(enabled=self.config.use_mixed_precision):
                    outputs, _ = self.model(x)
                    outputs = outputs.view(-1, outputs.size(-1))
                    y = y.view(-1)
                    loss = self.criterion(outputs, y)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return {'val_loss': avg_loss}
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        val_dataloader: Optional[DataLoader] = None,
        save_dir: Optional[Path] = None
    ):
        """Full training loop"""
        save_dir = save_dir or self.config.checkpoint_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.metrics.start_epoch()
            epoch_metrics = {'epoch': epoch}
            
            # Training loop
            for batch_idx, batch in enumerate(train_dataloader):
                self.metrics.start_batch()
                step_metrics = self.train_step(batch)
                
                # Record metrics
                self.metrics.record_batch(**step_metrics)
                epoch_metrics.update(step_metrics)
                
                # Log progress
                if batch_idx % 100 == 0:
                    self.monitor.log_stats(epoch_metrics)
            
            # Validation
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader)
                epoch_metrics.update(val_metrics)
                
                # Save best model
                if val_metrics['val_loss'] < self.best_loss:
                    self.best_loss = val_metrics['val_loss']
                    self.save_checkpoint(save_dir / 'best_model.pt', epoch_metrics)
            
            # End of epoch
            epoch_stats = self.metrics.get_epoch_stats()
            epoch_metrics.update(epoch_stats)
            
            # Save checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(
                    save_dir / f'checkpoint_epoch_{epoch}.pt',
                    epoch_metrics
                )
            
            # Log epoch metrics
            logger.info(f"Epoch {epoch} completed:")
            for key, value in epoch_metrics.items():
                logger.info(f"  {key}: {value:.4f}")
            
            self.metrics.reset_epoch_stats()
        
        # Save final model and metrics
        self.save_checkpoint(save_dir / 'final_model.pt', epoch_metrics)
        self.metrics.save_metrics(save_dir / 'training_metrics.json')
        self.metrics.plot_metrics(save_dir / 'training_plots.png')
        self.monitor.save_stats(save_dir / 'resource_stats.json')
        self.monitor.plot_stats(save_dir / 'resource_plots.png')
        
        logger.info("Training completed")
    
    def save_checkpoint(self, path: Path, metrics: Dict[str, Any]):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': metrics.get('epoch', 0),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.to_dict()
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = NLNNConfig.from_dict(checkpoint['config'])
        metrics = checkpoint['metrics']
        logger.info(f"Loaded checkpoint from epoch {metrics['epoch']}")
        return metrics