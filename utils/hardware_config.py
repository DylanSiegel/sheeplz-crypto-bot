import os
import torch
import torch.backends.cudnn as cudnn
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from pathlib import Path

@dataclass
class HardwareConfig:
    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    num_workers: int = 12
    pin_memory: bool = True
    cuda_non_blocking: bool = True
    torch_threads: int = 24
    cudnn_benchmark: bool = True
    amp_dtype: torch.dtype = torch.float16
    memory_format: torch.memory_format = torch.channels_last
    vram_allocation: float = 0.85
    batch_power: int = 9
    gradient_accumulation_steps: int = 2
    prefetch_factor: int = 4
    persistent_workers: bool = True
    
    def __post_init__(self):
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
            torch.cuda.empty_cache()
            torch.cuda.memory.set_per_process_memory_fraction(self.vram_allocation)
            
        torch.set_float32_matmul_precision('high')
        torch.set_num_threads(self.torch_threads)
        cudnn.benchmark = self.cudnn_benchmark
        cudnn.deterministic = False
        cudnn.enabled = True
        
        os.environ['OMP_NUM_THREADS'] = str(self.torch_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.torch_threads)
        
        self.batch_size = 2 ** self.batch_power
        
    def optimize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(
            device=self.device,
            dtype=self.amp_dtype if tensor.is_floating_point() else tensor.dtype,
            memory_format=self.memory_format,
            non_blocking=self.cuda_non_blocking
        )
    
    def get_dataloader_args(self) -> Dict:
        return {
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'prefetch_factor': self.prefetch_factor,
            'persistent_workers': self.persistent_workers
        }
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
    
    def memory_stats(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {}
            
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,
            'reserved': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
        }

class OptimizedDataPipeline:
    def __init__(self, config: HardwareConfig):
        self.config = config
        
    def create_dataloaders(
        self,
        dataset_train,
        dataset_val,
        collate_fn=None
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        dataloader_args = self.config.get_dataloader_args()
        
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            **dataloader_args
        )
        
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            collate_fn=collate_fn,
            **dataloader_args
        )
        
        return train_loader, val_loader

def create_optimized_config() -> HardwareConfig:
    return HardwareConfig(
        num_workers=12,
        torch_threads=24,
        batch_power=9,
        vram_allocation=0.85
    )