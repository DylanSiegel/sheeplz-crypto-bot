from dataclasses import dataclass, asdict
from typing import Dict, Optional
from pathlib import Path
import json

@dataclass
class NLNNConfig:
    """Configuration for Normalized Liquid Neural Network"""
    # Architecture
    hidden_size: int = 256
    input_size: int = 128
    output_size: int = 64
    num_layers: int = 2
    
    # Training
    batch_size: int = 512
    sequence_length: int = 128
    learning_rate: float = 1e-3
    grad_clip: float = 1.0
    dropout: float = 0.1
    
    # Optimization
    epsilon: float = 1e-8
    use_mixed_precision: bool = True
    num_threads: int = 24
    
    # Normalization
    normalization_strategy: str = 'full'  # 'full', 'group', or 'selective'
    use_slerp: bool = True  # If False, uses NLERP
    antipodal_strategy: str = 'random'  # 'random' or 'nlerp'
    activation_fn: str = 'tanh'  # 'tanh', 'modified_relu'
    
    # Paths
    checkpoint_dir: Optional[Path] = None
    log_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Set up directories after initialization"""
        if self.checkpoint_dir is None:
            self.checkpoint_dir = Path('checkpoints')
        if self.log_dir is None:
            self.log_dir = Path('logs')
            
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            key: str(value) if isinstance(value, Path) else value 
            for key, value in asdict(self).items()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'NLNNConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def save(self, filepath: str):
        """Save config to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'NLNNConfig':
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)