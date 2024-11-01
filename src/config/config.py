from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel, Field
import torch
import psutil

class HardwareConfig(BaseModel):
    """Hardware-specific configuration"""
    num_cpu_threads: int = Field(
        default_factory=lambda: psutil.cpu_count(logical=True),
        description="Number of CPU threads to use"
    )
    cuda_available: bool = Field(
        default_factory=lambda: torch.cuda.is_available(),
        description="Whether CUDA is available"
    )
    gpu_memory_limit: int = Field(
        default=7168,  # 7GB for NVIDIA 3070, leaving 1GB for system
        description="GPU memory limit in MB"
    )
    torch_num_threads: int = Field(
        default=12,  # Optimized for Ryzen 9 7900X
        description="Number of threads for PyTorch"
    )
    pin_memory: bool = Field(
        default=True,
        description="Use pinned memory for faster GPU transfer"
    )

class ModelConfig(BaseModel):
    """Neural network configuration"""
    feature_dim: int = Field(64, description="Input feature dimension")
    hidden_size: int = Field(256, description="Hidden layer size")
    num_layers: int = Field(4, description="Number of n-LNN layers")
    dropout: float = Field(0.1, description="Dropout rate")
    learning_rate: float = Field(3e-4, description="Learning rate")
    batch_size: int = Field(512, description="Training batch size")
    sequence_length: int = Field(100, description="Sequence length for training")
    gradient_clip: float = Field(1.0, description="Gradient clipping value")

class RiskConfig(BaseModel):
    """Risk management configuration"""
    max_position_size: float = Field(0.1, description="Maximum position size as fraction of capital")
    stop_loss_pct: float = Field(0.02, description="Stop loss percentage")
    take_profit_pct: float = Field(0.04, description="Take profit percentage")
    max_leverage: float = Field(5.0, description="Maximum allowed leverage")
    min_trade_interval: int = Field(5, description="Minimum intervals between trades")
    max_drawdown: float = Field(0.2, description="Maximum allowed drawdown")
    position_sizing_method: str = Field("kelly", description="Position sizing method")

class FeatureConfig(BaseModel):
    """Feature extraction configuration"""
    rolling_window_size: int = Field(100, description="Size of rolling window for features")
    technical_indicators: List[str] = Field(
        default=["rsi", "macd", "bbands", "volatility"],
        description="List of technical indicators to use"
    )
    market_features: List[str] = Field(
        default=[
            "price",
            "volume",
            "spread",
            "depth",
            "funding_rate"
        ],
        description="List of market features to use"
    )
    normalization_method: str = Field(
        "standard",
        description="Feature normalization method"
    )

class TrainingConfig(BaseModel):
    """Training configuration"""
    num_episodes: int = Field(10000, description="Number of training episodes")
    gamma: float = Field(0.99, description="Discount factor")
    gae_lambda: float = Field(0.95, description="GAE lambda parameter")
    entropy_coef: float = Field(0.01, description="Entropy coefficient")
    value_loss_coef: float = Field(0.5, description="Value loss coefficient")
    max_grad_norm: float = Field(0.5, description="Maximum gradient norm")
    update_interval: int = Field(2048, description="Steps between policy updates")
    num_minibatches: int = Field(4, description="Number of minibatches per update")
    warmup_steps: int = Field(1000, description="Number of warmup steps")

class TradingConfig(BaseModel):
    """Master configuration class"""
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    
    def optimize_for_hardware(self):
        """Optimize configuration for current hardware"""
        # Optimize batch size based on GPU memory
        if self.hardware.cuda_available:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            self.model.batch_size = min(
                self.model.batch_size,
                int(gpu_mem * 0.7 / (self.model.feature_dim * 4))  # 4 bytes per float
            )
        
        # Optimize number of threads
        torch.set_num_threads(self.hardware.torch_num_threads)
        
        # Enable TF32 for faster training on Ampere GPUs
        if self.hardware.cuda_available:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        return self

def create_default_config() -> TradingConfig:
    """Create and return a default configuration"""
    config = TradingConfig()
    return config.optimize_for_hardware()