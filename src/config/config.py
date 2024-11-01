# config.py

from pydantic import BaseModel, Field
from typing import List
from .hardware import HardwareConfig
from .model import ModelConfig
from .risk import RiskConfig
from .features import FeatureConfig
from .training import TrainingConfig

class TradingConfig(BaseModel):
    """Master configuration class.

    This class aggregates all configuration settings for the trading agent and its environment.

    Attributes:
        hardware (HardwareConfig): Configuration for hardware resources.
        model (ModelConfig): Configuration for the neural network model.
        risk (RiskConfig): Configuration for risk management parameters.
        feature (FeatureConfig): Configuration for feature extraction.
        training (TrainingConfig): Configuration for the training process.

    Methods:
        optimize_for_hardware(): Optimizes the configuration based on the detected hardware capabilities.

    """
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    def optimize_for_hardware(self):
        """Optimizes configuration for current hardware.

        This method adjusts certain parameters (primarily batch size) based on available GPU memory.
        It also sets the number of threads used by PyTorch to optimize performance.
        """
        if self.hardware.cuda_available:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            self.model.batch_size = min(self.model.batch_size, int(gpu_mem * 0.7 / (self.model.feature_dim * 4)))
        torch.set_num_threads(self.hardware.torch_num_threads)
        if self.hardware.cuda_available:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        return self

def create_default_config() -> TradingConfig:
    """Creates and returns a default configuration, optimized for the current hardware."""
    config = TradingConfig()
    return config.optimize_for_hardware()

import torch

# Example of how to save the default config to a file (config.pt)
config = create_default_config()
torch.save(config.model_dump(), "config.pt")