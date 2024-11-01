from pydantic import BaseModel, Field
import psutil
import torch

class HardwareConfig(BaseModel):
    """Hardware-specific configuration.

    This class encapsulates settings related to the hardware resources used by the trading agent.
    It automatically detects certain hardware features and provides options for optimization.

    Attributes:
        num_cpu_threads (int): The number of CPU threads to utilize for parallel processing. 
                               Defaults to the number of logical CPU cores detected by `psutil`.
        cuda_available (bool): A boolean indicating whether CUDA (GPU acceleration) is available. 
                               Automatically detected by PyTorch.
        gpu_memory_limit (int): The maximum amount of GPU memory (in MB) to allocate for training. 
                               Defaults to 7168 MB (7GB), leaving 1GB for system use on an NVIDIA 3070.  Adjust this based on your GPU's VRAM.
        torch_num_threads (int): The number of threads PyTorch should use. Defaults to 12 (optimized for Ryzen 9 7900X). Adjust for your CPU.
        pin_memory (bool): A boolean flag indicating whether to use pinned memory for faster data transfer between CPU and GPU. Defaults to True.  Generally recommended for GPU usage.

    """
    num_cpu_threads: int = Field(default_factory=lambda: psutil.cpu_count(logical=True), description="Number of CPU threads to use")
    cuda_available: bool = Field(default_factory=lambda: torch.cuda.is_available(), description="Whether CUDA is available")
    gpu_memory_limit: int = Field(default=7168, description="GPU memory limit in MB")
    torch_num_threads: int = Field(default=12, description="Number of threads for PyTorch")
    pin_memory: bool = Field(default=True, description="Use pinned memory for faster GPU transfer")