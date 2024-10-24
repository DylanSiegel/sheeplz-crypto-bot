# File: utils/monitoring.py

import psutil
import GPUtil
import time
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from utils.logging import get_logger

logger = get_logger('Monitoring')

@dataclass
class SystemStats:
    """System resource statistics"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used: float
    memory_total: float
    process_cpu_percent: float
    process_memory: float
    gpu_load: Optional[float] = None
    gpu_memory_used: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    gpu_temperature: Optional[float] = None

class ResourceMonitor:
    """Monitor system resources during training"""
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.stats_history: List[SystemStats] = []

    def get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU statistics if available"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Get primary GPU
                return {
                    'gpu_load': gpu.load * 100,
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_total': gpu.memoryTotal,
                    'gpu_temperature': gpu.temperature
                }
        except Exception as e:
            logger.warning(f"Could not get GPU stats: {e}")
        return {}

    def get_system_stats(self) -> Dict[str, float]:
        """Get system statistics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used': memory.used / (1024 ** 3),  # Convert to GB
            'memory_total': memory.total / (1024 ** 3)  # Convert to GB
        }

    def get_process_stats(self) -> Dict[str, float]:
        """Get statistics for current process"""
        return {
            'process_cpu_percent': self.process.cpu_percent(),
            'process_memory': self.process.memory_info().rss / (1024 ** 3),  # Convert to GB
            'elapsed_time': time.time() - self.start_time
        }

    def log_stats(self, extra_info: Optional[Dict] = None):
        """Log current system statistics"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            **self.get_system_stats(),
            **self.get_gpu_stats(),
            **self.get_process_stats()
        }

        # Create SystemStats object and store
        system_stats = SystemStats(**stats)
        self.stats_history.append(system_stats)

        # Log information
        logger.info("\nResource Usage:")
        logger.info(f"CPU Usage: {stats['cpu_percent']:.1f}%")
        logger.info(f"Memory Usage: {stats['memory_used']:.1f}GB/{stats['memory_total']:.1f}GB "
                    f"({stats['memory_percent']:.1f}%)")

        if 'gpu_load' in stats and stats['gpu_load'] is not None:
            logger.info(f"GPU Usage: {stats['gpu_load']:.1f}%")
            logger.info(f"GPU Memory: {stats['gpu_memory_used']:.1f}MB/{stats['gpu_memory_total']:.1f}MB")
            logger.info(f"GPU Temperature: {stats['gpu_temperature']:.1f}°C")

        logger.info(f"Process Memory: {stats['process_memory']:.2f}GB")
        logger.info(f"Elapsed Time: {stats['elapsed_time']:.1f}s")

        if extra_info:
            logger.info("\nTraining Stats:")
            for key, value in extra_info.items():
                logger.info(f"{key}: {value}")

    def save_stats(self, path: Path):
        """Save resource statistics to file"""
        stats_dict = [vars(stats) for stats in self.stats_history]
        with open(path, 'w') as f:
            json.dump(stats_dict, f, indent=2)

    def plot_stats(self, path: Path):
        """Create visualization of resource usage"""
        if not self.stats_history:
            logger.warning("No stats to plot")
            return

        plt.figure(figsize=(15, 10))

        # CPU and Memory Usage
        plt.subplot(2, 2, 1)
        plt.plot([s.cpu_percent for s in self.stats_history], label='CPU Usage %')
        plt.plot([s.memory_percent for s in self.stats_history], label='Memory Usage %')
        plt.title('CPU and Memory Usage')
        plt.xlabel('Time')
        plt.ylabel('Percentage')
        plt.legend()

        # GPU Usage if available
        if any(s.gpu_load is not None for s in self.stats_history):
            plt.subplot(2, 2, 2)
            plt.plot([s.gpu_load for s in self.stats_history], label='GPU Usage %')
            plt.plot([s.gpu_temperature for s in self.stats_history if s.gpu_temperature is not None], label='GPU Temp °C')
            plt.title('GPU Usage and Temperature')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()

        # Process Memory
        plt.subplot(2, 2, 3)
        plt.plot([s.process_memory for s in self.stats_history], label='Process Memory (GB)')
        plt.title('Process Memory Usage')
        plt.xlabel('Time')
        plt.ylabel('GB')
        plt.legend()

        # Elapsed Time
        plt.subplot(2, 2, 4)
        plt.plot([s.elapsed_time for s in self.stats_history], label='Elapsed Time (s)')
        plt.title('Elapsed Time')
        plt.xlabel('Time')
        plt.ylabel('Seconds')
        plt.legend()

        plt.tight_layout()
        plt.savefig(path)
        plt.close()
