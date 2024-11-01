import psutil
import GPUtil
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from loguru import logger

@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    gpu_utilization: List[float]
    gpu_memory_utilization: List[float]

class SystemMonitor:
    def __init__(self):
        self.metrics_history: List[SystemMetrics] = []
        
    def get_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # GPU metrics
            gpus = GPUtil.getGPUs()
            gpu_utilization = [gpu.load * 100 for gpu in gpus]
            gpu_memory_utilization = [gpu.memoryUtil * 100 for gpu in gpus]
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_utilization=gpu_utilization,
                gpu_memory_utilization=gpu_memory_utilization
            )
            
            self.metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return SystemMetrics(0.0, 0.0, [], [])
            
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary statistics of system metrics"""
        if not self.metrics_history:
            return {}
            
        metrics_array = np.array([
            [m.cpu_percent, m.memory_percent] +
            m.gpu_utilization +
            m.gpu_memory_utilization
            for m in self.metrics_history
        ])
        
        return {
            'cpu_mean': np.mean(metrics_array[:, 0]),
            'cpu_max': np.max(metrics_array[:, 0]),
            'memory_mean': np.mean(metrics_array[:, 1]),
            'memory_max': np.max(metrics_array[:, 1]),
            'gpu_util_mean': np.mean(metrics_array[:, 2:2+len(self.metrics_history[0].gpu_utilization)]),
            'gpu_memory_mean': np.mean(metrics_array[:, -len(self.metrics_history[0].gpu_memory_utilization):])
        }