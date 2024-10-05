from collections import deque

class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self.processing_times = deque(maxlen=window_size)

    def record(self, start_time: float, end_time: float):
        self.processing_times.append(end_time - start_time)

    @property
    def average_processing_time(self):
        return sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0