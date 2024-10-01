# src/data/acquisition/utils/rate_limiter.py

import asyncio
import time

class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    async def wait(self):
        now = time.time()
        self.calls = [call for call in self.calls if call > now - self.period]
        if len(self.calls) >= self.max_calls:
            sleep_time = self.calls[0] - (now - self.period)
            await asyncio.sleep(sleep_time)
        self.calls.append(time.time())