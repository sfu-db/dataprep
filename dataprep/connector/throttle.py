"""
Throttler limits how many requests can issue given a specific time window
Copied from https://github.com/hallazzang/asyncio-throttle
"""
import time
import asyncio
from collections import deque
from typing import Deque, Any


class Throttler:
    """
    Throttler
    """

    req_per_window: int
    window: float
    retry_interval: float

    def __init__(
        self, req_per_window: int, window: float = 1.0, retry_interval: float = 0.01
    ):
        """
        Create a throttler.
        """
        self.req_per_window = req_per_window
        self.window = window
        self.retry_interval = retry_interval

        self._task_logs: Deque[float] = deque()

    def _flush(self) -> None:
        now = time.time()
        while self._task_logs:
            if now - self._task_logs[0] > self.window:
                self._task_logs.popleft()
            else:
                break

    async def _acquire(self) -> None:
        while True:
            self._flush()
            if len(self._task_logs) < self.req_per_window:
                break
            await asyncio.sleep(self.retry_interval)

        self._task_logs.append(time.time())

    async def __aenter__(self) -> None:
        await self._acquire()

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        pass
