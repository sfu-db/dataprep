"""
Throttler limits how many requests can issue given a specific time window
Copied from https://github.com/hallazzang/asyncio-throttle
"""

import time
import asyncio
from collections import deque
from typing import Deque


# class Throttler:
#     """
#     Throttler
#     """

#     req_per_window: int
#     window: float
#     retry_interval: float

#     def __init__(
#         self, req_per_window: int, window: float = 1.0, retry_interval: float = 0.01
#     ):
#         """
#         Create a throttler.
#         """
#         self.req_per_window = req_per_window
#         self.window = window
#         self.retry_interval = retry_interval

#         self._task_logs: Deque[float] = deque()

#     def _flush(self) -> None:
#         now = time.time()
#         while self._task_logs:
#             if now - self._task_logs[0] > self.window:
#                 self._task_logs.popleft()
#             else:
#                 break

#     async def _acquire(self) -> None:
#         while True:
#             self._flush()
#             if len(self._task_logs) < self.req_per_window:
#                 break
#             await asyncio.sleep(self.retry_interval)

#         self._task_logs.append(time.time())

#     def release(self) -> None:
#         self._task_logs.pop()

#     async def __aenter__(self) -> None:
#         await self._acquire()

#     async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
#         pass


class OrderedThrottler:
    """
    Throttler, but also keeps request in order by
    requiring them a seq number
    """

    req_per_window: int
    window: float
    retry_interval: float

    def __init__(self, req_per_window: int, window: float = 1.0, retry_interval: float = 0.01):
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

    def session(self) -> "ThrottleSession":
        """returns a session"""
        return ThrottleSession(self)


class ThrottleSession:  # pylint: disable=protected-access
    """
    ThrottleSessions share a same rate throttler but
    can have independent sequence numbers
    """

    thr: OrderedThrottler
    seq: int = -1

    def __init__(self, thr: OrderedThrottler) -> None:
        self.thr = thr

    async def acquire(self, i: int) -> None:
        """
        Wait for the request being allowed to send out,
        without violating the # reqs/sec constraint and the order constraint.
        """
        if self.seq >= i:
            raise RuntimeError(f"{i} already acquired")

        while len(self.thr._task_logs) >= self.thr.req_per_window or self.seq != i - 1:
            await asyncio.sleep(self.thr.retry_interval)
            self.thr._flush()

        self.seq = i
        self.thr._task_logs.append(time.time())

    def release(self) -> None:
        """
        Cancel the last acquire, so the next acquire call will immediately return.
        Use this if no requests issued after the last acquire call.
        """
        self.thr._task_logs.pop()
