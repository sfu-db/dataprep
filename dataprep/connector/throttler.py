"""
Throttler limits how many requests can issue given a specific time window
Copied from https://github.com/hallazzang/asyncio-throttle
"""
import asyncio
import contextlib
import sys
from time import time
from typing import AsyncIterator, Callable, Deque, Dict, NamedTuple, Set, Tuple
from uuid import UUID, uuid4

import numpy as np


class TaskLog(NamedTuple):
    start: float
    end: float
    success: bool


class Throttler:
    """
    Throttler, but also keeps request in order by
    requiring them a seq number
    """

    _min_concurrency: int = 1
    _max_concurrency: int = 2 << 10
    _req_per_window: int
    _window: float
    _retry_interval: float
    _backoff: int = 0
    _last_backoff: float = 0
    _task_logs: Deque[TaskLog]
    _running_tasks: Dict[UUID, float]  # key, start time

    def __init__(self, req_per_window: int, window: float = 1.0, retry_interval: float = 0.01):
        """Create a throttler."""
        self._req_per_window = req_per_window
        self._max_concurrency = req_per_window
        self._window = window
        self._retry_interval = retry_interval

        self._task_logs = Deque()
        self._running_tasks = dict()

    def flush(self) -> None:
        """Clear tasks that are out of the window."""
        now = time()
        while self._task_logs:
            if now - self._task_logs[0].end > self._window:
                self._task_logs.popleft()
            else:
                break

    def ntasks_in_window(self) -> int:
        """How many tasks are in current window."""
        return len(self._task_logs) + len(self._running_tasks)

    @property
    def running_tasks(self) -> Dict[UUID, float]:
        return self._running_tasks

    @property
    def window_size(self) -> float:
        return self._window

    def running(self, task_id: UUID) -> None:
        """Add a running task."""
        self._running_tasks[task_id] = time()

    def complete(self, task_id: UUID, status: str = "completed") -> None:
        """Finish a running task.
        This removes the task from the running queue and
        add the finished time to the task log."""

        task_begin = self._running_tasks.pop(task_id)
        task_end = time()
        if status == "completed":
            self._task_logs.append(TaskLog(task_begin, task_end, True))  # append the finish time
            self.bound_min_concurrency()
        elif status == "failed":
            self._task_logs.append(TaskLog(task_begin, task_end, False))  # append the finish time
            self.bound_max_concurrency()
        elif status == "cancelled":
            pass
        else:
            raise RuntimeError(f"Unknown status {status}")

    def bound_min_concurrency(self) -> None:
        succeeded = [tl for tl in self._task_logs if tl.success]
        l = len(succeeded)
        begins = sorted(range(l), key=lambda i: succeeded[i].start)
        ends = sorted(range(l), key=lambda i: succeeded[i].end)

        # Calculating the minimal concurrency
        i = l - 1

        for j in range(l - 1, -1, -1):
            win_end = succeeded[ends[j]][1]
            window = (win_end - self.window_size, win_end)

            while i >= 0 and window[0] <= succeeded[begins[i]].start:
                i -= 1

            # Now begins[i+1:] are all larger than window start
            # ends[:j] are all smaller than window end
            if i == -1:
                break

            self._min_concurrency = max(
                len(set(begins[i + 1 :]) & set(ends[:j])), self._min_concurrency
            )

    def bound_max_concurrency(self) -> None:
        l = len(self._task_logs)
        running = list(self._running_tasks.values())
        l2 = len(running)

        begins = sorted(
            range(l + l2),
            key=lambda i: self._task_logs[i][0] if i < l else running[i - l],
        )
        ends = sorted(range(l), key=lambda i: self._task_logs[i][1])

        i = l - 1

        for j in range(l + l2 - 1, -1, -1):
            win_end = self._task_logs[begins[j]][0] if j < l else running[j - l]
            window = (win_end - self._window, win_end)

            if window[0] > self._task_logs[-1][1]:
                continue

            if window[1] < self._task_logs[-1][0]:
                break

            while i >= 0 and window[0] <= self._task_logs[ends[i]][1]:
                i -= 1
            if self._task_logs[ends[i + 1]][1] < self._task_logs[-1][0] - self._window:
                break

            if i == -1:
                break

            remainder = max(j - l + 1, 0)
            j -= remainder

            self._max_concurrency = min(
                len(set(ends[i + 1 :]) | set(begins[:j])) + remainder,
                self._min_concurrency,
            )

    def ordered(self) -> "OrderedThrottleSession":
        """returns an ordered throttler session"""
        return OrderedThrottleSession(self)

    @property
    def retry_interval(self) -> int:
        return self._retry_interval

    @property
    def req_per_window(self) -> int:
        return np.clip(
            round(self._req_per_window / (2 ** self._backoff)),
            self._min_concurrency,
            self._max_concurrency,
        )

    def backoff(self) -> None:
        self._last_backoff = time()
        self._backoff += 1

    @property
    def last_backoff(self) -> float:
        return self._last_backoff


class OrderedThrottleSession:  # pylint: disable=protected-access
    """OrderedThrottleSession share a same rate throttler but
    can have independent sequence numbers."""

    thr: Throttler
    seqs: Set[int]

    def __init__(self, thr: Throttler) -> None:
        self.thr = thr
        self.seqs = set()

    @contextlib.asynccontextmanager
    async def acquire(
        self, i: int
    ) -> AsyncIterator[Tuple[Callable[[], None], Callable[[bool], None]]]:
        """Wait for the request being allowed to send out,
        without violating the # reqs/sec constraint and the order constraint."""
        while self.thr.ntasks_in_window() >= self.thr.req_per_window or self.next_seq() != i:
            await asyncio.sleep(self.thr.retry_interval)
            self.thr.flush()

        self.seqs.add(i)
        task_id = uuid4()
        self.thr.running(task_id)

        status = "completed"

        def fail(backoff: bool = True) -> None:
            nonlocal status
            status = "failed"
            self.seqs.remove(i)
            # only if the task is sent after last backoff will trigger a new backoff
            if (
                backoff
                and self.thr.running_tasks[task_id] > self.thr.last_backoff + self.thr.window_size
            ):
                self.thr.backoff()
                print(
                    f"Request failed, decreasing the concurrency level to {self.req_per_window}",
                    file=sys.stderr,
                )

        def cancel() -> None:
            nonlocal status
            status = "cancelled"

        yield cancel, fail
        self.thr.complete(task_id, status)

    def next_seq(self) -> int:
        if not self.seqs:
            return 0

        for i in range(max(self.seqs) + 2):
            if i not in self.seqs:
                return i
        raise RuntimeError("Unreachable")

    def backoff(self) -> None:
        self.thr.backoff()

    @property
    def req_per_window(self) -> int:
        return self.thr.req_per_window
