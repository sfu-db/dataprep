"""ProgressBar shows the how many dask tasks finished/remains using tqdm."""

from typing import Any, Optional, Dict, Tuple, Union
from time import time

from dask.callbacks import Callback

from .utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# pylint: disable=method-hidden,too-many-instance-attributes
class ProgressBar(Callback):  # type: ignore
    """A progress bar for DataPrep.EDA.

    Parameters
    ----------
    minimum : int, optional
        Minimum time threshold in seconds before displaying a progress bar.
        Default is 0 (always display)
    _min_tasks : int, optional
        Minimum graph size to show a progress bar, default is 5
    width : int, optional
        Width of the bar. None means auto width.
    interval : float, optional
        Update resolution in seconds, default is 0.1 seconds
    """

    _minimum: float = 0
    _min_tasks: int = 5
    _width: Optional[int] = None
    _interval: float = 0.1
    _last_duration: float = 0
    _pbar: Optional[tqdm] = None
    _state: Optional[Dict[str, Any]] = None
    _started: Optional[float] = None
    _last_task: Optional[str] = None  # in case we initialize the pbar in _finish

    def __init__(
        self,
        minimum: float = 0,
        min_tasks: int = 5,
        width: Optional[int] = None,
        interval: float = 0.1,
    ) -> None:
        super().__init__()
        self._minimum = minimum
        self._min_tasks = min_tasks
        self._width = width
        self._interval = interval

    def _start(self, _dsk: Any) -> None:
        """A hook to start this callback."""

    def _start_state(self, _dsk: Any, state: Dict[str, Any]) -> None:
        """A hook called before every task gets executed."""
        self._started = time()
        self._state = state
        _, ntasks = self._count_tasks()

        if ntasks > self._min_tasks:
            self._init_bar()

    def _pretask(
        self, key: Union[str, Tuple[str, ...]], _dsk: Any, _state: Dict[str, Any]
    ) -> None:
        """A hook called before one task gets executed."""
        if self._started is None:
            raise ValueError("ProgressBar not started properly")

        if self._pbar is None and time() - self._started > self._minimum:
            self._init_bar()

        if isinstance(key, tuple):
            key = key[0]

        if self._pbar is not None:
            self._pbar.set_description(f"Computing {key}")
        else:
            self._last_task = key

    def _posttask(
        self,
        _key: str,
        _result: Any,
        _dsk: Any,
        _state: Dict[str, Any],
        _worker_id: Any,
    ) -> None:
        """A hook called after one task gets executed."""

        if self._pbar is not None:
            self._update_bar()

    def _finish(self, _dsk: Any, _state: Dict[str, Any], _errored: bool) -> None:
        """A hook called after all tasks get executed."""
        if self._started is None:
            raise ValueError("ProgressBar not started properly")

        if self._pbar is None and time() - self._started > self._minimum:
            self._init_bar()

        if self._pbar is not None:
            self._update_bar()
            self._pbar.close()

        self._state = None
        self._started = None
        self._pbar = None

    def _update_bar(self) -> None:
        if self._pbar is None:
            return
        ndone, _ = self._count_tasks()

        self._pbar.update(max(0, ndone - self._pbar.n))

    def _init_bar(self) -> None:
        if self._pbar is not None:
            raise ValueError("ProgressBar already initialized.")
        ndone, ntasks = self._count_tasks()

        if self._last_task is not None:
            desc = f"Computing {self._last_task}"
        else:
            desc = ""

        if self._width is None:
            self._pbar = tqdm(
                total=ntasks,
                dynamic_ncols=True,
                mininterval=self._interval,
                initial=ndone,
                desc=desc,
            )
        else:
            self._pbar = tqdm(
                total=ntasks,
                ncols=self._width,
                mininterval=self._interval,
                initial=ndone,
                desc=desc,
            )

        self._pbar.start_t = self._started
        self._pbar.refresh()

    def _count_tasks(self) -> Tuple[int, int]:
        if self._state is None:
            raise ValueError("ProgressBar not started properly")

        state = self._state
        ndone = len(state["finished"])
        ntasks = sum(len(state[k]) for k in ["ready", "waiting", "running"]) + ndone

        return ndone, ntasks
