"""ProgressBar shows the how many dask tasks finished/remains using tqdm."""

import warnings
from time import time
from typing import Any, Dict, Optional, Tuple, Union

from dask.callbacks import Callback

from .utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# pylint: disable=method-hidden,too-many-instance-attributes
class ProgressBar(Callback):  # type: ignore
    """A progress bar for DataPrep.EDA.
    Not thread safe.

    Parameters
    ----------
    minimum : int, optional
        Minimum time threshold in seconds before displaying a progress bar.
        Default is 0 (always display)
    min_tasks : int, optional
        Minimum graph size to show a progress bar, default is 5
    width : int, optional
        Width of the bar. None means auto width.
    interval : float, optional
        Update resolution in seconds, default is 0.1 seconds.
    disable : bool, optional
        Disable the progress bar.
    warn_cost: bool, optional
        Whether print the warning of computational time of progress bar if it is high.
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
    _pbar_runtime: float = 0
    _last_updated: Optional[float] = None
    _disable: bool = False
    _warn_cost: bool = False

    def __init__(  # pylint: disable=too-many-arguments
        self,
        minimum: float = 0,
        min_tasks: int = 5,
        width: Optional[int] = None,
        interval: float = 0.1,
        disable: bool = False,
        warn_cost: bool = False,
    ) -> None:
        super().__init__()
        self._minimum = minimum
        self._min_tasks = min_tasks
        self._width = width
        self._interval = interval
        self._disable = disable
        self._warn_cost = warn_cost

    def _start(self, _dsk: Any) -> None:
        """A hook to start this callback."""

    def _start_state(self, _dsk: Any, state: Dict[str, Any]) -> None:
        """A hook called before every task gets executed."""
        if self._disable:
            return

        then = time()

        self._last_updated = self._started = time()

        self._state = state
        _, ntasks = self._count_tasks()

        if ntasks > self._min_tasks:
            self._init_pbar()

        self._pbar_runtime += time() - then

    def _pretask(self, key: Union[str, Tuple[str, ...]], _dsk: Any, _state: Dict[str, Any]) -> None:
        """A hook called before one task gets executed."""
        if self._disable:
            return

        then = time()

        if self._started is None:
            raise ValueError("ProgressBar not started properly")

        if self._pbar is None and then - self._started > self._minimum:
            self._init_pbar()

        if isinstance(key, tuple):
            key = key[0]

        self._last_task = key

        self._pbar_runtime += time() - then

    def _posttask(
        self,
        _key: str,
        _result: Any,
        _dsk: Any,
        _state: Dict[str, Any],
        _worker_id: Any,
    ) -> None:
        """A hook called after one task gets executed."""

        if self._disable:
            return

        then = time()

        if self._pbar is not None:
            if self._last_updated is None:
                raise ValueError("ProgressBar not started properly")

            if time() - self._last_updated > self._interval:
                self._update_bar()
                self._last_updated = time()

        self._pbar_runtime += time() - then

    def _finish(self, _dsk: Any, _state: Dict[str, Any], _errored: bool) -> None:
        """A hook called after all tasks get executed."""
        if self._disable:
            return

        then = time()

        if self._started is None:
            raise ValueError("ProgressBar not started properly")

        if self._pbar is None and time() - self._started > self._minimum:
            self._init_pbar()

        if self._pbar is not None:
            self._update_bar()
            self._pbar.close()

        self._pbar_runtime += time() - then

        if self._warn_cost and (
            self._pbar_runtime > 0.1 * (time() - self._started) and self._pbar_runtime > 1
        ):
            warnings.warn(
                "[ProgressBar] ProgressBar takes additional 10%+ of the computation time,"
                " consider disable it by passing 'progress=False' to the plot function.",
            )

        self._state = None
        self._started = None
        self._pbar = None

    def _update_bar(self) -> None:
        if self._pbar is None:
            return
        ndone, _ = self._count_tasks()

        self._pbar.set_description(f"Computing {self._last_task}", refresh=False)
        self._pbar.update(max(0, ndone - self._pbar.n))

    def _init_pbar(self) -> None:
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
                leave=False,
            )
        else:
            self._pbar = tqdm(
                total=ntasks,
                ncols=self._width,
                mininterval=self._interval,
                initial=ndone,
                leave=False,
            )

        self._pbar.set_description(desc)
        self._pbar.start_t = self._started
        self._pbar.refresh()

    def _count_tasks(self) -> Tuple[int, int]:
        if self._state is None:
            raise ValueError("ProgressBar not started properly")

        state = self._state
        ndone = len(state["finished"])
        ntasks = sum(len(state[k]) for k in ["ready", "waiting", "running"]) + ndone

        return ndone, ntasks

    def register(self) -> None:
        raise ValueError("ProgressBar is not thread safe thus cannot be regestered globally")

    def unregister(self) -> None:
        raise ValueError("ProgressBar is not thread safe thus cannot be unregestered globally")
