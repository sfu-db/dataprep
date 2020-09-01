"""Miscellaneous functions
"""
import logging
from math import ceil
from typing import Any, Union, List
import dask.dataframe as dd
import numpy as np
import pandas as pd
import dask
from bokeh.models import Legend
from bokeh.plotting import Figure

LOGGER = logging.getLogger(__name__)


def is_notebook() -> Any:
    """
    :return: whether it is running in jupyter notebook
    """
    try:
        # pytype: disable=import-error
        from IPython import get_ipython  # pylint: disable=import-outside-toplevel

        # pytype: enable=import-error

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        return False
    except (NameError, ImportError):
        return False


def to_dask(df: Union[pd.DataFrame, dd.DataFrame]) -> dd.DataFrame:
    """
    Convert a dataframe to a dask dataframe.
    """
    if isinstance(df, dd.DataFrame):
        return df

    df_size = df.memory_usage(deep=True).sum()
    npartitions = ceil(df_size / 128 / 1024 / 1024)
    return dd.from_pandas(df, npartitions=npartitions)


def sample_n(arr: np.ndarray, n: int) -> np.ndarray:  # pylint: disable=C0103
    """
    Sample n values uniformly from the range of the `arr`,
    not from the distribution of `arr`'s elems.
    """
    if len(arr) <= n:
        return arr

    subsel = np.linspace(0, len(arr) - 1, n)
    subsel = np.floor(subsel).astype(int)
    return arr[subsel]


def relocate_legend(fig: Figure, loc: str) -> Figure:
    """
    Relocate legend(s) from center to `loc`
    """
    remains = []
    targets = []
    for layout in fig.center:
        if isinstance(layout, Legend):
            targets.append(layout)
        else:
            remains.append(layout)
    fig.center = remains
    for layout in targets:
        fig.add_layout(layout, loc)

    return fig


def cut_long_name(name: str, max_len: int = 12) -> str:
    """
    If the name is longer than `max_len`,
    cut it to `max_len` length and append "..."
    """
    # Bug 136 Fixed
    name = str(name)
    if len(name) <= max_len:
        return name
    return f"{name[:max_len]}..."


def fuse_missing_perc(name: str, perc: float) -> str:
    """
    Append (x.y%) to the name if `perc` is not 0
    """
    if perc == 0:
        return name

    return f"{name} ({perc:.1%})"


def get_intervals(minv: float, maxv: float, bins: int) -> List[float]:
    """
    returns the bin intervals used for the histogram as a list of floats
    """
    # compute the min and max of the column
    minv, maxv = dask.compute(minv, maxv)
    # calculate the gap (interval)
    gap = abs((maxv - minv) / bins)
    # get exponent from scientific notation
    _, after = f"{gap:.0e}".split("e")
    # round to this amount
    round_to = -1 * int(after)
    # round value between ticks
    gap = np.round(gap, round_to)

    if minv <= 0 <= maxv:
        ticks = [0.0]
        # iterate to go from 0 to minv
        while min(ticks) > minv:
            ticks.insert(0, min(ticks) - gap)
        # iterate to go from 0 to maxv
        while max(ticks) < maxv:
            ticks.append(max(ticks) + gap)
    else:
        # round the first x tick
        rounded_minv = np.round(minv, round_to)
        ticks = [rounded_minv - gap] if rounded_minv > minv else [rounded_minv]
        while max(ticks) < maxv:
            ticks.append(max(ticks) + gap)

    return ticks
