"""Miscellaneous functions
"""
import logging
from math import ceil
from typing import Any, Union, Optional
import dask.dataframe as dd
import numpy as np
import pandas as pd
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


def nullity_filter(
    df: pd.DataFrame,
    filter_type: Optional[str] = None,
    p_cut_off: int = 0,
    n_cut_off: int = 0,
) -> pd.DataFrame:
    """
    This function is designed to filters a DataFrame according to its nullity,
    using some combination of 'top' and 'bottom' numerical
    and percentage values.
    Percentages and numerical thresholds can be specified simultaneously
	Parameters
    ----------
    df
	 The DataFrame whose columns are being filtered.
	filter
	 The orientation of the filter being applied to the DataFrame.
     One of, "top", "bottom", or None (default).
     The filter will simply return the DataFrame if you leave the filter
     argument unspecified or as None.
    p
	 A completeness ratio cut-off.
     If non-zero the filter will limit the DataFrame to columns with at least p
     completeness.Input should be in the range [0, 1].
    n
	 A numerical cut-off. If non-zero no more than this number of columns will be returned.
    return
	 The nullity-filtered `DataFrame`.
	Examples
    ----------
	to get a DataFrame with columns of at least 75% completeness but with no more than 5 columns
	>>> nullity_filter(df, filter='top', p=.75, n=5)
    """

    if filter_type == "top":
        if p_cut_off:
            df = df.iloc[
                :, [c >= p_cut_off for c in df.count(axis="rows").values / len(df)]
            ]
        if n_cut_off:
            df = df.iloc[
                :, np.sort(np.argsort(df.count(axis="rows").values)[-n_cut_off:])
            ]
    elif filter_type == "bottom":
        if p_cut_off:
            df = df.iloc[
                :, [c <= p_cut_off for c in df.count(axis="rows").values / len(df)]
            ]
        if n_cut_off:
            df = df.iloc[
                :, np.sort(np.argsort(df.count(axis="rows").values)[:n_cut_off])
            ]
    return df


def nullity_sort(
    df: pd.DataFrame, sort: Optional[str] = None, axis: str = "columns"
) -> pd.DataFrame:
    """
    This function is designed to Sorts a DataFrame according to its nullity,
    in either ascending or descending order.
	Parameters
    ----------
    df
	    the pandas data_frame object being sorted.
    sort
		the sorting method: either "ascending", "descending", or None (default).
    return
		the nullity-sorted DataFrame.
    """
    if sort is None:
        return df

    if axis == "columns":
        if sort == "ascending":
            return df.iloc[np.argsort(df.count(axis="columns").values), :]
        elif sort == "descending":
            return df.iloc[np.flipud(np.argsort(df.count(axis="columns").values)), :]
        else:
            raise ValueError(
                'The "sort" parameter must be set to "ascending" or "descending".'
            )
    elif axis == "rows":
        if sort == "ascending":
            return df.iloc[:, np.argsort(df.count(axis="rows").values)]
        elif sort == "descending":
            return df.iloc[:, np.flipud(np.argsort(df.count(axis="rows").values))]
        else:
            raise ValueError(
                'The "sort" parameter must be set to "ascending" or "descending".'
            )
    else:
        raise ValueError('The "axis" parameter must be set to "rows" or "columns".')
