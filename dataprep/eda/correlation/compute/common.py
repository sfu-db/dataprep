"""Common components for compute correlation."""

from enum import Enum, auto

import dask
import numpy as np
from scipy.stats.mstats import rankdata
from scipy.stats import kendalltau as kendalltau_


class CorrelationMethod(Enum):
    """Supported correlation methods"""

    # pylint: disable=invalid-name
    Pearson = auto()
    Spearman = auto()
    KendallTau = auto()


# @dask.delayed(name="rankdata-bottleneck", pure=True)  # pylint: disable=no-value-for-parameter
# def nanrankdata(data: np.ndarray, axis: int = 0) -> np.ndarray:
#     """delayed version of rankdata."""
#     return nanrankdata_(data, axis=axis)


@dask.delayed(name="nanrankdata", pure=True)  # pylint: disable=no-value-for-parameter
def nanrankdata(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """delayed version of rankdata."""
    ranks = rankdata(np.ma.masked_invalid(data), axis=axis)
    ranks[ranks == 0] = np.nan
    return ranks


@dask.delayed(name="kendalltau-scipy", pure=True)  # pylint: disable=no-value-for-parameter
def kendalltau(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    """delayed version of kendalltau."""
    corr = kendalltau_(a, b).correlation
    return np.float64(corr)  # Sometimes corr is a float, causes dask error


@dask.delayed(name="kendalltau-scipy", pure=True)  # pylint: disable=no-value-for-parameter
def corrcoef(arr: np.ndarray) -> np.ndarray:
    """delayed version of np.corrcoef."""
    _, (corr, _) = np.corrcoef(arr, rowvar=False)
    return corr
