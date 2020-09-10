"""Common components for compute correlation."""

from enum import Enum, auto

import dask
import numpy as np
from bottleneck import rankdata as rankdata_, nanrankdata as nanrankdata_
from scipy.stats import kendalltau as kendalltau_


class CorrelationMethod(Enum):
    """Supported correlation methods"""

    Pearson = auto()
    Spearman = auto()
    KendallTau = auto()


@dask.delayed(  # pylint: disable=no-value-for-parameter
    name="rankdata-bottleneck", pure=True
)
def rankdata(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """delayed version of rankdata"""
    return rankdata_(data, axis=axis)


@dask.delayed(  # pylint: disable=no-value-for-parameter
    name="rankdata-bottleneck", pure=True
)
def nanrankdata(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """delayed version of rankdata."""
    return nanrankdata_(data, axis=axis)


@dask.delayed(  # pylint: disable=no-value-for-parameter
    name="kendalltau-scipy", pure=True
)
def kendalltau(  # pylint: disable=invalid-name
    a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """delayed version of kendalltau."""
    corr = kendalltau_(a, b).correlation
    return np.float64(corr)  # Sometimes corr is a float, causes dask error


@dask.delayed(  # pylint: disable=no-value-for-parameter
    name="kendalltau-scipy", pure=True
)
def corrcoef(arr: np.ndarray) -> np.ndarray:
    """delayed version of np.corrcoef."""
    _, (corr, _) = np.corrcoef(arr, rowvar=False)
    return corr
