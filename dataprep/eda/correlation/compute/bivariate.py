"""This module implements the intermediates computation
for plot_correlation(df) function."""

from operator import itruediv
from typing import Optional, Tuple

import dask
import dask.dataframe as dd
import dask.array as da
import numpy as np

from ...intermediate import Intermediate


def _calc_bivariate(
    df: dd.DataFrame,
    x: str,
    y: str,
    *,
    k: Optional[int] = None,
) -> Intermediate:
    if x not in df.columns:
        raise ValueError(f"{x} not in columns names")
    if y not in df.columns:
        raise ValueError(f"{y} not in columns names")

    df = df[[x, y]].dropna()
    coeffs, df_smp, influences = scatter_with_regression(df, sample_size=1000, k=k)

    coeffs, df_smp, influences = dask.compute(coeffs, df_smp, influences)

    result = {"coeffs": coeffs, "data": df_smp}

    if (influences is None) != (k is None):
        raise RuntimeError("Not possible")

    if influences is not None and k is not None:
        infidx = np.argsort(influences)
        labels = np.full(len(influences), "=")
        # pylint: disable=invalid-unary-operand-type
        labels[infidx[-k:]] = "-"  # type: ignore
        # pylint: enable=invalid-unary-operand-type
        labels[infidx[:k]] = "+"
        result["data"]["influence"] = labels  # type: ignore

    return Intermediate(**result, visual_type="correlation_scatter")


def scatter_with_regression(
    df: dd.DataFrame, sample_size: int, k: Optional[int] = None
) -> Tuple[Tuple[da.Array, da.Array], Tuple[da.Array, da.Array], Optional[da.Array]]:
    """Calculate pearson correlation on 2 given arrays.

    Parameters
    ----------
    df
        dataframe
    sample_size
        Number of points to show in the scatter plot
    k : Optional[int] = None
        Highlight k points which influence pearson correlation most
    """
    df["ones"] = 1
    arr = df.to_dask_array(lengths=True)

    (coeffa, coeffb), _, _, _ = da.linalg.lstsq(arr[:, [0, 2]], arr[:, 1])

    df = df.drop(columns=["ones"])
    df_smp = df.map_partitions(lambda x: x.sample(min(sample_size, x.shape[0])), meta=df)
    # TODO influences should not be computed on a sample
    influences = (
        pearson_influence(
            df_smp[df.columns[0]].to_dask_array(lengths=True),
            df_smp[df.columns[1]].to_dask_array(lengths=True),
        )
        if k
        else None
    )

    return (coeffa, coeffb), df_smp, influences


def pearson_influence(xarr: da.Array, yarr: da.Array) -> da.Array:
    """Calculating the influence for deleting a point on the pearson correlation"""

    if xarr.shape != yarr.shape:
        raise ValueError(
            f"The shape of xarr and yarr should be same, got {xarr.shape}, {yarr.shape}"
        )

    # Fast calculating the influence for removing one element on the correlation
    n = xarr.shape[0]

    x2, y2 = da.square(xarr), da.square(yarr)
    xy = xarr * yarr

    # The influence is vectorized on xarr and yarr, so we need to repeat all the sums for n times

    xsum = da.ones(n) * da.sum(xarr)
    ysum = da.ones(n) * da.sum(yarr)
    xysum = da.ones(n) * da.sum(xy)
    x2sum = da.ones(n) * da.sum(x2)
    y2sum = da.ones(n) * da.sum(y2)

    # Note: in we multiply (n-1)^2 to both denominator and numerator to avoid divisions.
    numerator = (n - 1) * (xysum - xy) - (xsum - xarr) * (ysum - yarr)

    varx = (n - 1) * (x2sum - x2) - da.square(xsum - xarr)
    vary = (n - 1) * (y2sum - y2) - da.square(ysum - yarr)
    denominator = da.sqrt(varx * vary)

    return da.map_blocks(itruediv, numerator, denominator, dtype=numerator.dtype)
