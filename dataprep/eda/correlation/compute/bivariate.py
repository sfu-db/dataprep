"""This module implements the intermediates computation
for plot_correlation(df) function."""

from operator import itruediv
from typing import Optional, Tuple

import dask
import dask.array as da
import numpy as np
import pandas as pd


from ...data_array import DataArray
from ...intermediate import Intermediate


def _calc_bivariate(
    df: DataArray,
    x: Optional[str] = None,
    y: Optional[str] = None,
    *,
    k: Optional[int] = None,
) -> Intermediate:
    if x not in df.columns:
        raise ValueError(f"{x} not in columns names")
    if y not in df.columns:
        raise ValueError(f"{y} not in columns names")

    xname, yname = x, y

    df.compute_length()

    xloc = df.columns.get_loc(x)
    yloc = df.columns.get_loc(y)

    x = df.values[:, xloc]
    y = df.values[:, yloc]
    coeffs, (x, y), influences = scatter_with_regression(x, y, k=k, sample_size=1000,)

    coeffs, (x, y), influences = dask.compute(coeffs, (x, y), influences)

    # lazy/eager border line
    result = {
        "coeffs": coeffs,
        "data": pd.DataFrame({xname: x, yname: y}),
    }

    if (influences is None) != (k is None):
        raise RuntimeError("Not possible")

    if influences is not None and k is not None:
        infidx = np.argsort(influences)
        labels = np.full(len(influences), "=")
        # pylint: disable=invalid-unary-operand-type
        labels[infidx[-k:]] = "-"  # type: ignore
        # pylint: enable=invalid-unary-operand-type
        labels[infidx[:k]] = "+"
        result["data"]["influence"] = labels

    return Intermediate(**result, visual_type="correlation_scatter")


def scatter_with_regression(
    x: da.Array, y: da.Array, sample_size: int, k: Optional[int] = None
) -> Tuple[Tuple[da.Array, da.Array], Tuple[da.Array, da.Array], Optional[da.Array]]:
    """Calculate pearson correlation on 2 given arrays.

    Parameters
    ----------
    xarr : da.Array
    yarr : da.Array
    sample_size : int
    k : Optional[int] = None
        Highlight k points which influence pearson correlation most
    """
    if k == 0:
        raise ValueError("k should be larger than 0")

    xp1 = da.vstack([x, da.ones_like(x)]).T
    xp1 = xp1.rechunk((xp1.chunks[0], -1))

    mask = ~(da.isnan(x) | da.isnan(y))
    # if chunk size in the first dimension is 1, lstsq will use sfqr instead of tsqr,
    # where the former does not support nan in shape.

    if len(xp1.chunks[0]) == 1:
        xp1 = xp1.rechunk((2, -1))
        y = y.rechunk((2, -1))
        mask = mask.rechunk((2, -1))

    (coeffa, coeffb), _, _, _ = da.linalg.lstsq(xp1[mask], y[mask])

    if sample_size < x.shape[0]:
        samplesel = da.random.choice(x.shape[0], int(sample_size), chunks=x.chunksize)
        x = x[samplesel]
        y = y[samplesel]

    if k is None:
        return (coeffa, coeffb), (x, y), None

    influences = pearson_influence(x, y)
    return (coeffa, coeffb), (x, y), influences


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
