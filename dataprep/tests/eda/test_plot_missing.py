"""
    This module for testing plot_missing(df, x, y) function.
"""
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from ...eda.dtypes import Numerical
from ...eda.missing import compute_missing, render_missing
from ...eda.utils import to_dask


@pytest.fixture(scope="module")  # type: ignore
def simpledf() -> dd.DataFrame:
    df = pd.DataFrame(np.random.rand(1000, 3), columns=["a", "b", "c"])

    df = pd.concat(
        [df, pd.Series(np.random.choice(["a", "b", "c"], 1000, replace=True))], axis=1
    )

    df.columns = ["a", "b", "c", "d"]
    idx = np.arange(1000)
    np.random.shuffle(idx)
    df.iloc[idx[:500], 0] = None

    ddf = to_dask(df)

    return ddf


def test_sanity_compute_1(simpledf: dd.DataFrame) -> None:
    itmdt = compute_missing(simpledf)
    render_missing(itmdt)


def test_sanity_compute_2(simpledf: dd.DataFrame) -> None:
    itmdt = compute_missing(simpledf, x="a")
    render_missing(itmdt)


def test_sanity_compute_3(simpledf: dd.DataFrame) -> None:
    itmdt = compute_missing(simpledf, x="d")
    render_missing(itmdt)


def test_sanity_compute_4(simpledf: dd.DataFrame) -> None:
    itmdt = compute_missing(simpledf, x="a", y="b")
    render_missing(itmdt)


def test_sanity_compute_5(simpledf: dd.DataFrame) -> None:
    itmdt = compute_missing(simpledf, x="a", y="d")
    render_missing(itmdt)


def test_specify_column_type(simpledf: dd.DataFrame) -> None:
    itmdt = compute_missing(simpledf, x="b", dtype={"a": Numerical()})
    render_missing(itmdt)


@pytest.mark.xfail  # type: ignore
def test_sanity_compute_6(simpledf: dd.DataFrame) -> None:
    compute_missing(simpledf, y="b")
