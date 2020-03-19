"""
    module for testing plot(df, x, y) function.
"""
import logging

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from ...eda import plot
from ...eda.utils import to_dask

LOGGER = logging.getLogger(__name__)


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
    plot(simpledf, simpledf.columns[0])


def test_sanity_compute_2(simpledf: dd.DataFrame) -> None:
    plot(simpledf, simpledf.columns[-1])


def test_sanity_compute_3(simpledf: dd.DataFrame) -> None:
    plot(simpledf)
