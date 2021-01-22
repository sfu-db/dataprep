"""
    module for testing plot_diff([df1, df2, ..., dfn]) function.
"""
import logging

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from ...eda import plot_diff
from ...eda.dtypes import Nominal
from ...eda.utils import to_dask

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")  # type: ignore
def simpledf() -> dd.DataFrame:
    df = pd.DataFrame(np.random.rand(1000, 3), columns=["a", "b", "c"])

    df = pd.concat([df, pd.Series(np.random.choice(["a", "b", "c"], 1000, replace=True))], axis=1)
    df = pd.concat(
        [
            df,
            pd.Series(
                np.random.choice(["2020/03/29", "2020/01/10", "2019/11/21"], 1000, replace=True)
            ),
        ],
        axis=1,
    )
    df = pd.concat([df, pd.Series(np.zeros(1000))], axis=1)
    df.columns = ["a", "b", "c", "d", "e", "f"]
    df["e"] = pd.to_datetime(df["e"])
    # test when column is object but some cells are numerical
    df["g"] = pd.Series([0, "x"] * 500)

    idx = np.arange(1000)
    np.random.shuffle(idx)
    df.iloc[idx[:500], 0] = None

    ddf = to_dask(df)

    return ddf


def test_sanity_compute_mulitple_df(simpledf: dd.DataFrame) -> None:
    plot_diff([simpledf, simpledf])


def test_specify_column_type(simpledf: dd.DataFrame) -> None:
    plot_diff([simpledf, simpledf], dtype={"a": Nominal()})
    plot_diff([simpledf, simpledf], dtype=Nominal())


def test_specify_color(simpledf: dd.DataFrame) -> None:
    plot_diff([simpledf, simpledf], config={"bar.color": "#123456", "hist.color": "orange"})
    plot_diff([simpledf, simpledf], config={"kde.hist_color": (1, 2, 3)})


def test_specify_label(simpledf: dd.DataFrame) -> None:
    plot_diff([simpledf, simpledf], config={"diff.label": ["label_1", "label_2"]})


def test_specify_baseline(simpledf: dd.DataFrame) -> None:
    plot_diff([simpledf, simpledf], config={"diff.baseline": 1})
