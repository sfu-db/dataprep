"""
    module for testing plot(df, x, y) function.
"""

import logging

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from ...datasets import load_dataset

from ...eda import plot
from ...eda.dtypes_v2 import Nominal, LatLong
from ...eda.utils import to_dask
from ...datasets import load_dataset
from .random_data_generator import random_df

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
    df["h"] = pd.Series(np.ones(1000))
    df["i"] = pd.Series(np.random.normal(0, 0.1, 1000))

    idx = np.arange(1000)
    np.random.shuffle(idx)
    df.iloc[idx[:500], 0] = None

    ddf = to_dask(df)

    return ddf


@pytest.fixture(scope="module")  # type: ignore
def geodf() -> dd.DataFrame:
    df = df = load_dataset("countries")

    ddf = to_dask(df)

    return ddf


def test_sanity_compute_univariate(simpledf: dd.DataFrame) -> None:
    plot(simpledf, "a")
    plot(simpledf, "e")
    plot(simpledf, "g")


def test_sanity_compute_overview(simpledf: dd.DataFrame) -> None:
    plot(simpledf)
    plot(simpledf, config={"hist.yscale": "log"})


def test_sanity_compute_bivariate(simpledf: dd.DataFrame) -> None:
    plot(simpledf, "a", "e")
    plot(simpledf, "d", "e")
    plot(simpledf, "a", "g")
    plot(simpledf, "d", "g")


def test_sanity_compute_7(simpledf: dd.DataFrame) -> None:
    plot(simpledf, "a", "b")


def test_specify_column_type(simpledf: dd.DataFrame) -> None:
    plot(simpledf, dtype={"a": Nominal()})
    plot(simpledf, dtype=Nominal())


def test_specify_color(simpledf: dd.DataFrame) -> None:
    plot(simpledf, config={"bar.color": "#123456", "hist.color": "orange"})
    plot(simpledf, "a", config={"kde.hist_color": (1, 2, 3)})


def test_geo(geodf: dd.DataFrame) -> None:
    plot(geodf, "Country")
    plot(geodf, "Country", "Population")
    covid = load_dataset("covid19")
    plot(covid, LatLong("Lat", "Long"), "2/16/2020")


def test_random_df(random_df: pd.DataFrame) -> None:
    plot(random_df)
    plot(random_df, display=["Bar Chart"])


def test_plot_dt() -> None:
    srs = pd.Series(
        [
            "3/11/2001",
            "3/12/2002",
            "3/12/2003",
            "3/13/2003",
            "4/13/2003",
            "4/13/2003",
            "4/13/2003",
            "4/13/2003",
            "4/13/2003",
            "4/13/2003",
            "4/13/2003",
            "4/13/2003",
            "4/13/2003",
            "4/13/2003",
            "4/13/2003",
        ]
    )
    dt_col = pd.to_datetime(srs, infer_datetime_format=True)
    df = pd.DataFrame()
    df["dt"] = dt_col
    df["num"] = [1.0, 2.1, 3.5, 4.5, 2.5, 1.5, 2.3, 6.1, 8.1, 1.0, 3, 10.6, 7.8, 9.1, 20.6]
    plot(df, "dt", "num")


def test_plot_titanic() -> None:
    df = load_dataset("titanic")
    plot(df, "Sex", display=["Value Table"])
    plot(df, "Age", display=["Value Table"])
    plot(df, "Name", "Ticket")
