"""
    module for testing plot_diff([df1, df2, ..., dfn]) function.
"""

import logging

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from ...eda import plot_diff
from ...datasets import load_dataset
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


def test_sanity_compute_mulitple_column(simpledf: dd.DataFrame) -> None:
    plot_diff([simpledf, simpledf], "a")


def test_specify_column_type(simpledf: dd.DataFrame) -> None:
    plot_diff([simpledf, simpledf], dtype={"a": Nominal()})
    plot_diff([simpledf, simpledf], dtype=Nominal())


def test_specify_label(simpledf: dd.DataFrame) -> None:
    plot_diff([simpledf, simpledf], config={"diff.label": ["label_1", "label_2"]})


def test_specify_label_col(simpledf: dd.DataFrame) -> None:
    plot_diff([simpledf, simpledf], "a", config={"diff.label": ["label_1", "label_2"]})


def test_specify_baseline(simpledf: dd.DataFrame) -> None:
    plot_diff([simpledf, simpledf], config={"diff.baseline": 1})


def test_specify_baseline_col(simpledf: dd.DataFrame) -> None:
    plot_diff([simpledf, simpledf], "a", config={"diff.baseline": 1})


def test_col_not_align() -> None:
    df2 = pd.DataFrame({"a": [1, 2], "c": ["a", "b"], "d": [2, 3]})
    df1 = pd.DataFrame({"a": [2, 3], "e": ["a", "c"]})
    plot_diff([df1, df2], config={"diff.label": ["train_df", "test_df"]})


def test_dataset() -> None:
    df = load_dataset("titanic")
    df1 = df[df["Survived"] == 0]
    df2 = df[df["Survived"] == 1]
    plot_diff([df1, df2])
    plot_diff([df1, df2], config={"diff.density": True})
