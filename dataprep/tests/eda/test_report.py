"""
    module for testing plot(df, x, y) function.
"""
import logging

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from ...eda import plot, plot_missing, plot_correlation
from ...eda.utils import to_dask

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")  # type: ignore
def simpledf() -> dd.DataFrame:
    df = pd.DataFrame(np.random.rand(1000, 3), columns=["a", "b", "c"])

    df = pd.concat(
        [df, pd.Series(np.random.choice(["a", "b", "c"], 1000, replace=True))], axis=1
    )
    df = pd.concat(
        [df, pd.Series(np.random.choice([list("a"), set("b"),], 1000, replace=True)),],
        axis=1,
    )
    df = pd.concat(
        [
            df,
            pd.Series(
                np.random.choice(
                    [pd.datetime(6, 4, 1), pd.to_datetime("today")], 1000, replace=True
                )
            ),
        ],
        axis=1,
    )

    df.columns = ["a", "b", "c", "d", "e", "f"]

    idx = np.arange(1000)
    np.random.shuffle(idx)
    df.iloc[idx[:500], 0] = None

    ddf = to_dask(df)

    return ddf


def test_plot_report(simpledf: dd.DataFrame) -> None:
    report = plot(simpledf)
    report.save(filename="plot_report.html")
    report._repr_html_()


def test_plot_correlation_report(simpledf: dd.DataFrame) -> None:
    report = plot_correlation(simpledf)
    report.save(filename="plot_correlation_report.html")
    report._repr_html_()


def test_plot_missing_report(simpledf: dd.DataFrame) -> None:
    report = plot_missing(simpledf)
    report.save(filename="plot_missing_report.html")
    report._repr_html_()
