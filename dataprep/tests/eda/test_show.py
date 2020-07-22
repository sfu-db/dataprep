# type: ignore
from os import environ
import dask.dataframe as dd
import pandas as pd
import numpy as np
import pytest

from ...eda import plot, plot_correlation, plot_missing
from ...eda.utils import to_dask


@pytest.fixture(scope="module")  # type: ignore
def simpledf() -> dd.DataFrame:
    df = pd.DataFrame(np.random.rand(10, 3), columns=["a", "b", "c"])
    df = pd.concat([df, pd.Series(["a"] * 10)], axis=1)
    df.columns = ["a", "b", "c", "d"]
    df = to_dask(df)
    return df


def test_show(simpledf: dd.DataFrame) -> None:
    plot(simpledf).show()
    plot_correlation(simpledf).show()
    plot_missing(simpledf).show()


@pytest.mark.skipif(
    environ.get("DATAPREP_BROWSER_TESTS", "0") == "0",
    reason="Skip tests that requires opening browser",
)
def test_show_browser(simpledf: dd.DataFrame) -> None:
    plot(simpledf).show_browser()
    plot_correlation(simpledf).show_browser()
    plot_missing(simpledf).show_browser()
