"""
    module for testing create_report(df) function.
"""
import logging
import numpy as np
import pandas as pd
import pytest
from ...eda import create_report

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")  # type: ignore
def simpledf() -> pd.DataFrame:
    df = pd.DataFrame(np.random.rand(1000, 3), columns=["a", "b", "c"])

    df = pd.concat([df, pd.Series(np.random.choice(["a", "b", "c"], 1000, replace=True))], axis=1)
    df = pd.concat([df, pd.Series([["foo"] * 1000])], axis=1)
    df = pd.concat(
        [
            df,
            pd.Series(
                np.random.choice(["2020/03/29", "2020/01/10", "2019/11/21"], 1000, replace=True)
            ),
        ],
        axis=1,
    )
    df.columns = ["a", "b", "c", "d", "e", "f"]
    df["g"] = pd.to_datetime(df["f"])
    # test when column is object but some cells are numerical
    df["h"] = pd.Series([0, "x"] * 500)

    idx = np.arange(1000)
    np.random.shuffle(idx)
    df.iloc[idx[:500], 0] = None

    return df


@pytest.fixture(scope="module")  # type: ignore
def constantdf() -> pd.DataFrame:
    df = pd.DataFrame({"a": [0] * 10, "b": [1] * 10, "c": [np.nan] * 10})

    return df


def test_report(simpledf: pd.DataFrame) -> None:
    from sys import platform

    if platform == "darwin":
        import matplotlib

        matplotlib.use("PS")
    create_report(simpledf, mode="basic")


def test_report_show(simpledf: pd.DataFrame) -> None:
    from sys import platform

    if platform == "darwin":
        import matplotlib

        matplotlib.use("PS")
    report = create_report(simpledf, mode="basic")
    report.show()


def test_report_constant(constantdf: pd.DataFrame) -> None:
    from sys import platform

    if platform == "darwin":
        import matplotlib

        matplotlib.use("PS")
    create_report(constantdf, mode="basic")


def test_report_single_column(simpledf: pd.DataFrame) -> None:
    from sys import platform

    if platform == "darwin":
        import matplotlib

        matplotlib.use("PS")
    create_report(simpledf[["a"]], mode="basic")
