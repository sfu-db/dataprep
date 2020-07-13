"""
    module for testing create_report(df) function.
"""
import logging
import numpy as np
import pandas as pd
import pytest
from ...eda.create_report.formatter import format_report

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")  # type: ignore
def simpledf() -> pd.DataFrame:
    df = pd.DataFrame(np.random.rand(1000, 3), columns=["a", "b", "c"])

    df = pd.concat(
        [df, pd.Series(np.random.choice(["a", "b", "c"], 1000, replace=True))], axis=1
    )
    df = pd.concat(
        [
            df,
            pd.Series(
                np.random.choice(
                    ["2020/03/29", "2020/01/10", "2019/11/21"], 1000, replace=True
                )
            ),
        ],
        axis=1,
    )
    df = pd.concat([df, pd.Series(np.zeros(1000))], axis=1)
    df.columns = ["a", "b", "c", "d", "e", "f"]
    df["e"] = pd.to_datetime(df["e"])

    idx = np.arange(1000)
    np.random.shuffle(idx)
    df.iloc[idx[:500], 0] = None

    return df


def test_report(simpledf: pd.DataFrame) -> None:
    format_report(simpledf, mode="basic")
