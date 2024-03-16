"""
This module is for performance testing of EDA module in github action.
"""

from functools import partial
import pandas as pd
from typing import Any
from ...datasets import load_dataset
from ...eda import create_report


def report_func(df: pd.DataFrame, **kwargs: Any) -> None:
    """
    Create report function, used for performance testing.
    """
    create_report(df, **kwargs)


def test_create_report(benchmark: Any) -> None:
    """
    Performance test of create report on titanic dataset.
    """
    df = load_dataset("titanic")
    benchmark(partial(report_func), df)
