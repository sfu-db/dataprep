"""
    module for testing plot(df, x, y) function.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd

from ...eda import plot


def test_corner() -> None:
    """

    :return:
    """
    df_2 = pd.DataFrame(
        {"all_nan": [np.nan for _ in range(10)], "all_one": np.ones(10),
         "all_zeros": np.zeros(10), "random": np.array(
             [0.38538395, 0.13609054, 0.15973238, 0.96192966, 0.03708882,
              0.03633855, 0.25260128, 0.72139843, 0.74553949,
              0.41102021])})

    df_1_expected = {"all_one": {"bar_plot": {1.0: 10}},
                     "all_zeros": {"bar_plot": {0.0: 10}},
                     "random": {"bar_plot": np.array([2, 2, 1, 1, 1, 0, 0, 2, 0, 1],
                                                     dtype=np.int64)}}

    res = plot(df_2, force_cat=["all_one", "all_zeros"])

    assert res["all_one"] == df_1_expected["all_one"]
    assert res["all_zeros"] == df_1_expected["all_zeros"]

    df_2 = pd.DataFrame({
        "empty": [],
        "another_empty": []
    })

    df_2_expected: Dict[str, Any] = {'scatter_plot': {}}

    res = plot(df_2, "empty", "another_empty")
    assert res == df_2_expected
