"""
    module for testing plot(df, x, y) function.
"""
import datetime
import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from pandas import Timestamp
from ...eda.common import Intermediate
from ...eda import plot

LOGGER = logging.getLogger(__name__)


def test_normal_1() -> None:
    """

    :return:
    """
    # TEST - 1 for plot(df)
    data_1 = {
        "id": [chr(97 + c) for c in range(1, 10)],
        "x": [50, 50, -10, 0, 0, 5, 15, -3, None],
        "y": [0.000001, 654.152, None, 15.984512, 3122, -3.1415926535, 111, 15.9, 13.5],
        "s1": np.ones(9),
        "somedate": [
            datetime.date(2011, 7, 4),
            datetime.datetime(2022, 1, 1, 13, 57),
            datetime.datetime(1990, 12, 9),
            np.nan,
            datetime.datetime(1990, 12, 9),
            datetime.datetime(1950, 12, 9),
            datetime.datetime(1898, 1, 2),
            datetime.datetime(1950, 12, 9),
            datetime.datetime(1950, 12, 9),
        ],
        "bool_tf": [True, True, False, True, False, True, True, False, True],
        "bool_tf_with_nan": [
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            np.nan,
        ],
        "bool_01": [1, 1, 0, 1, 1, 0, 0, 0, 1],
        "bool_01_with_nan": [1, 0, 1, 0, 0, 1, 1, 0, np.nan],
        "mixed": [1, 2, "a", 4, 5, 6, 7, 8, 9],
    }

    df_1 = pd.DataFrame(data_1)

    df_1_expected: Dict[str, Dict[str, Union[Dict[Any, Any], Tuple[Any, Any]]]] = {
        "bool_01": {"bar_plot": {0: 4, 1: 5}},
        "bool_01_with_nan": {"bar_plot": {0.0: 4, 1.0: 4}},
        "bool_tf": {"bar_plot": {False: 3, True: 6}},
        "bool_tf_with_nan": {"bar_plot": {False: 5, True: 3}},
        "s1": {"bar_plot": {1.0: 9}},
        "x": {
            "histogram": (
                np.array([1, 3, 1, 0, 1, 0, 0, 0, 0, 2], dtype=np.int64),
                np.array(
                    [-10.0, -4.0, 2.0, 8.0, 14.0, 20.0, 26.0, 32.0, 38.0, 44.0, 50.0]
                ),
            )
        },
        "y": {
            "histogram": (
                np.array([6, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64),
                np.array(
                    [
                        -3.14159265,
                        309.37256661,
                        621.88672588,
                        934.40088514,
                        1246.91504441,
                        1559.42920367,
                        1871.94336294,
                        2184.4575222,
                        2496.97168147,
                        2809.48584073,
                        3122.0,
                    ]
                ),
            )
        },
    }
    returned_1: List[Intermediate] = plot(
        df_1, force_cat=["bool_01", "bool_01_with_nan", "s1"]
    )

    # TESTING
    for intermediate in returned_1:
        result = intermediate.result
        field = intermediate.raw_data["col_x"]
        if field in df_1_expected:
            LOGGER.info("Testing %s", field)
            if "histogram" in result:
                assert np.allclose(
                    result["histogram"][0],
                    df_1_expected[field]["histogram"][0],
                    equal_nan=True,
                )
                assert np.allclose(
                    result["histogram"][1],
                    df_1_expected[field]["histogram"][1],
                    equal_nan=True,
                )
            else:
                assert result == df_1_expected[intermediate.raw_data["col_x"]]
            LOGGER.info("....Checked")

    # TEST - 2 for plot(df, x, y)
    data_2 = {
        "id": [chr(97 + c) for c in range(1, 21)],
        "x": [
            "d",
            "c",
            "b",
            "a",
            "b",
            "d",
            "c",
            "a",
            "a",
            "a",
            "c",
            "b",
            "c",
            "a",
            "d",
            "b",
            "b",
            "b",
            "b",
            "b",
        ],
        "y": [
            794,
            652,
            158,
            134,
            448,
            682,
            135,
            795,
            353,
            395,
            403,
            498,
            622,
            80,
            654,
            772,
            867,
            676,
            670,
            736,
        ],
        "s1": np.ones(20),
        "somedate": [
            datetime.date(2011, 7, 4),
            datetime.datetime(1898, 1, 2),
            datetime.datetime(1950, 12, 9),
            datetime.datetime(1950, 12, 9),
            datetime.datetime(1898, 1, 2),
            datetime.datetime(1990, 12, 9),
            np.nan,
            datetime.datetime(1990, 12, 9),
            datetime.datetime(1950, 12, 9),
            datetime.datetime(1898, 1, 2),
            datetime.datetime(1950, 12, 9),
            datetime.datetime(1950, 12, 9),
            datetime.datetime(1950, 12, 9),
            datetime.datetime(1950, 12, 9),
            datetime.datetime(1950, 12, 9),
            datetime.datetime(1950, 12, 9),
            datetime.datetime(1950, 12, 9),
            datetime.datetime(1950, 12, 9),
            datetime.datetime(1950, 12, 9),
            datetime.datetime(1950, 12, 9),
        ],
    }

    df_2 = pd.DataFrame(data_2)

    df_2_expected: Dict[str, Dict[str, Any]] = {
        "box_plot": {
            "a": {
                "tf": 134.0,
                "fy": 353.0,
                "sf": 395.0,
                "iqr": 261.0,
                "max": 395,
                "min": 80,
                "max_outlier": 795,
                "outliers": [795],
            },
            "b": {
                "tf": 473.0,
                "fy": 673.0,
                "sf": 754.0,
                "iqr": 281.0,
                "max": 867,
                "min": 158,
                "max_outlier": np.nan,
                "outliers": [],
            },
            "c": {
                "tf": 269.0,
                "fy": 512.5,
                "sf": 637.0,
                "iqr": 368.0,
                "max": 652,
                "min": 135,
                "max_outlier": np.nan,
                "outliers": [],
            },
            "d": {
                "tf": 668.0,
                "fy": 682.0,
                "sf": 738.0,
                "iqr": 70.0,
                "max": 794,
                "min": 654,
                "max_outlier": np.nan,
                "outliers": [],
            },
        },
        "histogram": {
            "d": (
                np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64),
                np.array(
                    [
                        654.0,
                        668.0,
                        682.0,
                        696.0,
                        710.0,
                        724.0,
                        738.0,
                        752.0,
                        766.0,
                        780.0,
                        794.0,
                    ]
                ),
            ),
            "c": (
                np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 2], dtype=np.int64),
                np.array(
                    [
                        135.0,
                        186.7,
                        238.4,
                        290.1,
                        341.8,
                        393.5,
                        445.2,
                        496.9,
                        548.6,
                        600.3,
                        652.0,
                    ]
                ),
            ),
            "b": (
                np.array([1, 0, 0, 0, 2, 0, 0, 2, 2, 1], dtype=np.int64),
                np.array(
                    [
                        158.0,
                        228.9,
                        299.8,
                        370.7,
                        441.6,
                        512.5,
                        583.4,
                        654.3,
                        725.2,
                        796.1,
                        867.0,
                    ]
                ),
            ),
            "a": (
                np.array([2, 0, 0, 1, 1, 0, 0, 0, 0, 1], dtype=np.int64),
                np.array(
                    [
                        80.0,
                        151.5,
                        223.0,
                        294.5,
                        366.0,
                        437.5,
                        509.0,
                        580.5,
                        652.0,
                        723.5,
                        795.0,
                    ]
                ),
            ),
        },
    }
    returned_2: List[Intermediate] = plot(df_2, "y", "x")

    # TESTING
    for intermediate in returned_2:
        result = intermediate.result
        field_x = intermediate.raw_data["col_x"]
        field_y = intermediate.raw_data["col_y"]
        LOGGER.info("Testing %s and %s", field_x, field_y)
        if "box_plot" in result:
            assert result["box_plot"]["a"] == df_2_expected["box_plot"]["a"]
            assert result["box_plot"]["b"] == df_2_expected["box_plot"]["b"]
            assert result["box_plot"]["c"] == df_2_expected["box_plot"]["c"]
            assert result["box_plot"]["d"] == df_2_expected["box_plot"]["d"]
            LOGGER.info("....Checked.")
        elif "histogram" in result:
            assert np.allclose(
                result["histogram"]["a"][0],
                df_2_expected["histogram"]["a"][0],
                equal_nan=True,
            )
            assert np.allclose(
                result["histogram"]["b"][0],
                df_2_expected["histogram"]["b"][0],
                equal_nan=True,
            )
            assert np.allclose(
                result["histogram"]["c"][0],
                df_2_expected["histogram"]["c"][0],
                equal_nan=True,
            )
            assert np.allclose(
                result["histogram"]["d"][0],
                df_2_expected["histogram"]["d"][0],
                equal_nan=True,
            )
            LOGGER.info("....Checked.")

    # TEST - 3 for plot(df, x, y)
    df_2_expected_2 = {
        "stacked_column_plot": {
            ("a", Timestamp("1898-01-02 00:00:00")): 1,
            ("a", Timestamp("1950-12-09 00:00:00")): 3,
            ("a", Timestamp("1990-12-09 00:00:00")): 1,
            ("b", Timestamp("1898-01-02 00:00:00")): 1,
            ("b", Timestamp("1950-12-09 00:00:00")): 7,
            ("c", Timestamp("1898-01-02 00:00:00")): 1,
            ("c", Timestamp("1950-12-09 00:00:00")): 2,
            ("d", Timestamp("1950-12-09 00:00:00")): 1,
            ("d", Timestamp("1990-12-09 00:00:00")): 1,
            ("d", Timestamp("2011-07-04 00:00:00")): 1,
        }
    }
    returned_3: List[Intermediate] = plot(df_2, "x", "somedate")

    # TESTING
    for intermediate in returned_3:
        result = intermediate.result
        LOGGER.info(
            "Testing %s and %s",
            intermediate.raw_data["col_x"],
            intermediate.raw_data["col_y"],
        )
        if "stacked_column_plot" in result:
            assert (
                df_2_expected_2["stacked_column_plot"] == result["stacked_column_plot"]
            )
            LOGGER.info(".....Checked.")


# def test_normal_2() -> None:
#     """
#     test normal 2
#     :return:
#     """
#     # TEST - 4 for plot(df, x)
#     data_3 = {"num_col": np.random.rand(10), "cat_col": [random.choice(["a", "b", "c", "d"])
#                                                          for _ in range(10)]}
#     df_3 = pd.DataFrame(data_3)

#     returned_4: List[Intermediate] = plot(df_3, "num_col")

#     mat_box = plt.boxplot(df_3["num_col"])
#     df_3_expected = {"qqnorm_plot": {"sample": sm.probplot(df_3["num_col"])[0][0],
#                                      "theory": sm.probplot(df_3["num_col"])[0][1]},
#                      "histogram": np.histogram(df_3["num_col"]),
#                      "box_plot": {"tf": mat_box["boxes"][0].get_xydata()[0][1],
#                                   "sf": mat_box["boxes"][0].get_xydata()[2][1],
#                                   "fy": mat_box["medians"][0].get_xydata()[0][1],
#                                   "outliers": list(mat_box["fliers"][0].get_xydata()[:, 1])},
#                      "bar_plot": dict(df_3.groupby(["cat_col"])["cat_col"].count())
#                      }

#     for intermediate in returned_4:
#         result = intermediate.result
#         col_x = intermediate.raw_data["col_x"]
#         LOGGER.info("Testing %s", intermediate.raw_data["col_x"])
#         if "qqnorm_plot" in result:
#             assert np.allclose(result["qqnorm_plot"]["sample"],
#                                df_3_expected["qqnorm_plot"]["sample"], 0.1, 0.1)
#             assert np.allclose(result["qqnorm_plot"]["theory"],
#                                df_3_expected["qqnorm_plot"]["theory"], 0.1, 0.1)
#             LOGGER.info(".....Checked.")
#         elif "histogram" in result:
#             assert np.allclose(result["histogram"][0], df_3_expected["histogram"][0], 0.1, 0.1)
#             assert np.allclose(result["histogram"][1], df_3_expected["histogram"][1], 0.1, 0.1)
#             LOGGER.info(".....Checked.")
#         elif "box_plot" in result:
#             assert np.isclose(result["box_plot"][col_x]["tf"],
#                               np.round(df_3_expected["box_plot"]["tf"], 2), 0.1, 0.1)
#             assert np.isclose(result["box_plot"][col_x]["fy"],
#                               np.round(df_3_expected["box_plot"]["fy"], 2), 0.1, 0.1)
#             assert np.isclose(result["box_plot"][col_x]["sf"],
#                               np.round(df_3_expected["box_plot"]["sf"], 2), 0.1, 0.1)
#             assert np.allclose(result["box_plot"][col_x]["outliers"],
#                                df_3_expected["box_plot"]["outliers"], 0.1, 0.1)
#             LOGGER.info(".....Checked.")
#         elif "bar_plot" in result:
#             assert result["bar_plot"] == df_3_expected["bar_plot"]
#             LOGGER.info(".....Checked.")


# def test_corner() -> None:
#     """

#     :return:
#     """
#     df_2 = pd.DataFrame(
#         {"all_nan": [np.nan for _ in range(10)], "all_one": np.ones(10),
#          "all_zeros": np.zeros(10), "random": np.array(
#              [0.38538395, 0.13609054, 0.15973238, 0.96192966, 0.03708882,
#               0.03633855, 0.25260128, 0.72139843, 0.74553949,
#               0.41102021])})

#     df_1_expected = {"all_one": {"bar_plot": {1.0: 10}},
#                      "all_zeros": {"bar_plot": {0.0: 10}},
#                      "random": {"bar_plot": np.array([2, 2, 1, 1, 1, 0, 0, 2, 0, 1],
#                                                      dtype=np.int64)}}

#     res = plot(df_2, force_cat=["all_one", "all_zeros"])

#     assert res["all_one"] == df_1_expected["all_one"]
#     assert res["all_zeros"] == df_1_expected["all_zeros"]

#     df_2 = pd.DataFrame({
#         "empty": [],
#         "another_empty": []
#     })

#     df_2_expected: Dict[str, Any] = {'scatter_plot': {}}

#     res = plot(df_2, "empty", "another_empty")
#     assert res == df_2_expected
