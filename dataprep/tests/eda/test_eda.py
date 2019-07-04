"""
    module for testing plot(df, x, y) function.
"""
import datetime
from typing import Any, Dict, Union, cast, Tuple

from time import time
import numpy as np
import pandas as pd
from pandas import Timestamp

from ...eda.eda_plot import plot  # dataprep.tests.eda.test_eda
from ...eda.eda_plot_corr import plot_correlation


def test_normal() -> None:
    """

    :return:
    """
    data_1 = {

        "id": [chr(97 + c) for c in range(1, 10)],

        "x": [50, 50, -10, 0, 0, 5, 15, -3, None],

        "y": [0.000001, 654.152, None, 15.984512, 3122, -3.1415926535, 111,
              15.9, 13.5],

        "s1": np.ones(9),

        "somedate": [datetime.date(2011, 7, 4),
                     datetime.datetime(2022, 1, 1, 13, 57),
                     datetime.datetime(1990, 12, 9), np.nan,
                     datetime.datetime(1990, 12, 9),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1898, 1, 2),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1950, 12, 9)],

        "bool_tf": [True, True, False, True, False, True, True, False,
                    True],

        "bool_tf_with_nan": [True, False, False, False, False, True, True,
                             False, np.nan],

        "bool_01": [1, 1, 0, 1, 1, 0, 0, 0, 1],

        "bool_01_with_nan": [1, 0, 1, 0, 0, 1, 1, 0, np.nan],

        "mixed": [1, 2, "a", 4, 5, 6, 7, 8, 9]

    }

    df_1 = pd.DataFrame(data_1)

    df_1_expected: Dict[str, Dict[str, Union[Dict[Any, Any], Tuple[Any, Any]]]] = \
                    {"bool_01": {"bar_plot": {0: 4, 1: 5}},
                     "bool_01_with_nan": {"bar_plot": {0.0: 4, 1.0: 4}},
                     "bool_tf": {"bar_plot": {False: 3, True: 6}},
                     "bool_tf_with_nan": {"bar_plot": {False: 5, True: 3}},
                     "s1": {"bar_plot": {1.0: 9}},
                     "x": {"histogram": (np.array([1, 3, 1, 0, 1, 0, 0, 0, 0, 2], dtype=np.int64),
                                         np.array([-10., -4., 2., 8., 14., 20., 26.,
                                                   32., 38., 44., 50.]))},
                     'y': {'histogram': (np.array([6, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64),
                                         np.array([-3.14159265, 309.37256661, 621.88672588,
                                                   934.40088514, 1246.91504441, 1559.42920367,
                                                   1871.94336294, 2184.4575222, 2496.97168147,
                                                   2809.48584073, 3122.]))}
                    }
    res = cast(Dict[str, Dict[str, Union[Dict[Any, Any], Tuple[Any, Any]]]], plot(df_1,
                                                                                  force_cat=[
                                                                                      "bool_01",
                                                                                      "bool_01_ \
                                                                                       with_nan",
                                                                                      "s1"]))

    assert res["bool_01"] == df_1_expected["bool_01"]
    assert res["bool_01_with_nan"] == df_1_expected["bool_01_with_nan"]
    assert res["bool_tf"] == df_1_expected["bool_tf"]
    assert res["bool_tf_with_nan"] == df_1_expected["bool_tf_with_nan"]
    assert res["s1"] == df_1_expected["s1"]
    assert np.allclose(res["x"]["histogram"][0], df_1_expected["x"]["histogram"][0], equal_nan=True)
    assert np.allclose(res["x"]["histogram"][1], df_1_expected["x"]["histogram"][1], equal_nan=True)
    assert np.allclose(res["y"]["histogram"][0], df_1_expected["y"]["histogram"][0], equal_nan=True)
    assert np.allclose(res["y"]["histogram"][1], df_1_expected["y"]["histogram"][1], equal_nan=True)

    data = {

        "id": [chr(97 + c) for c in range(1, 21)],

        "x": ["d", "c", "b", "a", "b", "d", "c", "a", "a", "a", "c", "b",
              "c", "a", "d", "b", "b", "b", "b", "b"],

        "y": [794, 652, 158, 134, 448, 682, 135, 795, 353, 395, 403, 498,
              622, 80, 654, 772, 867, 676, 670, 736],

        "s1": np.ones(20),

        "somedate": [datetime.date(2011, 7, 4),
                     datetime.datetime(1898, 1, 2),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1898, 1, 2),
                     datetime.datetime(1990, 12, 9), np.nan,
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
                     datetime.datetime(1950, 12, 9)],
    }

    df_data = pd.DataFrame(data)

    df_expected: Dict[str, Dict[str, Any]] \
                = {"box_plot": {"a": {"25%": 134.0,
                                      "50%": 353.0,
                                      "75%": 395.0,
                                      "iqr": 261.0,
                                      "max": 395,
                                      "min": 80,
                                      "outliers": [795]},
                                "b": {"25%": 485.5,
                                      "50%": 673.0,
                                      "75%": 745.0,
                                      "iqr": 259.5,
                                      "max": 867,
                                      "min": 158,
                                      "outliers": []},
                                "c": {"25%": 336.0,
                                      "50%": 512.5,
                                      "75%": 629.5,
                                      "iqr": 293.5,
                                      "max": 652,
                                      "min": 135,
                                      "outliers": []},
                                "d": {"25%": 668.0,
                                      "50%": 682.0,
                                      "75%": 738.0,
                                      "iqr": 70.0,
                                      "max": 794,
                                      "min": 654,
                                      "outliers": []}},
                   "histogram": {"d": (np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64),
                                       np.array([654., 668., 682., 696., 710., 724., 738., 752.,
                                                 766., 780., 794.])),
                                 "c": (np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 2], dtype=np.int64),
                                       np.array([135., 186.7, 238.4, 290.1, 341.8, 393.5,
                                                 445.2, 496.9, 548.6, 600.3, 652.])),
                                 "b": (np.array([1, 0, 0, 0, 2, 0, 0, 2, 2, 1], dtype=np.int64),
                                       np.array([158., 228.9, 299.8, 370.7, 441.6, 512.5, 583.4,
                                                 654.3, 725.2, 796.1, 867.])),
                                 "a": (np.array([2, 0, 0, 1, 1, 0, 0, 0, 0, 1], dtype=np.int64),
                                       np.array([80., 151.5, 223., 294.5, 366., 437.5, 509.,
                                                 580.5, 652., 723.5, 795.]))}
                  }
    another_res = cast(Dict[str, Dict[str, Any]], plot(df_data, "y", "x"))

    assert another_res["box_plot"]["a"] == df_expected["box_plot"]["a"]
    assert another_res["box_plot"]["b"] == df_expected["box_plot"]["b"]
    assert another_res["box_plot"]["c"] == df_expected["box_plot"]["c"]
    assert another_res["box_plot"]["d"] == df_expected["box_plot"]["d"]

    assert np.allclose(another_res["histogram"]["a"][0], df_expected["histogram"]["a"][0],
                       equal_nan=True)
    assert np.allclose(another_res["histogram"]["b"][0], df_expected["histogram"]["b"][0],
                       equal_nan=True)
    assert np.allclose(another_res["histogram"]["c"][0], df_expected["histogram"]["c"][0],
                       equal_nan=True)
    assert np.allclose(another_res["histogram"]["d"][0], df_expected["histogram"]["d"][0],
                       equal_nan=True)

    df_expected_2 = {"stacked_column_plot": {("a", Timestamp("1898-01-02 00:00:00")): 1,
                                             ("a", Timestamp("1950-12-09 00:00:00")): 3,
                                             ("a", Timestamp("1990-12-09 00:00:00")): 1,
                                             ("b", Timestamp("1898-01-02 00:00:00")): 1,
                                             ("b", Timestamp("1950-12-09 00:00:00")): 7,
                                             ("c", Timestamp("1898-01-02 00:00:00")): 1,
                                             ("c", Timestamp("1950-12-09 00:00:00")): 2,
                                             ("d", Timestamp("1950-12-09 00:00:00")): 1,
                                             ("d", Timestamp("1990-12-09 00:00:00")): 1,
                                             ("d", Timestamp("2011-07-04 00:00:00")): 1
                                            }
                    }

    res_2 = plot(df_data, "x", "somedate")
    assert df_expected_2["stacked_column_plot"] == res_2["stacked_column_plot"]


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


def test_plot_corr_df() -> None:
    """
    :return:
    """
    df_data = pd.DataFrame({'a': np.random.normal(0, 10, 100)})
    df_data['b'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['c'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['d'] = df_data['a'] + np.random.normal(0, 10, 100)

    start_p_pd = time()
    res = df_data.corr(method='pearson')
    end_p_pd = time()
    print("pd pearson time: ", str(end_p_pd - start_p_pd) + " s")
    print("pd pearson: \n", res)

    start_p = time()
    _, res = plot_correlation(df_data, method='pearson',
                              show_intermediate=True)
    end_p = time()
    print("our pearson time: ", str(end_p - start_p) + " s")
    print("our pearson: \n", res['corr'])

    start_s_pd = time()
    res = df_data.corr(method='spearman')
    end_s_pd = time()
    print("pd spearman time: ", str(end_s_pd - start_s_pd) + " s")
    print("pd spearman: \n", res)

    start_s = time()
    _, res = plot_correlation(df_data, method='spearman',
                              show_intermediate=True)
    end_s = time()
    print("our spearman time: ", str(end_s - start_s) + " s")
    print("our spearman: \n", res['corr'])

    start_k_pd = time()
    res = df_data.corr(method='kendall')
    end_k_pd = time()
    print("pd kendall time: ", str(end_k_pd - start_k_pd) + " s")
    print("pd kendall: \n", res)

    start_k = time()
    _, res = plot_correlation(df_data, method='kendall',
                              show_intermediate=True)
    end_k = time()
    print("our kendall time: ", str(end_k - start_k) + " s")
    print("our kendall: \n", res['corr'])


def test_plot_corr_df_k() -> None:
    """
    :return:
    """
    df_data = pd.DataFrame({'a': np.random.normal(0, 10, 100)})
    df_data['b'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['c'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['d'] = df_data['a'] + np.random.normal(0, 10, 100)
    k = 5
    res = df_data.corr(method='pearson')
    print("df: \n", res)
    _, res = plot_correlation(pd_data_frame=df_data, k=k,
                              show_intermediate=True)
    print("result: \n", res['corr'])


def test_plot_corr_df_x_k() -> None:
    """
    :return:
    """
    df_data = pd.DataFrame({'a': np.random.normal(0, 10, 100)})
    df_data['b'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['c'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['d'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['e'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['f'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['g'] = df_data['a'] + np.random.normal(0, 10, 100)
    x_name = 'b'
    res = df_data.corr(method='pearson')
    print("pearson: \n", res)
    res = df_data.corr(method='spearman')
    print("spearman: \n", res)
    res = df_data.corr(method='kendall')
    print("kendall: \n", res)
    k = 3
    _, res = plot_correlation(pd_data_frame=df_data, x_name=x_name, k=k,
                              show_intermediate=True)
    print("top-k pearson: ", res['pearson'])
    print("top-k spearman: ", res['spearman'])
    print("top-k kendall: ", res['kendall'])


def test_plot_corr_df_x_y_k() -> None:
    """
    :return:
    """
    df_data = pd.DataFrame({'a': np.random.normal(0, 10, 100)})
    df_data['b'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['c'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['d'] = df_data['a'] + np.random.normal(0, 10, 100)
    x_name = 'b'
    y_name = 'c'
    k = 3
    _, res = plot_correlation(pd_data_frame=df_data,
                              x_name=x_name, y_name=y_name, k=k,
                              show_intermediate=True)
    print(res)

    df_data_cat = pd.DataFrame({'a': np.random.normal(0, 10, 5)})
    df_data_cat['b'] = pd.Categorical(['a', 'b', 'b', 'a', 'c'])
    df_data_cat['c'] = pd.Categorical(['a', 'b', 'a', 'b', 'a'])
    print(pd.crosstab(df_data_cat['b'], df_data_cat['c']))
    _, res = plot_correlation(pd_data_frame=df_data_cat,
                              x_name='b', y_name='c',
                              show_intermediate=True)
    print(res)
