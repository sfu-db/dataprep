"""
    module for testing plot_corr(df, x, y) function.
"""
import random
from time import time

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from ...eda.correlation import compute_correlation, to_dask


@pytest.fixture(scope="module")  # mypy: diable
def simpledf() -> dd.DataFrame:
    df = pd.DataFrame(np.random.rand(100, 3), columns=["a", "b", "c"])
    df = pd.concat([df, pd.Series(["a"] * 100)], axis=1)
    df.columns = ["a", "b", "c", "d"]
    df = to_dask(df)

    return df


def sanity_compute_1(simpledf: dd.DataFrame) -> None:
    df = simpledf
    compute_correlation(df)


def sanity_compute_2(simpledf: dd.DataFrame) -> None:
    df = simpledf
    compute_correlation(df, k=1)


def sanity_compute_3(simpledf: dd.DataFrame) -> None:
    df = simpledf
    compute_correlation(df, x="a")


def sanity_compute_4(simpledf: dd.DataFrame) -> None:
    df = simpledf
    compute_correlation(df, x="a", value_range=(0.5, 0.8))


def sanity_compute_5(simpledf: dd.DataFrame) -> None:
    df = simpledf
    compute_correlation(df, x="a", k=1)


def sanity_compute_6(simpledf: dd.DataFrame) -> None:
    df = simpledf
    compute_correlation(df, x="a", k=0)


def sanity_compute_7(simpledf: dd.DataFrame) -> None:
    df = simpledf
    compute_correlation(df, x="b", y="a")


def sanity_compute_8(simpledf: dd.DataFrame) -> None:
    df = simpledf
    compute_correlation(df, x="b", y="a", k=1)


@pytest.mark.xfail
def sanity_compute_fail_1(simpledf: dd.DataFrame) -> None:
    df = simpledf
    compute_correlation(df, value_range=(0.3, 0.7))


@pytest.mark.xfail
def sanity_compute_fail_2(simpledf: dd.DataFrame) -> None:
    df = simpledf
    compute_correlation(df, k=3, value_range=(0.3, 0.7))


@pytest.mark.xfail
def sanity_compute_fail_3(simpledf: dd.DataFrame) -> None:
    df = simpledf
    compute_correlation(df, x="a", value_range=(0.5, 0.8), k=3)


@pytest.mark.xfail
def sanity_compute_fail_4(simpledf: dd.DataFrame) -> None:
    df = simpledf
    compute_correlation(df, y="a")


@pytest.mark.xfail
def sanity_compute_fail_5(simpledf: dd.DataFrame) -> None:
    df = simpledf
    compute_correlation(df, x="d")


@pytest.mark.xfail
def sanity_compute_fail_6(simpledf: dd.DataFrame) -> None:
    df = simpledf
    compute_correlation(df, x="b", y="a", value_range=(0.5, 0.8))


@pytest.mark.xfail
def sanity_compute_fail_7(simpledf: dd.DataFrame) -> None:
    df = simpledf
    compute_correlation(df, x="b", y="a", value_range=(0.5, 0.8), k=3)


# def test_plot_corr_df() -> None:  # pylint: disable=too-many-locals
#     """
#     :return:
#     """
#     data = np.random.rand(100, 20)
#     df_data = pd.DataFrame(data)

#     start_p_pd = time()
#     res = df_data.corr(method="pearson")
#     end_p_pd = time()
#     print("pd pearson time: ", str(end_p_pd - start_p_pd) + " s")

#     start_p = time()
#     _, intermediate = plot_correlation(df=df_data, return_intermediate=True)
#     end_p = time()
#     print("our pearson time: ", str(end_p - start_p) + " s")
#     assert np.isclose(res, intermediate.result["corr_p"]).all()

#     start_s_pd = time()
#     res = df_data.corr(method="spearman")
#     end_s_pd = time()
#     print("pd spearman time: ", str(end_s_pd - start_s_pd) + " s")

#     start_s = time()
#     _, intermediate = plot_correlation(df=df_data, return_intermediate=True)
#     end_s = time()
#     print("our spearman time: ", str(end_s - start_s) + " s")
#     assert np.isclose(res, intermediate.result["corr_s"]).all()

#     start_k_pd = time()
#     res = df_data.corr(method="kendall")
#     end_k_pd = time()
#     print("pd kendall time: ", str(end_k_pd - start_k_pd) + " s")

#     start_k = time()
#     _, intermediate = plot_correlation(df=df_data, return_intermediate=True)
#     end_k = time()
#     print("our kendall time: ", str(end_k - start_k) + " s")
#     assert np.isclose(res, intermediate.result["corr_k"]).all()


# def test_plot_corr_df_k() -> None:
#     """
#     :return:
#     """
#     data = np.random.rand(100, 20)
#     df_data = pd.DataFrame(data)
#     k = 5
#     res = df_data.corr(method="pearson")
#     row, _ = np.shape(res)
#     res_re = np.reshape(np.triu(res, 1), (row * row,))
#     idx = np.argsort(np.absolute(res_re))
#     mask = np.zeros(shape=(row * row,))
#     for i in range(k):
#         mask[idx[-i - 1]] = 1
#     res = np.multiply(res_re, mask)
#     res = np.reshape(res, (row, row))
#     res = res.T
#     _, intermediate = plot_correlation(df=df_data, return_intermediate=True, k=k)
#     assert np.isclose(intermediate.result["corr_p"], res).all()
#     assert np.isclose(intermediate.result["mask_p"], mask).all()


# def test_plot_corr_df_x_k() -> None:
#     """
#     :return:
#     """
#     df_data = pd.DataFrame({"a": np.random.normal(0, 10, 100)})
#     df_data["b"] = df_data["a"] + np.random.normal(0, 10, 100)
#     df_data["c"] = df_data["a"] + np.random.normal(0, 10, 100)
#     df_data["d"] = df_data["a"] + np.random.normal(0, 10, 100)
#     x_name = "b"
#     k = 3
#     name_list = list(df_data.columns.values)
#     idx_name = name_list.index(x_name)
#     res_p = df_data.corr(method="pearson").values
#     res_p[idx_name][idx_name] = -1
#     res_s = df_data.corr(method="spearman").values
#     res_s[idx_name][idx_name] = -1
#     res_k = df_data.corr(method="kendall").values
#     res_k[idx_name][idx_name] = -1
#     _, _ = plot_correlation(df=df_data, x=x_name, return_intermediate=True, k=k)


# def test_plot_corr_df_x_y_k() -> None:
#     """
#     :return:
#     """
#     df_data = pd.DataFrame({"a": np.random.normal(0, 10, 100)})
#     df_data["b"] = df_data["a"] + np.random.normal(0, 10, 100)
#     df_data["c"] = df_data["a"] + np.random.normal(0, 10, 100)
#     df_data["d"] = df_data["a"] + np.random.normal(0, 10, 100)
#     x_name = "b"
#     y_name = "c"
#     k = 3
#     _ = plot_correlation(
#         df=df_data, x=x_name, y=y_name, return_intermediate=False, k=k,
#     )

#     letters = ["a", "b", "c"]
#     df_data_cat = pd.DataFrame({"a": np.random.normal(0, 10, 100)})
#     df_data_cat["b"] = pd.Categorical([random.choice(letters) for _ in range(100)])
#     df_data_cat["c"] = pd.Categorical([random.choice(letters) for _ in range(100)])
#     _, intermediate = plot_correlation(
#         df=df_data_cat, x="b", y="c", return_intermediate=True
#     )
#     assert np.isclose(
#         pd.crosstab(df_data_cat["b"], df_data_cat["c"]).values,
#         intermediate.result["cross_table"],
#     ).all()
