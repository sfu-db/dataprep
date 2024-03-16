"""
    module for testing plot_corr(df, x, y) function.
"""

import random
from time import time

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from ...eda.correlation import compute_correlation, plot_correlation
from ...eda.correlation.compute.univariate import (
    _kendall_tau_1xn,
    _pearson_1xn,
    _spearman_1xn,
    _corr_filter,
)
from ...eda.correlation.compute.overview import (
    _spearman_nxn,
    _pearson_nxn,
    _kendall_tau_nxn,
)
from ...eda.utils import to_dask
from ...eda.eda_frame import EDAFrame
from ...eda.configs import Config
from .random_data_generator import random_df


@pytest.fixture(scope="module")  # type: ignore
def simpledf() -> dd.DataFrame:
    df = pd.DataFrame(np.random.rand(100, 3), columns=["a", "b", "c"])
    df = pd.concat([df, pd.Series(["a"] * 100)], axis=1)
    df.columns = ["a", "b", "c", "d"]
    df = to_dask(df)

    return df


def test_random_df(random_df: pd.DataFrame) -> None:
    plot_correlation(random_df)


def test_sanity_compute_1(simpledf: dd.DataFrame) -> None:
    display = ["Stats", "Pearson"]
    config = {
        "height": 500,
        "width": 500,
    }
    cfg = Config.from_dict(display=display, config=config)
    compute_correlation(simpledf, cfg=cfg)
    plot_correlation(simpledf)


def test_sanity_compute_2(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, cfg=cfg, k=1)
    plot_correlation(simpledf, k=1)


def test_sanity_compute_3(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, col1="a", cfg=cfg)
    plot_correlation(simpledf, col1="a")


def test_sanity_compute_4(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, col1="a", cfg=cfg, value_range=(0.5, 0.8))
    plot_correlation(simpledf, col1="a", value_range=(0.5, 0.8))


def test_sanity_compute_5(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, col1="a", cfg=cfg, k=1)
    plot_correlation(simpledf, col1="a", k=1)


def test_sanity_compute_6(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, col1="a", cfg=cfg, k=0)
    plot_correlation(simpledf, col1="a", k=0)


def test_sanity_compute_7(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, col1="b", cfg=cfg, col2="a")
    plot_correlation(simpledf, col1="b", col2="a")


def test_sanity_compute_8(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, col1="b", cfg=cfg, col2="a", k=1)
    plot_correlation(simpledf, col1="b", col2="a", k=1)


def test_sanity_compute_9(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, cfg=cfg, value_range=(0.3, 0.7))
    plot_correlation(simpledf, value_range=(0.3, 0.7))


@pytest.mark.xfail  # type: ignore
def test_sanity_compute_fail_2(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, cfg=cfg, k=3, value_range=(0.3, 0.7))
    plot_correlation(simpledf, k=3, value_range=(0.3, 0.7))


@pytest.mark.xfail  # type: ignore
def test_sanity_compute_fail_3(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, col1="a", cfg=cfg, value_range=(0.5, 0.8), k=3)
    plot_correlation(simpledf, col1="a", value_range=(0.5, 0.8), k=3)


@pytest.mark.xfail  # type: ignore
def test_sanity_compute_fail_4(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, col2="a", cfg=cfg)
    plot_correlation(simpledf, col2="a")


@pytest.mark.xfail  # type: ignore
def test_sanity_compute_fail_5(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, col1="d", cfg=cfg)
    plot_correlation(simpledf, col1="d")


@pytest.mark.xfail  # type: ignore
def test_test_sanity_compute_fail_6(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, col1="b", col2="a", cfg=cfg, value_range=(0.5, 0.8))
    plot_correlation(simpledf, col1="b", col2="a", value_range=(0.5, 0.8))


@pytest.mark.xfail  # type: ignore
def test_sanity_compute_fail_7(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_correlation(simpledf, col1="b", col2="a", cfg=cfg, value_range=(0.5, 0.8), k=3)
    plot_correlation(simpledf, col1="b", col2="a", value_range=(0.5, 0.8), k=3)


def test_compute_pearson(simpledf: dd.DataFrame) -> None:
    df = EDAFrame(simpledf)
    df = df.select_num_columns()
    darray = df.values
    array = darray.compute()

    corr_eda = _pearson_nxn(df).compute()
    corr_pd = pd.DataFrame(data=array).corr("pearson").values
    assert np.isclose(corr_eda, corr_pd).all()

    for i in range(array.shape[1]):
        corr_eda = _pearson_1xn(darray[:, i : i + 1], darray).compute()
        assert np.isclose(_corr_filter(corr_eda)[1], np.sort(corr_pd[:, i])).all()


def test_compute_spearman(simpledf: dd.DataFrame) -> None:
    df = EDAFrame(simpledf)
    df = df.select_num_columns()
    darray = df.values
    array = darray.compute()

    corr_eda = _spearman_nxn(df).compute()
    corr_pd = pd.DataFrame(data=array).corr("spearman").values
    assert np.isclose(corr_eda, corr_pd).all()

    for i in range(array.shape[1]):
        corr_eda = _spearman_1xn(darray[:, i : i + 1], darray).compute()
        assert np.isclose(_corr_filter(corr_eda)[1], np.sort(corr_pd[:, i])).all()


def test_compute_kendall(simpledf: dd.DataFrame) -> None:
    df = EDAFrame(simpledf)
    df = df.select_num_columns()
    darray = df.values
    array = darray.compute()

    corr_eda = _kendall_tau_nxn(df).compute()
    corr_pd = pd.DataFrame(data=array).corr("kendall").values
    assert np.isclose(corr_eda, corr_pd).all()

    for i in range(array.shape[1]):
        corr_eda = _kendall_tau_1xn(darray[:, i : i + 1], darray).compute()
        assert np.isclose(_corr_filter(corr_eda)[1], np.sort(corr_pd[:, i])).all()
