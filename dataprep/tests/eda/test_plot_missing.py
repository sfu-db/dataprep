"""
    This module for testing plot_missing(df, x, y) function.
"""
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest

from ...eda.dtypes import Numerical
from ...eda.missing import compute_missing, render_missing, plot_missing
from ...eda.utils import to_dask
from ...eda.configs import Config


@pytest.fixture(scope="module")  # type: ignore
def simpledf() -> dd.DataFrame:
    df = pd.DataFrame(np.random.rand(1000, 3), columns=["a", "b", "c"])

    df = pd.concat([df, pd.Series(np.random.choice(["a", "b", "c"], 1000, replace=True))], axis=1)

    df.columns = ["a", "b", "c", "d"]
    idx = np.arange(1000)
    np.random.shuffle(idx)
    df.iloc[idx[:500], 0] = None

    ddf = to_dask(df)

    return ddf


def test_sanity_compute_1(simpledf: dd.DataFrame) -> None:
    display = ["Stats", "Bar Chart", "Spectrum"]
    config = {
        "spectrum.bins": 10,
        "height": 500,
        "width": 500,
    }
    cfg = Config.from_dict(display=display, config=config)

    itmdt = compute_missing(simpledf, cfg=cfg)
    render_missing(itmdt, cfg)


def test_sanity_compute_2(simpledf: dd.DataFrame) -> None:
    config = {"hist.bins": 20, "bar.bars": 15}
    cfg = Config.from_dict(config=config)

    itmdt = compute_missing(simpledf, x="a", cfg=cfg)
    render_missing(itmdt, cfg)


def test_sanity_compute_3(simpledf: dd.DataFrame) -> None:
    config = {"hist.bins": 20, "bar.bars": 15}
    cfg = Config.from_dict(config=config)
    itmdt = compute_missing(simpledf, x="d", cfg=cfg)
    render_missing(itmdt, cfg)


def test_sanity_compute_4(simpledf: dd.DataFrame) -> None:
    display = ["Histogram", "PDF"]
    config = {
        "hist.bins": 30,
        "hist.yscale": "linear",
        "height": 500,
        "width": 500,
    }
    cfg = Config.from_dict(display=display, config=config)
    itmdt = compute_missing(simpledf, x="a", y="b", cfg=cfg)
    render_missing(itmdt, cfg)


def test_sanity_compute_5(simpledf: dd.DataFrame) -> None:
    display = ["Histogram", "PDF"]
    config = {
        "hist.bins": 30,
        "hist.yscale": "linear",
        "height": 500,
        "width": 500,
    }
    cfg = Config.from_dict(display=display, config=config)
    itmdt = compute_missing(simpledf, x="a", y="d", cfg=cfg)
    render_missing(itmdt, cfg)


def test_specify_column_type(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    itmdt = compute_missing(simpledf, x="b", cfg=cfg, dtype={"a": Numerical()})
    render_missing(itmdt, cfg)


@pytest.mark.xfail  # type: ignore
def test_sanity_compute_6(simpledf: dd.DataFrame) -> None:
    cfg = Config.from_dict()
    compute_missing(simpledf, y="b", cfg=cfg)


def test_sanity_compute_7() -> None:
    df = pd.DataFrame([[1, 2, 3]])
    simpledf = to_dask(df)
    cfg = Config.from_dict()
    itmdt = compute_missing(simpledf, cfg=cfg)
    render_missing(itmdt, cfg)


def test_no_missing() -> None:
    from sys import platform

    if platform == "darwin" or platform == "win32":
        import matplotlib

        matplotlib.use("PS")
    import seaborn as sns

    df = sns.load_dataset("iris")
    plot_missing(df)
