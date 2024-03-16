"""
    This module for testing config parameter
"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
import random
import pytest

from ...eda import (
    plot,
    plot_correlation,
    plot_missing,
    compute,
    compute_correlation,
    compute_missing,
    create_report,
)
from ...eda.configs import Config
from ...eda.utils import to_dask


@pytest.fixture(scope="module")
def simpledf() -> dd.DataFrame:
    df = pd.DataFrame(np.random.rand(1000, 3), columns=["a", "b", "c"])
    df = pd.concat([df, pd.Series(np.random.choice(["a", "b", "c"], 1000, replace=True))], axis=1)
    df = pd.concat([df, pd.Series(np.random.choice(["a", "d"], 1000, replace=True))], axis=1)
    df.columns = ["a", "b", "c", "d", "e"]
    idx = np.arange(1000)
    np.random.shuffle(idx)
    df.iloc[idx[:500], 0] = None
    df = to_dask(df)
    return df


def test_sanity_compute_1(simpledf: dd.DataFrame) -> None:
    for _ in range(5):
        hist_bins = random.randint(20, 50)
        bar_bars = random.randint(20, 50)
        kde_bins = random.randint(20, 50)
        wordfreq_top_words = random.randint(20, 50)
        heatmap_ngroups = random.randint(20, 50)

        config = {"hist.bins": hist_bins, "hist.yscale": "log"}
        compute(simpledf, cfg=config)
        plot(simpledf, config=config)

        config = {"bar.bars": bar_bars, "bar.yscale": "log", "bar.color": "#123456"}
        compute(simpledf, cfg=config)
        plot(simpledf, config=config)

        config = {"kde.bins": kde_bins, "kde.yscale": "log"}
        compute(simpledf, cfg=config)
        plot(simpledf, config=config)

        config = {"wordfreq.top_words": wordfreq_top_words}
        compute(simpledf, cfg=config)
        plot(simpledf, config=config)

        config = {"heatmap.ngroups": heatmap_ngroups}
        compute(simpledf, cfg=config)
        plot(simpledf, config=config)


def test_sanity_compute_2(simpledf: dd.DataFrame) -> None:
    for _ in range(5):
        spectrum_bins = random.randint(5, 20)
        config = {"spectrum.bins": spectrum_bins}
        compute_missing(simpledf, cfg=config)
        plot_missing(simpledf, config=config)


def test_sanity_compute_3(simpledf: dd.DataFrame) -> None:
    config = {"scatter.sample_rate": 0.1}
    compute_correlation(simpledf, col1="a", col2="b", cfg=config)
    plot_correlation(simpledf, "a", "b", config=config)

    config = {"scatter.sample_size": 10000}
    compute_correlation(simpledf, col1="a", col2="b", cfg=config)
    plot_correlation(simpledf, "a", "b", config=config)

    config = {"scatter.sample_size": 100}
    compute_correlation(simpledf, col1="a", col2="b", cfg=config)
    plot_correlation(simpledf, "a", "b", config=config)


def test_report(simpledf: dd.DataFrame) -> None:
    create_report(simpledf, display=["Overview", "Interactions"])
    create_report(simpledf, display=["Interactions"], config={"interactions.cat_enable": True})
