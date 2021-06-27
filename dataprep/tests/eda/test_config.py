"""
    This module for testing config parameter
"""
import dask.dataframe as dd
import pandas as pd
import numpy as np
import random
import pytest

from ...eda import plot, plot_correlation, plot_missing
from ...eda.utils import to_dask


@pytest.fixture(scope="module")
def simpledf() -> dd.DataFrame:
    df = pd.DataFrame(np.random.rand(1000, 3), columns=["a", "b", "c"])
    df = pd.concat([df, pd.Series(np.random.choice(["a", "b", "c"], 1000, replace=True))], axis=1)
    df = pd.concat([df, pd.Series(np.random.choice(["a", "d"], 1000, replace=True))], axis=1)
    df.columns = ["a", "b", "c", "d"]
    idx = np.arange(1000)
    np.random.shuffle(idx)
    df.iloc[idx[:500], 0] = None
    df = to_dask(df)
    return df


def test_sanity_compute_1(simpledf: dd.DataFrame) -> None:
    for _ in range(10):
        hist_bins = random.randint(20, 50)
        bar_bars = random.randint(20, 50)
        kde_bins = random.randint(20, 50)
        wordfreq_top_words = random.randint(20, 50)
        heatmap_ngroups = random.randint(20, 50)
        plot(simpledf, config={"hist.bins": hist_bins, "hist.yscale": "log"})
        plot(simpledf, config={"bar.bars": bar_bars, "bar.yscale": "log", "bar.color": "#123456"})
        plot(simpledf, config={"kde.bins": kde_bins, "kde.yscale": "log"})
        plot(simpledf, config={"wordfreq.top_words": wordfreq_top_words})
        plot(simpledf, config={"heatmap.ngroups": heatmap_ngroups})


def test_sanity_compute_2(simpledf: dd.DataFrame) -> None:
    for _ in range(10):
        spectrum_bins = random.randint(5, 20)
        plot_missing(simpledf, config={"spectrum.bins": spectrum_bins})
