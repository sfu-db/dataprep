"""
This file defines palettes used for EDA.
"""
# pylint: disable=no-name-in-module
from bokeh.palettes import Category20  # type: ignore
from holoviews.plotting.util import process_cmap

PALETTE = Category20[20]
BIPALETTE = list(reversed(process_cmap("RdBu")))
BRG = ["#1f78b4", "#d62728", "#2ca02c"]
