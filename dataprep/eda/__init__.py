"""Docstring
    Data preparation module
"""
import tempfile

from bokeh.io import output_file, output_notebook

from ..utils import _rand_str, is_notebook
from .basic import plot

from .correlation import compute_correlation, plot_correlation, render_correlation
from .missing import plot_missing

__all__ = [
    "plot",
    "plot_correlation",
    "compute_correlation",
    "render_correlation",
    "plot_missing",
]

if is_notebook():
    output_notebook(hide_banner=True)
else:
    output_file(filename=tempfile.gettempdir() + "/" + _rand_str() + ".html")
