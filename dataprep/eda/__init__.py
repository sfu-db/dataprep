"""Docstring
    Data preparation module
"""
import tempfile

from bokeh.io import output_file, output_notebook
from .basic import compute, plot, render
from .correlation import compute_correlation, plot_correlation, render_correlation
from .missing import compute_missing, plot_missing, render_missing
from .utils import is_notebook, rand_str

__all__ = [
    "plot_correlation",
    "compute_correlation",
    "render_correlation",
    "compute_missing",
    "render_missing",
    "plot_missing",
    "plot",
    "compute",
    "render",
]

if is_notebook():
    output_notebook(hide_banner=True)
else:
    output_file(filename=tempfile.gettempdir() + "/" + rand_str() + ".html")
