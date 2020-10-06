"""
dataprep.eda
============
"""
from bokeh.io import output_notebook

from .correlation import compute_correlation, plot_correlation, render_correlation
from .create_report import create_report
from .distribution import compute, plot, render
from .dtypes import (
    Categorical,
    Continuous,
    DateTime,
    Discrete,
    DType,
    Nominal,
    Numerical,
    Ordinal,
    Text,
)
from .missing import compute_missing, plot_missing, render_missing
from ..utils import is_notebook

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
    "DType",
    "Categorical",
    "Nominal",
    "Ordinal",
    "Numerical",
    "Continuous",
    "Discrete",
    "DateTime",
    "Text",
    "create_report",
]


if is_notebook():
    output_notebook(hide_banner=True)
