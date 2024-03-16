"""
dataprep.eda
============
"""

from bokeh.io import output_notebook

from ..utils import is_notebook
from .correlation import compute_correlation, plot_correlation, render_correlation
from .create_report import create_report
from .create_db_report import create_db_report
from .create_diff_report import create_diff_report
from .distribution import compute, plot, render
from .dtypes import (
    Categorical,
    Continuous,
    GeoGraphy,
    GeoPoint,
    DateTime,
    Discrete,
    DType,
    Nominal,
    Numerical,
    Ordinal,
    Text,
)
from .missing import compute_missing, plot_missing, render_missing
from .diff import plot_diff, compute_diff, render_diff

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
    "create_db_report",
    "create_diff_report",
    "plot_diff",
    "compute_diff",
    "render_diff",
]


if is_notebook():
    output_notebook(hide_banner=True)
