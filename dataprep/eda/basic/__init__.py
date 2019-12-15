"""
This module implements the code for plot(df,...)
"""
import holoviews as hv

from .compute import compute
from .render import render

hv.extension("bokeh", logo=False)
