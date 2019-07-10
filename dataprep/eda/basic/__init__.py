"""
This module implements the code for plot(df,...)
"""
import holoviews as hv

from .computation import plot

hv.extension("bokeh", logo=False)
