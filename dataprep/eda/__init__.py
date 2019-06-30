
"""Docstring
    Data preparation module
"""
import logging
import tempfile

from bokeh.io import output_file, output_notebook

from ..utils import _rand_str, is_notebook

# Dask Default partitions
DEFAULT_PARTITIONS = 1

if is_notebook():
    output_notebook(hide_banner=True)
else:
    output_file(filename=tempfile.gettempdir() + '/' + _rand_str() + '.html')
