
"""Docstring
    Data preparation module
"""
import logging
import tempfile
from bokeh.io import output_notebook, output_file
from ..utils import is_notebook, _rand_str

# Dask Default partitions
DEFAULT_PARTITIONS = 1

logging.basicConfig(level=logging.INFO, format="%(message)")
LOGGER = logging.getLogger(__name__)

if is_notebook():
    output_notebook(hide_banner=True)
else:
    output_file(filename=tempfile.gettempdir() + '/' + _rand_str() + '.html')
