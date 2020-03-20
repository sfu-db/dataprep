"""Docstring
    Data preparation module
"""
import logging
from pathlib import Path
from typing import cast

import toml

DEFAULT_PARTITIONS = 1

logging.basicConfig(level=logging.INFO, format="%(message)s")

__version__ = "0.2.1"
