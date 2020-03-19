"""Docstring
    Data preparation module
"""
import logging
from pathlib import Path
from typing import cast

import toml

DEFAULT_PARTITIONS = 1

logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_version() -> str:
    """
    Get the library version from pyproject.toml
    """
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = toml.loads(open(str(path)).read())
    return cast(str, pyproject["tool"]["poetry"]["version"])


__version__ = get_version()
