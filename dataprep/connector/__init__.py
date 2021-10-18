"""Connector"""

from typing import Any, Dict, Optional

from .connector import Connector
from .generator import ConfigGenerator, ConfigGeneratorUI
from .info import info, websites
from .sql import read_sql

__all__ = [
    "Connector",
    "ConfigGenerator",
    "ConfigGeneratorUI",
    "connect",
    "info",
    "websites",
    "read_sql",
]


def connect(
    config_path: str,
    *,
    update: bool = False,
    _auth: Optional[Dict[str, Any]] = None,
    _concurrency: int = 1,
    **kwargs: Any,
) -> Connector:
    """Connect to a website.

    Parameters
    ----------
    config_path
        The path to the config. It can be hosted, e.g. "yelp", or from
        local filesystem, e.g. "./yelp"
    _auth: Optional[Dict[str, Any]] = None
        The parameters for authentication, e.g. OAuth2
    _concurrency: int = 5
        The concurrency setting. By default it is 1 reqs/sec.
    update: bool = True
        Force update the config file even if the local version exists.
    **kwargs
        Parameters that shared by different queries.

    Returns
    -------
    Connector
        a Connector object.

    Example
    -------
    >>> from dataprep.connector import connect
    >>> dc = connect("yelp", _auth={"access_token": access_token}, _concurrency=3)
    """
    return Connector(config_path, update=update, _auth=_auth, _concurrency=_concurrency, **kwargs)


def config_generator_ui(existing: Optional[Dict[str, Any]] = None) -> None:
    """Create a Config Generator UI.

    Parameters
    ----------
    existing: Optional[Dict[str, Any]] = None
        Optionally pass in an existing configuration.
    """

    ConfigGeneratorUI(existing).display()
