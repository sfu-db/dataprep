"""This module implements the generation of connector configuration files."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.parse import parse_qs, urlparse

import requests
from dataprep.connector.schema.base import BaseDef

from ..schema import AuthorizationDef, ConfigDef, PaginationDef
from .state import ConfigState
from .table import gen_schema_from_path, search_table_path

# class Example(TypedDict):
#     url: str
#     method: str
#     params: Dict[str, str]
#     authorization: Tuple[Dict[str, Any], Dict[str, Any]]
#     pagination: Dict[str, Any]


class ConfigGenerator:
    """Config Generator.

    Parameters
    ----------
    config
        Initialize the config generator with existing config file.

    """

    config: ConfigState
    storage: Dict[str, Any]  # for auth usage

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is None:
            self.config = ConfigState(None)
        else:
            self.config = ConfigState(ConfigDef(**config))
        self.storage = {}

    def add_example(  # pylint: disable=too-many-locals
        self, example: Dict[str, Any], table_path: Optional[str] = None
    ) -> None:  # pylint: disable=too-many-locals
        """Add an example to the generator. The example
        should be in the dictionary format.

        class Example(TypedDict):
            url: str
            method: str
            params: Dict[str, str]
            # 0 for def and 1 for params
            authorization: Optional[Tuple[Dict[str, Any], Dict[str, Any]]]
            pagination: Optional[Dict[str, Any]]

        Parameters
        ----------
        req_example
            The request example.
        """
        url = example["url"]
        method = example["method"]
        if method not in {"POST", "GET", "PUT"}:
            raise ValueError(f"{method} not allowed.")
        if method != "GET":
            raise NotImplementedError(f"{method} not implemented.")

        params = example.get("params", {})

        # Do sanity check on url. For all the parameters that already in the URL we keep them.
        # For all the parameters that is not in the url we make it as free variables.
        parsed = urlparse(url)

        query_string = parse_qs(parsed.query)
        for key, (val, *_) in query_string.items():
            if key in params and params[key] != val:
                raise ValueError(
                    f"{key} appears in both url and params, but have different values."
                )
            # params[key] = val

        # url = urlunparse((*parsed[:4], "", *parsed[5:]))
        req = {
            "method": method,
            "url": url,
            "headers": {},
            "params": params,
        }

        # Parse authorization and build authorization into request
        authdef: Optional[AuthorizationDef] = None
        authparams: Optional[Dict[str, Any]] = None
        if example.get("authorization") is not None:
            authorization, authparams = example["authorization"]
            authdef = AuthUnion(val=authorization).val

        if authdef is not None and authparams is not None:
            authdef.build(req, authparams, self.storage)

        # Send out request and construct config
        config = _create_config(req, table_path)

        # Add pagination information into the config
        pagination = example.get("pagination")
        if pagination is not None:
            pagdef = PageUnion(val=pagination).val
            config.request.pagination = pagdef
        if authdef is not None:
            config.request.authorization = authdef

        self.config += config

    def to_string(self) -> str:
        """Output the string format of the current config."""
        return str(self.config)

    def save(self, path: Union[str, Path]) -> None:
        """Save the current config to a file.

        Parameters
        ----------
        path
            The path to the saved file, with the file extension.
        """
        path = Path(path)

        with open(path, "w") as f:
            f.write(self.to_string())


def _create_config(req: Dict[str, Any], table_path: Optional[str] = None) -> ConfigDef:
    resp = requests.request(
        req["method"].lower(),
        req["url"],
        params=req["params"],
        headers=req["headers"],
    )

    if resp.status_code != 200:
        raise RuntimeError(
            f"Request to HTTP endpoint not successful: {resp.status_code}: {resp.text}"
        )
    payload = resp.json()

    if table_path is None:
        table_path = search_table_path(payload)

    ret: Dict[str, Any] = {
        "version": 1,
        "request": {
            "url": req["url"],
            "method": req["method"],
            "params": {key: False for key in req["params"]},
        },
        "response": {
            "ctype": "application/json",
            "orient": "records",
            "tablePath": table_path,
            "schema": gen_schema_from_path(table_path, payload),
        },
    }

    return ConfigDef(**ret)


class AuthUnion(BaseDef):
    """Helper class for parsing authorization."""

    val: AuthorizationDef


class PageUnion(BaseDef):
    """Helper class for parsing pagination."""

    val: PaginationDef
