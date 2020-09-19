"""
This module contains the Connector class.
Every data fetching action should begin with instantiating this Connector class.
"""
import math
import sys
from asyncio import as_completed
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Optional, Union

import pandas as pd
from aiohttp import ClientSession
from jinja2 import Environment, StrictUndefined, Template

from ..errors import UnreachableError
from .config_manager import config_directory, ensure_config
from .errors import RequestError, UniversalParameterOverridden, InvalidParameterError
from .implicit_database import ImplicitDatabase, ImplicitTable
from .int_ref import IntRef
from .throttler import OrderedThrottler, ThrottleSession

INFO_TEMPLATE = Template(
    """{% for tb in tbs.keys() %}
Table {{dbname}}.{{tb}}

Parameters
----------
{% if tbs[tb].required_params %}{{", ".join(tbs[tb].required_params)}} required {% endif %}
{% if tbs[tb].optional_params %}{{", ".join(tbs[tb].optional_params)}} optional {% endif %}

Examples
--------
>>> dc.query({{", ".join(["\\\"{}\\\"".format(tb)] + tbs[tb].joined_query_fields)}})
>>> dc.show_schema("{{tb}}")
{% endfor %}
"""
)


class Connector:
    """
    This is the main class of the connector component.
    Initialize Connector class as the example code.

    Parameters
    ----------
    config_path
        The path to the config. It can be hosted, e.g. "yelp", or from
        local filesystem, e.g. "./yelp"
    _auth: Optional[Dict[str, Any]] = None
        The parameters for authentication, e.g. OAuth2
    _concurrency: int = 5
        The concurrency setting. By default it is 1 reqs/sec.
    **kwargs
        Parameters that shared by different queries.

    Example
    -------
    >>> from dataprep.connector import Connector
    >>> dc = Connector("yelp", _auth={"access_token": access_token})
    """

    _impdb: ImplicitDatabase
    _vars: Dict[str, Any]
    _auth: Dict[str, Any]
    _concurrency: int
    _jenv: Environment

    def __init__(
        self,
        config_path: str,
        _auth: Optional[Dict[str, Any]] = None,
        _concurrency: int = 1,
        **kwargs: Any,
    ) -> None:
        if (
            config_path.startswith(".")
            or config_path.startswith("/")
            or config_path.startswith("~")
        ):
            path = Path(config_path).resolve()
        else:
            # From Github!
            ensure_config(config_path)
            path = config_directory() / config_path

        self._impdb = ImplicitDatabase(path)

        self._vars = kwargs
        self._auth = _auth or {}
        self._concurrency = _concurrency
        self._jenv = Environment(undefined=StrictUndefined)
        self._throttler = OrderedThrottler(_concurrency)

    async def query(  # pylint: disable=too-many-locals
        self,
        table: str,
        _auth: Optional[Dict[str, Any]] = None,
        _count: Optional[int] = None,
        _concurrency: Optional[int] = None,
        **where: Any,
    ) -> Union[Awaitable[pd.DataFrame], pd.DataFrame]:
        """
        Query the API to get a table.

        Parameters
        ----------
        table
            The table name.
        _auth: Optional[Dict[str, Any]] = None
            The parameters for authentication. Usually the authentication parameters
            should be defined when instantiating the Connector. In case some tables have different
            authentication options, a different authentication parameter can be defined here.
            This parameter will override the one from Connector if passed.
        _count: Optional[int] = None
            Count of returned records.
        **where
            The additional parameters required for the query.
        """
        allowed_params = self._impdb.tables[table].config["request"]["params"]
        for key in where:
            if key not in allowed_params:
                raise InvalidParameterError(key)

        return await self._query_imp(table, where, _auth=_auth, _count=_count)

    @property
    def table_names(self) -> List[str]:
        """
        Return all the names of the available tables in a list.
        Note
        ----
        We abstract each website as a database containing several tables.
        For example in Spotify, we have artist and album table.
        """
        return list(self._impdb.tables.keys())

    def info(self) -> None:
        """
        Show the basic information and provide guidance for users to issue queries.
        """

        # get info
        tbs: Dict[str, Any] = {}
        for cur_table in self._impdb.tables:
            table_config_content = self._impdb.tables[cur_table].config
            params_required = []
            params_optional = []
            example_query_fields = []
            count = 1
            for k, val in table_config_content["request"]["params"].items():
                if isinstance(val, bool) and val:
                    params_required.append(k)
                    example_query_fields.append(f"""{k}="word{count}\"""")
                    count += 1
                elif isinstance(val, bool):
                    params_optional.append(k)
            tbs[cur_table] = {}
            tbs[cur_table]["required_params"] = params_required
            tbs[cur_table]["optional_params"] = params_optional
            tbs[cur_table]["joined_query_fields"] = example_query_fields

        # show table info
        print(
            INFO_TEMPLATE.render(
                ntables=len(self.table_names), dbname=self._impdb.name, tbs=tbs
            )
        )

    def show_schema(self, table_name: str) -> pd.DataFrame:
        """
        This method shows the schema of the table that will be returned,
        so that the user knows what information to expect.
        Parameters
        ----------
        table_name
            The table name.
        Returns
        -------
        pd.DataFrame
            The returned data's schema.
        Note
        ----
        The schema is defined in the configuration file.
        The user can either use the default one or change it by editing the configuration file.
        """
        print(f"table: {table_name}")
        table_config_content = self._impdb.tables[table_name].config
        schema = table_config_content["response"]["schema"]
        new_schema_dict: Dict[str, List[Any]] = {}
        new_schema_dict["column_name"] = []
        new_schema_dict["data_type"] = []
        for k in schema.keys():
            new_schema_dict["column_name"].append(k)
            new_schema_dict["data_type"].append(schema[k]["type"])
        return pd.DataFrame.from_dict(new_schema_dict)

    async def _query_imp(  # pylint: disable=too-many-locals,too-many-branches
        self,
        table: str,
        kwargs: Dict[str, Any],
        *,
        _auth: Optional[Dict[str, Any]] = None,
        _count: Optional[int] = None,
    ) -> pd.DataFrame:
        if table not in self._impdb.tables:
            raise ValueError(f"No such table {table} in {self._impdb.name}")

        itable = self._impdb.tables[table]
        if itable.pag_params is None and _count is not None:
            print(
                f"ignoring _count since {table} has no pagination settings",
                file=sys.stderr,
            )

        if _count is not None and _count <= 0:
            raise RuntimeError("_count should be larger than 0")

        async with ClientSession() as client:
            throttler = self._throttler.session()

            if itable.pag_params is None or _count is None:
                df = await self._fetch(
                    itable, kwargs, _client=client, _throttler=throttler, _auth=_auth,
                )
                return df

            pag_type = itable.pag_params.type

            # pagination begins
            max_per_page = itable.pag_params.max_count
            total = _count
            n_page = math.ceil(total / max_per_page)

            if pag_type == "cursor":
                last_id = 0
                dfs = []
                # No way to parallelize for cursor type
                for i in range(n_page):
                    count = min(total - i * max_per_page, max_per_page)

                    df = await self._fetch(
                        itable,
                        kwargs,
                        _client=client,
                        _throttler=throttler,
                        _auth=_auth,
                        _count=count,
                        _cursor=last_id - 1,
                    )

                    if df is None:
                        raise NotImplementedError

                    if len(df) == 0:
                        # The API returns empty for this page, maybe we've reached the end
                        break

                    last_id = int(df[itable.pag_params.cursor_id][len(df) - 1]) - 1
                    dfs.append(df)

            elif pag_type == "limit":
                resps_coros = []
                allowed_page = IntRef(n_page)
                for i in range(n_page):
                    count = min(total - i * max_per_page, max_per_page)
                    resps_coros.append(
                        self._fetch(
                            itable,
                            kwargs,
                            _client=client,
                            _throttler=throttler,
                            _page=i,
                            _allowed_page=allowed_page,
                            _auth=_auth,
                            _count=count,
                            _cursor=i * max_per_page,
                        )
                    )

                dfs = []
                for resp_coro in as_completed(resps_coros):
                    df = await resp_coro
                    if df is not None:
                        dfs.append(df)

            else:
                raise NotImplementedError

        df = pd.concat(dfs, axis=0).reset_index(drop=True)

        return df

    async def _fetch(  # pylint: disable=too-many-locals,too-many-branches
        self,
        table: ImplicitTable,
        kwargs: Dict[str, Any],
        *,
        _client: ClientSession,
        _throttler: ThrottleSession,
        _page: int = 0,
        _allowed_page: Optional[IntRef] = None,
        _count: Optional[int] = None,
        _cursor: Optional[int] = None,
        _auth: Optional[Dict[str, Any]] = None,
    ) -> Optional[pd.DataFrame]:
        if (_count is None) != (_cursor is None):
            raise ValueError("_cursor and _count should both be None or not None")

        method = table.method
        url = table.url
        req_data: Dict[str, Dict[str, Any]] = {
            "headers": {},
            "params": {},
            "cookies": {},
        }
        merged_vars = {**self._vars, **kwargs}

        if table.authorization is not None:
            table.authorization.build(req_data, _auth or self._auth)

        for key in ["headers", "params", "cookies"]:
            if getattr(table, key) is not None:
                instantiated_fields = getattr(table, key).populate(
                    self._jenv, merged_vars
                )
                req_data[key].update(**instantiated_fields)

        if table.body is not None:
            # TODO: do we support binary body?
            instantiated_fields = table.body.populate(self._jenv, merged_vars)
            if table.body_ctype == "application/x-www-form-urlencoded":
                req_data["data"] = instantiated_fields
            elif table.body_ctype == "application/json":
                req_data["json"] = instantiated_fields
            else:
                raise NotImplementedError(table.body_ctype)

        if table.pag_params is not None and _count is not None:
            pag_type = table.pag_params.type
            count_key = table.pag_params.count_key
            if pag_type == "cursor":
                if table.pag_params.cursor_key is None:
                    raise ValueError(
                        "pagination type is cursor but no cursor_key set in the configuration file."
                    )
                cursor_key = table.pag_params.cursor_key
            elif pag_type == "limit":
                if table.pag_params.anchor_key is None:
                    raise ValueError(
                        "pagination type is limit but no anchor_key set in the configuration file."
                    )
                cursor_key = table.pag_params.anchor_key
            else:
                raise UnreachableError()

            if count_key in req_data["params"]:
                raise UniversalParameterOverridden(count_key, "_count")
            req_data["params"][count_key] = _count

            if cursor_key in req_data["params"]:
                raise UniversalParameterOverridden(cursor_key, "_cursor")
            req_data["params"][cursor_key] = _cursor

        await _throttler.acquire(_page)

        if _allowed_page is not None and int(_allowed_page) <= _page:
            # cancel current throttler counter since the request is not sent out
            _throttler.release()
            return None

        async with _client.request(
            method=method,
            url=url,
            headers=req_data["headers"],
            params=req_data["params"],
            json=req_data.get("json"),
            data=req_data.get("data"),
            cookies=req_data["cookies"],
        ) as resp:
            if resp.status != 200:
                raise RequestError(status_code=resp.status, message=await resp.text())
            content = await resp.text()
            df = table.from_response(content)

            if len(df) == 0 and _allowed_page is not None and _page is not None:
                _allowed_page.set(_page)
                return None
            else:
                return df
