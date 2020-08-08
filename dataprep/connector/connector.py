"""
This module contains the Connector class.
Every data fetching action should begin with instantiating this Connector class.
"""
import asyncio
import math
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Optional, Union

import pandas as pd
from jinja2 import Environment, StrictUndefined, Template

from aiohttp import ClientSession

from ..errors import UnreachableError
from .config_manager import config_directory, ensure_config
from .errors import RequestError, UniversalParameterOverridden
from .implicit_database import ImplicitDatabase, ImplicitTable
from .throttle import Throttler

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
        self._throttler = Throttler(_concurrency)

    async def query(  # pylint: disable=too-many-locals
        self,
        table: str,
        _auth: Optional[Dict[str, Any]] = None,
        _count: Optional[int] = None,
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

    async def _fetch(  # pylint: disable=too-many-locals,too-many-branches
        self,
        table: ImplicitTable,
        kwargs: Dict[str, Any],
        *,
        _client: ClientSession,
        _throttler: Throttler,
        _count: Optional[int] = None,
        _cursor: Optional[int] = None,
        _auth: Optional[Dict[str, Any]] = None,
    ) -> str:
        assert (_count is None) == (
            _cursor is None
        ), "_cursor and _count should both be None or not None"

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
                assert table.pag_params.cursor_key is not None
                cursor_key = table.pag_params.cursor_key
            elif pag_type == "limit":
                assert table.pag_params.anchor_key is not None
                cursor_key = table.pag_params.anchor_key
            else:
                raise UnreachableError()

            if count_key in req_data["params"]:
                raise UniversalParameterOverridden(count_key, "_count")
            req_data["params"][count_key] = _count

            if cursor_key in req_data["params"]:
                raise UniversalParameterOverridden(cursor_key, "_cursor")
            req_data["params"][cursor_key] = _cursor

        async with _throttler, _client.request(
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
            content: str = await resp.text()
            return content

    async def _query_imp(  # pylint: disable=too-many-locals
        self,
        table: str,
        kwargs: Dict[str, Any],
        *,
        _auth: Optional[Dict[str, Any]] = None,
        _count: Optional[int] = None,
    ) -> pd.DataFrame:
        assert (
            table in self._impdb.tables
        ), f"No such table {table} in {self._impdb.name}"

        async with ClientSession() as client:
            itable = self._impdb.tables[table]
            if itable.pag_params is None:
                resp = await self._fetch(
                    itable,
                    kwargs,
                    _client=client,
                    _throttler=self._throttler,
                    _auth=_auth,
                )
                df = itable.from_response(resp)
                return df

            pag_type = itable.pag_params.type

            if _count is None:
                # User doesn't specify _count
                resp = await self._fetch(
                    itable,
                    kwargs,
                    _client=client,
                    _throttler=self._throttler,
                    _auth=_auth,
                )
                df = itable.from_response(resp)
                return df

            # pagination begins
            max_count = itable.pag_params.max_count
            count = _count or 1
            n_page = math.ceil(count / max_count)
            remain = count % max_count

            if pag_type == "cursor":
                last_id = 0
                dfs = []
                # No way to parallel for cursor type
                for i in range(n_page):
                    remain = remain if remain > 0 else max_count
                    cnt_to_fetch = max_count if i < n_page - 1 else remain
                    resp = await self._fetch(
                        itable,
                        kwargs,
                        _client=client,
                        _throttler=self._throttler,
                        _auth=_auth,
                        _count=cnt_to_fetch,
                        _cursor=last_id - 1,
                    )

                    df_ = itable.from_response(resp)

                    if len(df_) == 0:
                        # The API returns empty for this page, maybe we've reached the end
                        break

                    last_id = int(df_[itable.pag_params.cursor_id][len(df_) - 1]) - 1
                    dfs.append(df_)
            elif pag_type == "limit":
                df_coros = set()
                for i in range(n_page):
                    remain = remain if remain > 0 else max_count
                    cnt_to_fetch = max_count if i < n_page - 1 else remain
                    df_coros.add(
                        self._fetch(
                            itable,
                            kwargs,
                            _client=client,
                            _throttler=self._throttler,
                            _auth=_auth,
                            _count=cnt_to_fetch,
                            _cursor=i * max_count,
                        )
                    )
                df_futs, _ = await asyncio.wait(df_coros)

                dfs = []
                for fut in df_futs:
                    resp = fut.result()
                    df = itable.from_response(resp)
                    dfs.append(df)
                # dfs = [itable.from_response(fut.result()) for fut in df_futs]
            else:
                raise NotImplementedError

        df = pd.concat(dfs, axis=0).reset_index(drop=True)

        return df
