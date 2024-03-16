"""
This module contains the Connector class.
Every data fetching action should begin with instantiating this Connector class.
"""

import math
import sys
from asyncio import as_completed
from typing import Any, Awaitable, Dict, Optional, Set, Tuple, Union
from warnings import warn

import pandas as pd
from aiohttp import ClientSession
from aiohttp.client_reqrep import ClientResponse
from jinja2 import Environment, StrictUndefined, UndefinedError
from jsonpath_ng import parse as jparse

from .config_manager import initialize_path
from .errors import InvalidParameterError, RequestError, UniversalParameterOverridden
from .implicit_database import ImplicitDatabase, ImplicitTable
from .info import info
from .ref import Ref
from .schema import (
    FieldDef,
    FieldDefUnion,
    OffsetPaginationDef,
    PagePaginationDef,
    SeekPaginationDef,
    TokenLocation,
    TokenPaginationDef,
)
from .throttler import OrderedThrottler, ThrottleSession


class Connector:  # pylint: disable=too-many-instance-attributes
    """This is the main class of the connector component.
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
    update: bool = True
        Force update the config file even if the local version exists.
    **kwargs
        Parameters that shared by different queries.

    Example
    -------
    >>> from dataprep.connector import Connector
    >>> dc = Connector("yelp", _auth={"access_token": access_token})
    """

    _impdb: ImplicitDatabase
    # Varibles that used across different queries, can be overriden by query
    _vars: Dict[str, Any]
    _auth: Dict[str, Any]
    # storage for authorization
    _storage: Dict[str, Any]
    _concurrency: int
    _update: bool
    _jenv: Environment

    def __init__(
        self,
        config_path: str,
        *,
        update: bool = False,
        _auth: Optional[Dict[str, Any]] = None,
        _concurrency: int = 1,
        **kwargs: Any,
    ) -> None:
        path = initialize_path(config_path, update)

        self._impdb = ImplicitDatabase(path)

        self._vars = kwargs
        self._auth = _auth or {}
        self._storage = {}
        self._concurrency = _concurrency
        self._update = update
        self._jenv = Environment(undefined=StrictUndefined)
        self._throttler = OrderedThrottler(_concurrency)

    async def query(  # pylint: disable=too-many-locals
        self,
        table: str,
        *,
        _q: Optional[str] = None,
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
        _q: Optional[str] = None
            Search string to be matched in the response.
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
        allowed_params: Set[str] = set()

        for key, val in self._impdb.tables[table].config.request.params.items():
            if isinstance(val, FieldDef):
                if isinstance(val.from_key, list):
                    allowed_params.update(val.from_key)
                elif isinstance(val.from_key, str):
                    allowed_params.add(val.from_key)
                else:
                    allowed_params.add(key)
            else:
                allowed_params.add(key)

        allowed_params.update(self._impdb.tables[table].config.request.url_path_params())

        for key in where:
            if key not in allowed_params:
                raise InvalidParameterError(key)

        return await self._query_imp(table, where, _auth=_auth, _q=_q, _count=_count)

    def info(self) -> None:
        """Show the basic information and provide guidance for users
        to issue queries."""
        info(self._impdb.name)

    async def _query_imp(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        table: str,
        kwargs: Dict[str, Any],
        *,
        _auth: Optional[Dict[str, Any]] = None,
        _count: Optional[int] = None,
        _q: Optional[str] = None,
    ) -> pd.DataFrame:
        if table not in self._impdb.tables:
            raise ValueError(f"No such table {table} in {self._impdb.name}")

        itable = self._impdb.tables[table]
        reqconf = itable.config.request

        if reqconf.pagination is None and _count is not None:
            print(
                f"ignoring _count since {table} has no pagination settings",
                file=sys.stderr,
            )

        if _count is not None and _count <= 0:
            raise RuntimeError("_count should be larger than 0")

        async with ClientSession() as client:

            throttler = self._throttler.session()

            if reqconf.pagination is None or _count is None:
                df = await self._fetch(
                    itable,
                    kwargs,
                    _client=client,
                    _throttler=throttler,
                    _auth=_auth,
                    _q=_q,
                )
                return df

            pagdef = reqconf.pagination

            # pagination begins
            max_per_page = pagdef.max_count
            total = _count
            n_page = math.ceil(total / max_per_page)

            if isinstance(pagdef, SeekPaginationDef):
                last_id = 0
                dfs = []
                # No way to parallelize for seek type
                for i in range(n_page):
                    count = min(total - i * max_per_page, max_per_page)

                    df = await self._fetch(
                        itable,
                        kwargs,
                        _client=client,
                        _throttler=throttler,
                        _page=i,
                        _auth=_auth,
                        _q=_q,
                        _limit=count,
                        _anchor=last_id - 1,
                    )

                    if df is None:
                        raise NotImplementedError

                    if len(df) == 0:
                        # The API returns empty for this page, maybe we've reached the end
                        break

                    cid = df.columns.get_loc(pagdef.seek_id)  # type: ignore
                    last_id = int(df.iloc[-1, cid]) - 1  # type: ignore

                    dfs.append(df)
            elif isinstance(pagdef, TokenPaginationDef):
                next_token = None
                dfs = []
                # No way to parallelize for seek type
                for i in range(n_page):
                    count = min(total - i * max_per_page, max_per_page)
                    df, resp = await self._fetch(  # type: ignore
                        itable,
                        kwargs,
                        _client=client,
                        _throttler=throttler,
                        _page=i,
                        _auth=_auth,
                        _q=_q,
                        _limit=count,
                        _anchor=next_token,
                        _raw=True,
                    )

                    if pagdef.token_location == TokenLocation.Header:
                        next_token = resp.headers[pagdef.token_accessor]
                    elif pagdef.token_location == TokenLocation.Body:
                        # only json body implemented
                        token_expr = jparse(pagdef.token_accessor)
                        (token_elem,) = token_expr.find(await resp.json())
                        next_token = token_elem.value

                    dfs.append(df)
            elif isinstance(pagdef, (OffsetPaginationDef, PagePaginationDef)):
                resps_coros = []
                allowed_page = Ref(n_page)
                for i in range(n_page):
                    count = min(total - i * max_per_page, max_per_page)
                    if pagdef.type == "offset":
                        anchor = i * max_per_page
                    elif pagdef.type == "page":
                        anchor = i + 1
                    else:
                        raise ValueError(f"Unknown pagination type {pagdef.type}")

                    resps_coros.append(
                        self._fetch(
                            itable,
                            kwargs,
                            _client=client,
                            _throttler=throttler,
                            _page=i,
                            _allowed_page=allowed_page,
                            _auth=_auth,
                            _q=_q,
                            _limit=count,
                            _anchor=anchor,
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

    async def _fetch(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self,
        table: ImplicitTable,
        kwargs: Dict[str, Any],
        *,
        _client: ClientSession,
        _throttler: ThrottleSession,
        _page: int = 0,
        _allowed_page: Optional[Ref[int]] = None,
        _limit: Optional[int] = None,
        _anchor: Optional[Any] = None,
        _auth: Optional[Dict[str, Any]] = None,
        _q: Optional[str] = None,
        _raw: bool = False,
    ) -> Union[Optional[pd.DataFrame], Tuple[Optional[pd.DataFrame], ClientResponse]]:

        reqdef = table.config.request
        method = reqdef.method

        req_data: Dict[str, Dict[str, Any]] = {
            "headers": {},
            "params": {},
            "cookies": {},
        }
        merged_vars = {**self._vars, **kwargs}

        if reqdef.authorization is not None:
            reqdef.authorization.build(req_data, _auth or self._auth, self._storage)

        if reqdef.body is not None:
            # TODO: do we support binary body?
            instantiated_fields = populate_field(reqdef.body.content, self._jenv, merged_vars)
            if reqdef.body.ctype == "application/x-www-form-urlencoded":
                req_data["data"] = instantiated_fields
            elif reqdef.body.ctype == "application/json":
                req_data["json"] = instantiated_fields
            else:
                raise NotImplementedError(reqdef.body.ctype)

        if reqdef.pagination is not None and _limit is not None:
            pagdef = reqdef.pagination
            limit_key = pagdef.limit_key

            if isinstance(pagdef, SeekPaginationDef):
                anchor = pagdef.seek_key
            elif isinstance(pagdef, OffsetPaginationDef):
                anchor = pagdef.offset_key
            elif isinstance(pagdef, PagePaginationDef):
                anchor = pagdef.page_key
            elif isinstance(pagdef, TokenPaginationDef):
                anchor = pagdef.token_key
            else:
                raise ValueError(f"Unknown pagination type {pagdef.type}.")

            if limit_key in req_data["params"]:
                raise UniversalParameterOverridden(limit_key, "_limit")
            req_data["params"][limit_key] = _limit

            if anchor in req_data["params"]:
                raise UniversalParameterOverridden(anchor, "_offset")

            if _anchor is not None:
                req_data["params"][anchor] = _anchor

        if _q is not None:
            if reqdef.search is None:
                raise ValueError("_q specified but the API does not support custom search.")

            searchdef = reqdef.search
            search_key = searchdef.key

            if search_key in req_data["params"]:
                raise UniversalParameterOverridden(search_key, "_q")
            req_data["params"][search_key] = _q

        for key in ["headers", "params", "cookies"]:
            field_def = getattr(reqdef, key, None)
            if field_def is not None:
                instantiated_fields = populate_field(
                    field_def,
                    self._jenv,
                    merged_vars,
                )

                for ikey in instantiated_fields:
                    if ikey in req_data[key]:
                        warn(
                            f"Query parameter {ikey}={req_data[key][ikey]}"
                            " is overriden by {ikey}={instantiated_fields[ikey]}",
                            RuntimeWarning,
                        )
                req_data[key].update(**instantiated_fields)

        for key in ["headers", "params", "cookies"]:
            field_def = getattr(reqdef, key, None)
            if field_def is not None:
                validate_fields(field_def, req_data[key])

        url = reqdef.populate_url(merged_vars)

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
                df = None

            if _raw:
                return df, resp
            else:
                return df


def validate_fields(fields: Dict[str, FieldDefUnion], data: Dict[str, Any]) -> None:
    """Check required fields are provided."""

    for key, def_ in fields.items():
        to_key = key

        if isinstance(def_, bool):
            required = def_
            if required and to_key not in data:
                raise KeyError(f"'{to_key}' is required but not provided")
        elif isinstance(def_, str):
            pass
        else:
            to_key = def_.to_key or to_key
            required = def_.required
            if required and to_key not in data:
                raise KeyError(f"'{to_key}' is required but not provided")


def populate_field(  # pylint: disable=too-many-branches
    fields: Dict[str, FieldDefUnion],
    jenv: Environment,
    params: Dict[str, Any],
) -> Dict[str, str]:
    """Populate a dict based on the fields definition and provided vars."""
    ret: Dict[str, str] = {}

    for key, def_ in fields.items():
        to_key = key

        if isinstance(def_, bool):
            value = params.get(to_key)
            remove_if_empty = False
        elif isinstance(def_, str):
            # is a template
            tmplt = jenv.from_string(def_)
            value = tmplt.render(**params)
            remove_if_empty = False
        else:
            template = def_.template
            remove_if_empty = def_.remove_if_empty
            to_key = def_.to_key or to_key

            if template is None:
                value = params.get(to_key)
            else:
                tmplt = jenv.from_string(template)
                try:
                    value = tmplt.render(**params)
                except UndefinedError:
                    value = ""  # This empty string will be removed if `remove_if_empty` is True

        if value is not None:
            str_value = str(value)
            if not remove_if_empty or str_value:
                if to_key in ret:
                    warn(
                        f"{to_key}={ret[to_key]} overriden by {to_key}={str_value}",
                        RuntimeWarning,
                    )
                ret[to_key] = str_value
                continue
    return ret
