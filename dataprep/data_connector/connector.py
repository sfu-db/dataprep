"""
This module contains the Connector class,
where every data fetching should begin with instantiating
the Connector class.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from jinja2 import Environment, Template, StrictUndefined
from requests import Request, Response, Session

from ..errors import UnreachableError
from .config_manager import config_directory, ensure_config
from .errors import RequestError
from .implicit_database import ImplicitDatabase, ImplicitTable

from pathlib import Path
from json import load as jload


class Connector:
    """
    The main class of DataConnector.
    """

    impdb: ImplicitDatabase
    vars: Dict[str, Any]
    auth_params: Dict[str, Any]
    session: Session
    jenv: Environment

    def __init__(
        self,
        config_path: str,
        auth_params: Optional[Dict[str, Any]] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Connector

        parameters
        ----------
        config_path : str
            The path to the config. It can be hosted, e.g. "yelp", or from
            local filesystem, e.g. "./yelp"
        **kwargs : Dict[str, Any]
            Additional parameters
        """

        self.session = Session()
        if (
            config_path.startswith(".")
            or config_path.startswith("/")
            or config_path.startswith("~")
        ):
            path = Path(config_path).resolve()
            self.impdb = ImplicitDatabase(path)
        else:
            # From Github!
            ensure_config(config_path)
            path = config_directory() / config_path
            self.impdb = ImplicitDatabase(path)

        self.vars = kwargs
        self.auth_params = auth_params or {}
        self.jenv = Environment(undefined=StrictUndefined)
        self.config_path = config_path

    def _fetch(
        self,
        table: ImplicitTable,
        auth_params: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> Response:
        method = table.method
        url = table.url
        req_data: Dict[str, Dict[str, Any]] = {
            "headers": {},
            "params": {},
            "cookies": {},
        }

        merged_vars = {**self.vars, **kwargs}
        if table.authorization is not None:
            table.authorization.build(req_data, auth_params or self.auth_params)

        for key in ["headers", "params", "cookies"]:
            if getattr(table, key) is not None:
                instantiated_fields = getattr(table, key).populate(
                    self.jenv, merged_vars
                )
                req_data[key].update(**instantiated_fields)
        if table.body is not None:
            # TODO: do we support binary body?
            instantiated_fields = table.body.populate(self.jenv, merged_vars)
            if table.body_ctype == "application/x-www-form-urlencoded":
                req_data["data"] = instantiated_fields
            elif table.body_ctype == "application/json":
                req_data["json"] = instantiated_fields
            else:
                raise UnreachableError

        resp: Response = self.session.send(  # type: ignore
            Request(
                method=method,
                url=url,
                headers=req_data["headers"],
                params=req_data["params"],
                json=req_data.get("json"),
                data=req_data.get("data"),
                cookies=req_data["cookies"],
            ).prepare()
        )

        if resp.status_code != 200:
            raise RequestError(status_code=resp.status_code, message=resp.text)

        return resp

    def query(
        self,
        table: str,
        auth_params: Optional[Dict[str, Any]] = None,
        **where: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Query the API to get a table.

        Parameters
        ----------
        table : str
            The table name.
        auth_params : Optional[Dict[str, Any]] = None
            The parameters for authentication. Usually the authentication parameters
            should be defined when instantiating the Connector. In case some tables have different
            authentication options, a different authentication parameter can be defined here.
            This parameter will override the one from Connector if passed.
        **where: Dict[str, Any]
            The additional parameters required for the query.

        Returns
        -------
            pd.DataFrame
        """
        assert table in self.impdb.tables, f"No such table {table} in {self.impdb.name}"

        itable = self.impdb.tables[table]

        resp = self._fetch(itable, auth_params, where)

        return itable.from_response(resp)

    def info(self) -> None:
        # show tables available for connection
        print(
            len(self.table_names),
            "table(s) available in",
            Path(self.config_path).stem,
            ":\n",
        )

        # create templates for showing table information
        # 1. amndatory parameters for a query
        t_params = Template("--- requried parameters for quering:\n>>> {{params}}")
        # 2. example query:
        t_query = Template(
            "--- example query:\n>>> dc.query('{{table}}', {{joined_query_fields}})"
        )

        for t in self.impdb.tables.keys():
            print(t, "table:")
            table_config_content = self.impdb.tables[t].config
            params_required = []
            example_query_fields = []
            c = 1
            for k in table_config_content["request"]["params"].keys():
                if table_config_content["request"]["params"][k] == True:
                    params_required.append(k)
                    example_query_fields.append(k + "='word" + str(c) + "'")
                    c += 1
            print(t_params.render(params=params_required))
            print(
                t_query.render(
                    table=t, joined_query_fields=", ".join(example_query_fields)
                )
            )

        # other methods in the connector class:
        print("\nother methods:")
        print(">>>", "dc.table_names")
        print(">>>", "dc.show_schema('table name')")

    @property
    def table_names(self) -> List[str]:
        """
        Return all the table names contained in this database.
        """
        return list(self.impdb.tables.keys())

    def show_schema(self, table_name: str) -> pd.DataFrame:
        print("table:", table_name)
        table_config_content = self.impdb.tables[table_name].config
        schema = table_config_content["response"]["schema"]
        new_schema_dict: Dict[str, List[Any]] = {}
        c = 0
        new_schema_dict["column_name"] = []
        new_schema_dict["data_type"] = []
        for k in schema.keys():
            new_schema_dict["column_name"].append(k)
            new_schema_dict["data_type"].append(schema[k]["type"])
            c += 1
            # print("attribute name:", k, ", data type:", schema[k]['type'])
        return pd.DataFrame.from_dict(new_schema_dict)
