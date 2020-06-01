"""
Module defines ImplicitDatabase and ImplicitTable,
where ImplicitDatabase is a conceptual model describes
a website and ImplicitTable describes an API endpoint.
"""
from io import StringIO
from json import load as jload
from json import loads as jloads
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union

import jsonschema
import pandas as pd
from jsonpath2 import Path as JPath
from lxml import etree  # pytype: disable=import-error
from requests import Response

from ..errors import UnreachableError
from .schema import CONFIG_SCHEMA
from .types import Authorization, AuthorizationType, Fields, Orient

_TYPE_MAPPING = {
    "int": int,
    "string": str,
    "float": float,
    "boolean": bool,
}


class SchemaField(NamedTuple):
    """
    Schema of one table field
    """

    target: str
    type: str
    description: Optional[str]


class Pagination:
    """
    Schema of Pagination field
    """

    type: str
    count_key: str
    max_count: int
    anchor_key: Optional[str]
    cursor_id: Optional[str]
    cursor_key: Optional[str]

    def __init__(self, pdef: Dict[str, Any]) -> None:

        self.type = pdef["type"]
        self.max_count = pdef["max_count"]
        self.count_key = pdef["count_key"]
        self.anchor_key = pdef.get("anchor_key")
        self.cursor_id = pdef.get("cursor_id")
        self.cursor_key = pdef.get("cursor_key")


class ImplicitTable:  # pylint: disable=too-many-instance-attributes
    """
    ImplicitTable class abstracts the request and the response to a Restful API,
    so that the remote API can be treated as a database table.
    """

    name: str
    config: Dict[str, Any]
    # Request related
    method: str
    url: str
    authorization: Optional[Authorization] = None
    headers: Optional[Fields] = None
    params: Optional[Fields] = None
    body_ctype: str
    body: Optional[Fields] = None
    cookies: Optional[Fields] = None
    pag_params: Optional[Pagination] = None

    # Response related
    ctype: str
    table_path: str
    schema: Dict[str, SchemaField]
    orient: Orient

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        jsonschema.validate(
            config, CONFIG_SCHEMA
        )  # This will throw errors if validate failed
        self.name = name
        self.config = config

        request_def = config["request"]

        self.method = request_def["method"]
        self.url = request_def["url"]

        if "authorization" in request_def:
            auth_def = request_def["authorization"]
            if isinstance(auth_def, str):
                auth_type = AuthorizationType[auth_def]
                auth_params: Dict[str, str] = {}
            elif isinstance(auth_def, dict):
                auth_type = AuthorizationType[auth_def.pop("type")]
                auth_params = {**auth_def}
            else:
                raise NotImplementedError
            self.authorization = Authorization(auth_type=auth_type, params=auth_params)

        if "pagination" in request_def:
            self.pag_params = Pagination(request_def["pagination"])

        for key in ["headers", "params", "cookies"]:
            if key in request_def:
                setattr(self, key, Fields(request_def[key]))

        if "body" in request_def:
            body_def = request_def["body"]
            self.body_ctype = body_def["ctype"]
            self.body = Fields(body_def["content"])

        response_def = config["response"]
        self.ctype = response_def["ctype"]
        self.table_path = response_def["tablePath"]
        self.schema = {
            name: SchemaField(def_["target"], def_["type"], def_.get("description"))
            for name, def_ in response_def["schema"].items()
        }
        self.orient = Orient(response_def["orient"])

    def from_response(self, resp: Response) -> pd.DataFrame:
        """
        Create a dataframe from a http response.
        """
        if self.ctype == "application/json":
            rows = self.from_json(resp.text)
        elif self.ctype == "application/xml":
            rows = self.from_xml(resp.text)
        else:
            raise UnreachableError

        return pd.DataFrame(rows)

    def from_json(self, data: str) -> Dict[str, List[Any]]:
        """
        Create rows from json string.
        """
        data = jloads(data)
        table_data = {}
        root = self.table_path

        if self.orient == Orient.Records:
            data_rows = [
                row_node.current_value for row_node in JPath.parse_str(root).match(data)
            ]

            for column_name, column_def in self.schema.items():
                column_target = column_def.target
                column_type = column_def.type

                target_matcher = JPath.parse_str(column_target)

                col: List[Any] = []
                for data_row in data_rows:
                    maybe_cell_value = [
                        m.current_value for m in target_matcher.match(data_row)
                    ]

                    if not maybe_cell_value:  # If no match
                        col.append(None)
                    elif len(maybe_cell_value) == 1 and column_type != "object":
                        (cell_value,) = maybe_cell_value
                        if cell_value is not None:
                            # Even we have value matched,
                            # the value might be None so we don't do type conversion.
                            cell_value = _TYPE_MAPPING[column_type](cell_value)
                        col.append(cell_value)
                    else:
                        assert (
                            column_type == "object"
                        ), f"{column_name}: {maybe_cell_value} is not {column_type}"
                        col.append(maybe_cell_value)

                table_data[column_name] = col
        else:
            # TODO: split orient
            raise NotImplementedError

        return table_data

    def from_xml(self, data: str) -> Dict[str, List[Any]]:
        """
        Create rows from xml string.
        """
        table_data = {}

        data = data.replace('<?xml version="1.0" encoding="UTF-8"?>', "")

        root = etree.parse(StringIO(data))
        data_rows = root.xpath(self.table_path)

        if self.orient.value == Orient.Records.value:
            for column_name, column_def in self.schema.items():
                column_target = column_def.target
                column_type = column_def.type

                col: List[Any] = []
                for data_row in data_rows:
                    maybe_cell_value = data_row.xpath(column_target)

                    if not maybe_cell_value:
                        col.append(None)
                    elif len(maybe_cell_value) == 1 and column_type != "object":
                        (cell_value,) = maybe_cell_value
                        if cell_value is not None:
                            # Even we have value matched,
                            # the value might be None so we don't do type conversion.
                            cell_value = _TYPE_MAPPING[column_type](cell_value)
                        col.append(cell_value)
                    else:
                        assert (
                            column_type == "object"
                        ), f"{column_name}: {maybe_cell_value} is not {column_type}"
                        col.append(maybe_cell_value)

                table_data[column_name] = col
        else:
            # TODO: split orient
            raise NotImplementedError

        return table_data


class ImplicitDatabase:
    """
    A website that provides data can be treat as a database, represented
    as ImplicitDatabase in DataConnector.
    """

    name: str
    tables: Dict[str, ImplicitTable]

    def __init__(self, config_path: Union[str, Path]) -> None:
        path = Path(config_path)

        self.name = path.name
        self.tables = {}

        for table_config_path in path.iterdir():
            if not table_config_path.is_file():
                # ignore configs that are not file
                continue
            if table_config_path.name == "_meta.json":
                # ignore meta file
                continue
            if table_config_path.suffix != ".json":
                # ifnote non json file
                continue

            with open(table_config_path) as f:
                table_config = jload(f)

            table = ImplicitTable(table_config_path.stem, table_config)
            if table.name in self.tables:
                raise RuntimeError(f"Duplicated table name {table.name}")
            self.tables[table.name] = table
