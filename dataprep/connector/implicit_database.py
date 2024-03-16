"""
Module defines ImplicitDatabase and ImplicitTable,
where ImplicitDatabase is a conceptual model describes
a website and ImplicitTable describes an API endpoint.
"""

from json import load as jload
from json import loads as jloads
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
from jsonpath_ng import parse as jparse
from dataprep.connector.schema.defs import ConfigDef

from .schema import ConfigDef

_TYPE_MAPPING = {
    "int": int,
    "string": str,
    "float": float,
    "boolean": bool,
    "list": list,
}


class ImplicitTable:  # pylint: disable=too-many-instance-attributes
    """ImplicitTable class abstracts the request and the response
    to a Restful API, so that the remote API can be treated as a database
    table."""

    name: str
    config: ConfigDef

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name = name
        self.config = ConfigDef(**config)

    def from_response(self, payload: str) -> pd.DataFrame:
        """Create a dataframe from a http body payload."""

        ctype = self.config.response.ctype  # pylint: disable=no-member
        if ctype == "application/json":
            rows = self.from_json(payload)
        else:
            raise NotImplementedError(f"{ctype} not supported")

        return pd.DataFrame(rows)

    def from_json(self, data: str) -> Dict[str, List[Any]]:
        """Create rows from json string."""

        data = jloads(data)
        table_data = {}
        respdef = self.config.response
        table_expr = jparse(respdef.table_path)  # pylint: disable=no-member

        if respdef.orient == "records":  # pylint: disable=no-member
            data_rows = [match.value for match in table_expr.find(data)]

            for (
                column_name,
                column_def,
            ) in respdef.schema_.items():
                column_target = column_def.target
                column_type = column_def.type

                target_matcher = jparse(column_target)

                col: List[Any] = []
                for data_row in data_rows:
                    maybe_cell_value = [m.value for m in target_matcher.find(data_row)]

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
                # ignote non json file
                continue

            with open(table_config_path) as f:
                table_config = jload(f)

            table = ImplicitTable(table_config_path.stem, table_config)
            if table.name in self.tables:
                raise RuntimeError(f"Duplicated table name {table.name}")
            self.tables[table.name] = table
