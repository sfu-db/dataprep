"""Table parsing utilities."""

from collections import defaultdict
from operator import itemgetter
from typing import Any, Dict, Set, Tuple

from jsonpath_ng import parse as jparse

from ..schema import SchemaFieldDef


def search_table_path(val: Dict[str, Any]) -> str:
    """Search table path in a json dict."""

    paths = _search_table_path("$", val)
    if not paths:
        raise ValueError("No tables found.")
    return max(paths, key=itemgetter(1))[0]


def _search_table_path(base: str, val: Dict[str, Any]) -> Set[Tuple[str, int]]:
    table_paths = set()
    for key, value in val.items():
        cur = f"{base}.{key}"
        if is_table_node(value):
            table_paths.add((f"{cur}[*]", len(value)))
        else:
            if isinstance(value, dict):
                table_paths.update(_search_table_path(cur, value))

    return table_paths


def is_table_node(node: Any) -> bool:
    """Detect if a node is a table node."""

    if isinstance(node, list):
        for row in node:
            if not isinstance(row, dict):
                return False
            for key in row.keys():
                if not isinstance(key, str):
                    return False

        # Better solutions? For different rows we might get different key sets
        # keys = node[0].keys()
        # for row in node[1:]:
        #     if row.keys() != keys:
        #         return False
        return True
    else:
        return False


def gen_schema_from_path(path: str, val: Dict[str, Any]) -> Dict[str, SchemaFieldDef]:
    """Generate the table schema from a path to the table."""

    finder = jparse(path)
    rows = finder.find(val)
    ret = {}

    for row in rows:
        for key, value in row.value.items():
            if key in ret:
                continue
            target = f"$.{key}"
            typ = _TYPE_MAPPING[type(value)]
            description = "auto generated"
            ret[key] = SchemaFieldDef(target=target, type=typ, description=description)

    return ret


_TYPE_MAPPING = defaultdict(
    lambda: "object",
    {
        int: "int",
        str: "string",
        float: "float",
        bool: "boolean",
    },
)
