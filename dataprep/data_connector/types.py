"""
Defines useful types in this library.
"""
from typing import Any, Dict, NamedTuple, Optional, cast
from enum import Enum
from jinja2 import Environment

from ..errors import UnreachableError


class AuthorizationType(Enum):
    """
    Enum class defines the supported authorization methods
    in this library.
    """

    Bearer = "Bearer"


class Authorization(NamedTuple):
    """
    Class carries the authorization type and
    the corresponding parameter.
    """

    type: AuthorizationType
    params: Dict[str, str]

    def build(self, req_data: Dict[str, Any], params: Dict[str, Any]) -> None:
        """
        Populate some required fields to the request data.
        Complex logic may also happens in this function (e.g. start a server to do OAuth).
        """
        if self.type == AuthorizationType.Bearer:  # pylint: disable=no-member
            req_data["headers"]["Authorization"] = f"Bearer {params['token']}"


class Fields:
    """
    A data structure that stores the fields information (e.g. headers, cookies, ...).
    This class is useful to populate concrete fields data with required variables provided.
    """

    fields: Dict[str, Any]

    def __init__(self, fields_config: Dict[str, Any]) -> None:
        self.fields = fields_config

    def populate(self, jenv: Environment, params: Dict[str, Any]) -> Dict[str, str]:
        """
        Populate a dict based on the fields definition and provided vars.
        """
        ret = {}

        for key, def_ in self.fields.items():
            if isinstance(def_, bool):
                required = def_
                value = params.get(key)
                if value is None and required:
                    raise KeyError(key)
                remove_if_empty = False
            elif isinstance(def_, str):
                # is a template
                template: Optional[str] = def_
                expr = jenv.compile_expression(cast(str, template))
                value = expr(**params)
                remove_if_empty = False
            elif isinstance(def_, dict):
                template = def_.get("template")
                remove_if_empty = def_["removeIfEmpty"]

                if template is None:
                    required = def_["required"]
                    true_key = def_.get("valueFrom") or key
                    value = params.get(true_key)
                    if value is None and required:
                        raise KeyError(key)
                else:
                    expr = jenv.compile_expression(template)
                    value = expr(**params)
            else:
                raise UnreachableError()

            if value is not None:
                str_value = str(value)

                if not (remove_if_empty and not str_value):
                    ret[key] = str_value
                    continue

        return ret


class Orient(Enum):
    """
    Different types of table orientations
    ref: (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html).
    Currently, DataConnector supports two different types of orientaions:
        1. Split, which is column store.
        2. Records, which is row store.
    Details can be found in the pandas page.
    """

    Split = "split"
    Records = "records"
