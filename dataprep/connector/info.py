"""This module contains back end functions helping developers use data connector."""

from typing import Any, Dict, List

import pandas as pd
from IPython.display import display

from ..utils import get_styled_schema, is_notebook
from .implicit_database import ImplicitDatabase
from .info_ui import info_ui
from .schema import ConfigDef
from .config_manager import initialize_path


def info(config_path: str, update: bool = False) -> None:  # pylint: disable=too-many-locals
    """Show the basic information and provide guidance for users
    to issue queries.

    Parameters
    ----------
    config_path
        The path to the config. It can be hosted, e.g. "yelp", or from
        local filesystem, e.g. "./yelp"
    update
        Force update the config file even if the local version exists.
    """

    path = initialize_path(config_path, update)
    impdb = ImplicitDatabase(path)

    # get info
    tbs: Dict[str, Any] = {}
    for cur_table in impdb.tables:
        table_config_content: ConfigDef = impdb.tables[cur_table].config

        joined_auth_params = []
        required_params = []
        optional_params = []
        examples = []
        example_query_fields = []
        count = False
        required_param_count = 1
        auth = table_config_content.request.authorization
        config_examples = table_config_content.examples

        if auth is None:
            pass
        elif auth.type == "OAuth2":
            joined_auth_params.append(
                "client_id':'QNf0IeGSeMq3K*********NIU9mfHFMfX3cYe'"
                + " , "
                + "'client_secret':'eeNspLqiRoVfX*********3V2ntIiXKui9A6X'"
            )
        else:
            joined_auth_params.append("access_token':'cCMHU4M4t7rdt*********vp3whGzFjgIKIm0'")

        for k, val in table_config_content.request.params.items():
            if isinstance(val, bool) and val:
                required_params.append(k)
                if config_examples:
                    examples.append(k + "=" + config_examples[k])
                required_param_count += 1
            elif isinstance(val, bool):
                optional_params.append(k)

        separator = ", "

        if examples:
            example_query_fields.append(separator.join(examples))

        if table_config_content.request.pagination is not None:
            count = True

        schema = get_schema(table_config_content.response.schema_)
        styled_schema = get_styled_schema(schema)

        tbs[cur_table] = {}
        tbs[cur_table]["joined_auth_params"] = joined_auth_params
        tbs[cur_table]["required_params"] = required_params
        tbs[cur_table]["optional_params"] = optional_params
        tbs[cur_table]["joined_query_fields"] = example_query_fields
        tbs[cur_table]["count"] = count
        tbs[cur_table]["schemas"] = styled_schema

    # show table info
    info_ui(impdb.name, tbs)


def get_schema(schema: Dict[str, Any]) -> pd.DataFrame:
    """This method returns the schema of the table that will be returned,
    so that the user knows what information to expect.

    Parameters
    ----------
    schema
        The schema for the table from the config file.

    Returns
    -------
    pandas.DataFrame
        The returned data's schema.

    Note
    ----
    The schema is defined in the configuration file.
    The user can either use the default one or change it by editing the configuration file.
    """
    new_schema_dict: Dict[str, List[Any]] = {}
    new_schema_dict["column_name"] = []
    new_schema_dict["data_type"] = []
    for k in schema.keys():
        new_schema_dict["column_name"].append(k)
        new_schema_dict["data_type"].append(schema[k].type)

    return pd.DataFrame.from_dict(new_schema_dict)


def websites() -> None:
    """Displays names of websites supported by data connector."""
    websites = {
        "business": ["yelp"],
        "finance": ["finnhub"],
        "geocoding": ["mapquest"],
        "lifestyle": ["spoonacular"],
        "music": ["musixmatch", "spotify"],
        "news": ["guardian", "times"],
        "science": ["dblp"],
        "shopping": ["etsy"],
        "social": ["twitch", "twitter"],
        "video": ["youtube"],
        "weather": ["openweathermap"],
    }

    supported_websites = pd.DataFrame.from_dict(websites, orient="index")
    supported_websites = supported_websites.transpose()
    supported_websites.columns = supported_websites.columns.str.upper()

    # replace "None" values with empty string
    mask = supported_websites.applymap(lambda x: x is None)
    cols = supported_websites.columns[(mask).any()]
    for col in supported_websites[cols]:
        supported_websites.loc[mask[col], col] = ""

    if is_notebook():
        display(supported_websites)
    else:
        print(supported_websites.to_string(index=False))
