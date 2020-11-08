"""This module contains back end functions helping developers use data connector."""
from .implicit_database import ImplicitDatabase
from .config_manager import config_directory, ensure_config
from pathlib import Path
import pandas as pd
from .connector_UI import info_UI 
from ..utils import get_styled_schema
from typing import Any, Dict, List
import requests

GIT_REF_URL = "https://api.github.com/repos/sfu-db/DataConnectorConfigs/contents"

def info(config_path: str, update: bool = True) -> None:
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
        params_required = []
        params_optional = []
        example_query_fields = []
        count = False
        required_param_count = 1
        auth = table_config_content.request.authorization
        
        if auth == None:
            pass
        elif auth.type == 'OAuth2':
            joined_auth_params.append('client_id\':client_id, \'client_secret\':client_secret')
        else:
            joined_auth_params.append('access_token\':access_token')
         
        for k, val in table_config_content.request.params.items():
            if isinstance(val, bool) and val:
                params_required.append(k)
                example_query_fields.append(k + '={insert_value}')
                required_param_count += 1
            elif isinstance(val, bool):
                params_optional.append(k)
                 
        if table_config_content.request.pagination != None:
            count = True
             
        schema = get_schema(table_config_content.response.schema_)
        styled_schema = get_styled_schema(schema)
             
        tbs[cur_table] = {}
        tbs[cur_table]["joined_auth_params"] = joined_auth_params
        tbs[cur_table]["required_params"] = params_required
        tbs[cur_table]["optional_params"] = params_optional
        tbs[cur_table]["joined_query_fields"] = example_query_fields
        tbs[cur_table]["count"] = count
        tbs[cur_table]["schemas"] = styled_schema

    # show table info
    info_UI(impdb.name, tbs)

def initialize_path(config_path: str, update: bool) -> str:
    """Determines if the given config_path is local or in GitHub. 
    Fetches the full path."""
    if (
        config_path.startswith(".")
        or config_path.startswith("/")
        or config_path.startswith("~")
    ):
        path = Path(config_path).resolve()
    else:
        # From Github!
        ensure_config(config_path, update)
        path = config_directory() / config_path
    return path

def get_schema(schema: Dict[str, List[Any]]) -> pd.DataFrame:
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