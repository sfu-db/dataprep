"""
This module contains the method of lineagex.
It is a wrapper on lineagex.lineagex function.
"""

from typing import Optional, Union, List

try:
    import lineagex as lx

    _WITH_LX = True
except ImportError:
    _WITH_LX = False


def lineagex(
    sql: Optional[Union[List, str]] = None,
    target_schema: Optional[str] = "",
    conn_string: Optional[str] = None,
    search_path_schema: Optional[str] = "",
) -> dict:
    """
    Produce the lineage information.
    Please check out https://github.com/sfu-db/lineagex for more details.
    :param sql: The input of the SQL files, it can be a path to a file, a path to a folder containing SQL files, a list of SQLs or a list of view names and/or schemas
    :param target_schema: The schema where the SQL files would be created, defaults to public, or the first schema in the search_path_schema if provided
    :param conn_string: The postgres connection string in the format postgresql://username:password@server:port/database, defaults to None
    :param search_path_schema: The SET search_path TO ... schemas, defaults to public or the target_schema if provided
    :return:
    """

    if _WITH_LX:
        output_dict = lx.lineagex(
            sql=sql,
            target_schema=target_schema,
            conn_string=conn_string,
            search_path_schema=search_path_schema,
        ).output_dict
        return output_dict
    else:
        raise ImportError("lineagex is not installed." "Please run pip install lineagex")
