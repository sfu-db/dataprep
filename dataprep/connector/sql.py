"""
This module contains the method of read_sql.
It is a wrapper on connectorx.read_sql function.
"""

from typing import Optional, Tuple, Union, List, Any

try:
    import connectorx as cx

    _WITH_CX = True
except ImportError:
    _WITH_CX = False


def read_sql(
    conn: str,
    query: Union[List[str], str],
    *,
    return_type: str = "pandas",
    protocol: str = "binary",
    partition_on: Optional[str] = None,
    partition_range: Optional[Tuple[int, int]] = None,
    partition_num: Optional[int] = None,
) -> Any:
    """
    Run the SQL query, download the data from database into a dataframe.
    Please check out https://github.com/sfu-db/connector-x for more details.

    Parameters
    ----------
    conn
      the connection string.
    query
      a SQL query or a list of SQL query.
    return_type
      the return type of this function. It can be "arrow", "pandas", "modin", "dask" or "polars".
    protocol
      the protocol used to fetch data from source. Valid protocols are database dependent
      (https://github.com/sfu-db/connector-x/blob/main/Types.md).
    partition_on
      the column to partition the result.
    partition_range
      the value range of the partition column.
    partition_num
      how many partition to generate.

    Example
    --------
    >>> db_url = "postgresql://username:password@server:port/database"
    >>> query = "SELECT * FROM lineitem"
    >>> read_sql(db_url, query, partition_on="partition_col", partition_num=10)
    """
    if _WITH_CX:
        df = cx.read_sql(
            conn=conn,
            query=query,
            return_type=return_type,
            protocol=protocol,
            partition_on=partition_on,
            partition_range=partition_range,
            partition_num=partition_num,
        )
        return df
    else:
        raise ImportError("connectorx is not installed." "Please run pip install connectorx")
