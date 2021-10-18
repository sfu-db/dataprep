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

    Supported databases
    ==========
    - Postgres
    - Mysql
    - Sqlite
    - SQL Server
    - Oracle
    - Redshift (through postgres protocol)
    - Clickhouse (through mysql protocol)

    Supported dataframes
    ==========
    - Pandas
    - Arrow
    - Dask
    - Modin
    - Polars

    Parameters
    ==========
    conn
      the connection string.
    query
      a SQL query or a list of SQL query.
    return_type
      the return type of this function. It can be "arrow", "pandas", "modin", "dask" or "polars".
    protocol
      the protocol used to fetch data from source. Valid protocols are database dependent (https://github.com/sfu-db/connector-x/blob/main/Types.md).
    partition_on
      the column to partition the result.
    partition_range
      the value range of the partition column.
    partition_num
      how many partition to generate.

    Examples
    ========
    Read a DataFrame from a SQL using a single thread:
    >>> postgres_url = "postgresql://username:password@server:port/database"
    >>> query = "SELECT * FROM lineitem"
    >>> read_sql(postgres_url, query)
    Read a DataFrame parallelly using 10 threads by automatically partitioning the provided SQL on the partition column:
    >>> postgres_url = "postgresql://username:password@server:port/database"
    >>> query = "SELECT * FROM lineitem"
    >>> read_sql(postgres_url, query, partition_on="partition_col", partition_num=10)
    Read a DataFrame parallelly using 2 threads by manually providing two partition SQLs:
    >>> postgres_url = "postgresql://username:password@server:port/database"
    >>> queries = ["SELECT * FROM lineitem WHERE partition_col <= 10", "SELECT * FROM lineitem WHERE partition_col > 10"]
    >>> read_sql(postgres_url, queries)
    """
    if _WITH_CX:
        df = cx.read_sql(
            conn=conn,
            query=query,
            return_type=return_type,
            partition_on=partition_on,
            partition_range=partition_range,
            partition_num=partition_num,
        )
        return df
    else:
        raise ImportError("connectorx is not installed." "Please run pip install connectorx")
