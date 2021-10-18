# type: ignore
from os import environ
import pytest
import pandas as pd

from ...utils import display_dataframe
from ...connector import read_sql


@pytest.mark.skipif(
    environ.get("DB_URL", "") == "" or environ.get("DB_SQL", "") == "",
    reason="Skip tests that requires database setup and sql query specified",
)
def test_read_sql() -> None:
    db_url = environ["DB_URL"]
    sql = environ["DB_SQL"]
    df = read_sql(db_url, sql)
    display_dataframe(df)
