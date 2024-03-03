# type: ignore
from os import environ
import os
import pytest

from ...lineage import lineagex


@pytest.mark.skipif(
    environ.get("DB_URL", "") == "",
    reason="Skip tests that requires database setup and sql query specified",
)
def test_read_sql() -> None:
    db_url = environ["DB_URL"]
    sql = os.path.join(os.getcwd(), "dependency_example")
    lx = lineagex(sql, "mimiciii_derived", db_url, "mimiciii_clinical, public")
    print("dependency test with database connection", lx)
    lx = lineagex(
        sql=sql, target_schema="mimiciii_derived", search_path_schema="mimiciii_clinical, public"
    )
    print("dependency test without database connection", lx)


if __name__ == "__main__":
    test_read_sql()
