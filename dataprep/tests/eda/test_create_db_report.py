import os
from ...eda.create_db_report.run_function import generate_db_report
from ...datasets import load_db


def test_create_db_report_sqlite() -> None:
    db_engine = load_db("sakila.db")
    generate_db_report(db_engine)

    # Check if table files were generated properly
    table_folder_location = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "../../eda/create_db_report/layout/tables")
    )
    table_files = os.listdir(table_folder_location)
    assert len(table_files) == 23
