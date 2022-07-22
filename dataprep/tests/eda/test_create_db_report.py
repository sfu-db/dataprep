import os
from ...eda.create_db_report.run_function import generate_db_report
from ...datasets import load_db


def test_create_db_report_sqlite() -> None:
    db_engine = load_db("sakila.db")
    generate_db_report(db_engine)

    # Check if output files were generated properly
    assert get_folder_file_num("../../eda/create_db_report/layout/tables") == 23
    assert get_folder_file_num("../../eda/create_db_report/layout/diagrams/summary") == 1
    assert get_folder_file_num("../../eda/create_db_report/layout/diagrams/tables") == 21


def get_folder_file_num(path):
    file = os.path.realpath(os.path.join(os.path.dirname(__file__), path))
    files = os.listdir(file)
    return len(files)
