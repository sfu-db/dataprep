import os
from ...eda.create_db_report.run_function import generate_db_report
from sqlalchemy import create_engine

def test_create_db_report_sqlite() -> None:
    db_file = str(os.path.realpath(os.path.join(os.path.dirname(__file__), "test_database", "sakila.db")))
    engine = create_engine(f'sqlite:///{db_file}')
    generate_db_report(engine)