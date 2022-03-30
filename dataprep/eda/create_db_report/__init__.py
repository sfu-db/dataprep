"""
    This module implements the create_db_report(sql_engine) function.
"""
import warnings
import webbrowser
from sqlalchemy.engine.base import Engine
from .run_function import generate_db_report

__all__ = ["create_db_report"]


def create_db_report(sql_engine: Engine, analyze: bool = False) -> None:
    """
    This function is to generate and render database report and show in browser.

    Parameters
    ----------
    sql_engine
        SQL Alchemy Engine object returned from create_engine() with an url passed
        E.g. sql_engine = create_engine(url)
    analyze
        Whether to execute ANALYZE to write database statistics to the database

    Examples
    --------
    >>> from dataprep.eda import create_db_report
    >>> from dataprep.datasets import load_db
    >>> db_engine = load_db('sakila.db')
    >>> create_db_report(db_engine)
    """
    _suppress_warnings()
    output_file = generate_db_report(sql_engine, analyze)
    webbrowser.open(f"file://{output_file}", new=2)


def _suppress_warnings() -> None:
    """
    suppress warnings in create_diff_report
    """
    warnings.filterwarnings(
        "ignore",
        "The default value of regex will change from True to False in a future version",
        category=FutureWarning,
    )
