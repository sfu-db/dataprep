"""
    This module implements the create_db_report(sql_engine) function.
"""
import warnings
from .run_function import generate_db_report

__all__ = ["create_db_report"]


def create_db_report(sql_engine) -> None:
    """
    This function is to generate and render database report and show in browser.

    Parameters
    ----------
    sql_engine
        The database engine object
    """
    _suppress_warnings()
    return generate_db_report(sql_engine)


def _suppress_warnings() -> None:
    """
    suppress warnings in create_diff_report
    """
    warnings.filterwarnings(
        "ignore",
        "The default value of regex will change from True to False in a future version",
        category=FutureWarning,
    )
