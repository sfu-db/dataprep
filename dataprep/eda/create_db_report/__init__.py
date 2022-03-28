"""
    This module implements the create_db_report(sql_engine) function.
"""
import warnings
# import run_function.generate_db_report as generate_db_report
from .run_function import generate_db_report
from .report import Report

__all__ = ["create_db_report"]


def create_db_report(
    sql_engine
) -> Report:
    """
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
