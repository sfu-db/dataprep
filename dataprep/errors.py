"""
Library-wise errors
"""


class DataprepError(Exception):
    """
    Base exception, used library-wise
    """


class UnreachableError(DataprepError):
    """
    Error indicating some path of the code is unreachable.
    """
