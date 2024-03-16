"""
Module defines errors used in this library.
"""

from typing import Set

from ..errors import DataprepError


class RequestError(DataprepError):
    """
    A error indicating the status code of the API response
    is not 200.
    """

    status_code: int
    message: str

    def __init__(self, status_code: int, message: str) -> None:
        """
        Constructor

        parameters
        ----------
        status_code : int
            The http status code
        messsage : str
            The message from the response
        """

        super().__init__()

        self.status_code = status_code
        self.message = message

    def __str__(self) -> str:
        return f"RequestError: status={self.status_code}, message={self.message}"


class UniversalParameterOverridden(Exception):
    """
    The parameter is overrided by the universal parameter
    """

    param: str
    uparam: str

    def __init__(self, param: str, uparam: str) -> None:
        super().__init__()
        self.param = param
        self.uparam = uparam

    def __str__(self) -> str:
        return f"the parameter {self.param} is overridden by {self.uparam}"


class InvalidParameterError(Exception):
    """
    The parameter used in the query is invalid
    """

    param: str

    def __init__(self, param: str) -> None:
        super().__init__()
        self.param = param

    def __str__(self) -> str:
        return f"the parameter {self.param} is invalid, refer info method"


class MissingRequiredAuthParams(ValueError):
    """Some parameters for Authorization are missing."""

    params: Set[str]

    def __init__(self, params: Set[str]) -> None:
        super().__init__()
        self.params = params

    def __str__(self) -> str:
        return f"Missing required authorization parameter(s) {self.params} in _auth"


class InvalidAuthParams(ValueError):
    """The parameters used for Authorization are invalid."""

    params: Set[str]

    def __init__(self, params: Set[str]) -> None:
        super().__init__()
        self.params = params

    def __str__(self) -> str:
        return f"Authorization parameter(s) {self.params} in _auth are not required."
