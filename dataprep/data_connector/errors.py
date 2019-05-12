"""
Module defines errors used in this library.
"""
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
