"""
    Intermediate class
"""
from typing import Any, Dict


class Intermediate:  # pylint: disable=too-few-public-methods
    """
       This is class of intermediate
       There are two variables
       result: intermediate result
       raw_data: input data
    """

    result: Dict[str, Any]
    raw_data: Dict[str, Any]

    def __init__(self,
                 result: Dict[str, Any],
                 raw_data: Dict[str, Any]) -> None:
        """
        :param result: The intermediate result calculated by us
        :param raw_data: The raw data input by the users
        """
        self.result = result
        self.raw_data = raw_data

    def __setattr__(self,
                    name: str,
                    value: Dict[str, Any]) -> None:
        """
        :param name: object's attribute name
        :param value: object's attribute type
        :return:
        """
        if name == 'result' and not isinstance(value, dict):
            raise TypeError('A.result must be a dict')
        if name == 'raw_data' and not isinstance(value, dict):
            raise TypeError('A.raw_data must be a dict')
        super().__setattr__(name, value)
