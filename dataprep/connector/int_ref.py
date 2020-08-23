"""int_ref: defines a reference type of int
"""


class IntRef:
    """
    A reference to an int
    """

    n: int

    def __init__(self, n: int) -> None:
        self.n = n

    def __int__(self) -> int:
        return self.n

    def set(self, n: int) -> None:
        """set the int value"""
        self.n = n
