"""ref: defines a reference type of value."""

from typing import TypeVar, Generic

T = TypeVar("T")  # pylint: disable=invalid-name


class Ref(Generic[T]):
    """A reference to a value."""

    __slots__ = ("val",)

    val: T

    def __init__(self, val: T) -> None:
        self.val = val

    def __int__(self) -> int:
        return int(self.val)  # type: ignore

    def __bool__(self) -> bool:
        return bool(self.val)

    def set(self, val: T) -> None:
        """set the value."""
        self.val = val

    def __str__(self) -> str:
        return str(self.val)

    def __repr__(self) -> str:
        return str(self.val)
