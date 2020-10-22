"""Base class for schema definition."""

# pylint: disable=missing-function-docstring
import re
from copy import deepcopy
from typing import Callable, Dict, Optional, TypeVar, cast, Any

from pydantic import BaseModel  # pylint: disable=no-name-in-module

T = TypeVar("T")  # pylint: disable=invalid-name
BaseDefT = TypeVar("BaseDefT", bound="BaseDef")


# copied from stringcase package
def camelcase(string: str) -> str:
    """Convert string into camel case.

    Args:
        string: String to convert.

    Returns:
        string: Camel case string.

    """

    string = re.sub(r"\w[\s\W]+\w", "", str(string))
    if not string:
        return string
    return string[0].lower() + re.sub(
        r"[\-_\.\s]([a-z])", lambda matched: matched.group(1).upper(), string[1:]
    )


class Policy:
    """Merge policy. Defines how a field can be merged."""

    override_none: bool = False
    merge: str = "same"  # merge policy, values: same, override, keep or None

    def __init__(self, override_none: bool = False, merge: str = "same") -> None:
        self.override_none = override_none
        self.merge = merge


class BaseDef(BaseModel):
    """The base definition."""

    __merge_policy__: Dict[str, Policy] = {}

    class Config:  # pylint: disable=missing-class-docstring
        alias_generator: Callable[[str], str] = camelcase
        validate_assignment: bool = True
        extra: str = "forbid"
        validate_all: bool = True

    def merge(self, rhs: BaseDefT) -> BaseDefT:
        if not isinstance(rhs, type(self)):
            raise ValueError(f"Cannot merge {type(self)} with {type(rhs)}")

        cur: BaseDefT = cast(BaseDefT, self.copy())

        for attr, _ in self.__fields__.items():
            cur_value, rhs_value = getattr(cur, attr), getattr(rhs, attr)

            if cur_value is None and rhs_value is None:
                pass
            elif (cur_value is None) != (rhs_value is None):
                if self.__merge_policy__.get(attr, Policy()).override_none:
                    setattr(cur, attr, coalesce(cur_value, rhs_value))
                else:
                    raise ValueError(f"None {attr} cannot be overriden.")
            else:

                merged = merge_values(
                    cur_value,
                    rhs_value,
                    attr,
                    self.__merge_policy__.get(attr, Policy()),
                )
                setattr(cur, attr, merged)

        return cur


def merge_values(  # pylint: disable=too-many-branches
    lhs: Any, rhs: Any, attr: str, policy: Policy
) -> Any:
    """merge two not none values."""

    if not isinstance(rhs, type(lhs)):
        raise ValueError(
            f"Cannot merge {type(lhs)} with {type(rhs)} for {type(lhs).__name__}.{attr}"
        )

    if isinstance(lhs, BaseDef):
        return lhs.merge(rhs)
    elif isinstance(rhs, dict):
        lhs = cast(Dict[str, Any], lhs)
        rhs = cast(Dict[str, Any], rhs)

        for key in rhs.keys():
            if key in lhs:
                lhs[key] = merge_values(lhs[key], rhs[key], attr, policy)
            else:
                if isinstance(rhs[key], BaseDef):
                    lhs[key] = rhs[key].copy()
                else:
                    lhs[key] = deepcopy(rhs[key])
        return lhs
    elif isinstance(lhs, (int, float, str, bool)):
        if policy.merge is None or policy.merge == "same":
            if lhs != rhs:
                raise ValueError(
                    f"Cannot merge with different {attr}:{type(lhs).__name__} :  {lhs} != {rhs}."
                )
            return lhs
        elif policy.merge == "override":
            return rhs
        elif policy.merge == "keep":
            return lhs
        else:
            raise RuntimeError(f"Unknown merge policy {policy.merge}.")
    else:
        raise RuntimeError(f"Unknown type {type(lhs).__name__}.")


def coalesce(a: Optional[T], b: Optional[T]) -> Optional[T]:  # pylint: disable=invalid-name
    if a is None:
        return b
    else:
        return a
