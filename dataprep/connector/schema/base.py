"""Base class for schema definition."""

# pylint: disable=missing-function-docstring

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar

from pydantic import BaseModel  # pylint: disable=no-name-in-module
from stringcase import camelcase

T = TypeVar("T")  # pylint: disable=invalid-name


@dataclass
class Policy:
    """Merge policy. Defines how a field can be merged."""

    override_none: bool = False
    merge: str = "same"  # merge policy, values: same, override, keep or None


class BaseDef(BaseModel):
    """The base definition."""

    __merge_policy__: Dict[str, Policy] = {}

    class Config:  # pylint: disable=missing-class-docstring
        alias_generator: Callable[[str], str] = camelcase
        validate_assignment: bool = True
        extra: str = "forbid"
        validate_all: bool = True

    def merge(self, rhs: Any) -> "BaseDef":
        if not isinstance(rhs, type(self)):
            raise ValueError(f"Cannot merge {type(self)} with {type(rhs)}")

        cur: "BaseDef" = self.copy()

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
    lhs: T, rhs: T, attr: str, policy: Policy
) -> T:
    """merge two not none values."""

    if not isinstance(rhs, type(lhs)):
        raise ValueError(
            f"Cannot merge {type(lhs)} with {type(rhs)} for {type(lhs).__name__}.{attr}"
        )

    if isinstance(lhs, BaseDef):
        return lhs.merge(rhs)
    elif isinstance(rhs, dict):
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


def coalesce(  # pylint: disable=invalid-name
    a: Optional[T], b: Optional[T]
) -> Optional[T]:
    if a is None:
        return b
    else:
        return a
