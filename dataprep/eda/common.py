"""
    Intermediate class
"""
from typing import Any, Dict, Sequence
import numpy as np


class Intermediate(Dict[str, Any]):
    """
    This class contains intermediate results.
    """

    visual_type: str

    def __init__(self, *args: Any, **kwargs: Any):
        if (
            len(args) == 1
            and isinstance(args[0], dict)
            and len(kwargs) == 1
            and "visual_type" in kwargs
        ):
            super().__init__(args[0])
            self.visual_type = kwargs["visual_type"]
        elif len(args) == 0:
            visual_type = kwargs.pop("visual_type")
            super().__init__(**kwargs)
            self.visual_type = visual_type
        else:
            assert False, "Unsupported inivialization"


def sample_n(arr: np.ndarray, n: int) -> np.ndarray:  # pylint: disable=C0103
    """
    Sample n values uniformly from the range of the `arr`,
    not from the distribution of `arr`'s elems.
    """
    if len(arr) <= n:
        return arr

    subsel = np.linspace(0, len(arr) - 1, n)
    subsel = np.floor(subsel).astype(int)
    return arr[subsel]
