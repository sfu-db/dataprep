"""
Intermediate class
"""
from typing import Any, Dict


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
            assert False, "Unsupported initialization"
