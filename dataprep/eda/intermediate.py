"""
Intermediate class
"""
from typing import Any, Dict, Tuple, Union

import pandas as pd


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


class ColumnsMetadata:
    """
    Container for storing each column's metadata
    """

    metadata: pd.DataFrame

    def __init__(self) -> None:
        self.metadata = pd.DataFrame()
        self.metadata.index.name = "Column Name"

    def __setitem__(self, key: Tuple[str, str], val: Any) -> None:
        col, vtype = key
        if (
            isinstance(val, (tuple, list, dict))
            and vtype
            not in self.metadata.columns  # pylint: disable=unsupported-membership-test
        ):
            self.metadata[vtype] = pd.Series(dtype="object")

        self.metadata.loc[col, vtype] = val

    def __getitem__(self, key: Union[str, Tuple[str, str]]) -> Any:
        if isinstance(key, tuple):
            col, vtype = key
            return self.metadata.loc[col, vtype]
        else:
            return ColumnMetadata(self.metadata.loc[key])


class ColumnMetadata:
    """
    Container for storing a single column's metadata.
    This is immutable
    """

    metadata: pd.Series

    def __init__(self, meta: pd.Series) -> None:
        self.metadata = meta

    def __getitem__(self, key: str) -> Any:
        return self.metadata.loc[key]
