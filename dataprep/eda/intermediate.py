"""
Intermediate class
"""

from typing import Any, Dict, Tuple, Union, Optional

from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

from .dtypes_v2 import Continuous, Nominal, SmallCardNum


class Intermediate(Dict[str, Any]):
    """This class contains intermediate results."""

    visual_type: str

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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
            raise ValueError("Unsupported initialization")

    def save(self, path: Optional[str] = None) -> None:
        """
        Save intermediate to current working directory.

        Parameters
        ----------
        filename: Optional[str], default 'intermediate'
            The filename used for saving intermediate without the extension name.
        to: Optional[str], default Path.cwd()
            The path to where the intermediate will be saved.
        """
        saved_file_path = None

        if path:
            extension = os.path.splitext(path)[1]
            posix_path = Path(path).expanduser()

            if posix_path.is_dir():
                if path.endswith("/"):
                    path += "imdt.json"
                else:
                    path += "/imdt.json"

            elif extension:
                if extension != ".json":
                    raise ValueError(
                        "Format '{extension}' is not supported (supported formats: json)"
                    )

            else:
                path += ".json"

            saved_file_path = Path(path).expanduser()

        else:
            path = str(Path.cwd()) + "/imdt.json"
            saved_file_path = Path(path).expanduser()

        # pylint: disable=no-member
        inter_dict: Dict[str, Any] = {}
        for key in self.keys():
            inter_dict[key] = self[key]
        self._standardize_type(inter_dict)
        with open(path, "w") as outfile:
            json.dump(inter_dict, outfile, indent=4)
        print(f"Intermediate has been saved to {saved_file_path}!")

    def _standardize_type(self, inter_dict: Dict[str, Any]) -> None:
        # pylint: disable=too-many-nested-blocks, too-many-branches
        """
        In order to make intermediate could be saved as json file,
        check the type of data contained in the intermediate

        Parameters
        ----------
        inter_dict: Dict[str, Any], default "Intermediate"
            The intermediate result
        Returns
        -------

        """
        for key in inter_dict:
            if isinstance(inter_dict[key], dict):
                self._standardize_type(inter_dict[key])
            elif isinstance(
                inter_dict[key],
                (
                    np.int_,
                    np.intc,
                    np.intp,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                ),
            ):
                inter_dict[key] = int(inter_dict[key])
            elif isinstance(inter_dict[key], (np.float_, np.float16, np.float32, np.float64)):
                inter_dict[key] = float(inter_dict[key])
            elif isinstance(inter_dict[key], (np.ndarray,)):
                inter_dict[key] = inter_dict[key].tolist()
            elif isinstance(inter_dict[key], tuple):
                inter_dict[key] = list(inter_dict[key])
                for index in range(len(inter_dict[key])):
                    if isinstance(inter_dict[key][index], (np.ndarray,)):
                        inter_dict[key][index] = inter_dict[key][index].tolist()
                inter_dict[key] = tuple(inter_dict[key])
            elif isinstance(inter_dict[key], pd.DataFrame):
                inter_dict[key] = inter_dict[key].to_dict()
            elif isinstance(inter_dict[key], list):
                for idx, value in enumerate(inter_dict[key]):
                    if isinstance(value, tuple):
                        value = list(value)
                        for list_idx, list_value in enumerate(value):
                            if isinstance(list_value, Continuous):
                                value[list_idx] = "Continuous"
                            elif isinstance(list_value, Nominal):
                                value[list_idx] = "Nominal"
                            elif isinstance(list_value, SmallCardNum):
                                value[list_idx] = "SmallCardNum"
                            elif isinstance(value[list_idx], tuple):
                                ndy_value = list(list_value)
                                for ndy_idx, ndy_val in enumerate(ndy_value):
                                    if isinstance(ndy_val, (np.ndarray,)):
                                        ndy_value[ndy_idx] = ndy_val.tolist()
                                    elif isinstance(ndy_val, pd.DataFrame):
                                        ndy_value[ndy_idx] = ndy_val.to_dict()
                                    elif isinstance(
                                        ndy_value[ndy_idx],
                                        (
                                            np.int_,
                                            np.intc,
                                            np.intp,
                                            np.int8,
                                            np.int16,
                                            np.int32,
                                            np.int64,
                                            np.uint8,
                                            np.uint16,
                                            np.uint32,
                                            np.uint64,
                                        ),
                                    ):
                                        ndy_value[ndy_idx] = int(ndy_val)
                                value[list_idx] = tuple(ndy_value)
                            else:
                                pass
                    elif isinstance(value, (np.ndarray,)):
                        inter_dict[key][idx] = value.tolist()
                    inter_dict[key][idx] = tuple(value)
            else:
                pass


class ColumnsMetadata:
    """Container for storing each column's metadata."""

    metadata: pd.DataFrame

    def __init__(self) -> None:
        self.metadata = pd.DataFrame()
        self.metadata.index.name = "Column Name"

    def __setitem__(self, key: Tuple[str, str], val: Any) -> None:
        col, vtype = key
        if (
            isinstance(val, (tuple, list, dict))
            and vtype not in self.metadata.columns  # pylint: disable=unsupported-membership-test
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
    """Container for storing a single column's metadata.
    This is immutable.
    """

    metadata: pd.Series

    def __init__(self, meta: pd.Series) -> None:
        self.metadata = meta

    def __getitem__(self, key: str) -> Any:
        return self.metadata.loc[key]
