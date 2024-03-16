"""
Clean and validate a DataFrame column containing Latvian PVN (VAT) numbers (PVNs).
"""

# pylint: disable=too-many-lines, too-many-arguments, too-many-branches
from typing import Any, Union
from operator import itemgetter

import dask.dataframe as dd
import numpy as np
import pandas as pd

from stdnum.lv import pvn
from ..progress_bar import ProgressBar
from .utils import NULL_VALUES, to_dask


def clean_lv_pvn(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    output_format: str = "standard",
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean Latvian PVN (VAT) numbers (PVNs) type data in a DataFrame column.

    Parameters
    ----------
        df
            A pandas or Dask DataFrame containing the data to be cleaned.
        col
            The name of the column containing data of PVN type.
        output_format
            The output format of standardized number string.
            If output_format = 'compact', return string without any separators or whitespace.
            If output_format = 'standard', return string with proper separators and whitespace.
            If output_format = 'birthdate', return the birthdate of the person. Note only when
                PVN refers to a person (but not a legal entity) this format will be available.
            Note: in the case of PVN, the compact format is the same as the standard one.

            (default: "standard")
        inplace
           If True, delete the column containing the data that was cleaned.
           Otherwise, keep the original column.

           (default: False)
        errors
            How to handle parsing errors.
            - ‘coerce’: invalid parsing will be set to NaN.
            - ‘ignore’: invalid parsing will return the input.
            - ‘raise’: invalid parsing will raise an exception.

            (default: 'coerce')
        progress
            If True, display a progress bar.

            (default: True)

    Examples
    --------
    Clean a column of PVN data.

    >>> df = pd.DataFrame({
            "pvn": [
            '161175-19997',
            '40003521601',]
            })
    >>> clean_lv_pvn(df, 'pvn')
            pvn                     pvn_clean
    0       161175-19997            16117519997
    1       40003521601             NaN
    """

    if output_format not in {"compact", "standard", "birthdate"}:
        raise ValueError(
            f"output_format {output_format} is invalid. "
            'It needs to be "compact", "standard" or "birthdate".'
        )

    # convert to dask
    df = to_dask(df)

    # To clean, create a new column "clean_code_tup" which contains
    # the cleaned values and code indicating how the initial value was
    # changed in a tuple. Then split the column of tuples and count the
    # amount of different codes to produce the report
    df["clean_code_tup"] = df[column].map_partitions(
        lambda srs: [_format(x, output_format, errors) for x in srs],
        meta=object,
    )

    df = df.assign(
        _temp_=df["clean_code_tup"].map(itemgetter(0)),
    )

    df = df.rename(columns={"_temp_": f"{column}_clean"})

    df = df.drop(columns=["clean_code_tup"])

    if inplace:
        df[column] = df[f"{column}_clean"]
        df = df.drop(columns=f"{column}_clean")
        df = df.rename(columns={column: f"{column}_clean"})

    with ProgressBar(minimum=1, disable=not progress):
        df = df.compute()

    return df


def validate_lv_pvn(
    df: Union[str, pd.Series, dd.Series, pd.DataFrame, dd.DataFrame],
    column: str = "",
) -> Union[bool, pd.Series, pd.DataFrame]:
    """
    Validate if a data cell is PVN in a DataFrame column. For each cell, return True or False.

    Parameters
    ----------
    df
            A pandas or Dask DataFrame containing the data to be validated.
    col
            The name of the column to be validated.
    """
    if isinstance(df, (pd.Series, dd.Series)):
        return df.apply(pvn.is_valid)
    elif isinstance(df, (pd.DataFrame, dd.DataFrame)):
        if column != "":
            return df[column].apply(pvn.is_valid)
        else:
            return df.applymap(pvn.is_valid)
    return pvn.is_valid(df)


def _format(val: Any, output_format: str = "standard", errors: str = "coarse") -> Any:
    """
    Reformat a number string with proper separators and whitespace.

    Parameters
    ----------
    val
           The value of number string.
    output_format
           If output_format = 'compact', return string without any separators or whitespace.
           If output_format = 'standard', return string with proper separators and whitespace.
           If output_format = 'birthdate', return the birthdate of the person. Note only when
                PVN refers to a person (but not a legal entity) this format will be available.
           Note: in the case of PVN, the compact format is the same as the standard one.
    """
    # pylint: disable=bare-except

    val = str(val)
    result: Any = []

    if val in NULL_VALUES:
        return [np.nan]

    if not validate_lv_pvn(val):
        if errors == "raise":
            raise ValueError(f"Unable to parse value {val}")
        error_result = val if errors == "ignore" else np.nan
        return [error_result]

    if output_format in {"compact", "standard"}:
        result = [pvn.compact(val)] + result
    elif output_format == "birthdate":
        try:
            result = [pvn.get_birth_date(val)] + result
        except:  # PVN stands for a legal entity
            return [np.nan]

    return result
