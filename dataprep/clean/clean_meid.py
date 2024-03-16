"""
Clean and validate a DataFrame column containing
Mobile Equipment Identifiers (MEIDs).
"""

# pylint: disable=too-many-lines, too-many-arguments, too-many-branches, unused-argument, E1101, E1133
from typing import Any, Union
from operator import itemgetter

import dask.dataframe as dd
import numpy as np
import pandas as pd

from stdnum import meid
from ..progress_bar import ProgressBar
from .utils import NULL_VALUES, to_dask


def clean_meid(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    output_format: str = "standard",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean MEID type data in a DataFrame column.

    Parameters
    ----------
        df
            A pandas or Dask DataFrame containing the data to be cleaned.
        column
            The name of the column containing data of MEID type.
        output_format
            The output format of standardized number string.
            If output_format = 'compact', return string without any separators.
            If output_format = 'standard', return string with proper separators.
            If output_format = 'hex', return MEID string with hex format.
            If output_format = 'dec', return MEID string with dec format.
            If output_format = 'binary', return MEID string with binary format.
            If output_format = 'pseudo_esn', return MEID string with pseudo_esn format.

            (default: "standard")
        split
            If True,
                each component of derived from its number string will be put into its own column.

            (default: False)
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
    Clean a column of MEID data.

    >>> df = pd.DataFrame({
            "meid": [
            "af0123450abcDEC"]
            })
    >>> clean_meid(df, 'meid', inplace=True)
           meid_clean
    0  AF 01 23 45 0A BC DE C
    """

    if output_format not in {"compact", "standard", "hex", "dec", "binary", "pseudo_esn"}:
        raise ValueError(
            f"output_format {output_format} is invalid. "
            'It needs to be "compact", "standard", "hex", "dec", "binary" or "pseudo_esn".'
        )

    # convert to dask
    df = to_dask(df)

    # To clean, create a new column "clean_code_tup" which contains
    # the cleaned values and code indicating how the initial value was
    # changed in a tuple. Then split the column of tuples and count the
    # amount of different codes to produce the report
    df["clean_code_tup"] = df[column].map_partitions(
        lambda srs: [_format(x, output_format, split, errors) for x in srs],
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

    # output a report describing the result of clean_country
    return df


def validate_meid(
    df: Union[str, pd.Series, dd.Series, pd.DataFrame, dd.DataFrame],
    column: str = "",
) -> Union[bool, pd.Series, pd.DataFrame]:
    """
    Validate if a data cell is ISBN in a DataFrame column.
    For each cell, return True or False.

    Parameters
    ----------
    df
            A pandas or Dask DataFrame containing the data to be validated.
    column
            The name of the column to be validated.
    """
    if isinstance(df, (pd.Series, dd.Series)):
        return df.apply(meid.is_valid)
    elif isinstance(df, (pd.DataFrame, dd.DataFrame)):
        if column != "":
            return df[column].apply(meid.is_valid)
        else:
            return df.applymap(meid.is_valid)
    return meid.is_valid(df)


def _format(
    val: Any, output_format: str = "standard", split: bool = False, errors: str = "coarse"
) -> Any:
    """
    Reformat a number string with proper separators (formats).

    Parameters
    ----------
    val
           The value of number string.
    output_format
           If output_format = 'compact', return string without any separators.
           If output_format = 'standard', return string with proper separators function.
           If output_format = 'hex', return MEID string with hex format.
           If output_format = 'dec', return MEID string with dec format.
           If output_format = 'binary', return MEID string with binary format.
           If output_format = 'pseudo_esn', return MEID string with pseudo_esn format.
    """
    val = str(val)
    result: Any = []

    if val in NULL_VALUES:
        return [np.nan]

    if not validate_meid(val):
        if errors == "raise":
            raise ValueError(f"Unable to parse value {val}")
        error_result = val if errors == "ignore" else np.nan
        return [error_result]

    if output_format == "compact":
        result = [meid.compact(val)] + result
    elif output_format == "standard":
        result = [meid.format(val, add_check_digit=True)] + result
    elif output_format == "hex":
        result = [meid.format(val, format="hex", add_check_digit=True)] + result
    elif output_format == "dec":
        result = [meid.format(val, format="dec", add_check_digit=True)] + result
    elif output_format == "binary":
        result = [meid.to_binary(val)] + result
    elif output_format == "pseudo_esn":
        result = [meid.to_pseudo_esn(val)] + result

    return result
