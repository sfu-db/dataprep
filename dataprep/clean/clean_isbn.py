"""
Clean and validate a DataFrame column containing ISBN numbers.
"""

# pylint: disable=too-many-lines, too-many-arguments, too-many-branches
from typing import Any, Union
from operator import itemgetter

import dask.dataframe as dd
import numpy as np
import pandas as pd

from stdnum import isbn
from ..progress_bar import ProgressBar
from .utils import NULL_VALUES, to_dask


def clean_isbn(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    output_format: str = "standard",
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean ISBN type data in a DataFrame column.

    Parameters
    ----------
        df
            A pandas or Dask DataFrame containing the data to be cleaned.
        column
            The name of the column containing data of ISBN type.
        output_format
            The output format of standardized number string.
            If output_format = 'compact', return string without any separators.
            If output_format = 'standard', return string with proper separators.
            If output_format = 'isbn13', return ISBN string with 13 digits.
            If output_format = 'isbn10', return ISBN string with 10 digits.

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
    Clean a column of ISBN data.

    >>> df = pd.DataFrame({
            "isbn": [
            "978-9024538270",
            "978-9024538271"]
            })
    >>> clean_isbn(df, 'isbn', inplace=True)
           isbn_clean
    0  978-90-245-3827-0
    1         NaN
    """

    if output_format not in {"compact", "standard", "isbn13", "isbn10"}:
        raise ValueError(
            f"output_format {output_format} is invalid. "
            'It needs to be "compact", "standard", "isbn13" or "isbn10".'
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

    if split:
        # For some reason the meta data for the last 3 components needs to be
        # set. I think this is a dask bug
        df = df.assign(
            _temp_=df["clean_code_tup"].map(itemgetter(0), meta=("_temp", object)),
            prefix_code=df["clean_code_tup"].map(itemgetter(1), meta=("prefix_code", object)),
            group_code=df["clean_code_tup"].map(itemgetter(2), meta=("group_code", object)),
            publisher_code=df["clean_code_tup"].map(itemgetter(3), meta=("publisher_code", object)),
            item_code=df["clean_code_tup"].map(itemgetter(4), meta=("item_code", object)),
            check_digit=df["clean_code_tup"].map(itemgetter(5), meta=("check_digit", object)),
        )
    else:
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


def validate_isbn(
    df: Union[str, pd.Series, dd.Series, pd.DataFrame, dd.DataFrame],
    column: str = "",
) -> Union[bool, pd.Series, pd.DataFrame]:
    """
    Validate if a data cell is ISBN in a DataFrame column. For each cell, return True or False.

    Parameters
    ----------
    df
            A pandas or Dask DataFrame containing the data to be validated.
    column
            The name of the column to be validated.
    """
    if isinstance(df, (pd.Series, dd.Series)):
        return df.apply(isbn.is_valid)
    elif isinstance(df, (pd.DataFrame, dd.DataFrame)):
        if column != "":
            return df[column].apply(isbn.is_valid)
        else:
            return df.applymap(isbn.is_valid)
    return isbn.is_valid(df)


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
           If output_format = 'isbn13', return ISBN string with 13 digits.
           If output_format = 'isbn10', return ISBN string with 10 digits.
    """
    val = str(val)
    result: Any = []

    if val in NULL_VALUES:
        if split:
            return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        else:
            return [np.nan]

    if not validate_isbn(val):
        if errors == "raise":
            raise ValueError(f"Unable to parse value {val}")
        error_result = val if errors == "ignore" else np.nan
        if split:
            return [error_result, np.nan, np.nan, np.nan, np.nan, np.nan]
        else:
            return [error_result]

    if split:
        result = list(isbn.split(isbn.to_isbn13(val)))
        if len(result) == 0:
            return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    if output_format == "compact":
        result = [isbn.compact(val)] + result
    elif output_format == "standard":
        result = [isbn.format(val)] + result
    elif output_format == "isbn13":
        result = [isbn.format(isbn.to_isbn13(val))] + result
    elif output_format == "isbn10":
        result = [isbn.format(isbn.to_isbn10(val))] + result

    return result
