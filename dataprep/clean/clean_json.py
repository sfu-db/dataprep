"""
Clean and validate a DataFrame column containing JSON.
"""

from typing import Any, Union
import json

# import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

# from .utils import to_dask


def clean_json(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    #  split: bool = True,
    errors: str = "coerce",
) -> pd.DataFrame:
    """
    Clean and standardize JSON.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing JSON.
    split
        If True, split the JSON into the semantic columns.
        If False, return a column of dictionaries with the relavant
        information (e.g., scheme, hostname, etc.) as key-value pairs.

        (default: False)
    inplace
        If True, delete the column containing the data that was cleaned. Otherwise,
        keep the original column.

        (default: False)
    errors
        How to handle parsing errors.
            - ‘coerce’: invalid parsing will be set to NaN.
            - ‘ignore’: invalid parsing will return the input.
            - ‘raise’: invalid parsing will raise an exception.

        (default: 'coerce')

    Examples
    --------
    Split a json into its components.
    >>> df = pd.DataFrame({ "messy_json": [
                  '{"name": "jane doe", "salary": 9000, "email": "jane.doe@pynative.com"}',
                  '{"name": "jane doe", "salary": 9000, "email": "jane.doe@pynative.com"}'
                        ]})
    >>> clean_json(df, column)

    """

    if not validate_json(df[column]).all():
        if errors == "raise":
            raise ValueError("Unable to clean value")
        error_result = df if errors == "ignore" else np.nan
        return error_result
    # df[column] = df[column].apply(lambda x: json.loads(x))
    df[column] = df.apply(lambda x: json.loads(x[column]), axis=1)
    new = pd.json_normalize(df[column])
    new_df = pd.concat([df, new], axis=1)
    new_df = new_df.astype(str)
    # convert to dask
    # df = to_dask(new_df)

    # To clean, create a new column "clean_code_tup" which contains
    # the cleaned values and code indicating how the initial value was
    # changed in a tuple. Then split the column of tuples and count the
    # amount of different codes to produce the report

    # df["clean_code_tup"] = df[column].map_partitions(
    #     lambda srs: [_format_json(x, split, errors) for x in srs],
    #     meta=object,
    # )
    # print( dir(df["clean_code_tup"].map(itemgetter(0))))
    # df = df.assign(
    #     _temp_=df["clean_code_tup"].map(itemgetter(0)),
    # )

    # df = df.rename(columns={"_temp_": f"{column}_clean"})

    # df = df.drop(columns=["clean_code_tup"])

    return new_df


def validate_json(x: Union[str, pd.Series]) -> Union[bool, pd.Series]:
    """
    Validate JSON.

    Parameters
    ----------
    x
        pandas Series of JSON.

    Examples
    --------

    >>> df = pd.DataFrame(
        {
            "messy_json": [
              '{"name": "jane doe", "salary": 9000, "email": "jane.doe@pynative.com",}',
              '{"name": "jane doe", "salary": 9000, "email": "jane.doe@pynative.com"}'
            ]
        }
    )
    >>> validate_json(df["messy_json"])
    0    False
    1     True
    Name: messy_json, dtype: bool
    """
    # x = x.apply(str)
    if isinstance(x, pd.Series):
        return x.apply(_check_json, args=(False,))

    return _check_json(x, False)


def _check_json(json_data: Any, clean: bool) -> Any:
    """
    Function to check whether a value is a valid json
    """
    try:
        json.loads(json_data)
    except ValueError:
        return "unknown" if clean else False
    return "success" if clean else True
