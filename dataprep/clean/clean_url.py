# pylint: disable=too-many-arguments, global-statement, line-too-long, too-many-locals
"""
implement clean_url functionality
"""

import re
from typing import Any, Union, List
from urllib.parse import unquote, urlparse

import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd

from .utils import NULL_VALUES, to_dask

# to extract queries
QUERY_REGEX = re.compile(r"(\?|\&)([^=]+)\=([^&]+)")

# regex to validate url
VALID_URL_REGEX = re.compile(
    r"^(?:http(s)?:\/\/)?[\w.-]+(?:\.[\w\.-]+)+[\w\-\._~:/?#[\]@!\$&'\(\)\*\+,;=.]+$"
)

# removing auth params
AUTH_VALUES = {
    "access_token",
    "auth_key",
    "auth",
    "password",
    "username",
    "login",
    "token",
    "passcode",
    "access-token",
    "auth-key",
    "authentication",
    "authentication-key",
}

# unified_list
UNIFIED_AUTH_LIST = set()

# STATs count
# cleaned=number of queries removed,
# rows=number of rows from which querries were removed
# correct_format=number of rows in correct format
# incorrect_format=number of rows in incorrect format
# first_val is to indicate whether what format is the first values in (more details in `check_first` doc-string), 100 for correct format, 200 for incorrect format
STATS = {"cleaned": 0, "rows": 0, "correct_format": 0, "incorrect_format": 0, "first_val": 0}


def clean_url(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    inplace: bool = False,
    split: bool = False,
    remove_auth: Union[bool, List[str]] = False,
    report: bool = True,
    errors: str = "coerce",
) -> Union[pd.DataFrame, dd.DataFrame]:

    """
    This function cleans url
    Parameters
    ----------
    df
        pandas or Dask DataFrame
    column
        column name
    split
        If True, split a column containing into the scheme, hostname, queries, cleaned_url columns
        if set to False would return a new column of dictionaries with the relavant
        information (scheme, host, etc.) in form of key-value pairs
    inplace
        If True, delete the given column with dirty data, else, create a new
        column with cleaned data.
    remove_auth
        can be a bool, or list of string representing the names of Auth queries
        to be removed. By default it is set to False
    report
        Displays how many queries were removed from rows
    errors
        Specify ways to deal with broken value
        {'ignore', 'coerce', 'raise'}, default 'coerce'
        'raise': raise an exception when there is broken value
        'coerce': set invalid value to NaN
        'ignore': just return the initial input
    """

    # reset stats and convert to dask
    reset_stats()
    df = to_dask(df)

    # unified list of auth removal params
    if not isinstance(remove_auth, bool):
        global UNIFIED_AUTH_LIST
        UNIFIED_AUTH_LIST = {*AUTH_VALUES, *set(remove_auth)}

    # specify the metadata for dask apply
    meta = df.dtypes.to_dict()

    if split:
        meta.update(
            zip(("scheme", "host", f"{column}_clean", "queries"), (str, str, str, str, str))
        )
    else:
        meta[f"{column}_details"] = str

    df = df.apply(format_url, args=(column, split, remove_auth, errors), axis=1, meta=meta)

    df, nrows = dask.compute(df, df.shape[0])

    if inplace:
        df = df.drop(columns=[column])

    if report:
        report_url(nrows=nrows, errors=errors, split=split, column=column)

    return df


def format_url(
    row: pd.Series, column: str, split: bool, remove_auth: Union[bool, List[str]], errors: str
) -> pd.Series:
    """
    This function formats each row of a pd.Series containing the url column
    """
    if split:
        row["scheme"], row["host"], row[f"{column}_clean"], row["queries"] = get_url_params(
            row[column], split=split, remove_auth=remove_auth, column=column, errors=errors
        )
    else:
        val_dict = get_url_params(
            row[column], split=split, remove_auth=remove_auth, column=column, errors=errors
        )
        row[f"{column}_details"] = val_dict

    return row


def get_url_params(
    url: str, column: str, split: bool, remove_auth: Union[bool, List[str]], errors: str
) -> Any:
    """
    This function extracts all the params from a given url string
    """
    if not validate_url(url):
        # values based on errors
        if split:
            return np.nan, np.nan, np.nan, np.nan
        else:
            if errors == "raise":
                raise ValueError(f"Unable to parse value {url}")
            return url if errors == "ignore" else np.nan

    # regex for finding the query / params and values
    re_queries = re.findall(QUERY_REGEX, url)
    all_queries = dict((y, z) for x, y, z in re_queries)

    # removing auth queries
    if remove_auth:
        if isinstance(remove_auth, bool):
            filtered_queries = {k: v for k, v in all_queries.items() if k not in AUTH_VALUES}
        else:
            filtered_queries = {k: v for k, v in all_queries.items() if k not in UNIFIED_AUTH_LIST}

        # for stats display
        diff = len(all_queries) - len(filtered_queries)
        if diff > 0:
            STATS["rows"] += 1
        STATS["cleaned"] += len(all_queries) - len(filtered_queries)

    # parsing the url using urlib
    parsed = urlparse(url)

    # extracting params
    formatted_scheme = (parsed.scheme + "://") if parsed.scheme else ""
    scheme = parsed.scheme
    host = parsed.hostname if parsed.hostname else ""
    path = parsed.path if parsed.path else ""
    cleaned_url = unquote(formatted_scheme + host + path).replace(" ", "")
    queries = filtered_queries if remove_auth else all_queries

    # returning the type based upon the split parameter.
    if split:
        return scheme, host, cleaned_url, queries
    else:
        return {"scheme": scheme, "host": host, f"{column}_clean": cleaned_url, "queries": queries}


def validate_url(x: Union[str, pd.Series]) -> Union[bool, pd.Series]:
    """
    This function validates url
    Parameters
    ----------
    x
        pandas Series of urls or url instance
    """
    if isinstance(x, pd.Series):
        verfied_series = x.apply(check_url)
        return verfied_series
    else:
        return check_url(x)


def check_url(val: Union[str, Any]) -> Any:
    """
    Function to check whether a value is a valid url
    """
    # check if the url is parsable
    try:
        if val in NULL_VALUES:
            if check_first():
                # set first_val = 200, here 200 means incorrect_format
                STATS["first_val"] = 200
            STATS["incorrect_format"] += 1
            return False
        # for non-string datatypes
        else:
            val = str(val)

        val = unquote(val).replace(" ", "")

        if re.match(VALID_URL_REGEX, val):
            if check_first():
                # set first_val = 100, here 100 means incorrect_format
                STATS["first_val"] = 100
            STATS["correct_format"] += 1
            return True
        else:
            if check_first():
                # set first_val = 200, here 200 means incorrect_format
                STATS["first_val"] = 200
            STATS["incorrect_format"] += 1
            return False
    except TypeError:
        if check_first():
            # set first_val = 200, here 200 means incorrect_format
            STATS["first_val"] = 200
        STATS["incorrect_format"] += 1
        return False


def report_url(nrows: int, errors: str, split: bool, column: str) -> None:
    """
    This function displays the stats report
    """
    correct_format = (
        STATS["correct_format"] - 1 if (STATS["first_val"] == 100) else STATS["correct_format"]
    )
    correct_format_percentage = (correct_format / nrows) * 100

    incorrect_format = (
        STATS["incorrect_format"] - 1 if (STATS["first_val"] == 200) else STATS["incorrect_format"]
    )
    incorrect_format_percentage = (incorrect_format / nrows) * 100

    cleaned_queries = STATS["cleaned"]
    rows = STATS["rows"]

    rows_string = (
        f"\nRemoved {cleaned_queries} auth queries from {rows} rows" if STATS["rows"] > 0 else ""
    )
    set_to = "NaN" if (errors == "coerce" or split) else "their original values"
    result_null = "null values" if (errors == "coerce" or split) else "null / not parsable values"

    if split:
        result = (
            f"Result contains parsed values for {correct_format}"
            f"({(correct_format / nrows) * 100 :.2f} %) rows and {incorrect_format} {result_null}"
            f"({(incorrect_format / nrows) * 100:.2f} %)."
        )
    else:
        result = (
            f"Result contains parsed key-value pairs for {correct_format} "
            f"({(correct_format / nrows) * 100 :.2f} %) rows (stored in column `{column}_details`) and {incorrect_format} {result_null}"
            f"({(incorrect_format / nrows) * 100:.2f} %)."
        )

    print(
        f"""
Url Cleaning report:
        {correct_format} values parsed ({correct_format_percentage:.2f} %)
        {incorrect_format} values unable to be parsed ({incorrect_format_percentage:.2f} %), set to {set_to} {rows_string}
{result}
        """
    )


def reset_stats() -> None:
    """
    This function resets the STATS
    """
    STATS["cleaned"] = 0
    STATS["rows"] = 0
    STATS["correct_format"] = 0
    STATS["incorrect_format"] = 0
    STATS["first_val"] = 0


def check_first() -> bool:
    """
    Dask runs 2 times for the first value (hence the first value is counted twice),
    this function checks whether the value we are parsing is the first value,
    after we find the first value we check whether it is in the correct form or incorrect form
    we then set STATS["first_val"] according to the following convention
    100 is the code for correct form and 200 is the code for the incorrect form
    we will use this value (STATS["first_val"] == 100 or STATS["first_val"] == 200) in our stats
    report to compensate for the overcounting of the first value by reducing the value.
    """
    return STATS["correct_format"] == 0 and STATS["incorrect_format"] == 0
