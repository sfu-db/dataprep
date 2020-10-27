# pylint: disable=too-many-arguments, trailing-whitespace, global-statement, line-too-long
"""
implement clean_url functionality
"""

import re
from typing import Any, Union, Dict, List
from urllib.parse import unquote, urlparse

import pandas as pd
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
STATS = {"cleaned": 0, "rows": 0, "correct_format": 0, "incorrect_format": 0}


def clean_url(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    inplace: bool = False,
    split: bool = False,
    remove_auth: Union[bool, List[str]] = False,
    report: bool = True,
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
    report:
        Displays how many queries were removed from rows
    """

    # reset stats and convert to dask
    reset_stats()
    df = to_dask(df)

    # unified list of auth removal params
    if not isinstance(remove_auth, bool):

        if not isinstance(remove_auth, list):
            raise TypeError(
                "Parameter `remove_auth` should either be boolean value or list of strings"
            )

        global UNIFIED_AUTH_LIST
        UNIFIED_AUTH_LIST = {*AUTH_VALUES, *set(remove_auth)}

    # specify the metadata for dask apply
    meta = df.dtypes.to_dict()

    if split:
        meta.update(zip(("scheme", "host", "cleaned_url", "queries"), (str, str, str, str, str)))
    else:
        meta.update(zip(("url_details",), (str,)))

    df = df.apply(
        format_url,
        args=(column, split, remove_auth),
        axis=1,
        meta=meta,
    )

    df, nrows = dask.compute(df, df.shape[0])

    if inplace:
        df = df.drop(columns=[column])

    if report:
        report_url(STATS, nrows)

    return df


def format_url(
    row: pd.Series, column: str, split: bool, remove_auth: Union[bool, List[str]]
) -> pd.Series:
    """
    This function formats each row of a pd.Series containing the url column
    """

    if split:

        if row[column] in NULL_VALUES:
            row["scheme"], row["host"], row["cleaned_url"], row["queries"] = None, None, None, None

        else:
            row["scheme"], row["host"], row["cleaned_url"], row["queries"] = get_url_params(
                row[column], split=split, remove_auth=remove_auth
            )

    else:
        if row[column] in NULL_VALUES:
            row["url_details"] = None
        else:
            val_dict = get_url_params(row[column], split=split, remove_auth=remove_auth)
            row["url_details"] = val_dict
    return row


def get_url_params(url: str, split: bool, remove_auth: Union[bool, List[str]]) -> Any:
    """
    This function extracts all the params from a given url string
    """

    if not validate_url(url, report=False):
        if split:
            return None, None, None, None
        else:
            return {"scheme": None, "host": None, "cleaned_url": None, "queries": None}

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
        return {"scheme": scheme, "host": host, "cleaned_url": cleaned_url, "queries": queries}


def validate_url(x: Union[str, pd.Series], report: bool = True) -> Union[bool, pd.Series]:
    """
    This function validates url
    Parameters
    ----------
    x
        pandas Series of urls or url instance
    """
    if isinstance(x, pd.Series):
        reset_stats()
        verfied_series = x.apply(check_url)
        if report:
            report_url(stats=STATS, nrows=x.size, validate=True)
        return verfied_series
    else:
        return check_url(x)


def check_url(val: Union[str, Any]) -> Any:
    """
    Function to check whether a value is a valid url
    """
    # check if the url is parsable
    if not isinstance(val, str):

        # check for null values
        if val in NULL_VALUES:
            STATS["incorrect_format"] += 1
            return False
        # for non-string datatypes
        else:
            val = str(val)

    val = unquote(val).replace(" ", "")

    if re.match(VALID_URL_REGEX, val):
        STATS["correct_format"] += 1
        return True
    else:
        STATS["incorrect_format"] += 1
        return False


def report_url(stats: Dict[str, int], nrows: int, validate: bool = False) -> None:
    """
    This function displays the stats report
    """
    if validate:
        correct_format = stats["correct_format"]
        incorrect_format = stats["incorrect_format"]
        print(
            f"{correct_format} rows ({((correct_format / nrows) * 100) : .2f} %) in correct format"
        )
        print(
            f"{incorrect_format} rows ({((incorrect_format / nrows) * 100): .2f} %)  in incorrect format"
        )
    else:
        cleaned_queries = stats["cleaned"]
        rows = stats["rows"]
        print(f"Removed {cleaned_queries} auth queries from {rows} rows")


def reset_stats() -> None:
    """
    This function resets the STATS
    """
    STATS["cleaned"] = 0
    STATS["rows"] = 0
    STATS["correct_format"] = 0
    STATS["incorrect_format"] = 0
