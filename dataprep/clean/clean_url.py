"""
Implement clean_url functionality
"""
import re
from operator import itemgetter
from typing import Any, List, Union
from urllib.parse import unquote, urlparse

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..eda.progress_bar import ProgressBar
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


def clean_url(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    inplace: bool = False,
    split: bool = False,
    remove_auth: Union[bool, List[str]] = False,
    report: bool = True,
    errors: str = "coerce",
    progress: bool = True,
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
    errors {‘ignore’, ‘raise’, ‘coerce’}, default 'coerce'
        * If ‘raise’, then invalid parsing will raise an exception.
        * If ‘coerce’, then invalid parsing will be set as NaN.
        * If ‘ignore’, then invalid parsing will return the input.
    progress
        If True, enable the progress bar
    """
    # pylint: disable=too-many-arguments, global-statement

    # convert to dask
    df = to_dask(df)

    # unified list of auth removal params
    if not isinstance(remove_auth, bool):
        global UNIFIED_AUTH_LIST
        UNIFIED_AUTH_LIST = {*AUTH_VALUES, *set(remove_auth)}

    # To clean, create a new column "clean_code_tup" which contains
    # the cleaned values and code indicating how the initial value was
    # changed in a tuple. Then split the column of tuples and count the
    # amount of different codes to produce the report
    df["clean_code_tup"] = df[column].map_partitions(
        lambda srs: [_format_url(x, column, remove_auth, split, errors) for x in srs],
        meta=object,
    )
    if split:
        df = df.assign(
            scheme=df["clean_code_tup"].map(itemgetter(0)),
            host=df["clean_code_tup"].map(itemgetter(1)),
            _temp_=df["clean_code_tup"].map(itemgetter(2)),
            queries=df["clean_code_tup"].map(itemgetter(3), meta=("queries", object)),
            _code_=df["clean_code_tup"].map(itemgetter(4), meta=("_code_", object)),
            _nrem_=df["clean_code_tup"].map(itemgetter(5), meta=("_nrem_", object)),
        )
        df = df.rename(columns={"_temp_": f"{column}_clean"})

    else:
        df = df.assign(
            _temp_=df["clean_code_tup"].map(itemgetter(0)),
            _code_=df["clean_code_tup"].map(itemgetter(1)),
            _nrem_=df["clean_code_tup"].map(itemgetter(2)),
        )
        df = df.rename(columns={"_temp_": f"{column}_details"})

    # counts of codes indicating how values were changed
    stats = df["_code_"].value_counts(sort=False)
    # sum of auth tokens that were removed
    removed_auth_cnt = df["_nrem_"].sum()
    df = df.drop(columns=["clean_code_tup", "_code_", "_nrem_"])

    if inplace:
        df = df.drop(columns=column)

    with ProgressBar(minimum=1, disable=not progress):
        df, stats, removed_auth_cnt = dask.compute(df, stats, removed_auth_cnt)

    # output a report describing the result of clean_url
    if report:
        _report_url(stats, removed_auth_cnt, errors)

    return df


def validate_url(x: Union[str, pd.Series]) -> Union[bool, pd.Series]:
    """
    This function validates url

    Parameters
    ----------
    x
        pandas Series of urls or url instance
    """

    if isinstance(x, pd.Series):
        return x.apply(_check_url, args=(False,))
    return _check_url(x, False)


def _format_url(
    url: Any, col: str, remove_auth: Union[bool, List[str]], split: bool, errors: str
) -> Any:
    """
    This function formats the input value "url"

    The last two components of the returned tuple hold the following codes:
        the first component: code indicating how the value was transformed (see below)
        the second component: the count of auth queries removed, if applicable
    In the first component, there are the following four codes:
        0 := the value is null
        1 := the value could not be parsed
        2 := the value was parsed and DID NOT have authentication queries removed
        3 := the value was parsed and DID have authentication querires removed
    """
    # pylint: disable=too-many-locals

    # check if the url is a valid URL, returns a "status" value "null" (url is null),
    # "unknwon" (url is not a URL), and "success" (url is a URL)
    status = _check_url(url, True)

    if status == "null":
        return (np.nan, np.nan, np.nan, np.nan, 0, 0) if split else (np.nan, 0, 0)
    if status == "unknown":
        if errors == "raise":
            raise ValueError(f"Unable to parse value {url}")
        result = url if errors == "ignore" else np.nan
        return (result, np.nan, np.nan, np.nan, 1, 0) if split else (result, 1, 0)

    # regex for finding the query / params and values
    re_queries = re.findall(QUERY_REGEX, url)
    all_queries = dict((y, z) for _, y, z in re_queries)

    # initialize the removed authentication code and count for the stats
    rem_auth_code, rem_auth_cnt = 2, 0
    # removing auth queries
    if remove_auth:
        to_remove = AUTH_VALUES if isinstance(remove_auth, bool) else UNIFIED_AUTH_LIST
        filtered_queries = {k: v for k, v in all_queries.items() if k not in to_remove}

        # count of removed auth queries
        rem_auth_cnt = len(all_queries) - len(filtered_queries)
        # code to indicate whether queries were removed
        rem_auth_code = 2 if rem_auth_cnt == 0 else 3

    # parse the url using urllib
    parsed = urlparse(url)

    # extracting params
    scheme = parsed.scheme
    host = parsed.hostname if parsed.hostname else ""
    path = parsed.path if parsed.path else ""
    cleaned_url = unquote(f"{scheme}://{host}{path}").replace(" ", "")
    queries = filtered_queries if remove_auth else all_queries

    # returning the type based upon the split parameter.
    if split:
        return scheme, host, cleaned_url, queries, rem_auth_code, rem_auth_cnt
    return (
        {"scheme": scheme, "host": host, f"{col}_clean": cleaned_url, "queries": queries},
        rem_auth_code,
        rem_auth_cnt,
    )


def _check_url(url: Any, clean: bool) -> Any:
    """
    Function to check whether a value is a valid url
    """
    # check if the url is parsable
    try:
        if url in NULL_VALUES:
            return "null" if clean else False

        url = unquote(str(url)).replace(" ", "")

        if re.match(VALID_URL_REGEX, url):
            return "success" if clean else True
        return "unknown" if clean else False

    except TypeError:
        return "unknown" if clean else False


def _report_url(stats: pd.Series, removed_auth_cnt: int, errors: str) -> None:
    """
    This function displays the stats report

    In the stats DataFrame, the codes have the following meaning:
        0 := values that are null
        1 := values that could not be parsed
        2 := values that are parsed and DID NOT have authentication queries removed
        3 := values that are parsed and DID have authentication querires removed
    """
    print("URL Cleaning Report:")
    nrows = stats.sum()

    # count all values that were parsed (2 and 3 in stats)
    nclnd = (stats.loc[2] if 2 in stats.index else 0) + (stats.loc[3] if 3 in stats.index else 0)
    pclnd = round(nclnd / nrows * 100, 2)
    if nclnd > 0:
        print(f"\t{nclnd} values parsed ({pclnd}%)")

    # count all values that could not be parsed
    nunknown = stats.loc[1] if 1 in stats.index else 0
    if nunknown > 0:
        punknown = round(nunknown / nrows * 100, 2)
        expl = "set to NaN" if errors == "coerce" else "left unchanged"
        print(f"\t{nunknown} values unable to be parsed ({punknown}%), {expl}")

    # if auth queries were removed
    if removed_auth_cnt > 0:
        print(f"Removed {removed_auth_cnt} auth queries from {stats.loc[3]} rows")

    # count all null values
    nnull = stats.loc[0] if 0 in stats.index else 0
    if errors == "coerce":  # add unknown values that were set to NaN
        nnull += stats.loc[1] if 1 in stats.index else 0
    pnull = round(nnull / nrows * 100, 2)
    print(
        f"Result contains {nclnd} ({pclnd}%) parsed key-value pairs "
        f"and {nnull} null values ({pnull}%)"
    )


# def _report_url(nrows: int, errors: str, split: bool, column: str) -> None:
#     """
#     This function displays the stats report
#     """
#     correct_format = (
#         STATS["correct_format"] - 1 if (STATS["first_val"] == 100) else STATS["correct_format"]
#     )
#     correct_format_percentage = (correct_format / nrows) * 100

#     incorrect_format = (
#     STATS["incorrect_format"] - 1 if (STATS["first_val"] == 200) else STATS["incorrect_format"]
#     )
#     incorrect_format_percentage = (incorrect_format / nrows) * 100

#     cleaned_queries = STATS["cleaned"]
#     rows = STATS["rows"]

#     rows_string = (
#         f"\nRemoved {cleaned_queries} auth queries from {rows} rows" if STATS["rows"] > 0 else ""
#     )
#     set_to = "NaN" if (errors == "coerce" or split) else "their original values"
#     result_null = "null values" if (errors == "coerce" or split) else "null / not parsable values"

#     if split:
#         result = (
#             f"Result contains parsed values for {correct_format}"
#             f"({(correct_format / nrows) * 100 :.2f} %) rows and {incorrect_format} {result_null}"
#             f"({(incorrect_format / nrows) * 100:.2f} %)."
#         )
#     else:
#         result = (
#             f"Result contains parsed key-value pairs for {correct_format} "
#             f"({(correct_format / nrows) * 100 :.2f} %) rows (stored in column "\
#             f"`{column}_details`) and {incorrect_format} {result_null}"
#             f"({(incorrect_format / nrows) * 100:.2f} %)."
#         )

#     print(
#         f"""
# Url Cleaning report:
#         {correct_format} values parsed ({correct_format_percentage:.2f} %)
#         {incorrect_format} values unable to be parsed ({incorrect_format_percentage:.2f} %), " \
#         f"set to {set_to} {rows_string}
# {result}
#         """
#     )
