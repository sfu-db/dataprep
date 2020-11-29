"""
Implement clean_country function
"""
import os
from functools import lru_cache
from operator import itemgetter
from typing import Any, Union

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import regex as re

from ..eda.progress_bar import ProgressBar
from .utils import NULL_VALUES, create_report_new, to_dask

COUNTRY_DATA_FILE = os.path.join(os.path.split(os.path.abspath(__file__))[0], "country_data.tsv")

DATA = pd.read_csv(COUNTRY_DATA_FILE, sep="\t", encoding="utf-8", dtype=str)

REGEXES = [re.compile(entry, re.IGNORECASE) for entry in DATA.regex]
# alternative regex search strategy given on line 243
# REGEXES = re.compile("|".join(f"(?P<a{i}>{x})" for i, x in enumerate(DATA.regex)), re.IGNORECASE)


def clean_country(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    input_format: str = "auto",
    output_format: str = "name",
    fuzzy_dist: int = 0,
    strict: bool = False,
    inplace: bool = False,
    report: bool = True,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    This function cleans countries

    Parameters
    ----------
    df
        pandas or Dask DataFrame
    column
        column name containing messy country data
    input_format
        the ISO 3166 input format of the country:
        "auto" (infers the input format), country name ("name"),
        official state name ("official"), alpha-2 code ("alpha-2"),
        alpha-3 code ("alpha-3"), numeric code ("numeric")
    output_format
        the desired format of the country:
        country name ("name"), official state name ("official"), alpha-2 code ("alpha-2"),
        alpha-3 code ("alpha-3"), numeric code ("numeric")
    fuzzy_dist
        The maximum edit distance (number of single character insertions, deletions
        or substitutions required to change one word into the other) between a country value
        and input that will count as a match. Only applies to "auto", "name" and "official" formats.
    strict
        If True, matching for input formats "name" and "official" are done by looking
        for a direct match, if False, matching is done by searching the input for a
        regex match
    inplace
        If True, delete the given column with dirty data, else, create a new
        column with cleaned data.
    report
        If True, output the summary report. Otherwise, no report is outputted.
    errors {‘ignore’, ‘raise’, ‘coerce’}, default 'coerce'
        * If ‘raise’, then invalid parsing will raise an exception.
        * If ‘coerce’, then invalid parsing will be set as NaN.
        * If ‘ignore’, then invalid parsing will return the input.
    progress
        If True, enable the progress bar
    """
    # pylint: disable=too-many-arguments

    input_formats = {"auto", "name", "official", "alpha-2", "alpha-3", "numeric"}
    output_formats = {"name", "official", "alpha-2", "alpha-3", "numeric"}
    if input_format not in input_formats:
        raise ValueError(
            f'input_format {input_format} is invalid, it needs to be one of "auto", '
            '"name", "official", "alpha-2", "alpha-3" or "numeric'
        )
    if output_format not in output_formats:
        raise ValueError(
            f'output_format {output_format} is invalid, it needs to be "name", '
            '"official", "alpha-2", "alpha-3" or "numeric'
        )
    if strict and fuzzy_dist > 0:
        raise ValueError(
            "can't do fuzzy matching while strict mode is enabled, "
            "set strict=False for fuzzy matching or fuzzy_dist=0 for strict matching"
        )

    # convert to dask
    df = to_dask(df)

    # To clean, create a new column "clean_code_tup" which contains
    # the cleaned values and code indicating how the initial value was
    # changed in a tuple. Then split the column of tuples and count the
    # amount of different codes to produce the report
    df["clean_code_tup"] = df[column].map_partitions(
        lambda srs: [
            _format_country(x, input_format, output_format, fuzzy_dist, strict, errors) for x in srs
        ],
        meta=object,
    )
    df = df.assign(
        _temp_=df["clean_code_tup"].map(itemgetter(0)),
        _code_=df["clean_code_tup"].map(itemgetter(1)),
    )
    df = df.rename(columns={"_temp_": f"{column}_clean"})

    # counts of codes indicating how values were changed
    stats = df["_code_"].value_counts(sort=False)
    df = df.drop(columns=["clean_code_tup", "_code_"])

    if inplace:
        df = df.drop(columns=column)

    with ProgressBar(minimum=1, disable=not progress):
        df, stats = dask.compute(df, stats)

    # output a report describing the result of clean_country
    if report:
        create_report_new("Country", stats, errors)

    return df


def validate_country(
    x: Union[str, int, pd.Series], input_format: str = "auto", strict: bool = True
) -> Union[bool, pd.Series]:
    """
    This function validates countries

    Parameters
    ----------
    x
        pandas Series of countries or str/int country value
    input_format
        the ISO 3166 input format of the country:
        "auto" (infers the input format), country name ("name"),
        official state name ("official"), alpha-2 code ("alpha-2"),
        alpha-3 code ("alpha-3"), numeric code ("numeric")
    strict
        If True, matching for input formats "name" and "official" are done by
        looking for a direct match, if False, matching is done by searching
        the input for a regex match
    """

    if isinstance(x, pd.Series):
        x = x.astype(str).str.lower().str.strip()
        return x.apply(_check_country, args=(input_format, strict, False))

    x = str(x).lower().strip()
    return _check_country(x, input_format, strict, False)


def _format_country(
    val: Any,
    input_format: str,
    output_format: str,
    fuzzy_dist: int,
    strict: bool,
    errors: str,
) -> Any:
    """
    Function to transform a country instance into the desired format

    The last component of the returned tuple contains a code indicating how the
    input value was changed:
        0 := the value is null
        1 := the value could not be parsed
        2 := the value is cleaned and the cleaned value is DIFFERENT than the input value
        3 := the value is cleaned and is THE SAME as the input value (no transformation)
    """
    # pylint: disable=too-many-arguments
    # _check_country parses input value "val", and returns the index of the country
    # in the DATA dataframe. The returned value "status" can be either "null"
    # (which means val is a null value), "unknown" (in which case val
    # could not be parsed) or "success" (a successful parse of the value).

    country = str(val).lower().strip()
    result_index, status = _check_country(country, input_format, strict, True)

    if fuzzy_dist > 0 and status == "unknown" and input_format in ("auto", "name", "official"):
        result_index, status = _check_fuzzy_dist(country, fuzzy_dist)

    if status == "null":
        return np.nan, 0
    if status == "unknown":
        if errors == "raise":
            raise ValueError(f"unable to parse value {val}")
        return val if errors == "ignore" else np.nan, 1

    result = DATA.loc[result_index, output_format]
    if pd.isna(result):
        # country doesn't have the required output format
        if errors == "raise":
            raise ValueError(f"unable to parse value {val}")
        return val if errors == "ignore" else np.nan, 1

    return result, 2 if val != result else 3


@lru_cache(maxsize=2 ** 20)
def _check_country(country: str, input_format: str, strict: bool, clean: bool) -> Any:
    """
    Finds the index of the given country in the DATA dataframe.

    Parameters
    ----------
    country
        string containing the country value being cleaned
    input_format
        the ISO 3166 input format of the country
    strict
        If True, for input types "name" and "offical" the function looks for a direct match
        in the DATA dataframe. If False, the country input is searched for a regex match.
    clean
        If True, a tuple (index, status) is returned.
        If False, the function returns True/False to be used by the validate country function.
    """
    if country in NULL_VALUES:
        return (None, "null") if clean else False

    if input_format == "auto":
        input_format = _get_format_from_name(country)

    if strict and input_format == "regex":
        for form in ("name", "official"):
            ind = DATA[DATA[form].str.contains(f"^{country}$", flags=re.IGNORECASE, na=False)].index
            if np.size(ind) > 0:
                return (ind[0], "success") if clean else True

    elif not strict and input_format in ("regex", "name", "official"):
        for index, country_regex in enumerate(REGEXES):
            if country_regex.search(country):
                return (index, "success") if clean else True

        # alternative regex search strategy
        # match = REGEXES.search(country)
        # if match:
        #     return (int(match.lastgroup[1:]), "success") if clean else True
    else:
        ind = DATA[
            DATA[input_format].str.contains(f"^{country}$", flags=re.IGNORECASE, na=False)
        ].index
        if np.size(ind) > 0:
            return (ind[0], "success") if clean else True

    return (None, "unknown") if clean else False


@lru_cache(maxsize=2 ** 20)
def _check_fuzzy_dist(country: str, fuzzy_dist: int) -> Any:
    """
    A match is found if a country has an edit distance <= fuzzy_dist
    with a string that contains a match with one of the country regexes.
    Find the index of a match with a minimum edit distance.
    """
    results = []
    for i, country_regex in enumerate(DATA.regex):
        # {e<=fuzzy_dist} means the total number of errors
        # (insertions, deletions and substitutions) must be <= fuzzy_dist,
        # re.BESTMATCH looks for a match with minimum number of errors
        fuzzy_regex = f"({country_regex}){{e<={fuzzy_dist}}}"
        match = re.search(fuzzy_regex, country, flags=re.BESTMATCH | re.IGNORECASE)
        if match:
            # add total number of errors and the index to results
            results.append((sum(match.fuzzy_counts), i))

    if not results:
        return None, "unknown"

    return min(results)[1], "success"


def _get_format_from_name(name: str) -> str:
    """
    Function to infer the input format. Used when the input format is auto.
    """
    try:
        int(name)
        return "numeric"
    except ValueError:
        return "alpha-2" if len(name) == 2 else "alpha-3" if len(name) == 3 else "regex"
