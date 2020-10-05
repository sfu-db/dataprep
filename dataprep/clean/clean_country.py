"""
Implement clean_country function
"""
from typing import Union, Any
from functools import lru_cache
import os

import regex as re
import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask

from .utils import NULL_VALUES, create_report, to_dask

COUNTRY_DATA_FILE = os.path.join(os.path.split(os.path.abspath(__file__))[0], "country_data.tsv")

DATA = pd.read_csv(COUNTRY_DATA_FILE, sep="\t", encoding="utf-8", dtype=str)

REGEXES = [re.compile(entry, re.IGNORECASE) for entry in DATA.regex]
STATS = {"cleaned": 0, "null": 0, "unknown": 0}


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
    """
    # pylint: disable=too-many-arguments
    reset_stats()

    input_formats = {"auto", "name", "official", "alpha-2", "alpha-3", "numeric"}
    output_formats = {"name", "official", "alpha-2", "alpha-3", "numeric"}
    if input_format not in input_formats:
        raise ValueError(
            f'input_format {input_format} is invalid, it needs to be "auto", '
            f'"name", "official", "alpha-2", "alpha-3" or "numeric'
        )
    if output_format not in output_formats:
        raise ValueError(
            f'output_format {output_format} is invalid, it needs to be "name", '
            f'"official", "alpha-2", "alpha-3" or "numeric'
        )
    if strict and fuzzy_dist > 0:
        raise ValueError(
            "can't do fuzzy matching while strict mode is enabled, "
            "set strict = False for fuzzy matching or fuzzy_dist = 0 for strict matching"
        )

    df = to_dask(df)
    meta = df.dtypes.to_dict()
    meta[f"{column}_clean"] = str

    df = df.apply(
        format_country,
        args=(column, input_format, output_format, fuzzy_dist, strict, errors),
        axis=1,
        meta=meta,
    )
    if inplace:
        df = df.drop(columns=[column])

    df, nrows = dask.compute(df, df.shape[0])

    if report:
        create_report("Country", STATS, nrows)

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
        x = x.astype(str).str.lower()
        x = x.str.strip()
        return x.apply(check_country, args=(input_format, False, strict))

    x = str(x).lower()
    x = x.strip()
    return check_country(x, input_format, False, strict)


def format_country(
    row: pd.Series,
    col: str,
    input_format: str,
    output_format: str,
    fuzzy_dist: int,
    strict: bool,
    errors: str,
) -> pd.Series:
    """
    Function to transform a country instance into the
    desired format
    """
    # pylint: disable=too-many-arguments
    # check_country parses the value in row[col], and will return the index of the country
    # in the DATA dataframe. The returned value "status" can be either "null"
    # (which means row[col] contains a null value), "unknown" (in which case the value
    # in row[col] could not be parsed) or "success" (a successful parse of the value).

    country = str(row[col])
    country = country.lower().strip()
    result_index, status = check_country(country, input_format, True, strict)

    if fuzzy_dist > 0 and status == "unknown" and input_format in ("auto", "name", "official"):
        result_index, status = check_fuzzy_dist(country, fuzzy_dist)

    if status == "null":
        STATS["null"] += 1
        row[f"{col}_clean"] = np.nan
        return row
    if status == "unknown":
        if errors == "raise":
            raise ValueError(f"unable to parse value {row[col]}")
        STATS["unknown"] += 1
        row[f"{col}_clean"] = row[col] if errors == "ignore" else np.nan
        return row

    result = DATA.iloc[result_index][output_format]
    if pd.isna(result):
        # country doesn't have the required output format
        if errors == "raise":
            raise ValueError(f"unable to parse value {row[col]}")
        STATS["unknown"] += 1
        row[f"{col}_clean"] = row[col] if errors == "ignore" else np.nan
        return row

    row[f"{col}_clean"] = result
    if row[col] != row[f"{col}_clean"]:
        STATS["cleaned"] += 1
    return row


@lru_cache(maxsize=2 ** 20)
def check_country(country: str, input_format: str, clean: bool, strict: bool) -> Any:
    """
    Finds the index of the given country in the DATA dataframe.

    Parameters
    ----------
    country
        string containing the country value being cleaned
    input_format
        the ISO 3166 input format of the country
    clean
        If True, a tuple (index, status) is returned.
        If False, the function returns True/False to be used by the validate country function.
    strict
        If True, for input types "name" and "offical" the function looks for a direct match
        in the DATA dataframe. If False, the country input is searched for a regex match.
    """
    if country in NULL_VALUES:
        return ("", "null") if clean else False

    if input_format == "auto":
        input_format = get_format_from_name(country)

    if strict and input_format == "regex":
        for input_type in ("name", "official"):
            indices = DATA[
                DATA[input_type].str.contains("^" + country + "$", flags=re.IGNORECASE, na=False)
            ].index

            if np.size(indices) > 0:
                return (indices[0], "success") if clean else True

    elif not strict and input_format in ("regex", "name", "official"):
        for index, country_regex in enumerate(REGEXES):
            if country_regex.search(country):
                return (index, "success") if clean else True
    else:
        indices = DATA[
            DATA[input_format].str.contains("^" + country + "$", flags=re.IGNORECASE, na=False)
        ].index

        if np.size(indices) > 0:
            return (indices[0], "success") if clean else True

    return ("", "unknown") if clean else False


@lru_cache(maxsize=2 ** 20)
def check_fuzzy_dist(country: str, fuzzy_dist: int) -> Any:
    """
    A match is found if a country has an edit distance <= fuzzy_dist
    with a string that contains a match with one of the country regexes.
    Find the index of a match with a minimum edit distance.
    """
    results = []
    for index, country_regex in enumerate(DATA.regex):
        # {e<=fuzzy_dist} means the total number of errors
        # (insertions, deletions and substitutions) must be <= fuzzy_dist,
        # re.BESTMATCH looks for a match with minimum number of errors
        fuzzy_regex = "(" + country_regex + f"){{e<={fuzzy_dist}}}"
        match = re.search(fuzzy_regex, country, flags=re.BESTMATCH | re.IGNORECASE)
        if match:
            # add total number of errors and the index to results
            results.append((sum(match.fuzzy_counts), index))

    if not results:
        return "", "unknown"

    _, min_index = min(results)
    return min_index, "success"


def get_format_from_name(name: str) -> str:

    """
    Function to infer the input format. Used when the input format is auto.
    """
    try:
        int(name)
        src_format = "numeric"
    except ValueError:
        if len(name) == 2:
            src_format = "alpha-2"
        elif len(name) == 3:
            src_format = "alpha-3"
        else:
            src_format = "regex"

    return src_format


def reset_stats() -> None:
    """
    Reset global statistics dictionary
    """
    STATS["cleaned"] = 0
    STATS["null"] = 0
    STATS["unknown"] = 0
