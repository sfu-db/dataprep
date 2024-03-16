"""
Clean and validate a DataFrame column containing country names.
"""

from functools import lru_cache
from operator import itemgetter
from os import path
from typing import Any, Union, Tuple, Optional

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import regex as re

from ..progress_bar import ProgressBar
from .utils import NULL_VALUES, create_report_new, to_dask

COUNTRY_DATA_FILE = path.join(path.split(path.abspath(__file__))[0], "country_data.tsv")

DATA = pd.read_csv(COUNTRY_DATA_FILE, sep="\t", encoding="utf-8", dtype=str)

REGEXES = [re.compile(entry, re.IGNORECASE) for entry in DATA.regex]


def clean_country(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    input_format: Union[str, Tuple[str, ...]] = "auto",
    output_format: str = "name",
    fuzzy_dist: int = 0,
    strict: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    report: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean and standardize country names.

    Read more in the :ref:`User Guide <country_userguide>`.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing country names.
    input_format
        The ISO 3166 input format of the country.
            - 'auto': infer the input format
            - 'name': country name ('United States')
            - 'official': official state name ('United States of America')
            - 'alpha-2': alpha-2 code ('US')
            - 'alpha-3': alpha-3 code ('USA')
            - 'numeric': numeric code (840)

        Can also be a tuple containing any combination of input formats,
        for example to clean a column containing alpha-2 and numeric
        codes set input_format to ('alpha-2', 'numeric').

        (default: 'auto')
    output_format
        The desired ISO 3166 format of the country:
            - 'name': country name ('United States')
            - 'official': official state name ('United States of America')
            - 'alpha-2': alpha-2 code ('US')
            - 'alpha-3': alpha-3 code ('USA')
            - 'numeric': numeric code (840)

        (default: 'name')
    fuzzy_dist
        The maximum edit distance (number of single character insertions, deletions
        or substitutions required to change one word into the other) between a country value
        and input that will count as a match. Only applies to 'auto', 'name' and 'official'
        input formats.

        (default: 0)
    strict
        If True, matching for input formats 'name' and 'official' are done by looking
        for a direct match. If False, matching is done by searching the input for a
        regex match.

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
    report
        If True, output the summary report. Otherwise, no report is outputted.

        (default: True)
    progress
        If True, display a progress bar.

        (default: True)

    Examples
    --------

    >>> df = pd.DataFrame({'country': [' Canada ', 'US']})
    >>> clean_country(df, 'country')
    Country Cleaning Report:
        2 values cleaned (100.0%)
    Result contains 2 (100.0%) values in the correct format and 0 null values (0.0%)
        country  country_clean
    0   Canada          Canada
    1        US  United States
    """
    # pylint: disable=too-many-arguments
    output_formats = {"name", "official", "alpha-2", "alpha-3", "numeric"}
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
    input_formats = _input_format_to_tuple(input_format)

    # convert to dask
    df = to_dask(df)

    # To clean, create a new column "clean_code_tup" which contains
    # the cleaned values and code indicating how the initial value was
    # changed in a tuple. Then split the column of tuples and count the
    # amount of different codes to produce the report
    df["clean_code_tup"] = df[column].map_partitions(
        lambda srs: [
            _format_country(x, input_formats, output_format, fuzzy_dist, strict, errors)
            for x in srs
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
    x: Union[str, int, pd.Series],
    input_format: Union[str, Tuple[str, ...]] = "auto",
    strict: bool = True,
) -> Union[bool, pd.Series]:
    """
    Validate country names.

    Read more in the :ref:`User Guide <country_userguide>`.

    Parameters
    ----------
    x
        pandas Series of countries or str/int country value.
    input_format
        The ISO 3166 input format of the country.
            - 'auto': infer the input format
            - 'name': country name ('United States')
            - 'official': official state name ('United States of America')
            - 'alpha-2': alpha-2 code ('US')
            - 'alpha-3': alpha-3 code ('USA')
            - 'numeric': numeric code (840)

        Can also be a tuple containing any combination of input formats,
        for example to clean a column containing alpha-2 and numeric
        codes set input_format to ('alpha-2', 'numeric').

        (default: 'auto')
    strict
        If True, matching for input formats 'name' and 'official' are done by
        looking for a direct match, if False, matching is done by searching
        the input for a regex match.

        (default: False)

    Examples
    --------

    >>> validate_country('United States')
    True
    >>> df = pd.DataFrame({'country': ['Canada', 'NaN']})
    >>> validate_country(df['country'])
    0     True
    1    False
    Name: country, dtype: bool
    """
    input_formats = _input_format_to_tuple(input_format)
    if isinstance(x, pd.Series):
        x = x.astype(str).str.lower().str.strip()
        return x.apply(_check_country, args=(input_formats, strict, False))

    x = str(x).lower().strip()
    return _check_country(x, input_formats, strict, False)


def _format_country(
    val: Any,
    input_formats: Tuple[str, ...],
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
    result_index, status = _check_country(country, input_formats, strict, True)

    if (
        fuzzy_dist > 0
        and status == "unknown"
        and ("name" in input_formats or "official" in input_formats)
    ):
        result_index, status = _check_fuzzy_dist(country, fuzzy_dist)

    if status == "null":
        return np.nan, 0
    if status in "none" or status in "unknown":
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


@lru_cache(maxsize=2**20)
def _check_country(country: str, input_formats: Tuple[str, ...], strict: bool, clean: bool) -> Any:
    """
    Finds the index of the given country in the DATA dataframe.

    Parameters
    ----------
    country
        string containing the country value being cleaned
    input_formats
        Tuple containing potential ISO 3166 input formats of the country
    strict
        If True, for input types "name" and "offical" the function looks for a direct match
        in the DATA dataframe. If False, the country input is searched for a regex match.
    clean
        If True, a tuple (index, status) is returned.
        If False, the function returns True/False to be used by the validate country function.
    """
    if country in "none" or country in NULL_VALUES:
        return (None, "null") if clean else False

    country_format = _get_format_from_name(country)
    input_format = _get_format_if_allowed(country_format, input_formats)
    if not input_format:
        return (None, "unknown") if clean else False

    if strict and input_format == "regex":
        for form in ("name", "official"):
            ind = DATA[
                DATA[form].str.contains(f"^{re.escape(country)}$", flags=re.IGNORECASE, na=False)
            ].index
            if np.size(ind) > 0:
                return (ind[0], "success") if clean else True

    elif not strict and input_format in ("regex", "name", "official"):
        for index, country_regex in enumerate(REGEXES):
            if country_regex.search(country):
                return (index, "success") if clean else True

    else:
        ind = DATA[
            DATA[input_format].str.contains(
                f"^{re.escape(country)}$", flags=re.IGNORECASE, na=False
            )
        ].index
        if np.size(ind) > 0:
            return (ind[0], "success") if clean else True

    return (None, "unknown") if clean else False


@lru_cache(maxsize=2**20)
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


def _get_format_if_allowed(input_format: str, allowed_formats: Tuple[str, ...]) -> Optional[str]:
    """
    Returns the input format if it's an allowed format.
    "regex" input_format is only returned if "name" and "official are
    allowed. This is because when strict = True and input_format = "regex"
    both the "name" and "official" columns in the DATA dataframe are checked.
    """
    if input_format == "regex":
        if "name" in allowed_formats and "official" in allowed_formats:
            return "regex"

        return (
            "name"
            if "name" in allowed_formats
            else "official" if "official" in allowed_formats else None
        )

    return input_format if input_format in allowed_formats else None


def _input_format_to_tuple(input_format: Union[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    """
    Converts a string input format to a tuple of allowed input formats and raises an error
    if an input format is not valid.
    """
    input_formats = {"auto", "name", "official", "alpha-2", "alpha-3", "numeric"}
    if isinstance(input_format, str):
        if input_format == "auto":
            return ("name", "official", "alpha-2", "alpha-3", "numeric")
        input_format = (input_format,)

    for fmt in input_format:
        if fmt not in input_formats:
            raise ValueError(
                f'input_format {fmt} is invalid, it needs to be one of "auto", '
                '"name", "official", "alpha-2", "alpha-3" or "numeric'
            )
    return input_format
