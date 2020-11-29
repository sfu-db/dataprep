"""
Implement clean_phone function
"""
import re
from operator import itemgetter
from typing import Any, Union

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..eda.progress_bar import ProgressBar
from .utils import NULL_VALUES, create_report_new, to_dask

CA_US_PATTERN = re.compile(
    r"""
    ^\s*
    (?:[+(]?(?P<country>1)[)\/]?)?
    [-. (]*
    (?P<area>\d{3})?
    [-. )\/]*
    (?P<office>\d{3})
    [-. \/]*
    (?P<station>\d{4})
    (?:[ \t]*(?:\#|x[.:]?|[Ee]xt[.:]?|[Ee]xtension)[ \t]*(?P<ext>\d+))?
    \s*$
    """,
    re.VERBOSE,
)


def clean_phone(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    output_format: str = "nanp",
    fix_missing: str = "empty",
    split: bool = False,
    inplace: bool = False,
    report: bool = True,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    This function cleans phone numbers.

    Parameters
    ----------
    df
        Pandas or Dask DataFrame.
    column
        Column name containing phone numbers.
    output_format
        The desired format of the phone numbers.
        "nanp": NPA-NXX-XXXX
        "e164": +1NPANXXXXXX
        "national": (NPA) NXX-XXXX
    fix_missing
        Fix the missing country code of a parsed phone number. If "empty",
        leave the missing component as is. If "auto", set the country
        code to a default value.
    split
        If True, split a column containing a phone number into different
        columns containing individual components.
    inplace
        If True, delete the given column with dirty data. Else, create a new
        column with cleaned data.
    report
        If True, output the summary report. Else, no report is outputted.
    errors {'ignore', 'raise', 'coerce'}, default 'coerce'.
        * If 'raise', then invalid parsing will raise an exception.
        * If 'coerce', then invalid parsing will be set as NaN.
        * If 'ignore', then invalid parsing will return the input.
    progress
        If True, enable the progress bar
    """
    # pylint: disable=too-many-arguments

    if output_format not in {"nanp", "e164", "national"}:
        raise ValueError(
            f'output_format {output_format} is invalid, it needs to be "nanp", "e164" or "national"'
        )

    if fix_missing not in {"auto", "empty"}:
        raise ValueError(
            f'output_format {output_format} is invalid, it needs to be "auto" or "empty"'
        )

    # convert to dask
    df = to_dask(df)

    # To clean, create a new column "clean_code_tup" which contains
    # the cleaned values and code indicating how the initial value was
    # changed in a tuple. Then split the column of tuples and count the
    # amount of different codes to produce the report
    df["clean_code_tup"] = df[column].map_partitions(
        lambda srs: [_format_phone(x, output_format, fix_missing, split, errors) for x in srs],
        meta=object,
    )
    if split:
        # For some reason the meta data for the last 3 components needs to be
        # set. I think this is a dask bug
        df = df.assign(
            country_code=df["clean_code_tup"].map(itemgetter(0)),
            area_code=df["clean_code_tup"].map(itemgetter(1)),
            office_code=df["clean_code_tup"].map(itemgetter(2)),
            station_code=df["clean_code_tup"].map(itemgetter(3), meta=("station_code", object)),
            ext_num=df["clean_code_tup"].map(itemgetter(4), meta=("ext_num", object)),
            _code_=df["clean_code_tup"].map(itemgetter(5), meta=("_code_", object)),
        )
    else:
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

    # output a report describing the result of clean_phone
    if report:
        create_report_new("Phone Number", stats, errors)

    return df


def validate_phone(x: Union[str, pd.Series]) -> Union[bool, pd.Series]:
    """
    Function to validate phone numbers.

    Parameters
    ----------
    x
        String or Pandas Series of phone numbers to be validated.
    """

    if isinstance(x, pd.Series):
        return x.apply(_check_phone, clean=False)
    return _check_phone(x, False)


def _format_phone(
    phone: Any,
    output_format: str,
    fix_missing: str,
    split: bool,
    errors: str,
) -> Any:
    """
    Function to transform a phone number instance into the desired format.

    The last component of the returned tuple contains a code indicating how the
    input value was changed:
        0 := the value is null
        1 := the value could not be parsed
        2 := the value is cleaned and the cleaned value is DIFFERENT than the input value
        3 := the value is cleaned and is THE SAME as the input value (no transformation)
    """
    country_code, area_code, office_code, station_code, ext_num, status = _check_phone(phone, True)

    if status == "null":
        return (np.nan, np.nan, np.nan, np.nan, np.nan, 0) if split else (np.nan, 0)

    if status == "unknown":
        if errors == "raise":
            raise ValueError(f"unable to parse value {phone}")
        result = phone if errors == "ignore" else np.nan
        return (result, np.nan, np.nan, np.nan, np.nan, 1) if split else (result, 1)

    if split:
        missing_code = "1" if fix_missing == "auto" and area_code else np.nan
        country_code = country_code if country_code else missing_code
        area_code = area_code if area_code else np.nan
        ext_num = ext_num if ext_num else np.nan
        return country_code, area_code, office_code, station_code, ext_num, 2

    if output_format == "nanp":  # NPA-NXX-XXXX
        area_code = f"{area_code}-" if area_code else ""
        ext_num = f" ext. {ext_num}" if ext_num else ""
        result = f"{area_code}{office_code}-{station_code}{ext_num}"
    elif output_format == "e164":  # +1NPANXXXXXX
        country_code = "+1" if area_code else ""
        area_code = area_code if area_code else ""
        ext_num = f" ext. {ext_num}" if ext_num else ""
        result = f"{country_code}{area_code}{office_code}{station_code}{ext_num}"
    elif output_format == "national":  # (NPA) NXX-XXXX
        area_code = f"({area_code}) " if area_code else ""
        ext_num = f" ext. {ext_num}" if ext_num else ""
        result = f"{area_code}{office_code}-{station_code}{ext_num}"

    return result, 2 if phone != result else 3


def _check_phone(phone: Any, clean: bool) -> Any:
    """
    Function to parse a phone number and return the components if the
    parse is successful.

    Parameters
    ----------
    val
        Phone number to be parsed.
    clean
        If True, return the components of the parse (if successful) and
        the status "null" (if the value is null), "unknown" (if the value
        could not be parsed) or "success" (if the value was successfully
        parsed). Else, return False for an unsuccesful parse and True
        for a successful parse.
    """
    # If the value is null, return None for the components
    # and "null" for the "status"
    if phone in NULL_VALUES:
        return (None,) * 5 + ("null",) if clean else False

    mch = re.match(CA_US_PATTERN, re.sub(r"''", r'"', str(phone)))
    # Check if the value was able to be parsed
    if not mch:
        return (None,) * 5 + ("unknown",) if clean else False
    if mch.group("country") and not mch.group("area"):
        return (None,) * 5 + ("unknown",) if clean else False

    # Components for phone number
    country_code = mch.group("country")
    area_code = mch.group("area")
    office_code = mch.group("office")
    station_code = mch.group("station")
    ext_num = mch.group("ext")

    return (
        (country_code, area_code, office_code, station_code, ext_num, "success") if clean else True
    )
