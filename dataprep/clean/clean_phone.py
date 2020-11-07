"""
Implement clean_phone function
"""

import re
from typing import Any, Union

import dask.dataframe as dd
import dask
import numpy as np
import pandas as pd

from .utils import NULL_VALUES, create_report, to_dask


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
    (?:[ \t]*(?:\#|x[.:]?|ext[.:]?|extension)[ \t]*(?P<ext>\d+))?
    \s*$
    """,
    re.VERBOSE,
)

STATS = {"cleaned": 0, "null": 0, "unknown": 0}


def clean_phone(
    df: Union[pd.DataFrame, dd.DataFrame],
    col: str,
    output_format: str = "nanp",
    fix_missing: str = "empty",
    split: bool = False,
    inplace: bool = False,
    report: bool = True,
    errors: str = "coerce",
) -> pd.DataFrame:
    """
    This function cleans phone numbers.

    Parameters
    ----------
    df
        Pandas or Dask DataFrame.
    col
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
    """
    # pylint: disable=too-many-arguments
    reset_stats()

    if output_format not in {"nanp", "e164", "national"}:
        raise ValueError(
            f'output_format {output_format} is invalid, it needs to be "nanp", "e164" or "national"'
        )

    if fix_missing not in {"auto", "empty"}:
        raise ValueError(
            f'output_format {output_format} is invalid, it needs to be "auto" or "empty"'
        )

    df = to_dask(df)
    # specify the metadata for dask apply
    meta = df.dtypes.to_dict()
    if split:
        meta.update(
            zip(("country_code", "area_code", "office_code", "station_code", "ext_num"), (str,) * 5)
        )
    else:
        meta[f"{col}_clean"] = str

    df = df.apply(
        format_phone,
        args=(col, output_format, fix_missing, split, errors),
        axis=1,
        meta=meta,
    )

    if inplace:
        df = df.drop(columns=[col])

    df, nrows = dask.compute(df, df.shape[0])

    # output the report describing the changes to the column
    if report:
        create_report("Phone Number", STATS, nrows)

    return df


def format_phone(
    row: pd.Series,
    col: str,
    output_format: str,
    fix_missing: str,
    split: bool,
    errors: str,
) -> pd.Series:
    """
    Function to transform a phone number instance into the
    desired format.
    """
    # pylint: disable=too-many-arguments,too-many-branches
    country_code, area_code, office_code, station_code, ext_num, status = check_phone(
        row[col], True
    )

    if status == "null":
        STATS["null"] += 1
        if split:
            (
                row["country_code"],
                row["area_code"],
                row["office_code"],
                row["station_code"],
                row["ext_num"],
            ) = (np.nan,) * 5
        else:
            row[f"{col}_clean"] = np.nan
        return row

    if status == "unknown":
        if errors == "raise":
            raise ValueError(f"unable to parse value {row[col]}")

        STATS["unknown"] += 1
        if split:
            row["country_code"] = row[col] if errors == "ignore" else np.nan
            row["area_code"], row["office_code"], row["station_code"], row["ext_num"] = (
                np.nan,
            ) * 4
            row[f"{col}_clean"] = row[col] if errors == "ignore" else np.nan
        return row

    if split:
        STATS["cleaned"] += 1
        if fix_missing == "auto" and area_code is not None:
            country_code = country_code if country_code is not None else "1"
        else:
            country_code = country_code if country_code is not None else np.nan
        area_code = area_code if area_code is not None else np.nan
        ext_num = ext_num if ext_num is not None else np.nan
        (
            row["country_code"],
            row["area_code"],
            row["office_code"],
            row["station_code"],
            row["ext_num"],
        ) = (country_code, area_code, office_code, station_code, ext_num)
    else:
        if output_format == "nanp":
            area_code = f"{area_code}-" if area_code is not None else ""
            ext_num = f" ext. {ext_num}" if ext_num is not None else ""
            row[f"{col}_clean"] = f"{area_code}{office_code}-{station_code}{ext_num}"
        elif output_format == "e164":
            country_code = "+1" if area_code is not None else ""
            area_code = area_code if area_code is not None else ""
            ext_num = f" ext. {ext_num}" if ext_num is not None else ""
            row[f"{col}_clean"] = f"{country_code}{area_code}{office_code}{station_code}{ext_num}"
        elif output_format == "national":
            area_code = f"({area_code}) " if area_code is not None else ""
            ext_num = f" ext. {ext_num}" if ext_num is not None else ""
            row[f"{col}_clean"] = f"{area_code}{office_code}-{station_code}{ext_num}"
        if row[col] != row[f"{col}_clean"]:
            STATS["cleaned"] += 1

    return row


def check_phone(val: Union[str, int, Any], clean: bool) -> Any:
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
    val = str(val)

    # If the value is null, return empty strings for the components
    # and "null" for the "status"
    if val in NULL_VALUES:
        return [""] * 5 + ["null"] if clean else False

    mch = re.match(CA_US_PATTERN, re.sub(r"''", r'"', val))
    # Check if the value was able to be parsed
    if not mch:
        return [""] * 5 + ["unknown"] if clean else False
    if mch.group("country") and not mch.group("area"):
        return [""] * 5 + ["unknown"] if clean else False

    # Components for phone number
    country_code = mch.group("country")
    area_code = mch.group("area")
    office_code = mch.group("office")
    station_code = mch.group("station")
    ext_num = mch.group("ext")

    return (
        (country_code, area_code, office_code, station_code, ext_num, "success") if clean else True
    )


def validate_phone(x: Union[str, pd.Series]) -> Union[bool, pd.Series]:
    """
    Function to validate phone numbers.

    Parameters
    ----------
    x
        String or Pandas Series of phone numbers to be validated.
    """

    if isinstance(x, pd.Series):
        return x.apply(check_phone, clean=False)
    else:
        return check_phone(x, False)


def reset_stats() -> None:
    """
    Reset global statistics dictionary.
    """
    STATS["cleaned"] = 0
    STATS["null"] = 0
    STATS["unknown"] = 0
