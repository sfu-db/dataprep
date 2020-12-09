"""
Implement clean_lat_long function
"""
# pylint: disable=too-many-boolean-expressions
import re
from typing import Any, Optional, Tuple, Union

import dask.dataframe as dd
import dask
import numpy as np
import pandas as pd

from .utils import NULL_VALUES, create_report, to_dask

LAT_LONG_PATTERN = re.compile(
    r"""
    .*?[(]?
      (?P<dir_front>[NS])?[ ]*
        (?P<deg>-?%(FLOAT)s)(?:[%(DEGREE)sD\*\u00B0\s][ ]*
        (?:(?P<min>%(FLOAT)s)[%(PRIME)s'm]?[ ]*)?
        (?:(?P<sec>%(FLOAT)s)[%(DOUBLE_PRIME)s"s][ ]*)?
      )?(?P<dir_back>[NS])?
      \s*[,;/\s]\s*
      (?P<dir_front2>[EW])?[ ]*
        (?P<deg2>-?%(FLOAT)s)(?:[%(DEGREE)sD\*\u00B0\s][ ]*
        (?:(?P<min2>%(FLOAT)s)[%(PRIME)s'm]?[ ]*)?
        (?:(?P<sec2>%(FLOAT)s)[%(DOUBLE_PRIME)s"s][ ]*)?
      )?(?P<dir_back2>[EW])?
    [)]?\s*$
"""
    % {
        "FLOAT": r"\d+(?:\.\d+)?",
        "DEGREE": chr(176),
        "PRIME": chr(8242),
        "DOUBLE_PRIME": chr(8243),
    },
    re.VERBOSE | re.UNICODE,
)

LAT_PATTERN = re.compile(
    r"""
    .*?
      (?P<dir_front>[NS])?[ ]*
        (?P<deg>-?%(FLOAT)s)(?:[%(DEGREE)sD\*\u00B0\s][ ]*
        (?:(?P<min>%(FLOAT)s)[%(PRIME)s'm]?[ ]*)?
        (?:(?P<sec>%(FLOAT)s)[%(DOUBLE_PRIME)s"s][ ]*)?
      )?(?P<dir_back>[NS])?
    \s*$
"""
    % {
        "FLOAT": r"\d+(?:\.\d+)?",
        "DEGREE": chr(176),
        "PRIME": chr(8242),
        "DOUBLE_PRIME": chr(8243),
    },
    re.VERBOSE | re.UNICODE,
)

LONG_PATTERN = re.compile(
    r"""
    .*?
      (?P<dir_front>[EW])?[ ]*
        (?P<deg>-?%(FLOAT)s)(?:[%(DEGREE)sD\*\u00B0\s][ ]*
        (?:(?P<min>%(FLOAT)s)[%(PRIME)s'm]?[ ]*)?
        (?:(?P<sec>%(FLOAT)s)[%(DOUBLE_PRIME)s"s][ ]*)?
      )?(?P<dir_back>[EW])?
    \s*$
"""
    % {
        "FLOAT": r"\d+(?:\.\d+)?",
        "DEGREE": chr(176),
        "PRIME": chr(8242),
        "DOUBLE_PRIME": chr(8243),
    },
    re.VERBOSE | re.UNICODE,
)

STATS = {"cleaned": 0, "null": 0, "unknown": 0}


def clean_lat_long(
    df: Union[pd.DataFrame, dd.DataFrame],
    lat_long_col: Optional[str] = None,
    *,
    lat_col: Optional[str] = None,
    long_col: Optional[str] = None,
    output_format: str = "dd",
    split: bool = False,
    inplace: bool = False,
    report: bool = True,
    errors: str = "coerce",
) -> pd.DataFrame:
    """
    This function cleans latitdinal and longitudinal coordinates

    Parameters
    ----------
    df
        pandas or Dask DataFrame
    lat_long_col
        column name containing latitudinal and longitudinal coordinates
    lat_col
        column name containing latitudinal coordinates. If specified, lat_long_col
        must be None
    long_col
        column name containing longitudinal coordinates. If specified, lat_long_col
        must be None
    output_format
        the desired format of the coordinates: decimal degrees ("dd"),
        decimal degrees with hemisphere ("ddh"), degrees minutes ("dm"),
        degrees minutes seconds ("dms")
    split
        if True, split a column containing latitudinal and longitudinal
        coordinates into one column for latitude and one column for longitude
    inplace
        If True, delete the given column with dirty data, else, create a new
        column with cleaned data.
    report
        If True, output the summary report. Otherwise, no report is outputted.
    errors {‘ignore’, ‘raise’, ‘coerce’}, default 'coerce'
        * If ‘raise’, then invalid parsing will raise an exception.
        * If ‘coerce’, then invalid parsing will be set as NaT.
        * If ‘ignore’, then invalid parsing will return the input.
    """
    # pylint: disable=too-many-arguments,too-many-branches
    reset_stats()

    if lat_long_col and (lat_col or long_col):
        raise ValueError("lat_long_col must be None if either lat_col or long_col is not None")

    if output_format not in {"dd", "ddh", "dm", "dms"}:
        raise ValueError(
            f'output_format {output_format} is invalid, it needs to be "dd", "ddh", "dm", or "dms"'
        )

    df = to_dask(df)
    # specify the metadata for dask apply
    if lat_long_col:
        meta = df.dtypes.to_dict()
        if split:
            if output_format == "dd":
                meta.update(zip(("latitude", "longitude"), (float, float)))
            else:
                meta.update(zip(("latitude", "longitude"), (str, str)))
        else:
            meta[f"{lat_long_col}_clean"] = float if output_format == "dd" else str

        df = df.apply(
            format_lat_long,
            args=(lat_long_col, output_format, split, errors),
            axis=1,
            meta=meta,
        )
        if inplace:
            df = df.drop(columns=[lat_long_col])
    else:
        # clean a latitude column
        if lat_col:
            meta = df.dtypes.to_dict()
            meta[f"{lat_col}_clean"] = float if output_format == "dd" else str
            df = df.apply(
                format_lat_or_long,
                args=(lat_col, output_format, errors, "lat"),
                axis=1,
                meta=meta,
            )
            if inplace:
                df = df.drop(columns=[lat_col])
        # clean a longitude column
        if long_col:
            meta = df.dtypes.to_dict()
            meta[f"{long_col}_clean"] = float if output_format == "dd" else str
            df = df.apply(
                format_lat_or_long,
                args=(long_col, output_format, errors, "long"),
                axis=1,
                meta=meta,
            )
            if inplace:
                df = df.drop(columns=[long_col])
        # merge the cleaned latitude and longitude
        if lat_col and long_col and not split:
            if output_format == "dd":
                df["latitude_longitude"] = df[[f"{lat_col}_clean", f"{long_col}_clean"]].apply(
                    tuple, axis=1
                )
            else:
                df["latitude_longitude"] = df[f"{lat_col}_clean"] + ", " + df[f"{long_col}_clean"]
            df = df.drop(columns=[f"{lat_col}_clean", f"{long_col}_clean"])

    df, nrows = dask.compute(df, df.shape[0])

    # output the report describing the changes to the column
    if report:
        create_report("Latitude and Longitude", STATS, nrows)

    return df


def validate_lat_long(
    x: Union[str, float, pd.Series, Tuple[float, float]],
    *,
    lat_long: Optional[bool] = True,
    lat: Optional[bool] = False,
    lon: Optional[bool] = False,
) -> Union[bool, pd.Series]:
    """
    This function validates latitdinal and longitudinal coordinates

    Parameters
    ----------
    x
        pandas Series of coordinates or str/float coordinate to be validated
    lat_long
        latitudinal and longitudinal coordinates
    lat
        only latitudinal coordinates
    lon
        only longitudinal coordinates
    """

    if lat or lon:
        hor_dir = "lat" if lat else "long"
        if isinstance(x, pd.Series):
            return x.apply(check_lat_or_long, args=(False, hor_dir))
        else:
            return check_lat_or_long(x, False, hor_dir)
    elif lat_long:
        if isinstance(x, pd.Series):
            return x.apply(check_lat_long, args=(False,))
        else:
            return check_lat_long(x, False)

    return None


def format_lat_long(
    row: pd.Series,
    col: str,
    output_format: str,
    split: bool,
    errors: str,
) -> pd.Series:
    """
    Function to transform a coordinate instance into the
    desired format
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    # check_lat_long parses the value in row[col], and will return the components
    # if the parse is succesful. The returned value "status" can be either "null"
    # (which means row[col] contains a null value), "unknown" (in which case the value
    # in row[col] could not be parsed) or "success" (a succesful parse of the value).
    # dds, mins, secs, hem are the latitude components and # dds2, mins2, secs2, hem2
    # are the longitude components
    dds, mins, secs, hem, dds2, mins2, secs2, hem2, status = check_lat_long(row[col], True)

    if status == "null":  # row[col] contains a null value
        STATS["null"] += 1
        if split:
            row["latitude"], row["longitude"] = np.nan, np.nan
        else:
            row[f"{col}_clean"] = np.nan
        return row

    if status == "unknown":  # row[col] contains an unknown value
        if errors == "raise":
            raise ValueError(f"unable to parse value {row[col]}")

        STATS["unknown"] += 1
        if split:
            row["latitude"] = row[col] if errors == "ignore" else np.nan
            row["longitude"] = np.nan
        else:
            row[f"{col}_clean"] = row[col] if errors == "ignore" else np.nan
        return row

    # derive the hemisphere if not given in the initial coordinate
    if not hem:
        hem = "N" if dds >= 0 else "S"
    if not hem2:
        hem2 = "E" if dds2 >= 0 else "W"
    dds, dds2 = abs(dds), abs(dds2)

    # the following code if/elif blocks converts the
    # coordinate components to the desired output
    # https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#Change_of_units_and_format
    if output_format == "dd":
        fctr = -1 if hem == "S" else 1
        fctr2 = -1 if hem2 == "W" else 1
        lat, lon = round(fctr * dds, 4), round(fctr2 * dds2, 4)
    elif output_format == "ddh":
        lat = f"{round(dds, 4)}{chr(176)} {hem}"
        lon = f"{round(dds2, 4)}{chr(176)} {hem2}"
    elif output_format == "dm":
        mins = round(60 * (dds - int(dds)), 4)
        mins = int(mins) if mins.is_integer() else mins
        mins2 = round(60 * (dds2 - int(dds2)), 4)
        mins2 = int(mins2) if mins2.is_integer() else mins2
        lat = f"{int(dds)}{chr(176)} {mins}{chr(8242)} {hem}"
        lon = f"{int(dds2)}{chr(176)} {mins2}{chr(8242)} {hem2}"
    elif output_format == "dms":
        mins = int(60 * (dds - int(dds)))
        secs = round(3600 * (dds - int(dds)) - 60 * mins, 4)
        secs = int(secs) if secs.is_integer() else secs
        mins2 = int(60 * (dds2 - int(dds2)))
        secs2 = round(3600 * (dds2 - int(dds2)) - 60 * mins2, 4)
        secs2 = int(secs2) if secs2.is_integer() else secs2
        lat = f"{int(dds)}{chr(176)} {mins}{chr(8242)} {secs}{chr(8243)} {hem}"
        lon = f"{int(dds2)}{chr(176)} {mins2}{chr(8242)} {secs2}{chr(8243)} {hem2}"
    if split:
        STATS["cleaned"] += 1
        row["latitude"], row["longitude"] = lat, lon
    else:
        row[f"{col}_clean"] = (lat, lon) if output_format == "dd" else f"{lat}, {lon}"
        if row[col] != row[f"{col}_clean"]:
            STATS["cleaned"] += 1

    return row


def check_lat_long(val: Union[str, float, Any], clean: bool) -> Any:
    """
    Function to transform a coordinate instance into the
    desired format
    """
    # if the value is null, return empty strings for the components
    # and "null" for the "status"
    if val in NULL_VALUES:
        return [""] * 8 + ["null"] if clean else False

    mch = re.match(LAT_LONG_PATTERN, re.sub(r"''", r'"', str(val)))
    # check if the value was able to be parsed
    if not mch:
        return [""] * 8 + ["unknown"] if clean else False
    if not mch.group("deg") or not mch.group("deg2"):
        return [""] * 8 + ["unknown"] if clean else False

    # coordinates for latitude
    mins = float(mch.group("min")) if mch.group("min") else 0
    secs = float(mch.group("sec")) if mch.group("sec") else 0
    dds = float(mch.group("deg")) + mins / 60 + secs / 3600
    hem = mch.group("dir_back") or mch.group("dir_front")

    # coordinates for longitude
    mins2 = float(mch.group("min2")) if mch.group("min2") else 0
    secs2 = float(mch.group("sec2")) if mch.group("sec2") else 0
    dds2 = float(mch.group("deg2")) + mins2 / 60 + secs2 / 3600
    hem2 = mch.group("dir_back2") or mch.group("dir_front2")

    # minutes and seconds need to be in the interval [0, 60)
    # for degrees:
    #  if hemisphere is give, then 0<=lat<=90 and 0<=long<=180
    #  if hemisphere is not given, then -90<=lat<=90 and -180<=long<=180
    # decimal degrees must be -90<=lat<=90 and -180<=long<=180
    # the first given hemisphere and last hemisphere cannot both be set
    if (
        not 0 <= mins < 60
        or not 0 <= mins2 < 60
        or not 0 <= secs < 60
        or not 0 <= secs2 < 60
        or hem
        and not 0 <= float(mch.group("deg")) <= 90
        or hem2
        and not 0 <= float(mch.group("deg2")) <= 180
        or not hem
        and abs(float(mch.group("deg"))) > 90
        or not hem2
        and abs(float(mch.group("deg2"))) > 180
        or abs(dds) > 90
        or abs(dds2) > 180
        or sum([mch.group("dir_back") is not None, mch.group("dir_front") is not None]) > 1
        or sum([mch.group("dir_back2") is not None, mch.group("dir_front2") is not None]) > 1
    ):
        return [""] * 8 + ["unknown"] if clean else False

    return (dds, mins, secs, hem, dds2, mins2, secs2, hem2, "success") if clean else True


def format_lat_or_long(
    row: pd.Series, col: str, output_format: str, errors: str, hor_dir: str
) -> pd.Series:
    """
    Function to transform a coordinate instance into the
    desired format
    """
    dds, mins, secs, hem, status = check_lat_or_long(row[col], True, hor_dir)

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

    if not hem:
        if hor_dir == "lat":
            hem = "N" if dds >= 0 else "S"
        else:
            hem = "E" if dds >= 0 else "W"
    dds = abs(dds)

    if output_format == "dd":
        fctr = 1 if hem in {"N", "E"} else -1
        res = round(fctr * dds, 4)
    if output_format == "ddh":
        res = f"{round(dds, 4)}{chr(176)} {hem}"
    elif output_format == "dm":
        mins = round(60 * (dds - int(dds)), 4)
        mins = int(mins) if mins.is_integer() else mins
        res = f"{int(dds)}{chr(176)} {mins}{chr(8242)} {hem}"
    elif output_format == "dms":
        mins = int(60 * (dds - int(dds)))
        secs = round(3600 * (dds - int(dds)) - 60 * mins, 4)
        secs = int(secs) if secs.is_integer() else secs
        res = f"{int(dds)}{chr(176)} {mins}{chr(8242)} {secs}{chr(8243)} {hem}"

    row[f"{col}_clean"] = res
    if row[col] != row[f"{col}_clean"]:
        STATS["cleaned"] += 1

    return row


def check_lat_or_long(val: Union[str, float, Any], clean: bool, hor_dir: str) -> Any:
    """
    Function to transform a coordinate instance into the
    desired format
    """
    if val in NULL_VALUES:
        return [""] * 4 + ["null"] if clean else False

    pat = LAT_PATTERN if hor_dir == "lat" else LONG_PATTERN

    mch = re.match(pat, re.sub(r"''", r'"', str(val)))
    if not mch:
        return [""] * 4 + ["unknown"] if clean else False
    if not mch.group("deg"):
        return [""] * 4 + ["unknown"] if clean else False

    # coordinates
    mins = float(mch.group("min")) if mch.group("min") else 0
    secs = float(mch.group("sec")) if mch.group("sec") else 0
    dds = float(mch.group("deg")) + mins / 60 + secs / 3600
    hem = mch.group("dir_back") or mch.group("dir_front")

    # range is [-90, 90] for latitude and [-180, 180] for longitude
    bound = 90 if hor_dir == "lat" else 180

    # minutes and seconds need to be in the interval [0, 60)
    # for degrees:
    #  if hemisphere is give, then 0<=deg<=bound
    #  if hemisphere is not given, then -bound<=deg<=bound
    # decimal degrees must be -bound<=lat<=bound
    # the first given hemisphere and last hemisphere cannot both be set
    if (
        not 0 <= mins < 60
        or not 0 <= secs < 60
        or hem
        and not 0 <= float(mch.group("deg")) <= bound
        or not hem
        and abs(float(mch.group("deg"))) > bound
        or abs(dds) > bound
        or sum([mch.group("dir_back") is not None, mch.group("dir_front") is not None]) > 1
    ):
        return [""] * 4 + ["unknown"] if clean else False

    return (dds, mins, secs, hem, "success") if clean else True


def reset_stats() -> None:
    """
    Reset global statistics dictionary
    """
    STATS["cleaned"] = 0
    STATS["null"] = 0
    STATS["unknown"] = 0
