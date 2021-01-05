"""
Implement clean_lat_long function
"""
import re
from operator import itemgetter
from typing import Any, Optional, Tuple, Union

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..eda.progress_bar import ProgressBar
from .utils import NULL_VALUES, create_report_new, to_dask

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


def clean_lat_long(
    df: Union[pd.DataFrame, dd.DataFrame],
    lat_long: Optional[str] = None,
    *,
    lat_col: Optional[str] = None,
    long_col: Optional[str] = None,
    output_format: str = "dd",
    split: bool = False,
    inplace: bool = False,
    report: bool = True,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    This function cleans latitdinal and longitudinal coordinates

    Parameters
    ----------
    df
        pandas or Dask DataFrame
    lat_long
        column name containing latitudinal and longitudinal coordinates
    lat_col
        column name containing latitudinal coordinates. If specified, lat_long
        must be None
    long_col
        column name containing longitudinal coordinates. If specified, lat_long
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
    progress
        If True, enable the progress bar
    """
    # pylint: disable=too-many-branches

    if lat_long and (lat_col or long_col):
        raise ValueError("lat_long must be None if either lat_col or long_col is not None")

    if output_format not in {"dd", "ddh", "dm", "dms"}:
        raise ValueError(
            f'output_format {output_format} is invalid, it must be "dd", "ddh", "dm", or "dms"'
        )

    # convert to dask
    df = to_dask(df)

    # To clean, create a new column "clean_code_tup" which contains
    # the cleaned values and code indicating how the initial value was
    # changed in a tuple. Then split the column of tuples and count the
    # amount of different codes to produce the report
    if lat_long:
        # clean a latitude and longitude column
        df["clean_code_tup"] = df[lat_long].map_partitions(
            lambda srs: [_format_lat_long(x, output_format, split, errors) for x in srs],
            meta=object,
        )
        if split:
            df = df.assign(
                latitude=df["clean_code_tup"].map(itemgetter(0)),
                longitude=df["clean_code_tup"].map(itemgetter(1)),
                _code_=df["clean_code_tup"].map(itemgetter(2)),
            )
        else:
            df = df.assign(
                _temp_=df["clean_code_tup"].map(itemgetter(0)),
                _code_=df["clean_code_tup"].map(itemgetter(1)),
            )
            df = df.rename(columns={"_temp_": f"{lat_long}_clean"})
        if inplace:
            df = df.drop(columns=lat_long)
    else:
        # clean a latitude column
        if lat_col:
            df["clean_code_tup"] = df[lat_col].map_partitions(
                lambda srs: [_format_lat_or_long(x, output_format, errors, "lat") for x in srs],
                meta=object,
            )
            df = df.assign(
                _temp_=df["clean_code_tup"].map(itemgetter(0)),
                _code_=df["clean_code_tup"].map(itemgetter(1)),
            )
            df = df.rename(columns={"_temp_": f"{lat_col}_clean"})
            if inplace:
                df = df.drop(columns=lat_col)
        # clean a longitude column
        if long_col:
            df["clean_code_tup"] = df[long_col].map_partitions(
                lambda srs: [_format_lat_or_long(x, output_format, errors, "long") for x in srs],
                meta=object,
            )
            df = df.assign(
                _temp_=df["clean_code_tup"].map(itemgetter(0)),
                _code_=df["clean_code_tup"].map(itemgetter(1)),
            )
            df = df.rename(columns={"_temp_": f"{long_col}_clean"})
            if inplace:
                df = df.drop(columns=long_col)
        # merge the cleaned latitude and longitude
        if lat_col and long_col and not split:
            if output_format == "dd":
                df["latitude_longitude"] = df[[f"{lat_col}_clean", f"{long_col}_clean"]].apply(
                    tuple, axis=1, meta=object
                )
            else:
                df["latitude_longitude"] = df[f"{lat_col}_clean"] + ", " + df[f"{long_col}_clean"]

            # if seperate lat and long columns are merged, then all values are "cleaned"
            df["_code_"] = 2
            df = df.drop(columns=[f"{lat_col}_clean", f"{long_col}_clean"])

    # counts of codes indicating how values were changed
    stats = df["_code_"].value_counts(sort=False)
    df = df.drop(columns=["clean_code_tup", "_code_"])

    with ProgressBar(minimum=1, disable=not progress):
        df, stats = dask.compute(df, stats)

    # output a report describing the result of clean_lat_long
    if report:
        create_report_new("Latitude and Longitude", stats, errors)

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
            return x.apply(_check_lat_or_long, args=(False, hor_dir))
        return _check_lat_or_long(x, False, hor_dir)
    elif lat_long:
        if isinstance(x, pd.Series):
            return x.apply(_check_lat_long, args=(False,))
        return _check_lat_long(x, False)

    return None


def _format_lat_long(
    val: Any,
    output_format: str,
    split: bool,
    errors: str,
) -> Any:
    """
    Function to transform a coordinate instance into the desired format

    The last component of the returned tuple contains a code indicating how the
    input value was changed:
        0 := the value is null
        1 := the value could not be parsed
        2 := the value is cleaned and the cleaned value is DIFFERENT than the input value
        3 := the value is cleaned and is THE SAME as the input value (no transformation)
    """
    # pylint: disable=too-many-locals
    # _check_lat_long parses the value val, and will return the components
    # if the parse is succesful. The returned value "status" can be either 0 ie
    # "null" (which means val is a null value), 1 ie ("unkwonw") (in which case
    # val could not be parsed) or 2 ie "success" (a succesful parse of val).
    # dds, mins, secs, hem are the latitude components and # dds2, mins2, secs2, hem2
    # are the longitude components
    dds, mins, secs, hem, dds2, mins2, secs2, hem2, status = _check_lat_long(val, True)

    if status == 0:  # val is a null value
        return (np.nan, np.nan, 0) if split else (np.nan, 0)

    if status == 1:  # val contains an unknown value
        if errors == "raise":
            raise ValueError(f"unable to parse value {val}")
        result = val if errors == "ignore" else np.nan
        return (result, np.nan, 1) if split else (result, 1)

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
        return lat, lon, 2
    result = (lat, lon) if output_format == "dd" else f"{lat}, {lon}"
    return result, 2 if val != result else 3


def _check_lat_long(val: Any, clean: bool) -> Any:
    """
    Function to check if a coordinate instance is valid
    """
    # pylint: disable=too-many-boolean-expressions
    # if the value is null, return empty strings for the components
    # and the code 0 to indicate a null status
    if val in NULL_VALUES:
        return (None,) * 8 + (0,) if clean else False

    mch = re.match(LAT_LONG_PATTERN, re.sub(r"''", r'"', str(val)))
    # check if the value was able to be parsed
    if not mch:
        return (None,) * 8 + (1,) if clean else False
    if not mch.group("deg") or not mch.group("deg2"):
        return (None,) * 8 + (1,) if clean else False

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
    #  if hemisphere is given, then 0<=lat<=90 and 0<=long<=180
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
        return (None,) * 8 + (1,) if clean else False

    return (dds, mins, secs, hem, dds2, mins2, secs2, hem2, 2) if clean else True


def _format_lat_or_long(val: Any, output_format: str, errors: str, hor_dir: str) -> Any:
    """
    Function to transform a coordinate instance into the desired format
    """
    dds, mins, secs, hem, status = _check_lat_or_long(val, True, hor_dir)

    if status == 0:  # val contains a null value
        return np.nan, 0

    if status == 1:  # val contains an unknown value
        if errors == "raise":
            raise ValueError(f"unable to parse value {val}")
        return val if errors == "ignore" else np.nan, 1

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

    return res, 2 if val != res else 3


def _check_lat_or_long(val: Any, clean: bool, hor_dir: str) -> Any:
    """
    Function to check if a coordinate instance is valid
    """
    # pylint: disable=too-many-boolean-expressions
    if val in NULL_VALUES:
        return (None,) * 4 + (0,) if clean else False

    pat = LAT_PATTERN if hor_dir == "lat" else LONG_PATTERN

    mch = re.match(pat, re.sub(r"''", r'"', str(val)))
    if not mch:
        return (None,) * 4 + (1,) if clean else False
    if not mch.group("deg"):
        return (None,) * 4 + (1,) if clean else False

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
        return (None,) * 4 + (1,) if clean else False

    return (dds, mins, secs, hem, 2) if clean else True
