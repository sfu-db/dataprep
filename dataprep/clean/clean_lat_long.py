"""
Clean and validate a DataFrame column containing geographic coordinates.
"""

import re
from operator import itemgetter
from typing import Any, Optional, Tuple, Union

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..progress_bar import ProgressBar
from .utils import NULL_VALUES, create_report_new, to_dask

LAT_LONG_PATTERN = re.compile(
    r"""
    [^/-]*?[(]?
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
    [^/-]*?
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
    [^/-]*?
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
    errors: str = "coerce",
    report: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean and standardize latitude and longitude coordinates.

    Read more in the :ref:`User Guide <clean_lat_long_user_guide>`.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    lat_long
        The name of the column containing latitude and longitude coordinates.
    lat_col
        The name of the column containing latitude coordinates.

        If specified, the parameter lat_long must be None.
    long_col
        The name of the column containing longitude coordinates.

        If specified, the parameter lat_long must be None.
    output_format
        The desired format of the coordinates.
            - 'dd': decimal degrees (51.4934, 0.0098)
            - 'ddh': decimal degrees with hemisphere ('51.4934° N, 0.0098° E')
            - 'dm': degrees minutes ('51° 29.604′ N, 0° 0.588′ E')
            - 'dms': degrees minutes seconds ('51° 29′ 36.24″ N, 0° 0′ 35.28″ E')

        (default: 'dd')
    split
        If True, split the latitude and longitude coordinates into one column
        for latitude and a separate column for longitude. Otherwise, merge
        the latitude and longitude coordinates into one column.

        (default: False)
    inplace
        If True, delete the column(s) containing the data that was cleaned. Otherwise,
        keep the original column(s).

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
    Split a column containing latitude and longitude strings into separate
    columns in decimal degrees format.

    >>> df = pd.DataFrame({'coord': ['51° 29′ 36.24″ N, 0° 0′ 35.28″ E', '51.4934° N, 0.0098° E']})
    >>> clean_lat_long(df, 'coord', split=True)
    Latitude and Longitude Cleaning Report:
        2 values cleaned (100.0%)
    Result contains 2 (100.0%) values in the correct format and 0 null values (0.0%)
                            coord  latitude  longitude
    0  51° 29′ 36.24″ N, 0° 0′ 35.28″ E   51.4934     0.0098
    1             51.4934° N, 0.0098° E   51.4934     0.0098
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
    def clean_lat_long_helper(df, col, col_name):
        # A helper to clean a latitude and longitude column
        df["clean_code_tup"] = df[col].map_partitions(
            lambda srs: [_format_lat_or_long(x, output_format, errors, col_name) for x in srs],
            meta=object,
        )
        df = df.assign(
            _temp_=df["clean_code_tup"].map(itemgetter(0)),
            _code_=df["clean_code_tup"].map(itemgetter(1)),
        )
        df = df.rename(columns={"_temp_": f"{col}_clean"})
        if inplace:
            df = df.drop(columns=col)
        return df

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
            df = clean_lat_long_helper(df, lat_col, "lat")
        # clean a longitude column
        if long_col:
            df = clean_lat_long_helper(df, long_col, "long")
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
    x: Union[pd.Series, str, float, Tuple[float, float]],
    *,
    lat_long: bool = True,
    lat: bool = False,
    lon: bool = False,
) -> Union[bool, pd.Series]:
    """
    Validate latitude and longitude coordinates.

    Read more in the :ref:`User Guide <clean_lat_long_user_guide>`.

    Parameters
    ----------
    x
        A pandas Series, string, float, or tuple of floats, containing the latitude
        and/or longitude coordinates to be validated.
    lat_long
        If True, valid values contain latitude and longitude coordinates. Parameters
        lat and lon must be False if lat_long is True.

        (default: True)
    lat
        If True, valid values contain only latitude coordinates. Parameters
        lat_long and lon must be False if lat is True.

       (default: False)
    lon
        If True, valid values contain only longitude coordinates. Parameters
        lat_long and lat must be False if lon is True.

        (default: False)

    Examples
    --------
    Validate a coordinate string or series of coordinates.

    >>> validate_lat_long('51° 29′ 36.24″ N, 0° 0′ 35.28″ E')
    True
    >>> df = pd.DataFrame({'coordinates', ['51° 29′ 36.24″ N, 0° 0′ 35.28″ E', 'NaN']})
    >>> validate_lat_long(df['coordinates'])
    0     True
    1    False
    Name: coordinates, dtype: bool
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


def _format_lat_long(val: Any, output_format: str, split: bool, errors: str) -> Any:
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

    # minutes and seconds need to be in the interval [0, 60]
    # for degrees:
    #  if hemisphere is give, then 0<=deg<=bound
    #  if hemisphere is not given, then -bound<=deg<=bound
    # decimal degrees must be -bound<=lat<=bound
    # the first given hemisphere and last hemisphere cannot both be set
    if (
        not 0 <= mins <= 60
        or not 0 <= secs <= 60
        or hem
        and not 0 <= float(mch.group("deg")) <= bound
        or not hem
        and abs(float(mch.group("deg"))) > bound
        or abs(dds) > bound
        or sum([mch.group("dir_back") is not None, mch.group("dir_front") is not None]) > 1
    ):
        return (None,) * 4 + (1,) if clean else False

    return (dds, mins, secs, hem, 2) if clean else True
