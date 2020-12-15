"""
Implement clean_date function
"""
from typing import Any, Union, Dict, List
import datetime
from datetime import timedelta
from copy import deepcopy
import pytz
from pytz import all_timezones

import dask.dataframe as dd
import dask
import numpy as np
import pandas as pd

from .utils import create_report, to_dask
from .clean_date_utils import (
    JUMP,
    WEEKDAYS,
    MONTHS,
    AM,
    PM,
    ZONE,
    TARGET_YEAR,
    TARGET_MONTH,
    TARGET_DAY,
    TARGET_HOUR,
    TARGET_MINUTE,
    TARGET_SECOND,
    TEXT_MONTHS,
    TARGET_WEEKDAY,
    TEXT_WEEKDAYS,
    split,
    check_date,
    fix_empty_auto_minimum,
    fix_empty_auto_nearest,
)
from .clean_date_utils import ParsedDate, ParsedTargetFormat

STATS = {"cleaned": 0, "null": 0, "unknown": 0}


def clean_date(
    df: Union[pd.DataFrame, dd.DataFrame],
    col: str,
    target_format: str = "YYYY-MM-DD hh:mm:ss",
    origin_timezone: str = "UTC",
    target_timezone: str = "",
    fix_empty: str = "auto_minimum",
    show_report: bool = True,
) -> pd.DataFrame:
    """
    This function cleans date string.
    Parameters
    ----------
    df
        Pandas or Dask DataFrame.
    col
        Column name containing phone numbers.
    target_format
        The desired format of the date.
        Defalut value is 'YYYY-MM-DD hh:mm:ss'
    origin_timezone
        Timezone of origin data
    target_timezone
        Timezone of target data
    fix_empty
        The user can specify the way of fixing empty value from value set:
            {'empty', 'auto_nearest', 'auto_minimum'}.
        The default fixed_empty is "auto_minimum":
            For hours, minutes and seconds:
                Just fill them with zeros.
            For years, months and days:
                Fill it with the minimum value.
        "auto_nearest":
            For hours, minutes and seconds:
                Fill them with nearest hour, minutes and seconds.
            For years, months and days:
                Fill it with the nearest value.
        "empty":
            Just left the missing component as it is
    show_report
        If True, output the summary report. Else, no report is outputted.
    """
    # pylint: disable=too-many-arguments
    reset_stats()

    if fix_empty not in {"auto_minimum", "auto_nearest", "empty"}:
        raise ValueError(
            f"fix_empty {fix_empty} is invalid. "
            f'It needs to be "auto_minimum", "auto_nearest" or "empty"'
        )
    if origin_timezone not in all_timezones and origin_timezone not in ZONE:
        raise ValueError(f"origin_timezone {origin_timezone} doesn't exist")
    if (
        target_timezone not in all_timezones
        and target_timezone not in ZONE
        and target_timezone != ""
    ):
        raise ValueError(f"target_timezone {target_timezone} doesn't exist")
    df = to_dask(df)
    # specify the metadata for dask apply
    meta = df.dtypes.to_dict()
    meta[f"{col}_clean"] = str
    df = df.apply(
        format_date,
        args=(
            col,
            target_format,
            {"origin_tz": origin_timezone, "target_tz": target_timezone},
            fix_empty,
        ),
        axis=1,
        meta=meta,
    )
    df, nrows = dask.compute(df, df.shape[0])
    # output the report describing the changes to the column
    if show_report:
        create_report("Date", STATS, nrows)
    return df


def format_date(
    row: pd.Series,
    col: str,
    target_format: str,
    tz_info: Dict[str, Union[str, int]],
    fix_empty: str,
) -> pd.Series:
    """
    This function cleans date string.
    Parameters
    ----------
    df, col, target_format, tz_info, fix_empty
        same as explained in clean_date function
    """
    date = row[col]
    origin_timezone = tz_info["origin_tz"]
    target_timezone = tz_info["target_tz"]
    if check_date(date) == "null":
        STATS["null"] += 1
        row[f"{col}_clean"] = np.nan
    elif check_date(date) == "unknown":
        STATS["unknown"] += 1
        row[f"{col}_clean"] = np.nan
    elif check_date(date) == "cleaned":
        # Handle date data and timezone
        parsed_date_data = parse(date=date, fix_empty=fix_empty)
        parsed_date_data.set_tzinfo(timezone=origin_timezone)
        parsed_date_data = set_timezone_offset(origin_timezone, parsed_date_data)
        # Handle target format and timezone
        parsed_target_format_data = check_target_format(target_format=target_format)
        parsed_target_format_data = set_timezone_offset(target_timezone, parsed_target_format_data)
        if parsed_target_format_data.valid:
            if parsed_date_data.valid == "cleaned":
                transformed_date = transform(
                    parsed_date_data=parsed_date_data,
                    parsed_target_format_data=parsed_target_format_data,
                    target_format=target_format,
                    target_timezone=target_timezone,
                )
                row[f"{col}_clean"] = f"{transformed_date}"
                if row[col] != row[f"{col}_clean"]:
                    STATS["cleaned"] += 1
            else:
                STATS["unknown"] += 1
                row[f"{col}_clean"] = np.nan
        else:
            raise ValueError(
                f"target_format {target_format} is invalid. "
                f"Invalid tokens are {parsed_target_format_data.invalid_tokens}. "
                f"Please retype it."
            )
    return row


def set_timezone_offset(
    timezone: Union[str, Any], parsed_data: Union[ParsedDate, ParsedTargetFormat, Any]
) -> Any:
    """
    This function set timezone information for parsed date or parsed target format
    Parameters
    ----------
    timezone
        string name of timezone
    parsed_data
        parsed date or parsed target format
    """
    example_date = datetime.datetime(2009, 9, 1)
    if timezone in all_timezones:
        days, seconds = 0, 0
        pytz_offset = pytz.timezone(timezone).utcoffset(example_date)
        if not pytz_offset is None:
            days = pytz_offset.days
            seconds = pytz_offset.seconds
        parsed_data.set_tzinfo(utc_offset_hours=int(abs(days) * 24 + abs(seconds) / 3600))
        parsed_data.set_tzinfo(
            utc_offset_minutes=int((abs(seconds) - (abs(seconds) / 3600) * 3600) / 60)
        )
        if days >= 0 and seconds >= 0:
            parsed_data.set_tzinfo(utc_add="+")
        elif days <= 0 and seconds < 0:
            parsed_data.set_tzinfo(utc_add="-")
    elif timezone in ZONE:
        parsed_data.set_tzinfo(utc_offset_hours=abs(ZONE[timezone]))
        parsed_data.set_tzinfo(utc_offset_minutes=0)
        if ZONE[timezone] >= 0:
            parsed_data.set_tzinfo(utc_add="+")
        elif ZONE[timezone] < 0:
            parsed_data.set_tzinfo(utc_add="-")
    return parsed_data


def validate_date(date: Union[str, pd.Series]) -> Union[bool, pd.Series]:
    """
    This function validates date string
    Parameters
    ----------
    date
        pandas Series of date string
    """
    if isinstance(date, pd.Series):
        verfied_series = date.apply(check_date)
        return verfied_series
    else:
        return check_date(date)


def check_target_format(target_format: Union[str, Any]) -> Any:
    """
    This function check validation of target_format
    Parameters
    ----------
    target_format
        target_format string
    """
    result = ParsedTargetFormat()
    target_tokens = split(target_format, JUMP)
    remain_tokens = deepcopy(target_tokens)
    # Handle Timezone
    result, remain_tokens = figure_target_format_timezone(result, target_tokens, remain_tokens)
    # Handle year, month, day
    result, remain_tokens = figure_target_format_ymd(result, target_tokens, remain_tokens)
    # Handle AM, PM with JUMP seperators
    result, remain_tokens = figure_target_format_ampm(result, target_tokens, remain_tokens)
    # Handle hour, minute, second
    result, remain_tokens = figure_target_format_hms(result, remain_tokens)
    # If len(remain_tokens) = 0, then is valid format
    if len(remain_tokens) > 0:
        result.set_valid(False)
        for token in remain_tokens:
            result.add_invalid_token(token)
    return result


def figure_target_format_timezone(
    parsed_data: Union[ParsedTargetFormat, Any],
    target_tokens: Union[List[str], Any],
    remain_tokens: Union[List[str], Any],
) -> Any:
    """
    This function figure timezone token in target format
    Parameters
    ----------
    parsed_data
        paresed target format
    target_tokens
        parsed target tokens
    remain_tokens
        remained tokens after figuring tokens
    """
    for token in target_tokens:
        if token in all_timezones:
            parsed_data.set_tzinfo(timezone=token)
            remain_tokens.remove(token)
    for token in target_tokens:
        if token in ("z", "Z"):
            parsed_data.set_timezone_token(token)
            remain_tokens.remove(token)
    return parsed_data, remain_tokens


def figure_target_format_ymd(
    parsed_data: Union[ParsedTargetFormat, Any],
    target_tokens: Union[List[str], Any],
    remain_tokens: Union[List[str], Any],
) -> Any:
    """
    This function figure year, month and day token in target format
    Parameters
    ----------
    parsed_data
        paresed target format
    target_tokens
        parsed target tokens
    remain_tokens
        remained tokens after figuring tokens
    """
    for token in target_tokens:
        if token in TARGET_YEAR:
            parsed_data.set_year_token(token)
            remain_tokens.remove(token)
        if token in TARGET_MONTH:
            parsed_data.set_month_token(token)
            remain_tokens.remove(token)
        if token in TARGET_DAY:
            parsed_data.set_day_token(token)
            remain_tokens.remove(token)
        if token in TARGET_WEEKDAY:
            parsed_data.set_weekday_token(token)
            remain_tokens.remove(token)
    return parsed_data, remain_tokens


def figure_target_format_ampm(
    parsed_data: Union[ParsedTargetFormat, Any],
    target_tokens: Union[List[str], Any],
    remain_tokens: Union[List[str], Any],
) -> Any:
    """
    This function figure AM or PM token in target format
    Parameters
    ----------
    parsed_data
        paresed target format
    target_tokens
        parsed target tokens
    remain_tokens
        remained tokens after figuring tokens
    """
    for token in target_tokens:
        if token in AM:
            remain_tokens.remove(token)
        if token in PM:
            parsed_data.set_ispm(True)
            remain_tokens.remove(token)
    return parsed_data, remain_tokens


def figure_target_format_hms(
    parsed_data: Union[ParsedTargetFormat, Any], remain_tokens: Union[List[str], Any]
) -> Any:
    """
    This function figure hour, minute and second token in target format
    Parameters
    ----------
    parsed_data
        paresed target format
    remain_tokens
        remained tokens after figuring tokens
    """
    if len(remain_tokens) > 0:
        remain_str = ""
        for token in remain_tokens:
            if (
                not token in TARGET_MONTH
                and not token in TARGET_WEEKDAY
                and not token in AM
                and not token in PM
            ):
                remain_str = token
        parsed_data, hms_tokens = get_target_format_hms_tokens(parsed_data, remain_str)
        for token in hms_tokens:
            if token in TARGET_HOUR:
                parsed_data.set_hour_token(token)
            if token in TARGET_MINUTE:
                parsed_data.set_minute_token(token)
            if token in TARGET_SECOND:
                parsed_data.set_second_token(token)
        if len(remain_str) > 0:
            remain_tokens.remove(remain_str)
    return parsed_data, remain_tokens


def get_target_format_hms_tokens(
    parsed_data: Union[ParsedTargetFormat, Any], remain_str: Union[str, Any]
) -> Any:
    """
    This function get hour, minute and second token in target format
    Parameters
    ----------
    parsed_data
        paresed target format
    remain_str
        remained string after figuring tokens
    """
    if "z" in remain_str:
        parsed_data.timezone_token = "z"
        hms_tokens = split(remain_str, [":", parsed_data.timezone_token])
    elif "Z" in remain_str:
        parsed_data.timezone_token = "Z"
        hms_tokens = split(remain_str, [":", parsed_data.timezone_token])
    else:
        hms_tokens = split(remain_str, [":"])
    # ensure AM, PM tokens without JUMP seperators
    for token in AM:
        if token in remain_str:
            hms_tokens = split(remain_str, AM)
            break
    for token in PM:
        if token in remain_str:
            hms_tokens = split(remain_str, PM)
            break
    if len(hms_tokens) == 0:
        hms_tokens = split(remain_str, [":"])
    return parsed_data, hms_tokens


def ensure_ymd(tokes: Union[str, Any]) -> Any:
    """
    This function extract value of year, month, day
    Parameters
    ----------
    tokes
        generated tokens
    """
    result = ParsedDate()
    remain_tokens = deepcopy(tokes)
    result, remain_tokens = ensure_year(result, tokes, remain_tokens)
    if len(remain_tokens) == 0:
        return result, remain_tokens
    num_tokens = []
    for token in remain_tokens:
        if token.isnumeric():
            num_tokens.append(token)
    for token in num_tokens:
        remain_tokens.remove(token)
    if result.ymd["year"] != -1:
        result = ensure_month_day(result, num_tokens)
    else:
        result = ensure_year_month_day(result, num_tokens)
    return result, remain_tokens


def ensure_year(
    parsed_data: Union[ParsedDate, Any],
    tokes: Union[str, Any],
    remain_tokens: Union[List[str], Any],
) -> Any:
    """
    This function extract year number whose length is 4
    Parameters
    ----------
    parsed_data
        parsed date
    tokes
        parsed tokens
    remain_tokens
        remained tokens
    """
    for token in tokes:
        if token in MONTHS:
            parsed_data.set_month(MONTHS[token])
            remain_tokens.remove(token)
        if token in WEEKDAYS:
            parsed_data.set_weekday(WEEKDAYS[token])
            remain_tokens.remove(token)
    for token in remain_tokens:
        if len(token) == 4 and token.isnumeric():
            parsed_data.set_year(int(token))
            remain_tokens.remove(token)
            break
    return parsed_data, remain_tokens


def ensure_month_day(parsed_data: Union[ParsedDate, Any], num_tokens: Union[List[str], Any]) -> Any:
    """
    This function extract month and day when year is not None.
    Parameters
    ----------
    parsed_data
        parsed date
    num_tokens
        remained numerical tokens
    """
    if len(num_tokens) == 1:
        if parsed_data.ymd["month"] != -1:
            parsed_data.set_day(int(num_tokens[0]))
        else:
            parsed_data.set_month(int(num_tokens[0]))
    else:
        if int(num_tokens[0]) > 12:
            parsed_data.set_month(int(num_tokens[1]))
            parsed_data.set_day(int(num_tokens[0]))
        elif int(num_tokens[1]) > 12:
            parsed_data.set_month(int(num_tokens[0]))
            parsed_data.set_day(int(num_tokens[1]))
        else:
            parsed_data.set_month(int(num_tokens[0]))
            parsed_data.set_day(int(num_tokens[1]))
    return parsed_data


def ensure_year_month_day(
    parsed_data: Union[ParsedDate, Any], num_tokens: Union[List[str], Any]
) -> Any:
    """
    This function extract month and day when year is None.
    Parameters
    ----------
    parsed_data
        parsed date
    num_tokens
        remained numerical tokens
    """
    if len(num_tokens) == 1:
        parsed_data.set_year(int(num_tokens[-1]) + 2000)
    elif len(num_tokens) == 2:
        parsed_data.set_year(int(num_tokens[-1]) + 2000)
        if parsed_data.ymd["month"] == -1:
            parsed_data.set_month(int(num_tokens[0]))
        else:
            parsed_data.set_day(int(num_tokens[0]))
    elif len(num_tokens) == 3:
        parsed_data.set_year(int(num_tokens[-1]) + 2000)
        if int(num_tokens[0]) > 12:
            parsed_data.set_month(int(num_tokens[1]))
            parsed_data.set_day(int(num_tokens[0]))
        elif int(num_tokens[1]) > 12:
            parsed_data.set_month(int(num_tokens[0]))
            parsed_data.set_day(int(num_tokens[1]))
        else:
            parsed_data.set_month(int(num_tokens[0]))
            parsed_data.set_day(int(num_tokens[1]))
    return parsed_data


def ensure_hms(inner_result: Union[ParsedDate, Any], remain_tokens: Union[str, Any]) -> Any:
    """
    This function extract value of hour, minute, second
    Parameters
    ----------
    inner_result
        already generated year, month, day value
    remain_tokens
        remained tokens used for generating hour, minute, second
    """
    result = deepcopy(inner_result)
    remain_str = remain_tokens[0]
    hms_tokens = []
    # Judge the expression of am pm
    ispm = False
    for token in AM:
        if token in remain_str:
            hms_tokens = split(remain_str, AM)
            break
    for token in PM:
        if token in remain_str:
            ispm = True
            hms_tokens = split(remain_str, PM)
            break
    if len(hms_tokens) == 0:
        hms_tokens = split(remain_str, [":"])
    else:
        hms_tokens = split(hms_tokens[0], [":"])
    if ispm:
        result = ensure_pm(result, hms_tokens, 12)
    else:
        result = ensure_pm(result, hms_tokens, 0)
    return result


def ensure_pm(
    parsed_data: Union[ParsedDate, Any], hms_tokens: Union[List[str], Any], offset: Union[int, Any]
) -> Any:
    """
    This function extract values which stand for pm time
    Parameters
    ----------
    parsed_data
        already generated parsed value
    hms_tokens
        tokens of hour, minute, second
    offset
        if it is pm time, offset = 12
        otherwise, offset = 0
    """
    if len(hms_tokens) == 1:
        parsed_data.set_hour(int(hms_tokens[0]) + offset)
    elif len(hms_tokens) == 2:
        parsed_data.set_hour(int(hms_tokens[0]) + offset)
        parsed_data.set_minute(int(hms_tokens[1]))
    elif len(hms_tokens) == 3:
        parsed_data.set_hour(int(hms_tokens[0]) + offset)
        parsed_data.set_minute(int(hms_tokens[1]))
        parsed_data.set_second(int(hms_tokens[2]))
    return parsed_data


def fix_empty_element(parsed_res: Union[ParsedDate, Any], fix_empty: Union[str, Any]) -> Any:
    """
    This function fix empty part of transformed format
    Parameters
    ----------
    parsed_res
        generated year, month, day, hour, minute, second
    fix_empty
        the format of fixing empty part
    """
    if parsed_res.valid == "unknown":
        return parsed_res
    if fix_empty == "auto_nearest":
        parsed_res = fix_empty_auto_nearest(parsed_res)
    elif fix_empty == "auto_minimum":
        parsed_res = fix_empty_auto_minimum(parsed_res)
    return parsed_res


def parse(date: Union[str, Any], fix_empty: Union[str, Any]) -> Any:
    """
    This function parse string into tokens
    Parameters
    ----------
    date
        date string
    fix_empty
        format of fixing empty
    """
    tokens = split(date, JUMP)
    parsed_date_res, remain_tokens = ensure_ymd(tokens)
    if len(remain_tokens) > 0:
        parsed_time_res = ensure_hms(parsed_date_res, remain_tokens)
    else:
        parsed_time_res = parsed_date_res
    parsed_res = fix_empty_element(parsed_time_res, fix_empty)
    return parsed_res


def change_timezone(
    parsed_date_data: Union[ParsedDate, Any], target_timezone: Union[str, Any]
) -> Any:
    """
    This function change timezone for already parsed date string
    Parameters
    ----------
    parsed_date_data
        parsed date string
    target_timezone
        target timezone string
    """
    origin_tz_offset = timedelta(days=0, seconds=0)
    target_tz_offset = timedelta(days=0, seconds=0)
    origin_date = datetime.datetime(
        year=parsed_date_data.ymd["year"],
        month=parsed_date_data.ymd["month"],
        day=parsed_date_data.ymd["day"],
        hour=parsed_date_data.hms["hour"],
        minute=parsed_date_data.hms["minute"],
        second=parsed_date_data.hms["second"],
    )
    if parsed_date_data.tzinfo["timezone"] in all_timezones:
        pytz_offset = pytz.timezone(str(parsed_date_data.tzinfo["timezone"])).utcoffset(origin_date)
        if not pytz_offset is None:
            origin_tz_offset = timedelta(days=-pytz_offset.days, seconds=-pytz_offset.seconds)
    elif parsed_date_data.tzinfo["timezone"] in ZONE:
        offset_value = -1 * ZONE[str(parsed_date_data.tzinfo["timezone"])] * 3600
        origin_tz_offset = timedelta(seconds=offset_value)
    if target_timezone in all_timezones:
        pytz_offset = pytz.timezone(target_timezone).utcoffset(origin_date)
        if not pytz_offset is None:
            target_tz_offset = pytz_offset
    elif target_timezone in ZONE:
        target_tz_offset = timedelta(seconds=ZONE[target_timezone] * 3600)
    result = deepcopy(parsed_date_data)
    if -1 in [
        parsed_date_data.ymd["year"],
        parsed_date_data.ymd["month"],
        parsed_date_data.ymd["day"],
        parsed_date_data.hms["hour"],
        parsed_date_data.hms["minute"],
        parsed_date_data.hms["second"],
    ]:
        return parsed_date_data
    utc_date = origin_date + origin_tz_offset
    target_date = utc_date + target_tz_offset
    result.set_year(target_date.year)
    result.set_month(target_date.month)
    result.set_day(target_date.day)
    result.set_hour(target_date.hour)
    result.set_minute(target_date.minute)
    result.set_second(target_date.second)
    result.set_tzinfo(timezone=target_timezone)
    days = target_tz_offset.days
    seconds = target_tz_offset.seconds
    result.set_tzinfo(utc_offset_hours=int(abs(days) * 24 + abs(seconds) / 3600))
    result.set_tzinfo(utc_offset_minutes=int((abs(seconds) - (abs(seconds) / 3600) * 3600) / 60))
    if days >= 0 and seconds >= 0:
        result.set_tzinfo(utc_add="+")
    elif days <= 0 and seconds < 0:
        result.set_tzinfo(utc_add="-")
    return result


def transform_year(
    result_str: Union[str, Any], year_token: Union[str, Any], year: Union[int, Any]
) -> Any:
    """
    This function transform parsed year into target format
    Parameters
    ----------
    result_str
        result string
    year_token
        token of year
    year
        value of year
    """
    result = deepcopy(result_str)
    if year_token != "":
        if year == -1:
            if len(year_token) == 4:
                result = result.replace(year_token, str("----"))
            elif len(year_token) == 2:
                result = result.replace(year_token, str("--"))
            elif len(year_token) == 1:
                result = result.replace(year_token, str("-"))
        else:
            if len(year_token) == 4:
                result = result.replace(year_token, str(year))
            else:
                year = year - 2000
                if year < 10:
                    result = result.replace(year_token, "0" + str(year))
                else:
                    result = result.replace(year_token, str(year))
    return result


def transform_month(
    result_str: Union[str, Any], month_token: Union[str, Any], month: Union[int, Any]
) -> Any:
    """
    This function transform parsed month into target format
    Parameters
    ----------
    result_str
        result string
    month_token
        token of month
    month
        value of month
    """
    result = deepcopy(result_str)
    if month_token != "":
        if month == -1:
            if len(month_token) == 3:
                result = result.replace(month_token, str("---"))
            elif len(month_token) == 5:
                result = result.replace(month_token, str("-----"))
            elif len(month_token) == 2:
                result = result.replace(month_token, str("--"))
            elif len(month_token) == 1:
                result = result.replace(month_token, str("-"))
        else:
            if len(month_token) == 2:
                if month < 10:
                    result = result.replace(month_token, "0" + str(month), 1)
                else:
                    result = result.replace(month_token, str(month), 1)
            elif len(month_token) == 3:
                result = result.replace(month_token, TEXT_MONTHS[month - 1][0], 1)
            elif len(month_token) == 5:
                result = result.replace(month_token, TEXT_MONTHS[month - 1][1], 1)
            else:
                result = result.replace(month_token, str(month), 1)
    return result


def transform_day(
    result_str: Union[str, Any], day_token: Union[str, Any], day: Union[int, Any]
) -> Any:
    """
    This function transform parsed month into target format
    Parameters
    ----------
    result_str
        result string
    day_token
        token of day
    day
        value of day
    """
    result = deepcopy(result_str)
    if day_token != "":
        if day == -1:
            if len(day_token) == 2:
                result = result.replace(day_token, str("--"))
            elif len(day_token) == 1:
                result = result.replace(day_token, str("-"))
        else:
            if len(day_token) == 2:
                if day < 10:
                    result = result.replace(day_token, "0" + str(day), 1)
                else:
                    result = result.replace(day_token, str(day), 1)
            else:
                result = result.replace(day_token, str(day))
    return result


def transform_hms(
    result_str: Union[str, Any],
    hms_token: Union[str, Any],
    ispm: Union[bool, Any],
    hms_value: Union[int, Any],
) -> Any:
    """
    This function transform parsed month into target format
    Parameters
    ----------
    result_str
        result string
    hms_token, ispm
        token of hour, minute or second, justify whether it is time in P.M.
    hms_value
        value of hour, minute or second
    """
    result = deepcopy(result_str)
    if hms_token != "":
        if hms_value == -1:
            if len(hms_token) == 2:
                result = result.replace(hms_token, str("--"))
            elif len(hms_token) == 1:
                result = result.replace(hms_token, str("-"))
        else:
            if not ispm is None and ispm:
                hms_value = hms_value - 12
            if len(hms_token) == 2:
                if hms_value < 10:
                    result = result.replace(hms_token, "0" + str(hms_value), 1)
                else:
                    result = result.replace(hms_token, str(hms_value), 1)
            else:
                result = result.replace(hms_token, str(hms_value))
    return result


def transform_weekday(
    result_str: Union[str, Any], weekday_token: Union[str, Any], weekday: Union[int, Any]
) -> Any:
    """
    This function transform parsed month into target format
    Parameters
    ----------
    result_str
        result string
    weekday_token
        token of weekday
    weekday
        value of weekday
    """
    result = deepcopy(result_str)
    if weekday_token != "":
        if weekday == -1:
            if len(weekday_token) == 3:
                result = result.replace(weekday_token, str("---"))
            elif len(weekday_token) == 5:
                result = result.replace(weekday_token, str("-----"))
        else:
            if len(weekday_token) == 3:
                result = result.replace(weekday_token, TEXT_WEEKDAYS[weekday - 1][0])
            elif len(weekday_token) == 5:
                result = result.replace(weekday_token, TEXT_WEEKDAYS[weekday - 1][1])
    return result


def transform_timezone(
    result_str: Union[str, Any],
    timezone_token: Union[str, Any],
    timezone: Union[str, Any],
    tz_info: Union[Dict[str, str], Any],
) -> Any:
    """
    This function transform parsed month into target format
    Parameters
    ----------
    result_str
        result string
    timezone_token
        token of timezone in target format
    timezone
        value of timezone string
    tz_info
        information of timezone, including offset hours and mins comparing to UTC
    """
    result = deepcopy(result_str)
    utc_add = tz_info["utc_add"]
    utc_offset_hours = int(tz_info["utc_offset_hours"])
    utc_offset_minutes = int(tz_info["utc_offset_minutes"])
    if timezone_token != "":
        if timezone_token == "z":
            result = result.replace(timezone_token, timezone)
        elif timezone_token == "Z":
            offset_hours_str = str(int(utc_offset_hours))
            if len(offset_hours_str) == 1:
                offset_hours_str = "0" + offset_hours_str
            offset_minutes_str = str(int(utc_offset_minutes))
            if len(offset_minutes_str) == 1:
                offset_minutes_str = "0" + offset_minutes_str
            result = result.replace(
                timezone_token, "UTC" + utc_add + offset_hours_str + ":" + offset_minutes_str
            )
    return result


def transform(
    parsed_date_data: Union[ParsedDate, Any],
    parsed_target_format_data: Union[ParsedTargetFormat, Any],
    target_format: Union[str, Any],
    target_timezone: Union[str, Any],
) -> Any:
    """
    This function transform parsed result into target format
    Parameters
    ----------
    parsed_date_data
        generated year, month, day, hour, minute, second
    parsed_target_format_data
        generated year token, month token, day token, hour token,
        minute token, second token of target format
    target_format
        target format string
    target_timezone
        target timezone string
    """
    result = deepcopy(target_format)
    if target_timezone != "":
        parsed_date_data = change_timezone(parsed_date_data, target_timezone)
    # Handle year
    result = transform_year(
        result, parsed_target_format_data.ymd_token["year_token"], parsed_date_data.ymd["year"]
    )
    # Handle day
    result = transform_day(
        result, parsed_target_format_data.ymd_token["day_token"], parsed_date_data.ymd["day"]
    )
    # Handle hours
    result = transform_hms(
        result,
        parsed_target_format_data.hms_token["hour_token"],
        parsed_target_format_data.hms_token["ispm"],
        parsed_date_data.hms["hour"],
    )
    # Handle minutes
    result = transform_hms(
        result,
        parsed_target_format_data.hms_token["minute_token"],
        None,
        parsed_date_data.hms["minute"],
    )
    # Handle seconds
    result = transform_hms(
        result,
        parsed_target_format_data.hms_token["second_token"],
        None,
        parsed_date_data.hms["second"],
    )
    # Handle month
    result = transform_month(
        result, parsed_target_format_data.ymd_token["month_token"], parsed_date_data.ymd["month"]
    )
    # Handle weekday
    result = transform_weekday(
        result, parsed_target_format_data.weekday_token, parsed_date_data.weekday
    )
    # Handle timezone
    tz_info = {
        "utc_add": parsed_date_data.tzinfo["utc_add"],
        "utc_offset_hours": str(parsed_date_data.tzinfo["utc_offset_hours"]),
        "utc_offset_minutes": str(parsed_date_data.tzinfo["utc_offset_minutes"]),
    }
    result = transform_timezone(
        result,
        parsed_target_format_data.timezone_token,
        parsed_date_data.tzinfo["timezone"],
        tz_info,
    )
    return result


def reset_stats() -> None:
    """
    Reset global statistics dictionary.
    """
    STATS["cleaned"] = 0
    STATS["null"] = 0
    STATS["unknown"] = 0
