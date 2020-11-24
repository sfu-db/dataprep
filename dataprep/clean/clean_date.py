from copy import deepcopy
import datetime
from datetime import timedelta
import pytz
from pytz import all_timezones
from typing import Any, Union

import dask.dataframe as dd
import dask
import numpy as np
import pandas as pd

from .utils import NULL_VALUES, create_report, to_dask


JUMP = [" ", ".", ",", ";", "-", "/", "'",
        "st", "nd", "rd", "th",
        "at", "on", "and", "ad", "AD", "of"]
        #"T", "t"
WEEKDAYS = {"Mon": 1, "Monday": 1,
            "Tue": 2, "Tuesday": 2,
            "Wed": 3, "Wednesday": 3,
            "Thu" :4, "Thursday" :4,
            "Fri" :5, "Friday" :5,
            "Sat": 6, "Saturday" :6,
            "Sun": 7, "Sunday":7}
MONTHS = {"Jan": 1, "January": 1,
          "Feb": 2, "February":2,
          "Mar": 3, "March": 3,
          "Apr": 4, "April": 4,
          "May": 5,
          "Jun": 6, "June": 6,
          "Jul": 7, "July": 7,
          "Aug": 8, "August": 8,
          "Sep": 9, "Sept": 9, "September": 9,
          "Oct": 10, "October": 10,
          "Nov": 11, "November": 11,
          "Dec": 12, "December" : 12}
HMS = [("h", "hour", "hours"), ("m", "minute", "minutes"), ("s", "second", "seconds")]
AM = ["AM", "am", "a"]
PM = ["PM", "pm", "p"]
ZONE = {'UTC': 0,
        'ACT': -5,
        'ADT': -3,
        'AEDT': 11,
        'AEST': 10,
        'AKDT': -8,
        'AKST': -9,
        'AMST': -3,
        'AMT': -4,
        'ART': -3,
        'ArabiaST': 3,
        'AtlanticST': -4,
        'AWST': 8,
        'AZOST': 0,
        'AZOT': 0,
        'BOT': -4,
        'BRST': -2,
        'BRT': -3,
        'BST': 1,
        'BTT': 6,
        'CAT': 2,
        'CDT': -5,
        'CEST': 2,
        'CET': 1,
        'CHOST': 9,
        'CHOT': 8,
        'CHUT': 10,
        'CKT': -10,
        'CLST': -3,
        'CLT': -4,
        'CentralST': -6,
        'ChinaST': 8,
        'CubaST': -5,
        'ChST': 10,
        'EASST': -5,
        'EAST': -6,
        'EAT':3,
        'ECT':-5,
        'EDT': -4,
        'EEST': 3,
        'EET': 2,
        'EST': -5,
        'FKST': -3,
        'GFT': -3,
        'GILT':12,
        'GMT':0,
        'GST':4,
        'HKT':8,
        'HST':-10,
        'ICT':7,
        'IDT':3,
        'IrishST':1,
        'IsraelST':2,
        'JST':9,
        'KOST':11,
        'LINT':4,
        'MDT':-6,
        'MHT':12,
        'MSK':3,
        'MST':-7,
        'MYT':8,
        'NUT':-11,
        'NZDT':13,
        'NZST':12,
        'PDT':-7,
        'PET':-5,
        'PGT':10,
        'PHT':8,
        'PONT':11,
        'PST':-8,
        'SAST':2,
        'SBT':11,
        'SGT':8,
        'SRT':-3,
        'SST':-11,
        'TAHT':-10,
        'TLT':9,
        'TVT':12,
        'ULAST':9,
        'ULAT':8,
        'UYST':-2,
        'UYT':-3,
        'VET':-4,
        'WAST':2,
        'WAT':1,
        'WEST':1,
        'WET':0,
        'WIB':7,
        'WIT':9,
        'WITA':8}

TARGET_YEAR = ["yyyy", "yy", "YYYY", "YY", "Y", "y"]
TARGET_MONTH = ["MM", "M", "MMM", "MMMMM"]
TEXT_MONTHS = [("Jan", "January"),
              ("Feb", "February"),      # TODO: "Febr"
              ("Mar", "March"),
              ("Apr", "April"),
              ("May", "May"),
              ("Jun", "June"),
              ("Jul", "July"),
              ("Aug", "August"),
              ("Sep", "September"),
              ("Oct", "October"),
              ("Nov", "November"),
              ("Dec", "December")]
TARGET_DAY = ["dd", "d", "DD", "D"]
TARGET_HOUR = ["hh", "h", "HH", "H"]
TARGET_MINUTE = ["mm", "m"]
TARGET_SECOND = ["ss", "s", "SS", "S"]
TARGET_WEEKDAY = ["eee", "EEE", "eeeee", "EEEEE"]
TEXT_WEEKDAYS = [("Mon", "Monday"),
                ("Tue", "Tuesday"),     # TODO: "Tues"
                ("Wed", "Wednesday"),
                ("Thu", "Thursday"),    # TODO: "Thurs"
                ("Fri", "Friday"),
                ("Sat", "Saturday"),
                ("Sun", "Sunday")]

STATS = {"cleaned": 0, "null": 0, "unknown": 0}

class parsed_date():

    """Attributes of a parsed date.
    Attributes:
        year: Value of year.
        month: Value of month.
        day: Value of day.
        hour: Value of hour.
        minute: Value of minute.
        second: Value of second.
        weekday: Value of weekday.
        timezone: Value of timezone.
        utc_offset_hours: Hours of timezone offset.
        utc_offset_minutes: Minutes of timezone offset.
        utc_offset_add: Timezone offset is pos or neg.
        valid: if parsed values are all valid.
    """

    def __init__(self):
        """
        This function initiate parse_date
        """
        self.year = None
        self.month = None
        self.day = None
        self.hour = None
        self.minute = None
        self.second = None
        self.weekday = None
        self.timezone = None
        self.utc_offset_hours = None
        self.utc_offset_minutes = None
        self.utc_add = None
        self.valid = 'cleaned'

    def set_year(self, year):
        """
        This function set value of year
        Parameters
        ----------
        year
            year value
        """
        if 1700 <= year <= 2500:
            self.year = year
            return True
        self.valid = 'unknown'
        return False

    def set_month(self, month):
        """
        This function set value of month
        Parameters
        ----------
        month
            month value
        """
        if 1 <= month <= 12:
            self.month = month
            return True
        self.valid = 'unknown'
        return False

    def set_day(self, day):
        """
        This function set value of day
        Parameters
        ----------
        day
            day value
        """
        if self.month in [1, 3, 5, 7, 8, 10, 12]:
            if 1 <= day <= 31:
                self.day = day
                return True
        if self.month in [4, 6, 9, 11]:
            if 1 <= day <= 30:
                self.day = day
                return True
        if self.month in [2]:
            if self._is_leap_year():
                if 1 <= day <= 29:
                    self.day = day
                    return True
            else:
                if 1 <= day <= 28:
                    self.day = day
                    return True
        self.valid = 'unknown'
        return False

    def set_hour(self, hour):
        """
        This function set value of hour
        Parameters
        ----------
        hour
            hour value
        """
        if 0 <= hour < 24:
            self.hour = hour
            return True
        self.valid = 'unknown'
        return False

    def set_minute(self, minute):
        """
        This function set value of minute
        Parameters
        ----------
        minute
            minute value
        """
        if 0 <= minute < 60:
            self.minute = minute
            return True
        self.valid = 'unknown'
        return False

    def set_second(self, second):
        """
        This function set value of second
        Parameters
        ----------
        second
            second value
        """
        if 0 <= second < 60:
            self.second = second
            return True
        self.valid = 'unknown'
        return False

    def set_timezone(self, timezone):
        """
        This function set value of timezone
        Parameters
        ----------
        timezone
            timezone value
        """
        if timezone in all_timezones or timezone in ZONE:
            self.timezone = timezone
            return True
        self.valid = 'unknown'
        return False

    def set_weekday(self, weekday):
        """
        This function set value of weekday
        Parameters
        ----------
        weekday
            weekday value
        """
        if 1 <= weekday <= 7:
            self.weekday = weekday
            return True
        self.valid = 'unknown'
        return False

    def _is_leap_year(self):
        """
        This function judge if year is leap year
        """
        if self.year % 4 == 0:
            if self.year % 100 == 0:
                return self.year % 400 == 0
            else:
                return True
        else:
            return False

class parsed_target_format():

    """Attributes of a parsed target format.
    Attributes:
        year_token: Token standing of year.
        month_token: Token standing of month.
        day_token: Token standing of day.
        hour_token: Token standing of hour.
        minute_token: Token standing of minute.
        second_token: Token standing of second.
        weekday_token: Token standing of weekday.
        timezone_token: Token standing of timezone.
        timezone: Value of target timezone.
        utc_offset_hours: Hours of timezone offset.
        utc_offset_minutes: Minutes of timezone offset.
        utc_offset_add: Timezone offset is pos or neg.
        valid: if target format is valid.
        invalid_tokens: if target format is not valid,
                        what tokens lead to this result.
    """

    def __init__(self):
        """
        This function initiate parsed_target_fomat
        """
        self.year_token = None
        self.month_token = None
        self.day_token = None
        self.hour_token = None
        self.minute_token = None
        self.second_token = None
        self.weekday_token = None
        self.timezone = None
        self.timezone_token = None
        self.utc_offset_hours = None
        self.utc_offset_minutes = None
        self.utc_add = None
        self.ispm = False
        self.valid = True
        self.invalid_tokens = []

    def set_year_token(self, year_token):
        """
        This function set value of year_token
        Parameters
        ----------
        year_token
            token string of year
        """
        self.year_token = year_token
        return True

    def set_month_token(self, month_token):
        """
        This function set value of month_token
        Parameters
        ----------
        month_token
            token string of month
        """
        self.month_token = month_token
        return True

    def set_day_token(self, day_token):
        """
        This function set value of day_token
        Parameters
        ----------
        day_token
            token string of day
        """
        self.day_token = day_token
        return True

    def set_hour_token(self, hour_token):
        """
        This function set value of hour_token
        Parameters
        ----------
        hour_token
            token string of hour
        """
        self.hour_token = hour_token
        return True

    def set_minute_token(self, minute_token):
        """
        This function set value of minute_token
        Parameters
        ----------
        minute_token
            token string of minute
        """
        self.minute_token = minute_token
        return True

    def set_second_token(self, second_token):
        """
        This function set value of second_token
        Parameters
        ----------
        second_token
            token string of second
        """
        self.second_token = second_token
        return True

    def set_weekday_token(self, weekday_token):
        """
        This function set value of weekday_token
        Parameters
        ----------
        weekday_token
            token string of weekday
        """
        self.weekday_token = weekday_token
        return True

    def set_timezone_token(self, timezone_token):
        """
        This function set value of timezone_token
        Parameters
        ----------
        timezone_token
            token string of timezone
        """
        self.timezone_token = timezone_token
        return True

    def set_time_zone(self, timezone):
        """
        This function set value of timezone
        Parameters
        ----------
        timezone
            timezone string
        """
        self.timezone = timezone
        return True

    def set_valid(self, valid):
        """
        This function set valid status of target format
        Parameters
        ----------
        valid
            valid status
        """
        self.valid = valid
        return True

    def set_ispm(self, ispm):
        """
        This function set value of judgement
        of PM status for target format
        Parameters
        ----------
        ispm
            If is PM, True. If not, False
        """
        self.ispm = ispm
        return True

    def add_invalid_token(self, token):
        """
        This function set value of invalid tokens
        in target format
        Parameters
        ----------
        token
            invalid token
        """
        self.invalid_tokens.append(token)
        return True


def clean_date(
    df: Union[pd.DataFrame, dd.DataFrame],
    col: str,
    target_format: str = 'YYYY-MM-DD hh:mm:ss',
    origin_timezone: str = 'UTC',
    target_timezone: str = None,
    fix_empty = 'auto_minimum',
    show_report: bool = True
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
            f'fix_empty {fix_empty} is invalid. '
            f'It needs to be "auto_minimum", "auto_nearest" or "empty"'
        )

    if origin_timezone not in all_timezones and origin_timezone not in ZONE:
        raise ValueError(
            f'origin_timezone {origin_timezone} doesn\'t exist'
        )

    if target_timezone not in all_timezones and \
       target_timezone not in ZONE and \
       not target_timezone is None:
        raise ValueError(
            f'target_timezone {target_timezone} doesn\'t exist'
        )

    df = to_dask(df)
    # specify the metadata for dask apply
    meta = df.dtypes.to_dict()
    meta[f"{col}_clean"] = str

    df = df.apply(
        format_date,
        args=(col, target_format, origin_timezone, target_timezone, fix_empty),
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
    origin_timezone: str,
    target_timezone: str,
    fix_empty: str
) -> pd.Series:
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
    """

    date = row[col]

    if check_date(date) == 'null':
        STATS['null'] += 1
        row[f"{col}_clean"] = np.nan
    elif check_date(date) == 'unknown':
        STATS['unknown'] += 1
        row[f"{col}_clean"] = np.nan
    elif check_date(date) == 'cleaned':
        # Handle date data and timezone
        parsed_date_data = parse(date=date, fix_empty=fix_empty)
        parsed_date_data.set_timezone(origin_timezone)
        parsed_date_data = set_timezone_offset(origin_timezone, parsed_date_data)

        # Handle target format and timezone
        parsed_target_format_data = check_target_format(target_format=target_format)
        parsed_target_format_data = set_timezone_offset(target_timezone, parsed_target_format_data)
        if parsed_target_format_data.valid:
            if parsed_date_data.valid == 'cleaned':
                transformed_date = transform(parsed_date_data=parsed_date_data,
                                             parsed_target_format_data=parsed_target_format_data,
                                             target_format=target_format,
                                             target_timezone=target_timezone)
                row[f"{col}_clean"] = f"{transformed_date}"
                if row[col] != row[f"{col}_clean"]:
                    STATS["cleaned"] += 1
            else:
                STATS['unknown'] += 1
                row[f"{col}_clean"] = np.nan
        else:
            raise ValueError(
                f'target_format {target_format} is invalid. '
                f'Invalid tokens are {parsed_target_format_data.invalid_tokens}. '
                f'Please retype it.'
            )

    return row

def set_timezone_offset(timezone: Union[str, Any],
                        parsed_data: Union[parsed_date, parsed_target_format, Any]) -> Any:
    """
    This function set timezone information for
    parsed date or parsed target format
    Parameters
    ----------
    timezone
        string name of timezone
    parsed_data
        parsed date or parsed target format
    """
    if timezone in all_timezones:
        days = pytz.timezone(timezone)._utcoffset.days
        seconds = pytz.timezone(timezone)._utcoffset.seconds
        parsed_data.utc_offset_hours = abs(days) * 24 + abs(seconds) / 3600
        parsed_data.utc_offset_minutes = \
            int((abs(seconds) - (abs(seconds) / 3600) * 3600) / 60)
        if days >= 0 and seconds >= 0:
            parsed_data.utc_add = '+'
        elif days <= 0 and seconds < 0:
            parsed_data.utc_add= '-'
    elif timezone in ZONE:
        parsed_data.utc_offset_hours = abs(ZONE[timezone])
        parsed_data.utc_offset_minutes = 0
        if ZONE[timezone] >= 0:
            parsed_data.utc_add = '+'
        elif ZONE[timezone] < 0:
            parsed_data.utc_add= '-'
    return parsed_data

def validate_date(date: Union[str, pd.Series]) -> Union[bool, pd.Series]:
    """
    This function validates url
    Parameters
    ----------
    date
        pandas Series of urls or url instance
    """
    if isinstance(date, pd.Series):
        verfied_series = date.apply(check_date)
        return verfied_series
    else:
        return check_date(date)

def check_date(date: Union[str, Any]) -> Any:
    """
    This function check format of date
    Parameters
    ----------
    date
        date string
    """
    if str(date) in NULL_VALUES:
        return 'null'
    tokes = split(date, JUMP)
    remain_tokens = tokes.copy()

    # Handle timezone
    for token in tokes:
        if token in all_timezones or token in ZONE:
            remain_tokens.remove(token)

    for token in tokes:
        if token in MONTHS or token in WEEKDAYS:
            remain_tokens.remove(token)
    for token in remain_tokens:
        if token.isnumeric():
            remain_tokens.remove(token)
    for token in remain_tokens:
        token = split(token, AM + PM + [":"])
        invalid_judge = False in [temp_token.isnumeric() for temp_token in token]
        if invalid_judge:
            return 'unknown'
    return 'cleaned'

def validate_target_format(target_format: Union[str, Any]) -> Any:
    """
    This function check validation of target_format
    Parameters
    ----------
    target_format
        target_format string
    """

    return check_target_format(target_format)

def check_target_format(target_format: Union[str, Any]) -> Any:
    """
    This function check validation of target_format
    Parameters
    ----------
    target_format
        target_format string
    """
    result = parsed_target_format()
    target_tokens = split(target_format, JUMP)
    remain_tokens = deepcopy(target_tokens)

    # Handle Timezone
    result, remain_tokens = \
        figure_target_format_timezone(result, target_tokens, remain_tokens)

    # Handle year, month, day
    result, remain_tokens = \
        figure_target_format_ymd(result, target_tokens, remain_tokens)

    # Handle AM, PM with JUMP seperators
    result, remain_tokens = \
        figure_target_format_ampm(result, target_tokens, remain_tokens)

    # Handle hour, minute, second
    result, remain_tokens = \
        figure_target_format_hms(result, remain_tokens)

    # If len(remain_tokens) = 0, then is valid format
    if len(remain_tokens) > 0:
        result.set_valid(False)
        for token in remain_tokens:
            result.add_invalid_token(token)

    return result

def figure_target_format_timezone(parsed_data: Union[parsed_target_format, Any],
                           target_tokens: Union[list, Any],
                           remain_tokens: Union[list, Any]) -> Any:
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
            parsed_data.set_time_zone(token)
            remain_tokens.remove(token)

    for token in target_tokens:
        if token in ('z', 'Z'):
            parsed_data.set_timezone_token(token)
            remain_tokens.remove(token)

    return parsed_data, remain_tokens

def figure_target_format_ymd(parsed_data: Union[parsed_target_format, Any],
                           target_tokens: Union[list, Any],
                           remain_tokens: Union[list, Any]) -> Any:
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

def figure_target_format_ampm(parsed_data: Union[parsed_target_format, Any],
                           target_tokens: Union[list, Any],
                           remain_tokens: Union[list, Any]) -> Any:
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

def figure_target_format_hms(parsed_data: Union[parsed_target_format, Any],
                           remain_tokens: Union[list, Any]) -> Any:
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
        remain_str = ''
        for token in remain_tokens:
            if not token in TARGET_MONTH and not token in TARGET_WEEKDAY and \
               not token in AM and not token in PM:
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

def get_target_format_hms_tokens(parsed_data: Union[parsed_target_format, Any],
                                 remain_str: Union[str, Any]) -> Any:
    """
    This function get hour, minute and second token in target format
    Parameters
    ----------
    parsed_data
        paresed target format
    remain_str
        remained string after figuring tokens
    """

    if 'z' in remain_str:
        parsed_data.timezone_token = 'z'
        hms_tokens = split(remain_str, [":", parsed_data.timezone_token])
    elif 'Z' in remain_str:
        parsed_data.timezone_token = 'Z'
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
    result = parsed_date()
    remain_tokens = tokes.copy()
    result, remain_tokens = ensure_year(result, tokes, remain_tokens)

    if len(remain_tokens) == 0:
        return result, remain_tokens
    num_tokens = []
    for token in remain_tokens:
        if token.isnumeric():
            num_tokens.append(token)
    for token in num_tokens:
        remain_tokens.remove(token)

    if not result.year is None:
        result = ensure_month_day(result, num_tokens)
    else:
        result = ensure_year_month_day(result, num_tokens)
        
    return result, remain_tokens

def ensure_year(parsed_data: Union[parsed_date, Any],
                tokes: Union[str, Any],
                remain_tokens: Union[list, Any]) -> Any:
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

def ensure_month_day(parsed_data: Union[parsed_date, Any],
                     num_tokens: Union[list, Any]) -> Any:
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
        if not parsed_data.month is None:
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

def ensure_year_month_day(parsed_data: Union[parsed_date, Any],
                          num_tokens: Union[list, Any]) -> Any:
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
        if parsed_data.month is None:
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

def ensure_hms(inner_result: Union[parsed_date, Any], remain_tokens: Union[str, Any]) -> Any:
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
        if len(hms_tokens) == 1:
            result.set_hour(int(hms_tokens[0]) + 12)
        elif len(hms_tokens) == 2:
            result.set_hour(int(hms_tokens[0]) + 12)
            result.set_minute(int(hms_tokens[1]))
        elif len(hms_tokens) == 3:
            result.set_hour(int(hms_tokens[0]) + 12)
            result.set_minute(int(hms_tokens[1]))
            result.set_second(int(hms_tokens[2]))
    else:
        if len(hms_tokens) == 1:
            result.set_hour(int(hms_tokens[0]))
        elif len(hms_tokens) == 2:
            result.set_hour(int(hms_tokens[0]))
            result.set_minute(int(hms_tokens[1]))
        elif len(hms_tokens) == 3:
            result.set_hour(int(hms_tokens[0]))
            result.set_minute(int(hms_tokens[1]))
            result.set_second(int(hms_tokens[2]))
    return result

def split(txt: Union[str, Any], seps: Union[str, Any]) -> Any:
    """
    This function split string into tokens
    Parameters
    ----------
    txt
        string
    seps
        seprators
    """
    default_sep = seps[0]
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    result =  [i.strip() for i in txt.split(default_sep)]
    result = [value for value in result if value != '']
    return result

def fix_empty_element(parsed_res: Union[parsed_date, Any], fix_empty: Union[str, Any]) -> Any:
    """
    This function fix empty part of transformed format
    Parameters
    ----------
    parsed_res
        generated year, month, day, hour, minute, second
    fix_empty
        the format of fixing empty part
    """
    if parsed_res.valid == 'unknown':
        return parsed_res
    if fix_empty == 'auto_nearest':
        parsed_res = fix_empty_auto_nearest(parsed_res)
    elif fix_empty == 'auto_minimum':
        parsed_res = fix_empty_auto_minimum(parsed_res)
    return parsed_res

def fix_empty_auto_nearest(parsed_res: Union[parsed_date, Any]) -> Any:
    """
    This function fix empty part by nearest time
    Parameters
    ----------
    parsed_res
        parsed date result
    """

    now_time = datetime.datetime.now()
    if parsed_res.year is None:
        parsed_res.set_year(now_time.year)
    if parsed_res.month is None:
        parsed_res.set_month(now_time.month)
    if parsed_res.day is None:
        parsed_res.set_day(now_time.day)
    if parsed_res.hour is None:
        parsed_res.set_hour(now_time.hour)
    if parsed_res.minute is None:
        parsed_res.set_minute(now_time.minute)
    if parsed_res.second is None:
        parsed_res.set_second(now_time.second)
    if parsed_res.weekday is None:
        temp_date = datetime.datetime(parsed_res.year, parsed_res.month, parsed_res.day)
        parsed_res.set_weekday(temp_date.weekday() + 1)

    return parsed_res

def fix_empty_auto_minimum(parsed_res: Union[parsed_date, Any]) -> Any:
    """
    This function fix empty part by minimum time
    Parameters
    ----------
    parsed_res
        parsed date result
    """

    if parsed_res.year is None:
        parsed_res.set_year(2000)
    if parsed_res.month is None:
        parsed_res.set_month(1)
    if parsed_res.day is None:
        parsed_res.set_day(1)
    if parsed_res.hour is None:
        parsed_res.set_hour(0)
    if parsed_res.minute is None:
        parsed_res.set_minute(0)
    if parsed_res.second is None:
        parsed_res.set_second(0)
    if parsed_res.weekday is None:
        temp_date = datetime.datetime(parsed_res.year, parsed_res.month, parsed_res.day)
        parsed_res.set_weekday(temp_date.weekday() + 1)

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

def change_timezone(parsed_date_data: Union[parsed_date, Any],
                    target_timezone: Union[str, Any]) -> Any:
    """
    This function change timezone for already parsed date string
    Parameters
    ----------
    parsed_date_data
        parsed date string
    target_timezone
        target timezone string
    """
    origin_tz_offset = None
    target_tz_offset = None
    if parsed_date_data.timezone in all_timezones:
        origin_tz_offset = pytz.timezone(parsed_date_data.timezone)._utcoffset
        origin_tz_offset = timedelta(days=-origin_tz_offset.days,
                                              seconds=-origin_tz_offset.seconds)
    elif parsed_date_data.timezone in ZONE:
        origin_tz_offset = timedelta(seconds=-1 * ZONE[parsed_date_data.timezone] *  3600)
    if target_timezone in all_timezones:
        target_tz_offset = pytz.timezone(target_timezone)._utcoffset
    elif target_timezone in ZONE:
        target_tz_offset = timedelta(seconds=ZONE[target_timezone] * 3600)

    result = deepcopy(parsed_date_data)
    if parsed_date_data.year is None or \
       parsed_date_data.month is None or \
       parsed_date_data.day is None or \
       parsed_date_data.hour is None or \
       parsed_date_data.minute is None or \
       parsed_date_data.second is None:
        return parsed_date_data
    utc_date = datetime.datetime(year=parsed_date_data.year,
                                 month=parsed_date_data.month,
                                 day=parsed_date_data.day,
                                 hour=parsed_date_data.hour,
                                 minute=parsed_date_data.minute,
                                 second=parsed_date_data.second) + origin_tz_offset
    target_date = utc_date + target_tz_offset
    result.set_year(target_date.year)
    result.set_month(target_date.month)
    result.set_day(target_date.day)
    result.set_hour(target_date.hour)
    result.set_minute(target_date.minute)
    result.set_second(target_date.second)
    result.set_timezone(target_timezone)
    days = target_tz_offset.days
    seconds = target_tz_offset.seconds
    result.utc_offset_hours = abs(days) * 24 + abs(seconds) / 3600
    result.utc_offset_minutes = int((abs(seconds) - (abs(seconds) / 3600) * 3600) / 60)
    if days >= 0 and seconds >= 0:
        result.utc_add = '+'
    elif days <= 0 and seconds < 0:
        result.utc_add= '-'
    return result

def transform_year(result_str: Union[str, Any],
                   year_token: Union[str, Any],
                   year: Union[int, Any]) -> Any:
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
    if not year_token is None:
        if year is None:
            if len(year_token) == 4:
                result = result.replace(year_token, str('----'))
            elif len(year_token) == 2:
                result = result.replace(year_token, str('--'))
            elif len(year_token) == 1:
                result = result.replace(year_token, str('-'))
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

def transform_month(result_str: Union[str, Any],
                   month_token: Union[str, Any],
                   month: Union[int, Any]) -> Any:
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
    if not month_token is None:
        if month is None:
            if len(month_token) == 3:
                result = result.replace(month_token, str('---'))
            elif len(month_token) == 5:
                result = result.replace(month_token, str('-----'))
            elif len(month_token) == 2:
                result = result.replace(month_token, str('--'))
            elif len(month_token) == 1:
                result = result.replace(month_token, str('-'))
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

def transform_day(result_str: Union[str, Any],
                   day_token: Union[str, Any],
                   day: Union[int, Any]) -> Any:
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
    if not day_token is None:
        if day is None:
            if len(day_token) == 2:
                result = result.replace(day_token, str('--'))
            elif len(day_token) == 1:
                result = result.replace(day_token, str('-'))
        else:
            if len(day_token) == 2:
                if day < 10:
                    result = result.replace(day_token, "0" + str(day), 1)
                else:
                    result = result.replace(day_token, str(day), 1)
            else:
                result = result.replace(day_token, str(day))

    return result

def transform_hour(result_str: Union[str, Any],
                   hour_token: Union[str, Any],
                   ispm: Union[bool, Any],
                   hour: Union[int, Any]) -> Any:
    """
    This function transform parsed month into target format
    Parameters
    ----------
    result_str
        result string
    hour_token
        token of hour
    hour
        value of hour
    """
    result = deepcopy(result_str)
    if not hour_token is None:
        if hour is None:
            if len(hour_token) == 2:
                result = result.replace(hour_token, str('--'))
            elif len(hour_token) == 1:
                result = result.replace(hour_token, str('-'))
        else:
            if ispm:
                hour = hour - 12
            if len(hour_token) == 2:
                if hour < 10:
                    result = result.replace(hour_token, "0" + str(hour), 1)
                else:
                    result = result.replace(hour_token, str(hour), 1)
            else:
                result = result.replace(hour_token, str(hour))

    return result

def transform_minute(result_str: Union[str, Any],
                   minute_token: Union[str, Any],
                   minute: Union[int, Any]) -> Any:
    """
    This function transform parsed month into target format
    Parameters
    ----------
    result_str
        result string
    minute_token
        token of minute
    minute
        value of minute
    """
    result = deepcopy(result_str)
    if not minute_token is None:
        if minute is None:
            if len(minute_token) == 2:
                result = result.replace(minute_token, str('--'))
            elif len(minute_token) == 1:
                result = result.replace(minute_token, str('-'))
        else:
            if len(minute_token) == 2:
                if minute < 10:
                    result = result.replace(minute_token, "0" + str(minute), 1)
                else:
                    result = result.replace(minute_token, str(minute), 1)
            else:
                result = result.replace(minute_token, str(minute))

    return result

def transform_second(result_str: Union[str, Any],
                   second_token: Union[str, Any],
                   second: Union[int, Any]) -> Any:
    """
    This function transform parsed month into target format
    Parameters
    ----------
    result_str
        result string
    second_token
        token of second
    second
        value of second
    """
    result = deepcopy(result_str)
    if not second_token is None:
        if second is None:
            if len(second_token) == 2:
                result = result.replace(second_token, str('--'))
            elif len(second_token) == 1:
                result = result.replace(second_token, str('-'))
        else:
            if len(second_token) == 2:
                if second < 10:
                    result = result.replace(second_token, "0" + str(second), 1)
                else:
                    result = result.replace(second_token, str(second), 1)
            else:
                result = result.replace(second_token, str(second))

    return result

def transform_weekday(result_str: Union[str, Any],
                   weekday_token: Union[str, Any],
                   weekday: Union[int, Any]) -> Any:
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
    if not weekday_token is None:
        if weekday is None:
            if len(weekday_token) == 3:
                result = result.replace(weekday_token, str('---'))
            elif len(weekday_token) == 5:
                result = result.replace(weekday_token, str('-----'))
        else:
            if len(weekday_token) == 3:
                result = result.replace(weekday_token, TEXT_WEEKDAYS[weekday - 1][0])
            elif len(weekday_token) == 5:
                result = result.replace(weekday_token, TEXT_WEEKDAYS[weekday - 1][1])

    return result

def transform_timezone(result_str: Union[str, Any],
                   timezone_token: Union[str, Any],
                   timezone: Union[int, Any],
                   tz_info: Union[dict, Any]) -> Any:
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
    utc_add = tz_info['utc_add']
    utc_offset_hours = tz_info['utc_offset_hours']
    utc_offset_minutes = tz_info['utc_offset_minutes']
    if not timezone_token is None:
        if timezone_token == 'z':
            result = result.replace(timezone_token, timezone)
        elif timezone_token == 'Z':
            offset_hours_str = str(int(utc_offset_hours))
            if len(offset_hours_str) == 1:
                offset_hours_str = '0' + offset_hours_str
            offset_minutes_str = str(int(utc_offset_minutes))
            if len(offset_minutes_str) == 1:
                offset_minutes_str = '0' + offset_minutes_str
            result = result.replace(timezone_token,
                                    "UTC" + utc_add +
                                    offset_hours_str + ":"
                                    + offset_minutes_str)

    return result

def transform(parsed_date_data: Union[parsed_date, Any],
              parsed_target_format_data: Union[parsed_target_format, Any],
              target_format: Union[str, Any],
              target_timezone: Union[str, Any]) -> Any:
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
    if not target_timezone is None:
        parsed_date_data = change_timezone(parsed_date_data, target_timezone)

    # Handle year
    result = transform_year(result,
                            parsed_target_format_data.year_token,
                            parsed_date_data.year)
    # Handle day
    result = transform_day(result,
                           parsed_target_format_data.day_token,
                           parsed_date_data.day)
    # Handle hours
    result = transform_hour(result,
                            parsed_target_format_data.hour_token,
                            parsed_target_format_data.ispm,
                            parsed_date_data.hour)
    # Handle minutes
    result = transform_minute(result,
                             parsed_target_format_data.minute_token,
                             parsed_date_data.minute)
    # Handle seconds
    result = transform_second(result,
                             parsed_target_format_data.second_token,
                             parsed_date_data.second)
    # Handle month
    result = transform_month(result,
                             parsed_target_format_data.month_token,
                             parsed_date_data.month)
    # Handle weekday
    result = transform_weekday(result,
                             parsed_target_format_data.weekday_token,
                             parsed_date_data.weekday)
    # Handle timezone
    tz_info = {'utc_add': parsed_date_data.utc_add,
               'utc_offset_hours': parsed_date_data.utc_offset_hours,
               'utc_offset_minutes': parsed_date_data.utc_offset_minutes}
    result = transform_timezone(result,
                             parsed_target_format_data.timezone_token,
                             parsed_date_data.timezone,
                             tz_info)
    return result

def reset_stats() -> None:
    """
    Reset global statistics dictionary.
    """
    STATS["cleaned"] = 0
    STATS["null"] = 0
    STATS["unknown"] = 0
