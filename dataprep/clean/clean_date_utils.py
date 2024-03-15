"""
Common definitions and classes for the clean_date function.
"""

# pylint: disable-msg=too-many-branches

import datetime
from typing import Dict, List, Union

from pytz import all_timezones

from .utils import NULL_VALUES

JUMP = [
    " ",
    ".",
    ",",
    ";",
    "-",
    "/",
    "'",
    "st",
    "nd",
    "rd",
    "th",
    "at",
    "on",
    "and",
    "ad",
    "AD",
    "of",
]
WEEKDAYS = {
    "Mon": 1,
    "Monday": 1,
    "Tue": 2,
    "Tuesday": 2,
    "Wed": 3,
    "Wednesday": 3,
    "Thu": 4,
    "Thursday": 4,
    "Fri": 5,
    "Friday": 5,
    "Sat": 6,
    "Saturday": 6,
    "Sun": 7,
    "Sunday": 7,
}
MONTHS = {
    "Jan": 1,
    "January": 1,
    "Feb": 2,
    "February": 2,
    "Mar": 3,
    "March": 3,
    "Apr": 4,
    "April": 4,
    "May": 5,
    "Jun": 6,
    "June": 6,
    "Jul": 7,
    "July": 7,
    "Aug": 8,
    "August": 8,
    "Sep": 9,
    "Sept": 9,
    "September": 9,
    "Oct": 10,
    "October": 10,
    "Nov": 11,
    "November": 11,
    "Dec": 12,
    "December": 12,
}
HMS = [("h", "hour", "hours"), ("m", "minute", "minutes"), ("s", "second", "seconds")]
AM = ["AM", "am", "a"]
PM = ["PM", "pm", "p"]
ZONE = {
    "UTC": 0,
    "ACT": -5,
    "ADT": -3,
    "AEDT": 11,
    "AEST": 10,
    "AKDT": -8,
    "AKST": -9,
    "AMST": -3,
    "AMT": -4,
    "ART": -3,
    "ArabiaST": 3,
    "AtlanticST": -4,
    "AWST": 8,
    "AZOST": 0,
    "AZOT": 0,
    "BOT": -4,
    "BRST": -2,
    "BRT": -3,
    "BST": 1,
    "BTT": 6,
    "CAT": 2,
    "CDT": -5,
    "CEST": 2,
    "CET": 1,
    "CHOST": 9,
    "CHOT": 8,
    "CHUT": 10,
    "CKT": -10,
    "CLST": -3,
    "CLT": -4,
    "CentralST": -6,
    "ChinaST": 8,
    "CubaST": -5,
    "ChST": 10,
    "EASST": -5,
    "EAST": -6,
    "EAT": 3,
    "ECT": -5,
    "EDT": -4,
    "EEST": 3,
    "EET": 2,
    "EST": -5,
    "FKST": -3,
    "GFT": -3,
    "GILT": 12,
    "GMT": 0,
    "GST": 4,
    "HKT": 8,
    "HST": -10,
    "ICT": 7,
    "IDT": 3,
    "IrishST": 1,
    "IsraelST": 2,
    "JST": 9,
    "KOST": 11,
    "LINT": 4,
    "MDT": -6,
    "MHT": 12,
    "MSK": 3,
    "MST": -7,
    "MYT": 8,
    "NUT": -11,
    "NZDT": 13,
    "NZST": 12,
    "PDT": -7,
    "PET": -5,
    "PGT": 10,
    "PHT": 8,
    "PONT": 11,
    "PST": -8,
    "SAST": 2,
    "SBT": 11,
    "SGT": 8,
    "SRT": -3,
    "SST": -11,
    "TAHT": -10,
    "TLT": 9,
    "TVT": 12,
    "ULAST": 9,
    "ULAT": 8,
    "UYST": -2,
    "UYT": -3,
    "VET": -4,
    "WAST": 2,
    "WAT": 1,
    "WEST": 1,
    "WET": 0,
    "WIB": 7,
    "WIT": 9,
    "WITA": 8,
}

TARGET_YEAR = ["yyyy", "yy", "YYYY", "YY", "Y", "y"]
TARGET_MONTH = ["MM", "M", "MMM", "MMMMM"]
TEXT_MONTHS = [
    ("Jan", "January"),
    ("Feb", "February"),  # TODO: "Febr"
    ("Mar", "March"),
    ("Apr", "April"),
    ("May", "May"),
    ("Jun", "June"),
    ("Jul", "July"),
    ("Aug", "August"),
    ("Sep", "September"),
    ("Oct", "October"),
    ("Nov", "November"),
    ("Dec", "December"),
]
TARGET_DAY = ["dd", "d", "DD", "D"]
TARGET_HOUR = ["hh", "h", "HH", "H"]
TARGET_MINUTE = ["mm", "m"]
TARGET_SECOND = ["ss", "s", "SS", "S"]
TARGET_WEEKDAY = ["eee", "EEE", "eeeee", "EEEEE"]
TEXT_WEEKDAYS = [
    ("Mon", "Monday"),
    ("Tue", "Tuesday"),  # TODO: "Tues"
    ("Wed", "Wednesday"),
    ("Thu", "Thursday"),  # TODO: "Thurs"
    ("Fri", "Friday"),
    ("Sat", "Saturday"),
    ("Sun", "Sunday"),
]


class ParsedDate:
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

    def __init__(self) -> None:
        """
        This function initiate parse_date
        """
        self.ymd: Dict[str, int] = {"year": -1, "month": -1, "day": -1}
        self.hms: Dict[str, int] = {"hour": -1, "minute": -1, "second": -1}
        self.weekday: int = -1
        self.tzinfo: Dict[str, Union[int, str]] = {
            "timezone": "",
            "utc_add": "",
            "utc_offset_hours": -1,
            "utc_offset_minutes": -1,
        }
        self.valid: str = "cleaned"

    def set_year(self, year: int) -> None:
        """
        This function set value of year
        Parameters
        ----------
        year
            year value
        """
        if 1700 <= year <= 2500:
            self.ymd["year"] = year
        else:
            self.valid = "unknown"

    def set_month(self, month: int) -> None:
        """
        This function set value of month
        Parameters
        ----------
        month
            month value
        """
        if 1 <= month <= 12:
            self.ymd["month"] = month
        else:
            self.valid = "unknown"

    def set_day(self, day: int) -> None:
        """
        This function set value of day
        Parameters
        ----------
        day
            day value
        """
        # pylint: disable=too-many-branches
        if self.ymd["month"] in [1, 3, 5, 7, 8, 10, 12]:
            if 1 <= day <= 31:
                self.ymd["day"] = day
            else:
                self.valid = "unknown"
        elif self.ymd["month"] in [4, 6, 9, 11]:
            if 1 <= day <= 30:
                self.ymd["day"] = day
            else:
                self.valid = "unknown"
        elif self.ymd["month"] in [2]:
            if self._is_leap_year():
                if 1 <= day <= 29:
                    self.ymd["day"] = day
                else:
                    self.valid = "unknown"
            else:
                if 1 <= day <= 28:
                    self.ymd["day"] = day
                else:
                    self.valid = "unknown"
        else:
            self.valid = "unknown"

    def set_hour(self, hour: int) -> None:
        """
        This function set value of hour
        Parameters
        ----------
        hour
            hour value
        """
        if 0 <= hour < 24:
            self.hms["hour"] = hour
        else:
            self.valid = "unknown"

    def set_minute(self, minute: int) -> None:
        """
        This function set value of minute
        Parameters
        ----------
        minute
            minute value
        """
        if 0 <= minute < 60:
            self.hms["minute"] = minute
        else:
            self.valid = "unknown"

    def set_second(self, second: int) -> None:
        """
        This function set value of second
        Parameters
        ----------
        second
            second value
        """
        if 0 <= second < 60:
            self.hms["second"] = second
        else:
            self.valid = "unknown"

    def set_tzinfo(
        self,
        timezone: str = "",
        utc_add: str = "",
        utc_offset_hours: int = -1,
        utc_offset_minutes: int = -1,
    ) -> None:
        """
        This function set timezone info
        Parameters
        ----------
        timezone
            timezone value
        utc_add
            the offset is positive or negtive comaring to UTC
        utc_offset_hours
            value of offset hours
        utc_offset_minutes
            value of offset minutes
        """
        if timezone != "":
            if timezone in all_timezones or timezone in ZONE:
                self.tzinfo["timezone"] = timezone
            else:
                self.valid = "unknown"
        if utc_add != "":
            self.tzinfo["utc_add"] = utc_add
        if utc_offset_hours >= 0:
            self.tzinfo["utc_offset_hours"] = utc_offset_hours
        if utc_offset_minutes >= 0:
            self.tzinfo["utc_offset_minutes"] = utc_offset_minutes

    def set_weekday(self, weekday: int) -> None:
        """
        This function set value of weekday
        Parameters
        ----------
        weekday
            weekday value
        """
        if 1 <= weekday <= 7:
            self.weekday = weekday
        else:
            self.valid = "unknown"

    def _is_leap_year(self) -> bool:
        """
        This function judge if year is leap year
        """
        if self.ymd["year"] % 4 == 0:
            if self.ymd["year"] % 100 == 0:
                return self.ymd["year"] % 400 == 0
            else:
                return True
        return False


class ParsedTargetFormat:
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

    def __init__(self) -> None:
        """
        This function initiate parsed_target_fomat
        """
        self.ymd_token: Dict[str, str] = {"year_token": "", "month_token": "", "day_token": ""}
        self.hms_token: Dict[str, Union[str, bool]] = {
            "hour_token": "",
            "minute_token": "",
            "second_token": "",
            "ispm": False,
        }
        self.weekday_token: str = ""
        self.timezone_token: str = ""
        self.tzinfo: Dict[str, Union[int, str]] = {
            "timezone": "",
            "utc_add": "",
            "utc_offset_hours": -1,
            "utc_offset_minutes": -1,
        }
        self.valid: bool = True
        self.invalid_tokens: List[str] = []

    def set_year_token(self, year_token: str) -> None:
        """
        This function set value of year_token
        Parameters
        ----------
        year_token
            token string of year
        """
        self.ymd_token["year_token"] = year_token

    def set_month_token(self, month_token: str) -> None:
        """
        This function set value of month_token
        Parameters
        ----------
        month_token
            token string of month
        """
        self.ymd_token["month_token"] = month_token

    def set_day_token(self, day_token: str) -> None:
        """
        This function set value of day_token
        Parameters
        ----------
        day_token
            token string of day
        """
        self.ymd_token["day_token"] = day_token

    def set_hour_token(self, hour_token: str) -> None:
        """
        This function set value of hour_token
        Parameters
        ----------
        hour_token
            token string of hour
        """
        self.hms_token["hour_token"] = hour_token

    def set_minute_token(self, minute_token: str) -> None:
        """
        This function set value of minute_token
        Parameters
        ----------
        minute_token
            token string of minute
        """
        self.hms_token["minute_token"] = minute_token

    def set_second_token(self, second_token: str) -> None:
        """
        This function set value of second_token
        Parameters
        ----------
        second_token
            token string of second
        """
        self.hms_token["second_token"] = second_token

    def set_weekday_token(self, weekday_token: str) -> None:
        """
        This function set value of weekday_token
        Parameters
        ----------
        weekday_token
            token string of weekday
        """
        self.weekday_token = weekday_token

    def set_timezone_token(self, timezone_token: str) -> None:
        """
        This function set value of timezone_token
        Parameters
        ----------
        timezone_token
            token string of timezone
        """
        self.timezone_token = timezone_token

    def set_tzinfo(
        self,
        timezone: str = "",
        utc_add: str = "",
        utc_offset_hours: int = -1,
        utc_offset_minutes: int = -1,
    ) -> None:
        """
        This function set timezone info
        Parameters
        ----------
        timezone
            name of timezone
        utc_add
            the offset is positive or negtive comaring to UTC
        utc_offset_hours
            value of offset hours
        utc_offset_minutes
            value of offset minutes
        """
        if timezone != "":
            self.tzinfo["timezone"] = timezone
        if utc_add != "":
            self.tzinfo["utc_add"] = utc_add
        if utc_offset_hours >= 0:
            self.tzinfo["utc_offset_hours"] = utc_offset_hours
        if utc_offset_minutes >= 0:
            self.tzinfo["utc_offset_minutes"] = utc_offset_minutes

    def set_valid(self, valid: bool) -> None:
        """
        This function set valid status of target format
        Parameters
        ----------
        valid
            valid status
        """
        self.valid = valid

    def set_ispm(self, ispm: bool) -> None:
        """
        This function set value of judgement
        of PM status for target format
        Parameters
        ----------
        ispm
            If is PM, True. If not, False
        """
        self.hms_token["ispm"] = ispm

    def add_invalid_token(self, token: str) -> None:
        """
        This function set value of invalid tokens
        in target format
        Parameters
        ----------
        token
            invalid token
        """
        self.invalid_tokens.append(token)


def split(txt: str, seps: List[str]) -> List[str]:
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
    result = [value for value in [i.strip() for i in txt.split(default_sep)] if value != ""]
    return result


def check_date(date: str, clean: bool) -> Union[str, bool]:
    """
    This function check format of date
    Firstly, recognize timezone part in date string, and remove it
    Then, if month and weekdays are in text format, recognize and remove them
    Third, remove all numerical tokens because they are considered as valid vals
    Fourth, recognize AM, PM part and remove them
    Finally, remove all numerical tokens because they are considered as valid vals
    If there are non-numerical vals, we consider this string includes
    invalid tokens.
    Parameters
    ----------
    date
        date string
    """
    if date in NULL_VALUES:
        return "null" if clean else False
    date = str(date)
    tokens = split(date, JUMP)
    remain_tokens = tokens.copy()

    # Handle timezone
    for token in tokens:
        if token in all_timezones or token in ZONE:
            remain_tokens.remove(token)
    invalid_judge = not remain_tokens
    if invalid_judge:
        return "unknown" if clean else False

    # Handle weekdays text
    for token in tokens:
        if token in WEEKDAYS:
            remain_tokens.remove(token)
    invalid_judge = not remain_tokens
    if invalid_judge:
        return "unknown" if clean else False

    # Handle single AM and PM text
    for token in tokens:
        if token in AM + PM:
            remain_tokens.remove(token)
    invalid_judge = not remain_tokens
    if invalid_judge:
        return "unknown" if clean else False

    # Handle month text
    for token in tokens:
        if token in MONTHS:
            remain_tokens.remove(token)

    # Handle single numbers
    for token in remain_tokens:
        if token.isnumeric():
            remain_tokens.remove(token)

    # Handle connected AM, PM
    for token in remain_tokens:
        tokens = split(token, AM + PM + [":"])
        invalid_judge = False in [temp_token.isnumeric() for temp_token in tokens]
        if invalid_judge:
            return "unknown" if clean else False
    return "cleaned" if clean else True


def fix_missing_current(parsed_res: ParsedDate) -> ParsedDate:
    """
    This function fix empty part by nearest time
    Parameters
    ----------
    parsed_res
        parsed date result
    """
    now_time = datetime.datetime.now()
    if parsed_res.ymd["year"] == -1:
        parsed_res.set_year(now_time.year)
    if parsed_res.ymd["month"] == -1:
        parsed_res.set_month(now_time.month)
    if parsed_res.ymd["day"] == -1:
        parsed_res.set_day(now_time.day)
    if parsed_res.hms["hour"] == -1:
        parsed_res.set_hour(now_time.hour)
    if parsed_res.hms["minute"] == -1:
        parsed_res.set_minute(now_time.minute)
    if parsed_res.hms["second"] == -1:
        parsed_res.set_second(now_time.second)
    if parsed_res.weekday == -1:
        temp_date = datetime.datetime(
            parsed_res.ymd["year"], parsed_res.ymd["month"], parsed_res.ymd["day"]
        )
        parsed_res.set_weekday(temp_date.weekday() + 1)
    return parsed_res


def fix_missing_minimum(parsed_res: ParsedDate) -> ParsedDate:
    """
    This function fix empty part by minimum time
    Parameters
    ----------
    parsed_res
        parsed date result
    """
    if parsed_res.ymd["year"] == -1:
        parsed_res.set_year(2000)
    if parsed_res.ymd["month"] == -1:
        parsed_res.set_month(1)
    if parsed_res.ymd["day"] == -1:
        parsed_res.set_day(1)
    if parsed_res.hms["hour"] == -1:
        parsed_res.set_hour(0)
    if parsed_res.hms["minute"] == -1:
        parsed_res.set_minute(0)
    if parsed_res.hms["second"] == -1:
        parsed_res.set_second(0)
    if parsed_res.weekday == -1:
        temp_date = datetime.datetime(
            parsed_res.ymd["year"], parsed_res.ymd["month"], parsed_res.ymd["day"]
        )
        parsed_res.set_weekday(temp_date.weekday() + 1)
    return parsed_res
