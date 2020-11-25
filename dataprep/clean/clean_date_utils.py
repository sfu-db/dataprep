"""Common definitions and classes for clean_date function"""
from pytz import all_timezones

JUMP = [" ", ".", ",", ";", "-", "/", "'",
        "st", "nd", "rd", "th",
        "at", "on", "and", "ad", "AD", "of"]
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

class ParsedDate():

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
        self.ymd = {'year': None,
                    'month': None,
                    'day': None}
        self.hms = {'hour': None,
                    'minute': None,
                    'second': None}
        self.weekday = None
        self.tzinfo = {'timezone': None,
                       'utc_add': None,
                       'utc_offset_hours': None,
                       'utc_offset_minutes': None}
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
            self.ymd['year'] = year
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
            self.ymd['month'] = month
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
        if self.ymd['month'] in [1, 3, 5, 7, 8, 10, 12]:
            if 1 <= day <= 31:
                self.ymd['day'] = day
                return True
        if self.ymd['month'] in [4, 6, 9, 11]:
            if 1 <= day <= 30:
                self.ymd['day'] = day
                return True
        if self.ymd['month'] in [2]:
            if self._is_leap_year():
                if 1 <= day <= 29:
                    self.ymd['day'] = day
                    return True
            else:
                if 1 <= day <= 28:
                    self.ymd['day'] = day
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
            self.hms['hour'] = hour
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
            self.hms['minute'] = minute
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
            self.hms['second'] = second
            return True
        self.valid = 'unknown'
        return False

    def set_tzinfo(self,
                   timezone = None,
                   utc_add = None,
                   utc_offset_hours = None,
                   utc_offset_minutes = None):
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
        if not timezone is None:
            if timezone in all_timezones or timezone in ZONE:
                self.tzinfo['timezone'] = timezone
                return True
            self.valid = 'unknown'
            return False
        if not utc_add is None:
            self.tzinfo['utc_add'] = utc_add
        if not utc_offset_hours is None:
            self.tzinfo['utc_offset_hours'] = utc_offset_hours
        if not utc_offset_minutes is None:
            self.tzinfo['utc_offset_minutes'] = utc_offset_minutes
        return True

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
        if self.ymd['year'] % 4 == 0:
            if self.ymd['year'] % 100 == 0:
                return self.ymd['year'] % 400 == 0
            else:
                return True
        else:
            return False

class ParsedTargetFormat():

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
        self.ymd_token = {'year_token': None,
                          'month_token': None,
                          'day_token': None}
        self.hms_token = {'hour_token': None,
                          'minute_token': None,
                          'second_token': None,
                          'ispm': False}
        self.weekday_token = None
        self.timezone_token = None
        self.tzinfo = {'timezone': None,
                       'utc_add': None,
                       'utc_offset_hours': None,
                       'utc_offset_minutes': None}
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
        self.ymd_token['year_token'] = year_token
        return True

    def set_month_token(self, month_token):
        """
        This function set value of month_token
        Parameters
        ----------
        month_token
            token string of month
        """
        self.ymd_token['month_token'] = month_token
        return True

    def set_day_token(self, day_token):
        """
        This function set value of day_token
        Parameters
        ----------
        day_token
            token string of day
        """
        self.ymd_token['day_token'] = day_token
        return True

    def set_hour_token(self, hour_token):
        """
        This function set value of hour_token
        Parameters
        ----------
        hour_token
            token string of hour
        """
        self.hms_token['hour_token'] = hour_token
        return True

    def set_minute_token(self, minute_token):
        """
        This function set value of minute_token
        Parameters
        ----------
        minute_token
            token string of minute
        """
        self.hms_token['minute_token'] = minute_token
        return True

    def set_second_token(self, second_token):
        """
        This function set value of second_token
        Parameters
        ----------
        second_token
            token string of second
        """
        self.hms_token['second_token'] = second_token
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

    def set_tzinfo(self,
                   timezone = None,
                   utc_add = None,
                   utc_offset_hours = None,
                   utc_offset_minutes = None):
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
        if not timezone is None:
            self.tzinfo['timezone'] = timezone
        if not utc_add is None:
            self.tzinfo['utc_add'] = utc_add
        if not utc_offset_hours is None:
            self.tzinfo['utc_offset_hours'] = utc_offset_hours
        if not utc_offset_minutes is None:
            self.tzinfo['utc_offset_minutes'] = utc_offset_minutes
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
        self.hms_token['ispm'] = ispm
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
