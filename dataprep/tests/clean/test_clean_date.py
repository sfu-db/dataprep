"""
module for testing the functions clean_country() and validate_country()
"""

import logging

import numpy as np
import pandas as pd
import pytest

from ...clean import clean_date, validate_date

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")  # type: ignore
def df_dates() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "date": [
                "1996.07.10 AD at 15:08:56 PDT",
                "Thu Sep 25 10:36:28 2003",
                "Thu Sep 25 10:36:28 BRST 2003",
                "2003 10:36:28 BRST 25 Sep Thu",
                "Thu Sep 25 10:36:28 2003",
                "Thu 10:36:28",
                "Thu 10:36",
                "10:36",
                "Thu Sep 25 2003",
                "Sep 25 2003",
                "Sep 2003",
                "Sep",
                "2003",
                "2003-09-25",
                "2003-Sep-25",
                "25-Sep-2003",
                "Sep-25-2003",
                "09-25-2003",
                "10-09-2003",
                "10-09-03",
                "2003.Sep.25",
                "2003/09/25",
                "2003 Sep 25",
                "2003 09 25",
                "10pm",
                "12:00am",
                "Sep 03",
                "Sep of 03",
                "Wed, July 10, 96",
                "1996.07.10 AD at 15:08:56 PDT",
                "Tuesday, April 12, 1952 AD 3:30:42pm PST",
                "November 5, 1994, 8:15:30 am EST",
                "3rd of May 2001",
                "5:50 AM on June 13, 1990",
                "NULL",
                "nan",
                "I'm a little cat",
                "This is Sep.",
            ]
        }
    )
    return df


@pytest.fixture(scope="module")  # type: ignore
def df_messy_date() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "messy_date": [
                "T, Ap 12, 1952 AD 3:30:42p",
                "5:50 AM on June 13, 1990",
                "3rd of May 2001",
                "55/23/2014",
                "10pm",
                "10p",
                "2003-Sep-25",
                "Sepppppp",
                "23 4 1962",
                "2003 10:36:28 BRST 25 Sep Thu",
                "hello",
                np.nan,
                "NULL",
            ]
        }
    )
    return df


def test_clean_default(df_dates: pd.DataFrame) -> None:
    df_clean = clean_date(df_dates, "date")
    df_check = df_dates.copy()
    df_check["date_clean"] = [
        "1996-07-10 15:08:56",
        "2003-09-25 10:36:28",
        "2003-09-25 10:36:28",
        "2003-09-25 10:36:28",
        "2003-09-25 10:36:28",
        "2000-01-01 10:36:28",
        "2000-01-01 10:36:00",
        "2000-01-01 10:36:00",
        "2003-09-25 00:00:00",
        "2003-09-25 00:00:00",
        "2003-09-01 00:00:00",
        "2000-09-01 00:00:00",
        "2003-01-01 00:00:00",
        "2003-09-25 00:00:00",
        "2003-09-25 00:00:00",
        "2003-09-25 00:00:00",
        "2003-09-25 00:00:00",
        "2003-09-25 00:00:00",
        "2003-10-09 00:00:00",
        "2003-10-09 00:00:00",
        "2003-09-25 00:00:00",
        "2003-09-25 00:00:00",
        "2003-09-25 00:00:00",
        "2003-09-25 00:00:00",
        "2000-01-01 22:00:00",
        "2000-01-01 12:00:00",
        "2003-09-01 00:00:00",
        "2003-09-01 00:00:00",
        "2096-07-10 00:00:00",
        "1996-07-10 15:08:56",
        "1952-04-12 15:30:42",
        "1994-11-05 08:15:30",
        "2001-05-03 00:00:00",
        "1990-06-13 05:50:00",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    assert df_check.equals(df_clean)


def test_clean_output_format(df_dates: pd.DataFrame) -> None:
    df_clean1 = clean_date(df_dates, "date", output_format="YYYY-MM-DD")
    df_clean2 = clean_date(df_dates, "date", output_format="yyyy.MM.dd AD at HH:mm:ss Z")
    df_clean3 = clean_date(df_dates, "date", output_format="yyyy.MM.dd AD at HH:mm:ss z")
    df_clean4 = clean_date(df_dates, "date", output_format="EEE, d MMM yyyy HH:mm:ss Z")

    df_check1 = df_dates.copy()
    df_check1["date_clean"] = [
        "1996-07-10",
        "2003-09-25",
        "2003-09-25",
        "2003-09-25",
        "2003-09-25",
        "2000-01-01",
        "2000-01-01",
        "2000-01-01",
        "2003-09-25",
        "2003-09-25",
        "2003-09-01",
        "2000-09-01",
        "2003-01-01",
        "2003-09-25",
        "2003-09-25",
        "2003-09-25",
        "2003-09-25",
        "2003-09-25",
        "2003-10-09",
        "2003-10-09",
        "2003-09-25",
        "2003-09-25",
        "2003-09-25",
        "2003-09-25",
        "2000-01-01",
        "2000-01-01",
        "2003-09-01",
        "2003-09-01",
        "2096-07-10",
        "1996-07-10",
        "1952-04-12",
        "1994-11-05",
        "2001-05-03",
        "1990-06-13",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    df_check2 = df_dates.copy()
    df_check2["date_clean"] = [
        "1996.07.10 AD at 15:08:56 UTC+00:00",
        "2003.09.25 AD at 10:36:28 UTC+00:00",
        "2003.09.25 AD at 10:36:28 UTC+00:00",
        "2003.09.25 AD at 10:36:28 UTC+00:00",
        "2003.09.25 AD at 10:36:28 UTC+00:00",
        "2000.01.01 AD at 10:36:28 UTC+00:00",
        "2000.01.01 AD at 10:36:00 UTC+00:00",
        "2000.01.01 AD at 10:36:00 UTC+00:00",
        "2003.09.25 AD at 00:00:00 UTC+00:00",
        "2003.09.25 AD at 00:00:00 UTC+00:00",
        "2003.09.01 AD at 00:00:00 UTC+00:00",
        "2000.09.01 AD at 00:00:00 UTC+00:00",
        "2003.01.01 AD at 00:00:00 UTC+00:00",
        "2003.09.25 AD at 00:00:00 UTC+00:00",
        "2003.09.25 AD at 00:00:00 UTC+00:00",
        "2003.09.25 AD at 00:00:00 UTC+00:00",
        "2003.09.25 AD at 00:00:00 UTC+00:00",
        "2003.09.25 AD at 00:00:00 UTC+00:00",
        "2003.10.09 AD at 00:00:00 UTC+00:00",
        "2003.10.09 AD at 00:00:00 UTC+00:00",
        "2003.09.25 AD at 00:00:00 UTC+00:00",
        "2003.09.25 AD at 00:00:00 UTC+00:00",
        "2003.09.25 AD at 00:00:00 UTC+00:00",
        "2003.09.25 AD at 00:00:00 UTC+00:00",
        "2000.01.01 AD at 22:00:00 UTC+00:00",
        "2000.01.01 AD at 12:00:00 UTC+00:00",
        "2003.09.01 AD at 00:00:00 UTC+00:00",
        "2003.09.01 AD at 00:00:00 UTC+00:00",
        "2096.07.10 AD at 00:00:00 UTC+00:00",
        "1996.07.10 AD at 15:08:56 UTC+00:00",
        "1952.04.12 AD at 15:30:42 UTC+00:00",
        "1994.11.05 AD at 08:15:30 UTC+00:00",
        "2001.05.03 AD at 00:00:00 UTC+00:00",
        "1990.06.13 AD at 05:50:00 UTC+00:00",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    df_check3 = df_dates.copy()
    df_check3["date_clean"] = [
        "1996.07.10 AD at 15:08:56 UTC",
        "2003.09.25 AD at 10:36:28 UTC",
        "2003.09.25 AD at 10:36:28 UTC",
        "2003.09.25 AD at 10:36:28 UTC",
        "2003.09.25 AD at 10:36:28 UTC",
        "2000.01.01 AD at 10:36:28 UTC",
        "2000.01.01 AD at 10:36:00 UTC",
        "2000.01.01 AD at 10:36:00 UTC",
        "2003.09.25 AD at 00:00:00 UTC",
        "2003.09.25 AD at 00:00:00 UTC",
        "2003.09.01 AD at 00:00:00 UTC",
        "2000.09.01 AD at 00:00:00 UTC",
        "2003.01.01 AD at 00:00:00 UTC",
        "2003.09.25 AD at 00:00:00 UTC",
        "2003.09.25 AD at 00:00:00 UTC",
        "2003.09.25 AD at 00:00:00 UTC",
        "2003.09.25 AD at 00:00:00 UTC",
        "2003.09.25 AD at 00:00:00 UTC",
        "2003.10.09 AD at 00:00:00 UTC",
        "2003.10.09 AD at 00:00:00 UTC",
        "2003.09.25 AD at 00:00:00 UTC",
        "2003.09.25 AD at 00:00:00 UTC",
        "2003.09.25 AD at 00:00:00 UTC",
        "2003.09.25 AD at 00:00:00 UTC",
        "2000.01.01 AD at 22:00:00 UTC",
        "2000.01.01 AD at 12:00:00 UTC",
        "2003.09.01 AD at 00:00:00 UTC",
        "2003.09.01 AD at 00:00:00 UTC",
        "2096.07.10 AD at 00:00:00 UTC",
        "1996.07.10 AD at 15:08:56 UTC",
        "1952.04.12 AD at 15:30:42 UTC",
        "1994.11.05 AD at 08:15:30 UTC",
        "2001.05.03 AD at 00:00:00 UTC",
        "1990.06.13 AD at 05:50:00 UTC",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    df_check4 = df_dates.copy()
    df_check4["date_clean"] = [
        "Wed, 10 Jul 1996 15:08:56 UTC+00:00",
        "Thu, 25 Sep 2003 10:36:28 UTC+00:00",
        "Thu, 25 Sep 2003 10:36:28 UTC+00:00",
        "Thu, 25 Sep 2003 10:36:28 UTC+00:00",
        "Thu, 25 Sep 2003 10:36:28 UTC+00:00",
        "Thu, 1 Jan 2000 10:36:28 UTC+00:00",
        "Thu, 1 Jan 2000 10:36:00 UTC+00:00",
        "Sat, 1 Jan 2000 10:36:00 UTC+00:00",
        "Thu, 25 Sep 2003 00:00:00 UTC+00:00",
        "Thu, 25 Sep 2003 00:00:00 UTC+00:00",
        "Mon, 1 Sep 2003 00:00:00 UTC+00:00",
        "Fri, 1 Sep 2000 00:00:00 UTC+00:00",
        "Wed, 1 Jan 2003 00:00:00 UTC+00:00",
        "Thu, 25 Sep 2003 00:00:00 UTC+00:00",
        "Thu, 25 Sep 2003 00:00:00 UTC+00:00",
        "Thu, 25 Sep 2003 00:00:00 UTC+00:00",
        "Thu, 25 Sep 2003 00:00:00 UTC+00:00",
        "Thu, 25 Sep 2003 00:00:00 UTC+00:00",
        "Thu, 9 Oct 2003 00:00:00 UTC+00:00",
        "Thu, 9 Oct 2003 00:00:00 UTC+00:00",
        "Thu, 25 Sep 2003 00:00:00 UTC+00:00",
        "Thu, 25 Sep 2003 00:00:00 UTC+00:00",
        "Thu, 25 Sep 2003 00:00:00 UTC+00:00",
        "Thu, 25 Sep 2003 00:00:00 UTC+00:00",
        "Sat, 1 Jan 2000 22:00:00 UTC+00:00",
        "Sat, 1 Jan 2000 12:00:00 UTC+00:00",
        "Mon, 1 Sep 2003 00:00:00 UTC+00:00",
        "Mon, 1 Sep 2003 00:00:00 UTC+00:00",
        "Wed, 10 Jul 2096 00:00:00 UTC+00:00",
        "Wed, 10 Jul 1996 15:08:56 UTC+00:00",
        "Tue, 12 Apr 1952 15:30:42 UTC+00:00",
        "Sat, 5 Nov 1994 08:15:30 UTC+00:00",
        "Thu, 3 May 2001 00:00:00 UTC+00:00",
        "Wed, 13 Jun 1990 05:50:00 UTC+00:00",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    assert df_clean1.equals(df_check1)
    assert df_clean2.equals(df_check2)
    assert df_clean3.equals(df_check3)
    assert df_clean4.equals(df_check4)


def test_input_timezone_output_timezone(df_dates: pd.DataFrame) -> None:
    df_clean1 = clean_date(
        df_dates,
        "date",
        input_timezone="PDT",
        output_timezone="ChinaST",
        output_format="yyyy.MM.dd AD at HH:mm:ss Z",
    )
    df_clean2 = clean_date(
        df_dates,
        "date",
        input_timezone="EST",
        output_timezone="PDT",
        output_format="yyyy.MM.dd AD at HH:mm:ss Z",
    )
    df_clean3 = clean_date(
        df_dates,
        "date",
        input_timezone="PST",
        output_timezone="GMT",
        output_format="yyyy.MM.dd AD at HH:mm:ss Z",
    )
    df_check1 = df_dates.copy()
    df_check1["date_clean"] = [
        "1996.07.11 AD at 06:08:56 UTC+08:00",
        "2003.09.26 AD at 01:36:28 UTC+08:00",
        "2003.09.26 AD at 01:36:28 UTC+08:00",
        "2003.09.26 AD at 01:36:28 UTC+08:00",
        "2003.09.26 AD at 01:36:28 UTC+08:00",
        "2000.01.02 AD at 01:36:28 UTC+08:00",
        "2000.01.02 AD at 01:36:00 UTC+08:00",
        "2000.01.02 AD at 01:36:00 UTC+08:00",
        "2003.09.25 AD at 15:00:00 UTC+08:00",
        "2003.09.25 AD at 15:00:00 UTC+08:00",
        "2003.09.01 AD at 15:00:00 UTC+08:00",
        "2000.09.01 AD at 15:00:00 UTC+08:00",
        "2003.01.01 AD at 15:00:00 UTC+08:00",
        "2003.09.25 AD at 15:00:00 UTC+08:00",
        "2003.09.25 AD at 15:00:00 UTC+08:00",
        "2003.09.25 AD at 15:00:00 UTC+08:00",
        "2003.09.25 AD at 15:00:00 UTC+08:00",
        "2003.09.25 AD at 15:00:00 UTC+08:00",
        "2003.10.09 AD at 15:00:00 UTC+08:00",
        "2003.10.09 AD at 15:00:00 UTC+08:00",
        "2003.09.25 AD at 15:00:00 UTC+08:00",
        "2003.09.25 AD at 15:00:00 UTC+08:00",
        "2003.09.25 AD at 15:00:00 UTC+08:00",
        "2003.09.25 AD at 15:00:00 UTC+08:00",
        "2000.01.02 AD at 13:00:00 UTC+08:00",
        "2000.01.02 AD at 03:00:00 UTC+08:00",
        "2003.09.01 AD at 15:00:00 UTC+08:00",
        "2003.09.01 AD at 15:00:00 UTC+08:00",
        "2096.07.10 AD at 15:00:00 UTC+08:00",
        "1996.07.11 AD at 06:08:56 UTC+08:00",
        "1952.04.13 AD at 06:30:42 UTC+08:00",
        "1994.11.05 AD at 23:15:30 UTC+08:00",
        "2001.05.03 AD at 15:00:00 UTC+08:00",
        "1990.06.13 AD at 20:50:00 UTC+08:00",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    df_check2 = df_dates.copy()
    df_check2["date_clean"] = [
        "1996.07.12 AD at 03:08:56 UTC-07:00",
        "2003.09.26 AD at 22:36:28 UTC-07:00",
        "2003.09.26 AD at 22:36:28 UTC-07:00",
        "2003.09.26 AD at 22:36:28 UTC-07:00",
        "2003.09.26 AD at 22:36:28 UTC-07:00",
        "2000.01.02 AD at 22:36:28 UTC-07:00",
        "2000.01.02 AD at 22:36:00 UTC-07:00",
        "2000.01.02 AD at 22:36:00 UTC-07:00",
        "2003.09.26 AD at 12:00:00 UTC-07:00",
        "2003.09.26 AD at 12:00:00 UTC-07:00",
        "2003.09.02 AD at 12:00:00 UTC-07:00",
        "2000.09.02 AD at 12:00:00 UTC-07:00",
        "2003.01.02 AD at 12:00:00 UTC-07:00",
        "2003.09.26 AD at 12:00:00 UTC-07:00",
        "2003.09.26 AD at 12:00:00 UTC-07:00",
        "2003.09.26 AD at 12:00:00 UTC-07:00",
        "2003.09.26 AD at 12:00:00 UTC-07:00",
        "2003.09.26 AD at 12:00:00 UTC-07:00",
        "2003.10.10 AD at 12:00:00 UTC-07:00",
        "2003.10.10 AD at 12:00:00 UTC-07:00",
        "2003.09.26 AD at 12:00:00 UTC-07:00",
        "2003.09.26 AD at 12:00:00 UTC-07:00",
        "2003.09.26 AD at 12:00:00 UTC-07:00",
        "2003.09.26 AD at 12:00:00 UTC-07:00",
        "2000.01.03 AD at 10:00:00 UTC-07:00",
        "2000.01.03 AD at 00:00:00 UTC-07:00",
        "2003.09.02 AD at 12:00:00 UTC-07:00",
        "2003.09.02 AD at 12:00:00 UTC-07:00",
        "2096.07.11 AD at 12:00:00 UTC-07:00",
        "1996.07.12 AD at 03:08:56 UTC-07:00",
        "1952.04.14 AD at 03:30:42 UTC-07:00",
        "1994.11.06 AD at 20:15:30 UTC-07:00",
        "2001.05.04 AD at 12:00:00 UTC-07:00",
        "1990.06.14 AD at 17:50:00 UTC-07:00",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    df_check3 = df_dates.copy()
    df_check3["date_clean"] = [
        "1996.07.10 AD at 23:08:56 UTC+00:00",
        "2003.09.25 AD at 18:36:28 UTC+00:00",
        "2003.09.25 AD at 18:36:28 UTC+00:00",
        "2003.09.25 AD at 18:36:28 UTC+00:00",
        "2003.09.25 AD at 18:36:28 UTC+00:00",
        "2000.01.01 AD at 18:36:28 UTC+00:00",
        "2000.01.01 AD at 18:36:00 UTC+00:00",
        "2000.01.01 AD at 18:36:00 UTC+00:00",
        "2003.09.25 AD at 08:00:00 UTC+00:00",
        "2003.09.25 AD at 08:00:00 UTC+00:00",
        "2003.09.01 AD at 08:00:00 UTC+00:00",
        "2000.09.01 AD at 08:00:00 UTC+00:00",
        "2003.01.01 AD at 08:00:00 UTC+00:00",
        "2003.09.25 AD at 08:00:00 UTC+00:00",
        "2003.09.25 AD at 08:00:00 UTC+00:00",
        "2003.09.25 AD at 08:00:00 UTC+00:00",
        "2003.09.25 AD at 08:00:00 UTC+00:00",
        "2003.09.25 AD at 08:00:00 UTC+00:00",
        "2003.10.09 AD at 08:00:00 UTC+00:00",
        "2003.10.09 AD at 08:00:00 UTC+00:00",
        "2003.09.25 AD at 08:00:00 UTC+00:00",
        "2003.09.25 AD at 08:00:00 UTC+00:00",
        "2003.09.25 AD at 08:00:00 UTC+00:00",
        "2003.09.25 AD at 08:00:00 UTC+00:00",
        "2000.01.02 AD at 06:00:00 UTC+00:00",
        "2000.01.01 AD at 20:00:00 UTC+00:00",
        "2003.09.01 AD at 08:00:00 UTC+00:00",
        "2003.09.01 AD at 08:00:00 UTC+00:00",
        "2096.07.10 AD at 08:00:00 UTC+00:00",
        "1996.07.10 AD at 23:08:56 UTC+00:00",
        "1952.04.12 AD at 23:30:42 UTC+00:00",
        "1994.11.05 AD at 16:15:30 UTC+00:00",
        "2001.05.03 AD at 08:00:00 UTC+00:00",
        "1990.06.13 AD at 13:50:00 UTC+00:00",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    assert df_clean1.equals(df_check1)
    assert df_clean2.equals(df_check2)
    assert df_clean3.equals(df_check3)


def test_clean_fix_missing(df_dates: pd.DataFrame) -> None:
    df_clean_minimum = clean_date(df_dates, "date", fix_missing="minimum")
    df_clean_empty = clean_date(df_dates, "date", fix_missing="empty")
    df_check_minimum = df_dates.copy()
    df_check_minimum["date_clean"] = [
        "1996-07-10 15:08:56",
        "2003-09-25 10:36:28",
        "2003-09-25 10:36:28",
        "2003-09-25 10:36:28",
        "2003-09-25 10:36:28",
        "2000-01-01 10:36:28",
        "2000-01-01 10:36:00",
        "2000-01-01 10:36:00",
        "2003-09-25 00:00:00",
        "2003-09-25 00:00:00",
        "2003-09-01 00:00:00",
        "2000-09-01 00:00:00",
        "2003-01-01 00:00:00",
        "2003-09-25 00:00:00",
        "2003-09-25 00:00:00",
        "2003-09-25 00:00:00",
        "2003-09-25 00:00:00",
        "2003-09-25 00:00:00",
        "2003-10-09 00:00:00",
        "2003-10-09 00:00:00",
        "2003-09-25 00:00:00",
        "2003-09-25 00:00:00",
        "2003-09-25 00:00:00",
        "2003-09-25 00:00:00",
        "2000-01-01 22:00:00",
        "2000-01-01 12:00:00",
        "2003-09-01 00:00:00",
        "2003-09-01 00:00:00",
        "2096-07-10 00:00:00",
        "1996-07-10 15:08:56",
        "1952-04-12 15:30:42",
        "1994-11-05 08:15:30",
        "2001-05-03 00:00:00",
        "1990-06-13 05:50:00",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    df_check_empty = df_dates.copy()
    df_check_empty["date_clean"] = [
        "1996-07-10 15:08:56",
        "2003-09-25 10:36:28",
        "2003-09-25 10:36:28",
        "2003-09-25 10:36:28",
        "2003-09-25 10:36:28",
        "---------- 10:36:28",
        "---------- 10:36:--",
        "---------- 10:36:--",
        "2003-09-25 --:--:--",
        "2003-09-25 --:--:--",
        "2003-09--- --:--:--",
        "-----09--- --:--:--",
        "2003------ --:--:--",
        "2003-09-25 --:--:--",
        "2003-09-25 --:--:--",
        "2003-09-25 --:--:--",
        "2003-09-25 --:--:--",
        "2003-09-25 --:--:--",
        "2003-10-09 --:--:--",
        "2003-10-09 --:--:--",
        "2003-09-25 --:--:--",
        "2003-09-25 --:--:--",
        "2003-09-25 --:--:--",
        "2003-09-25 --:--:--",
        "---------- 22:--:--",
        "---------- 12:00:--",
        "2003-09--- --:--:--",
        "2003-09--- --:--:--",
        "2096-07-10 --:--:--",
        "1996-07-10 15:08:56",
        "1952-04-12 15:30:42",
        "1994-11-05 08:15:30",
        "2001-05-03 --:--:--",
        "1990-06-13 05:50:--",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    assert df_clean_minimum.equals(df_check_minimum)
    assert df_clean_empty.equals(df_check_empty)


def test_validate_value() -> None:
    assert validate_date("Novvvvvvvvember 5, 1994, 8:15:30 am EST hahaha") == False
    assert validate_date("1994, 8:15:30") == True
    assert validate_date("Hello.") == False


def test_validate_series(df_messy_date: pd.DataFrame) -> None:
    srs_valid = validate_date(df_messy_date["messy_date"])
    srs_check = pd.Series(
        [
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            False,
            False,
            False,
        ],
        name="valid",
    )
    assert srs_check.equals(srs_valid)
