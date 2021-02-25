"""
module for testing the functions `clean_url()` and `validate_url()`.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from ...clean import clean_url, validate_url

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")  # type: ignore
def df_urls() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "messy_url": [
                "random text which is not a url",
                "http://www.facebookee.com/otherpath?auth=facebookeeauth&token=iwusdkc&not_token=hiThere&another_token=12323423",
                "https://www.sfu.ca/ficticiouspath?auth=sampletoken1&studentid=1234&loc=van",
                "notaurl",
                np.nan,
                None,
                "https://www.sfu.ca/ficticiouspath?auth=sampletoken2&studentid=1230&loc=bur",
                "",
                {"not_a_url": True},
                "2345678",
                345345345,
                "https://www.sfu.ca/ficticiouspath?auth=sampletoken3&studentid=1231&loc=sur",
                "https://www.sfu.ca/ficticiouspath?auth=sampletoken1&studentid=1232&loc=van",
            ]
        }
    )
    return df


def test_clean_default(df_urls: pd.DataFrame) -> None:
    df_clean = clean_url(df_urls, column="messy_url")
    df_check = df_urls.copy()
    df_check["messy_url_details"] = [
        np.nan,
        {
            "scheme": "http",
            "host": "www.facebookee.com",
            "messy_url_clean": "http://www.facebookee.com/otherpath",
            "queries": {
                "auth": "facebookeeauth",
                "token": "iwusdkc",
                "not_token": "hiThere",
                "another_token": "12323423",
            },
        },
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
            "queries": {"auth": "sampletoken1", "studentid": "1234", "loc": "van"},
        },
        np.nan,
        np.nan,
        np.nan,
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
            "queries": {"auth": "sampletoken2", "studentid": "1230", "loc": "bur"},
        },
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
            "queries": {"auth": "sampletoken3", "studentid": "1231", "loc": "sur"},
        },
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
            "queries": {"auth": "sampletoken1", "studentid": "1232", "loc": "van"},
        },
    ]

    assert df_check.equals(df_clean)


def test_clean_split(df_urls: pd.DataFrame) -> None:
    df_clean = clean_url(df_urls, column="messy_url", split=True)
    df_check = df_urls.copy()

    df_check["scheme"] = [
        np.nan,
        "http",
        "https",
        np.nan,
        np.nan,
        np.nan,
        "https",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        "https",
        "https",
    ]
    df_check["host"] = [
        np.nan,
        "www.facebookee.com",
        "www.sfu.ca",
        np.nan,
        np.nan,
        np.nan,
        "www.sfu.ca",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        "www.sfu.ca",
        "www.sfu.ca",
    ]
    df_check["messy_url_clean"] = [
        np.nan,
        "http://www.facebookee.com/otherpath",
        "https://www.sfu.ca/ficticiouspath",
        np.nan,
        np.nan,
        np.nan,
        "https://www.sfu.ca/ficticiouspath",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        "https://www.sfu.ca/ficticiouspath",
        "https://www.sfu.ca/ficticiouspath",
    ]
    df_check["queries"] = [
        np.nan,
        {
            "auth": "facebookeeauth",
            "token": "iwusdkc",
            "not_token": "hiThere",
            "another_token": "12323423",
        },
        {"auth": "sampletoken1", "studentid": "1234", "loc": "van"},
        np.nan,
        np.nan,
        np.nan,
        {"auth": "sampletoken2", "studentid": "1230", "loc": "bur"},
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        {"auth": "sampletoken3", "studentid": "1231", "loc": "sur"},
        {"auth": "sampletoken1", "studentid": "1232", "loc": "van"},
    ]

    assert df_clean.equals(df_check)


def test_remove_auth_boolean(df_urls: pd.DataFrame) -> None:
    df_clean = clean_url(df_urls, column="messy_url", remove_auth=True, report=False)
    df_check = df_urls.copy()

    df_check["messy_url_details"] = [
        np.nan,
        {
            "scheme": "http",
            "host": "www.facebookee.com",
            "messy_url_clean": "http://www.facebookee.com/otherpath",
            "queries": {"not_token": "hiThere", "another_token": "12323423"},
        },
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
            "queries": {"studentid": "1234", "loc": "van"},
        },
        np.nan,
        np.nan,
        np.nan,
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
            "queries": {"studentid": "1230", "loc": "bur"},
        },
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
            "queries": {"studentid": "1231", "loc": "sur"},
        },
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
            "queries": {"studentid": "1232", "loc": "van"},
        },
    ]

    assert df_check.equals(df_clean)


def test_remove_auth_list(df_urls: pd.DataFrame) -> None:
    df_clean = clean_url(df_urls, column="messy_url", remove_auth=["not_token"], report=False)
    df_check = df_urls.copy()

    df_check["messy_url_details"] = [
        np.nan,
        {
            "scheme": "http",
            "host": "www.facebookee.com",
            "messy_url_clean": "http://www.facebookee.com/otherpath",
            "queries": {"another_token": "12323423"},
        },
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
            "queries": {"studentid": "1234", "loc": "van"},
        },
        np.nan,
        np.nan,
        np.nan,
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
            "queries": {"studentid": "1230", "loc": "bur"},
        },
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
            "queries": {"studentid": "1231", "loc": "sur"},
        },
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
            "queries": {"studentid": "1232", "loc": "van"},
        },
    ]

    assert df_check.equals(df_clean)


def test_clean_inplace(df_urls: pd.DataFrame) -> None:
    df_clean = clean_url(df_urls, column="messy_url", inplace=True, report=False)
    df_check = pd.DataFrame(
        {
            "messy_url_details": [
                np.nan,
                {
                    "scheme": "http",
                    "host": "www.facebookee.com",
                    "messy_url_clean": "http://www.facebookee.com/otherpath",
                    "queries": {
                        "auth": "facebookeeauth",
                        "token": "iwusdkc",
                        "not_token": "hiThere",
                        "another_token": "12323423",
                    },
                },
                {
                    "scheme": "https",
                    "host": "www.sfu.ca",
                    "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
                    "queries": {"auth": "sampletoken1", "studentid": "1234", "loc": "van"},
                },
                np.nan,
                np.nan,
                np.nan,
                {
                    "scheme": "https",
                    "host": "www.sfu.ca",
                    "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
                    "queries": {"auth": "sampletoken2", "studentid": "1230", "loc": "bur"},
                },
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                {
                    "scheme": "https",
                    "host": "www.sfu.ca",
                    "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
                    "queries": {"auth": "sampletoken3", "studentid": "1231", "loc": "sur"},
                },
                {
                    "scheme": "https",
                    "host": "www.sfu.ca",
                    "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
                    "queries": {"auth": "sampletoken1", "studentid": "1232", "loc": "van"},
                },
            ]
        }
    )
    assert df_check.equals(df_clean)


def test_clean_errors_ignore(df_urls: pd.DataFrame) -> None:
    df_clean = clean_url(df_urls, column="messy_url", report=False, errors="ignore")
    df_check = df_urls.copy()

    df_check["messy_url_details"] = [
        "random text which is not a url",
        {
            "scheme": "http",
            "host": "www.facebookee.com",
            "messy_url_clean": "http://www.facebookee.com/otherpath",
            "queries": {
                "auth": "facebookeeauth",
                "token": "iwusdkc",
                "not_token": "hiThere",
                "another_token": "12323423",
            },
        },
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
            "queries": {"auth": "sampletoken1", "studentid": "1234", "loc": "van"},
        },
        "notaurl",
        np.nan,
        np.nan,
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
            "queries": {"auth": "sampletoken2", "studentid": "1230", "loc": "bur"},
        },
        np.nan,
        {"not_a_url": True},
        "2345678",
        345345345,
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
            "queries": {"auth": "sampletoken3", "studentid": "1231", "loc": "sur"},
        },
        {
            "scheme": "https",
            "host": "www.sfu.ca",
            "messy_url_clean": "https://www.sfu.ca/ficticiouspath",
            "queries": {"auth": "sampletoken1", "studentid": "1232", "loc": "van"},
        },
    ]

    assert df_clean.equals(df_check)


def test_validate_value() -> None:
    assert validate_url("ksjdhfskjdh") == False
    assert validate_url("http://www.facebook.com") == True
    assert (
        validate_url("https://www.sfu.ca/ficticiouspath?auth=sampletoken2&studentid=1230&loc=bur")
        == True
    )
    assert validate_url(np.nan) == False
    assert validate_url("") == False


def test_validate_series(df_urls: pd.DataFrame) -> None:
    df_valid = validate_url(df_urls["messy_url"])
    df_check = pd.Series(
        [
            False,
            True,
            True,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
        ]
    )
    assert df_check.equals(df_valid)
