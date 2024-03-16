"""
module for testing the functions clean_email() and validate_email()
"""

import logging

import numpy as np
import pandas as pd
import pytest

from ...clean import clean_email, validate_email

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")  # type: ignore
def df_broken_email() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "messy_email": [
                "yi@gmali.com",
                "yi@sfu.ca",
                "y i@sfu.ca",
                "Yi@gmail.com",
                "H ELLO@hotmal.COM",
                "hello",
                np.nan,
                "NULL",
            ]
        }
    )
    return df


def test_clean_default(df_broken_email: pd.DataFrame) -> None:
    df_clean = clean_email(df_broken_email, "messy_email")
    df_check = df_broken_email.copy()
    df_check["messy_email_clean"] = [
        "yi@gmali.com",
        "yi@sfu.ca",
        None,
        "yi@gmail.com",
        None,
        None,
        None,
        None,
    ]
    assert df_check.equals(df_clean)


def test_clean_split(df_broken_email: pd.DataFrame) -> None:
    df_clean = clean_email(df_broken_email, "messy_email", split=True)
    df_check = df_broken_email.copy()
    df_check["username"] = ["yi", "yi", None, "yi", None, None, None, None]
    df_check["domain"] = [
        "gmali.com",
        "sfu.ca",
        None,
        "gmail.com",
        None,
        None,
        None,
        None,
    ]
    assert df_check.equals(df_clean)


def test_clean_remove_whitespace(df_broken_email: pd.DataFrame) -> None:
    df_clean = clean_email(df_broken_email, "messy_email", remove_whitespace=True)
    df_check = df_broken_email.copy()
    df_check["messy_email_clean"] = [
        "yi@gmali.com",
        "yi@sfu.ca",
        "yi@sfu.ca",
        "yi@gmail.com",
        "hello@hotmal.com",
        None,
        None,
        None,
    ]
    assert df_check.equals(df_clean)


def test_clean_fix_domain(df_broken_email: pd.DataFrame) -> None:
    df_clean = clean_email(df_broken_email, "messy_email", fix_domain=True)
    df_check = df_broken_email.copy()
    df_check["messy_email_clean"] = [
        "yi@gmail.com",
        "yi@sfu.ca",
        None,
        "yi@gmail.com",
        None,
        None,
        None,
        None,
    ]
    assert df_check.equals(df_clean)


def test_clean_inplace(df_broken_email: pd.DataFrame) -> None:
    df_clean = clean_email(df_broken_email, "messy_email", inplace=True)
    df_check = pd.DataFrame(
        {
            "messy_email_clean": [
                "yi@gmali.com",
                "yi@sfu.ca",
                None,
                "yi@gmail.com",
                None,
                None,
                None,
                None,
            ]
        }
    )
    assert df_check.equals(df_clean)


def test_validate_value() -> None:
    assert validate_email("Abc.example.com") == False
    assert validate_email("prettyandsimple@example.com") == True
    assert validate_email("disposable.style.email.with+symbol@example.com") == True
    assert validate_email('this is"not\allowed@example.com') == False


def test_validate_series(df_broken_email: pd.DataFrame) -> None:
    df_valid = validate_email(df_broken_email["messy_email"])
    df_check = pd.Series(
        [
            True,
            True,
            False,
            True,
            False,
            False,
            False,
            False,
        ],
        name="messy_lat_long",
    )
    assert df_check.equals(df_valid)
