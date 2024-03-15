"""
module for testing the functions clean_phone() and validate_phone()
"""

import logging

import numpy as np
import pandas as pd
import pytest

from ...clean import clean_phone, validate_phone

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")  # type: ignore
def df_phone() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "messy_phone": [
                "555-234-5678",
                "(555) 234-5678",
                "555.234.5678",
                "555/234/5678",
                15551234567,
                "(1) 555-234-5678",
                "+1 (234) 567-8901 x. 1234",
                "2345678901 extension 1234",
                "2345678",
                "800-299-JUNK",
                "1-866-4ZIPCAR",
                "1-800-G-O-T-J-U-N-K",
                "123 ABC COMPANY",
                "+66 91 889 8948",
                "hello",
                np.nan,
                "NULL",
            ]
        }
    )
    return df


def test_clean_default(df_phone: pd.DataFrame) -> None:
    df_clean = clean_phone(df_phone, "messy_phone")
    df_check = df_phone.copy()
    df_check["messy_phone_clean"] = [
        "555-234-5678",
        "555-234-5678",
        "555-234-5678",
        "555-234-5678",
        "555-123-4567",
        "555-234-5678",
        "234-567-8901 ext. 1234",
        "234-567-8901 ext. 1234",
        "234-5678",
        "800-299-5865",
        "866-494-7227",
        "800-468-5865",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    assert df_check.equals(df_clean)


def test_clean_output_format(df_phone: pd.DataFrame) -> None:
    df_clean_e164 = clean_phone(df_phone, "messy_phone", output_format="e164")
    df_clean_natl = clean_phone(df_phone, "messy_phone", output_format="national")
    df_check_e164 = df_phone.copy()
    df_check_e164["messy_phone_clean"] = [
        "+15552345678",
        "+15552345678",
        "+15552345678",
        "+15552345678",
        "+15551234567",
        "+15552345678",
        "+12345678901 ext. 1234",
        "+12345678901 ext. 1234",
        "2345678",
        "+18002995865",
        "+18664947227",
        "+18004685865",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check_natl = df_phone.copy()
    df_check_natl["messy_phone_clean"] = [
        "(555) 234-5678",
        "(555) 234-5678",
        "(555) 234-5678",
        "(555) 234-5678",
        "(555) 123-4567",
        "(555) 234-5678",
        "(234) 567-8901 ext. 1234",
        "(234) 567-8901 ext. 1234",
        "234-5678",
        "(800) 299-5865",
        "(866) 494-7227",
        "(800) 468-5865",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    assert df_check_e164.equals(df_clean_e164)
    assert df_check_natl.equals(df_clean_natl)


def test_clean_split(df_phone: pd.DataFrame) -> None:
    df_clean = clean_phone(df_phone, "messy_phone", split=True)
    df_check = df_phone.copy()
    df_check["country_code"] = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        "1",
        "1",
        "1",
        np.nan,
        np.nan,
        np.nan,
        "1",
        "1",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["area_code"] = [
        "555",
        "555",
        "555",
        "555",
        "555",
        "555",
        "234",
        "234",
        np.nan,
        "800",
        "866",
        "800",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["office_code"] = [
        "234",
        "234",
        "234",
        "234",
        "123",
        "234",
        "567",
        "567",
        "234",
        "299",
        "494",
        "468",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["station_code"] = [
        "5678",
        "5678",
        "5678",
        "5678",
        "4567",
        "5678",
        "8901",
        "8901",
        "5678",
        "5865",
        "7227",
        "5865",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["ext_num"] = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        "1234",
        "1234",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    assert df_check.equals(df_clean)


def test_clean_split_fix_missing(df_phone: pd.DataFrame) -> None:
    df_clean = clean_phone(df_phone, "messy_phone", split=True, fix_missing="auto")
    df_check = df_phone.copy()
    df_check["country_code"] = [
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        np.nan,
        "1",
        "1",
        "1",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["area_code"] = [
        "555",
        "555",
        "555",
        "555",
        "555",
        "555",
        "234",
        "234",
        np.nan,
        "800",
        "866",
        "800",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["office_code"] = [
        "234",
        "234",
        "234",
        "234",
        "123",
        "234",
        "567",
        "567",
        "234",
        "299",
        "494",
        "468",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["station_code"] = [
        "5678",
        "5678",
        "5678",
        "5678",
        "4567",
        "5678",
        "8901",
        "8901",
        "5678",
        "5865",
        "7227",
        "5865",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["ext_num"] = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        "1234",
        "1234",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    assert df_check.equals(df_clean)


def test_clean_inplace(df_phone: pd.DataFrame) -> None:
    df_clean = clean_phone(df_phone, "messy_phone", inplace=True)
    df_check = pd.DataFrame(
        {
            "messy_phone_clean": [
                "555-234-5678",
                "555-234-5678",
                "555-234-5678",
                "555-234-5678",
                "555-123-4567",
                "555-234-5678",
                "234-567-8901 ext. 1234",
                "234-567-8901 ext. 1234",
                "234-5678",
                "800-299-5865",
                "866-494-7227",
                "800-468-5865",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ]
        }
    )
    assert df_check.equals(df_clean)


def test_clean_split_inplace(df_phone: pd.DataFrame) -> None:
    df_clean = clean_phone(df_phone, "messy_phone", split=True, inplace=True)
    df_check = pd.DataFrame(
        {
            "country_code": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                "1",
                "1",
                "1",
                np.nan,
                np.nan,
                np.nan,
                "1",
                "1",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "area_code": [
                "555",
                "555",
                "555",
                "555",
                "555",
                "555",
                "234",
                "234",
                np.nan,
                "800",
                "866",
                "800",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "office_code": [
                "234",
                "234",
                "234",
                "234",
                "123",
                "234",
                "567",
                "567",
                "234",
                "299",
                "494",
                "468",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "station_code": [
                "5678",
                "5678",
                "5678",
                "5678",
                "4567",
                "5678",
                "8901",
                "8901",
                "5678",
                "5865",
                "7227",
                "5865",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "ext_num": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                "1234",
                "1234",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )
    assert df_check.equals(df_clean)


def test_clean_split_inplace_fix_missing(df_phone: pd.DataFrame) -> None:
    df_clean = clean_phone(df_phone, "messy_phone", split=True, inplace=True, fix_missing="auto")
    df_check = pd.DataFrame(
        {
            "country_code": [
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                "1",
                np.nan,
                "1",
                "1",
                "1",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "area_code": [
                "555",
                "555",
                "555",
                "555",
                "555",
                "555",
                "234",
                "234",
                np.nan,
                "800",
                "866",
                "800",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "office_code": [
                "234",
                "234",
                "234",
                "234",
                "123",
                "234",
                "567",
                "567",
                "234",
                "299",
                "494",
                "468",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "station_code": [
                "5678",
                "5678",
                "5678",
                "5678",
                "4567",
                "5678",
                "8901",
                "8901",
                "5678",
                "5865",
                "7227",
                "5865",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
            "ext_num": [
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                "1234",
                "1234",
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )
    assert df_check.equals(df_clean)


def test_validate_value() -> None:
    assert validate_phone(1234) == False
    assert validate_phone(2346789) == True
    assert validate_phone("1 800 234 6789") == True
    assert validate_phone("+44 7700 900077") == True
    assert validate_phone("555-234-6789 ext 32") == True
    assert validate_phone("1-866-4ZIPCAR") == True
    assert validate_phone("1-800-G-O-T-J-U-N-K") == True
    assert validate_phone("123 ABC COMPANY") == False


def test_validate_series(df_phone: pd.DataFrame) -> None:
    srs_valid = validate_phone(df_phone["messy_phone"])
    srs_check = pd.Series(
        [
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ],
        name="messy_phone",
    )
    assert srs_check.equals(srs_valid)


def test_npartition_type() -> None:
    """
    related to #901
    """
    df = pd.DataFrame(
        {
            "phone": [
                "555-234-5678",
                "(555) 234-5678",
                "555.234.5678",
                "555/234/5678",
                15551234567,
                "(1) 555-234-5678",
                "+1 (234) 567-8901 x. 1234",
                "2345678901 extension 1234",
                "2345678",
                "800-299-JUNK",
                "1-866-4ZIPCAR",
                "123 ABC COMPANY",
                "+66 91 889 8948",
                "hello",
                np.nan,
                "NULL",
            ]
        }
    )
    clean_phone(df, "phone")
