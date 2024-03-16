"""
module for testing the functions clean_address() and validate_address()
"""

import logging

import numpy as np
import pandas as pd
import pytest

from ...clean import clean_address, validate_address

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")  # type: ignore
def df_addresses() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "messy_address": [
                "123 pine ave",
                "1234 w main heights 57033",
                "apt 1 s maple rd manhattan",
                "robie house, 789 north main street manhattan new york",
                "1111 S Figueroa St, Los Angeles, CA 90015, United States",
                "(staples center) 1111 S Figueroa St, Los Angeles",
                "S Figueroa, Los Angeles",
                "hello",
                np.nan,
                "NULL",
            ]
        }
    )
    return df


def test_clean_default(df_addresses: pd.DataFrame) -> None:
    df_clean = clean_address(df_addresses, "messy_address")
    df_check = df_addresses.copy()
    df_check["messy_address_clean"] = [
        "123 Pine Ave.",
        "1234 W. Main Hts., 57033",
        np.nan,
        "(Robie House) 789 N. Main St., Manhattan, NY",
        "1111 S. Figueroa St., Los Angeles, CA 90015",
        "(Staples Center) 1111 S. Figueroa St., Los Angeles",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    assert df_check.equals(df_clean)


def test_clean_output_format(df_addresses: pd.DataFrame) -> None:
    df_clean = clean_address(
        df_addresses, "messy_address", output_format="(zipcode) street_name ~~state_full~~"
    )
    df_check = df_addresses.copy()
    df_check["messy_address_clean"] = [
        "Pine",
        "(57033) Main",
        np.nan,
        "Main ~~New York~~",
        "(90015) Figueroa ~~California~~",
        "Figueroa",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    df_clean2 = clean_address(
        df_addresses,
        "messy_address",
        output_format="house_number street_prefix_full street_name street_suffix_full (building)",
    )
    df_check2 = df_addresses.copy()
    df_check2["messy_address_clean"] = [
        "123 Pine Avenue",
        "1234 West Main Heights",
        np.nan,
        "789 North Main Street (Robie House)",
        "1111 South Figueroa Street",
        "1111 South Figueroa Street (Staples Center)",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    assert df_check.equals(df_clean)
    assert df_check2.equals(df_clean2)


def test_clean_output_format_with_tabs(df_addresses: pd.DataFrame) -> None:
    df_clean = clean_address(
        df_addresses, "messy_address", output_format="house_number street_name \t state_full"
    )
    df_check = df_addresses.copy()
    df_check["house_number street_name"] = [
        "123 Pine",
        "1234 Main",
        np.nan,
        "789 Main",
        "1111 Figueroa",
        "1111 Figueroa",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    df_check["state_full"] = [
        np.nan,
        np.nan,
        np.nan,
        "New York",
        "California",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    assert df_check.equals(df_clean)


def test_clean_must_contain(df_addresses: pd.DataFrame) -> None:
    df_clean = clean_address(df_addresses, "messy_address", must_contain=("street_name", "city"))
    df_check = df_addresses.copy()
    df_check["messy_address_clean"] = [
        np.nan,
        np.nan,
        "S. Maple Rd., Apt 1, Manhattan",
        "(Robie House) 789 N. Main St., Manhattan, NY",
        "1111 S. Figueroa St., Los Angeles, CA 90015",
        "(Staples Center) 1111 S. Figueroa St., Los Angeles",
        "S. Figueroa, Los Angeles",
        np.nan,
        np.nan,
        np.nan,
    ]

    assert df_check.equals(df_clean)


def test_clean_split(df_addresses: pd.DataFrame) -> None:
    df_clean = clean_address(df_addresses, "messy_address", split=True)

    df_check = df_addresses.copy()
    df_check["building"] = [
        np.nan,
        np.nan,
        np.nan,
        "Robie House",
        np.nan,
        "Staples Center",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["house_number"] = [
        "123",
        "1234",
        np.nan,
        "789",
        "1111",
        "1111",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["street_prefix_abbr"] = [
        np.nan,
        "W.",
        np.nan,
        "N.",
        "S.",
        "S.",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["street_name"] = [
        "Pine",
        "Main",
        np.nan,
        "Main",
        "Figueroa",
        "Figueroa",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["street_suffix_abbr"] = [
        "Ave.",
        "Hts.",
        np.nan,
        "St.",
        "St.",
        "St.",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["apartment"] = [
        np.nan,
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
    df_check["city"] = [
        np.nan,
        np.nan,
        np.nan,
        "Manhattan",
        "Los Angeles",
        "Los Angeles",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["state_abbr"] = [
        np.nan,
        np.nan,
        np.nan,
        "NY",
        "CA",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["zipcode"] = [
        np.nan,
        "57033",
        np.nan,
        np.nan,
        "90015",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    assert df_check.equals(df_clean)


def test_validate_value() -> None:
    assert validate_address("123 evergreen dr") == True
    assert validate_address("robie house, 789 north main street manhattan new york") == True
    assert validate_address("123 evergreen dr", must_contain=("city",)) == False
    assert validate_address("los angeles, california 57703", must_contain=("city",)) == True
    assert validate_address("apt 1 s maple rd manhattan", must_contain=("apartment",)) == True
    assert (
        validate_address("apt 1 maple rd manhattan", must_contain=("apartment", "street_prefix"))
        == False
    )
    assert validate_address("hello") == False


def test_validate_series(df_addresses: pd.DataFrame) -> None:
    srs_valid = validate_address(df_addresses["messy_address"])

    srs_check = pd.Series(
        [
            True,
            True,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        ],
        name="messy_address",
    )

    assert srs_check.equals(srs_valid)
