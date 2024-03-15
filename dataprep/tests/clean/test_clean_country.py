"""
module for testing the functions clean_country() and validate_country()
"""

import logging

import numpy as np
import pandas as pd
import pytest

from ...clean import clean_country, validate_country

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")  # type: ignore
def df_countries() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "messy_country": [
                "Canada",
                "foo canada bar",
                "cnada",
                "northern ireland",
                " ireland ",
                "congo, kinshasa",
                "congo, brazzaville",
                304,
                " 233 ",
                887,
                "AS",
                " tr ",
                "bZ",
                "ARG",
                " bvt ",
                "nzl",
                "hello",
                np.nan,
                "NULL",
            ]
        }
    )
    return df


@pytest.fixture(scope="module")  # type: ignore
def df_typo_countries() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "messy_country": [
                "canada",
                "cnada",
                "australa",
                "xntarctica",
                "koreea",
                "cxnda",
                "afghnitan",
                "country: cnada",
                "foo indnesia bar",
                "congo, kishasa",
                "hello",
                np.nan,
                "NULL",
            ]
        }
    )
    return df


def test_clean_default(df_countries: pd.DataFrame) -> None:
    df_clean = clean_country(df_countries, "messy_country")
    df_check = df_countries.copy()
    df_check["messy_country_clean"] = [
        "Canada",
        "Canada",
        np.nan,
        np.nan,
        "Ireland",
        "DR Congo",
        "Congo Republic",
        "Greenland",
        "Estonia",
        "Yemen",
        "American Samoa",
        "Turkey",
        "Belize",
        "Argentina",
        "Bouvet Island",
        "New Zealand",
        np.nan,
        np.nan,
        np.nan,
    ]
    assert df_check.equals(df_clean)


def test_clean_input_format(df_countries: pd.DataFrame) -> None:
    df_clean_name = clean_country(df_countries, "messy_country", input_format="name")
    df_clean_official = clean_country(df_countries, "messy_country", input_format="official")
    df_clean_alpha2 = clean_country(df_countries, "messy_country", input_format="alpha-2")
    df_clean_alpha3 = clean_country(df_countries, "messy_country", input_format="alpha-3")
    df_clean_numeric = clean_country(df_countries, "messy_country", input_format="numeric")

    df_check_name_and_official = df_countries.copy()
    df_check_name_and_official["messy_country_clean"] = [
        "Canada",
        "Canada",
        np.nan,
        np.nan,
        "Ireland",
        "DR Congo",
        "Congo Republic",
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
        np.nan,
        np.nan,
    ]
    df_check_alpha2 = df_countries.copy()
    df_check_alpha2["messy_country_clean"] = [
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
        "American Samoa",
        "Turkey",
        "Belize",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check_alpha3 = df_countries.copy()
    df_check_alpha3["messy_country_clean"] = [
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
        np.nan,
        np.nan,
        np.nan,
        "Argentina",
        "Bouvet Island",
        "New Zealand",
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check_numeric = df_countries.copy()
    df_check_numeric["messy_country_clean"] = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        "Greenland",
        "Estonia",
        "Yemen",
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

    assert df_clean_name.equals(df_check_name_and_official)
    assert df_clean_official.equals(df_check_name_and_official)
    assert df_clean_alpha2.equals(df_check_alpha2)
    assert df_clean_alpha3.equals(df_check_alpha3)
    assert df_clean_numeric.equals(df_check_numeric)


def test_clean_input_format_tuple(df_countries: pd.DataFrame) -> None:
    df_clean = clean_country(df_countries, "messy_country", input_format=("name", "alpha-2"))
    df_check = df_countries.copy()
    df_check["messy_country_clean"] = [
        "Canada",
        "Canada",
        np.nan,
        np.nan,
        "Ireland",
        "DR Congo",
        "Congo Republic",
        np.nan,
        np.nan,
        np.nan,
        "American Samoa",
        "Turkey",
        "Belize",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]

    assert df_clean.equals(df_check)


def test_clean_output_format(df_countries: pd.DataFrame) -> None:
    df_clean_official = clean_country(df_countries, "messy_country", output_format="official")
    df_clean_alpha2 = clean_country(df_countries, "messy_country", output_format="alpha-2")
    df_clean_alpha3 = clean_country(df_countries, "messy_country", output_format="alpha-3")
    df_clean_numeric = clean_country(df_countries, "messy_country", output_format="numeric")

    df_check_official = df_countries.copy()
    df_check_official["messy_country_clean"] = [
        "Canada",
        "Canada",
        np.nan,
        np.nan,
        "Ireland",
        "Democratic Republic of the Congo",
        "Republic of the Congo",
        "Greenland",
        "Republic of Estonia",
        "Republic of Yemen",
        "American Samoa",
        "Republic of Turkey",
        "Belize",
        "Argentine Republic",
        "Bouvet Island",
        "New Zealand",
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check_alpha2 = df_countries.copy()
    df_check_alpha2["messy_country_clean"] = [
        "CA",
        "CA",
        np.nan,
        np.nan,
        "IE",
        "CD",
        "CG",
        "GL",
        "EE",
        "YE",
        "AS",
        "TR",
        "BZ",
        "AR",
        "BV",
        "NZ",
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check_alpha3 = df_countries.copy()
    df_check_alpha3["messy_country_clean"] = [
        "CAN",
        "CAN",
        np.nan,
        np.nan,
        "IRL",
        "COD",
        "COG",
        "GRL",
        "EST",
        "YEM",
        "ASM",
        "TUR",
        "BLZ",
        "ARG",
        "BVT",
        "NZL",
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check_numeric = df_countries.copy()
    df_check_numeric["messy_country_clean"] = [
        "124",
        "124",
        np.nan,
        np.nan,
        "372",
        "180",
        "178",
        "304",
        "233",
        "887",
        "16",
        "792",
        "84",
        "32",
        "74",
        "554",
        np.nan,
        np.nan,
        np.nan,
    ]

    assert df_clean_official.equals(df_check_official)
    assert df_clean_alpha2.equals(df_check_alpha2)
    assert df_clean_alpha3.equals(df_check_alpha3)
    assert df_clean_numeric.equals(df_check_numeric)


def test_input_format_output_format(df_countries: pd.DataFrame) -> None:
    df_clean = clean_country(
        df_countries, "messy_country", input_format="alpha-2", output_format="numeric"
    )
    df_check = df_countries.copy()
    df_check["messy_country_clean"] = [
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
        "16",
        "792",
        "84",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    assert df_clean.equals(df_check)


def test_clean_strict(df_countries: pd.DataFrame) -> None:
    df_clean = clean_country(df_countries, "messy_country", strict=True)
    df_check = df_countries.copy()
    df_check["messy_country_clean"] = [
        "Canada",
        np.nan,
        np.nan,
        np.nan,
        "Ireland",
        np.nan,
        np.nan,
        "Greenland",
        "Estonia",
        "Yemen",
        "American Samoa",
        "Turkey",
        "Belize",
        "Argentina",
        "Bouvet Island",
        "New Zealand",
        np.nan,
        np.nan,
        np.nan,
    ]
    assert df_clean.equals(df_check)


def test_clean_output_format_strict(df_countries: pd.DataFrame) -> None:
    df_clean = clean_country(df_countries, "messy_country", output_format="alpha-2", strict=True)
    df_check = df_countries.copy()
    df_check["messy_country_clean"] = [
        "CA",
        np.nan,
        np.nan,
        np.nan,
        "IE",
        np.nan,
        np.nan,
        "GL",
        "EE",
        "YE",
        "AS",
        "TR",
        "BZ",
        "AR",
        "BV",
        "NZ",
        np.nan,
        np.nan,
        np.nan,
    ]
    assert df_clean.equals(df_check)


def test_clean_input_format_strict(df_countries: pd.DataFrame) -> None:
    df_clean = clean_country(df_countries, "messy_country", input_format="official", strict=True)
    df_check = df_countries.copy()
    df_check["messy_country_clean"] = [
        "Canada",
        np.nan,
        np.nan,
        np.nan,
        "Ireland",
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
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    assert df_clean.equals(df_check)


def test_clean_fuzzy_dist(df_typo_countries: pd.DataFrame) -> None:
    df_clean_dist1 = clean_country(df_typo_countries, "messy_country", fuzzy_dist=1)
    df_clean_dist2 = clean_country(df_typo_countries, "messy_country", fuzzy_dist=2)

    df_check_dist1 = df_typo_countries.copy()
    df_check_dist1["messy_country_clean"] = [
        "Canada",
        "Canada",
        "Australia",
        "Antarctica",
        "South Korea",
        np.nan,
        np.nan,
        "Canada",
        "Indonesia",
        "DR Congo",
        np.nan,
        np.nan,
        np.nan,
    ]

    df_check_dist2 = df_typo_countries.copy()
    df_check_dist2["messy_country_clean"] = [
        "Canada",
        "Canada",
        "Australia",
        "Antarctica",
        "South Korea",
        "Canada",
        "Afghanistan",
        "Canada",
        "Indonesia",
        "DR Congo",
        "Greece",
        np.nan,
        np.nan,
    ]

    assert df_clean_dist1.equals(df_check_dist1)
    assert df_clean_dist2.equals(df_check_dist2)


def test_clean_fuzzy_dist_input_format(df_countries: pd.DataFrame) -> None:
    df_clean_name = clean_country(df_countries, "messy_country", input_format="name", fuzzy_dist=1)
    df_clean_alpha3 = clean_country(
        df_countries, "messy_country", input_format="alpha-3", fuzzy_dist=1
    )

    df_check_name = df_countries.copy()
    df_check_name["messy_country_clean"] = [
        "Canada",
        "Canada",
        "Canada",
        "Iceland",
        "Ireland",
        "DR Congo",
        "Congo Republic",
        np.nan,
        np.nan,
        np.nan,
        "United States",
        np.nan,
        np.nan,
        np.nan,
        "British Virgin Islands",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check_alpha3 = df_countries.copy()
    df_check_alpha3["messy_country_clean"] = [
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
        np.nan,
        np.nan,
        np.nan,
        "Argentina",
        "Bouvet Island",
        "New Zealand",
        np.nan,
        np.nan,
        np.nan,
    ]

    assert df_clean_name.equals(df_check_name)
    assert df_clean_alpha3.equals(df_check_alpha3)


def test_clean_fuzzy_dist_output_format(df_typo_countries: pd.DataFrame) -> None:
    df_clean_dist1 = clean_country(
        df_typo_countries, "messy_country", output_format="alpha-2", fuzzy_dist=1
    )
    df_clean_dist2 = clean_country(
        df_typo_countries, "messy_country", output_format="alpha-2", fuzzy_dist=2
    )

    df_check_dist1 = df_typo_countries.copy()
    df_check_dist1["messy_country_clean"] = [
        "CA",
        "CA",
        "AU",
        "AQ",
        "KR",
        np.nan,
        np.nan,
        "CA",
        "ID",
        "CD",
        np.nan,
        np.nan,
        np.nan,
    ]

    df_check_dist2 = df_typo_countries.copy()
    df_check_dist2["messy_country_clean"] = [
        "CA",
        "CA",
        "AU",
        "AQ",
        "KR",
        "CA",
        "AF",
        "CA",
        "ID",
        "CD",
        "GR",
        np.nan,
        np.nan,
    ]

    assert df_clean_dist1.equals(df_check_dist1)
    assert df_clean_dist2.equals(df_check_dist2)


def test_clean_inplace(df_countries: pd.DataFrame) -> None:
    df_clean = clean_country(df_countries, "messy_country", inplace=True)
    df_check = pd.DataFrame(
        {
            "messy_country_clean": [
                "Canada",
                "Canada",
                np.nan,
                np.nan,
                "Ireland",
                "DR Congo",
                "Congo Republic",
                "Greenland",
                "Estonia",
                "Yemen",
                "American Samoa",
                "Turkey",
                "Belize",
                "Argentina",
                "Bouvet Island",
                "New Zealand",
                np.nan,
                np.nan,
                np.nan,
            ]
        }
    )
    assert df_check.equals(df_clean)


def test_validate_value() -> None:
    assert validate_country("switzerland") == True
    assert validate_country("foo united states bar") == False
    assert validate_country("cnada") == False
    assert validate_country("foo united states bar", strict=False) == True
    assert validate_country("ca") == True
    assert validate_country(" CAN ") == True
    assert validate_country(800) == True
    assert validate_country(" 800 ") == True


def test_validate_value_input_format() -> None:
    assert validate_country("switzerland", input_format="name") == True
    assert validate_country("switzerland", input_format="alpha-2") == False
    assert validate_country("HK", input_format="alpha-2") == True
    assert validate_country(" mkd ", input_format="alpha-3") == True
    assert validate_country(800, input_format="numeric") == True
    assert validate_country(800, input_format="alpha-3") == False


def test_validate_series(df_countries: pd.DataFrame) -> None:
    srs_valid = validate_country(df_countries["messy_country"])
    srs_check = pd.Series(
        [
            True,
            False,
            False,
            False,
            True,
            False,
            False,
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
        ],
        name="messy_country",
    )
    assert srs_check.equals(srs_valid)


def test_validate_series_input_format(df_countries: pd.DataFrame) -> None:
    srs_valid_name = validate_country(df_countries["messy_country"], input_format="name")
    srs_valid_alpha2 = validate_country(df_countries["messy_country"], input_format="alpha-2")
    srs_check_name = pd.Series(
        [
            True,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        name="messy_country",
    )
    srs_check_alpha2 = pd.Series(
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        name="messy_country",
    )
    assert srs_check_name.equals(srs_valid_name)
    assert srs_check_alpha2.equals(srs_valid_alpha2)


def test_validate_series_input_format_tuple(df_countries: pd.DataFrame) -> None:
    srs_valid = validate_country(df_countries["messy_country"], input_format=("name", "alpha-2"))
    srs_check = pd.Series(
        [
            True,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        name="messy_country",
    )
    assert srs_check.equals(srs_valid)


def test_validate_series_strict(df_countries: pd.DataFrame) -> None:
    srs_valid = validate_country(df_countries["messy_country"], strict=False)
    srs_check = pd.Series(
        [
            True,
            True,
            False,
            False,
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
        ],
        name="messy_country",
    )
    assert srs_check.equals(srs_valid)


def test_validate_series_strict_input_format(df_countries: pd.DataFrame) -> None:
    srs_valid_name = validate_country(
        df_countries["messy_country"], input_format="name", strict=False
    )
    srs_valid_alpha2 = validate_country(
        df_countries["messy_country"], input_format="alpha-2", strict=False
    )
    srs_check_name = pd.Series(
        [
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        name="messy_country",
    )
    srs_check_alpha2 = pd.Series(
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        name="messy_country",
    )
    assert srs_check_name.equals(srs_valid_name)
    assert srs_check_alpha2.equals(srs_valid_alpha2)
