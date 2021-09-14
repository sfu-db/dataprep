"""
module for testing the functions `clean_language()` and `validate_language()`.
"""

import logging

from os import path

import numpy as np
import pandas as pd
import pytest

from ...clean import clean_language, validate_language

LOGGER = logging.getLogger(__name__)

ALTERNATIVE_LANGUAGE_DATA_FILE = path.join(
    path.split(path.abspath(__file__))[0], "test_language_data.csv"
)


@pytest.fixture(scope="module")  # type: ignore
def df_languages() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "messy_language": [
                "eng",
                "zh",
                "Japanese",
                "english",
                "Zh",
                "tp",
                "233",
                304,
                "dd eng",
                " tr ",
                "hello",
                np.nan,
                "NULL",
            ]
        }
    )
    return df


@pytest.fixture(scope="module")  # type: ignore
def df_multicols_languages() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "some_messy_language": [
                "eng",
                "zh",
                "Japanese",
                "english",
                "Zh",
                "tp",
            ],
            "other_messy_language": [
                "233",
                304,
                " tr ",
                "hello",
                np.nan,
                "NULL",
            ],
        }
    )
    return df


def test_clean_default(df_languages: pd.DataFrame) -> None:
    df_clean = clean_language(df_languages, "messy_language")
    df_check = df_languages.copy()
    df_check["messy_language_clean"] = [
        "English",
        "Chinese",
        "Japanese",
        "English",
        "Chinese",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        "Turkish",
        np.nan,
        np.nan,
        np.nan,
    ]

    assert df_check.equals(df_clean)


def test_clean_input_formats(df_languages: pd.DataFrame) -> None:
    df_clean_name = clean_language(df_languages, "messy_language", input_format="name")
    df_clean_alpha2 = clean_language(df_languages, "messy_language", input_format="alpha-2")
    df_clean_alpha3 = clean_language(df_languages, "messy_language", input_format="alpha-3")

    df_check_name = df_languages.copy()
    df_check_name["messy_language_clean"] = [
        np.nan,
        np.nan,
        "Japanese",
        "English",
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
    df_check_alpha2 = df_languages.copy()
    df_check_alpha2["messy_language_clean"] = [
        np.nan,
        "Chinese",
        np.nan,
        np.nan,
        "Chinese",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        "Turkish",
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check_alpha3 = df_languages.copy()
    df_check_alpha3["messy_language_clean"] = [
        "English",
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

    assert df_clean_name.equals(df_check_name)
    assert df_clean_alpha2.equals(df_check_alpha2)
    assert df_clean_alpha3.equals(df_check_alpha3)


def test_clean_input_format_tuple(df_languages: pd.DataFrame) -> None:
    df_clean = clean_language(df_languages, "messy_language", input_format=("name", "alpha-3"))
    df_check = df_languages.copy()
    df_check["messy_language_clean"] = [
        "English",
        np.nan,
        "Japanese",
        "English",
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


def test_clean_output_format(df_languages: pd.DataFrame) -> None:
    df_clean_name = clean_language(df_languages, "messy_language", output_format="name")
    df_clean_alpha2 = clean_language(df_languages, "messy_language", output_format="alpha-2")
    df_clean_alpha3 = clean_language(df_languages, "messy_language", output_format="alpha-3")

    df_check_name = df_languages.copy()
    df_check_name["messy_language_clean"] = [
        "English",
        "Chinese",
        "Japanese",
        "English",
        "Chinese",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        "Turkish",
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check_alpha2 = df_languages.copy()
    df_check_alpha2["messy_language_clean"] = [
        "en",
        "zh",
        "ja",
        "en",
        "zh",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        "tr",
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check_alpha3 = df_languages.copy()
    df_check_alpha3["messy_language_clean"] = [
        "eng",
        "zho",
        "jpn",
        "eng",
        "zho",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        "tur",
        np.nan,
        np.nan,
        np.nan,
    ]

    assert df_clean_name.equals(df_check_name)
    assert df_clean_alpha2.equals(df_check_alpha2)
    assert df_clean_alpha3.equals(df_check_alpha3)


def test_clean_kb(df_languages: pd.DataFrame) -> None:
    df_clean = clean_language(
        df_languages, "messy_language", kb_path=ALTERNATIVE_LANGUAGE_DATA_FILE
    )
    df_check = df_languages.copy()
    df_check["messy_language_clean"] = [
        "English",
        "Chinese",
        "Japanese",
        "English",
        "Chinese",
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


def test_validate_value() -> None:
    assert validate_language("english") == True
    assert validate_language("zh") == True
    assert validate_language(" ZH ") == True
    assert validate_language("tp") == False
    assert validate_language("eng") == True
    assert validate_language("hello") == False
    assert validate_language("233") == False
    assert validate_language("dd eng") == False
    assert validate_language("") == False


def test_validate_series(df_languages: pd.DataFrame) -> None:
    srs_valid = validate_language(df_languages["messy_language"])
    srs_check = pd.Series(
        [
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
        ],
        name="messy_language",
    )
    assert srs_check.equals(srs_valid)


def test_validate_input_format(df_languages: pd.DataFrame) -> None:
    srs_valid = validate_language(df_languages["messy_language"], input_format="alpha-2")
    srs_check = pd.Series(
        [
            False,
            True,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
        ],
        name="messy_language",
    )
    assert srs_check.equals(srs_valid)


def test_validate_dataframe_col(df_multicols_languages: pd.DataFrame) -> None:
    srs_valid = validate_language(df_multicols_languages, "some_messy_language")
    srs_check = pd.Series(
        [
            True,
            True,
            True,
            True,
            True,
            False,
        ],
        name="some_messy_language",
    )
    assert srs_check.equals(srs_valid)


def test_validate_dataframe_all(df_multicols_languages: pd.DataFrame) -> None:
    df_valid = validate_language(df_multicols_languages)
    df_check = pd.DataFrame()

    df_check["some_messy_language"] = [
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    df_check["other_messy_language"] = [
        False,
        False,
        True,
        False,
        False,
        False,
    ]

    assert df_check.equals(df_valid)


def test_validate_kb(df_languages: pd.DataFrame) -> None:
    srs_valid = validate_language(
        df_languages["messy_language"], kb_path=ALTERNATIVE_LANGUAGE_DATA_FILE
    )
    srs_check = pd.Series(
        [
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
            False,
            False,
            False,
        ],
        name="messy_language",
    )
    assert srs_check.equals(srs_valid)
