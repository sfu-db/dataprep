"""
module for testing the functions clean_lat_long() and validate_lat_long()
"""

import logging

import numpy as np
import pandas as pd
import pytest

from ...clean import clean_lat_long, validate_lat_long

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")  # type: ignore
def df_lat_long_column() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "messy_lat_long": [
                (41.5, -81.0),
                "41.5;-81.0",
                "41.5,-81.0",
                "41.5 -81.0",
                "41.5° N, 81.0° W",
                "41.5 S;81.0 E",
                "-41.5 S;81.0 E",
                "23 26m 22s N 23 27m 30s E",
                "23 26' 22\" N 23 27' 30\" E",
                "UT: N 39°20' 0'' / W 74°35' 0''",
                "hello",
                np.nan,
                "NULL",
            ]
        }
    )
    return df


@pytest.fixture(scope="module")  # type: ignore
def df_separate_lat_long_columns() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "messy_lat": ["30° E", "41° 30′ N", "41 S", "80", "hello", "NA"],
            "messy_long": ["30° E", "41° 30′ N", "41 W", "80", "hello", "NA"],
        }
    )
    return df


def test_clean_default(df_lat_long_column: pd.DataFrame) -> None:
    df_clean = clean_lat_long(df_lat_long_column, "messy_lat_long")
    df_check = df_lat_long_column.copy()
    df_check["messy_lat_long_clean"] = [
        (41.5, -81.0),
        (41.5, -81.0),
        (41.5, -81.0),
        (41.5, -81.0),
        (41.5, -81.0),
        (-41.5, 81.0),
        np.nan,
        (23.4394, 23.4583),
        (23.4394, 23.4583),
        (39.3333, -74.5833),
        np.nan,
        np.nan,
        np.nan,
    ]
    assert df_check.equals(df_clean)


def test_clean_output_format(df_lat_long_column: pd.DataFrame) -> None:
    df_clean_ddh = clean_lat_long(df_lat_long_column, "messy_lat_long", output_format="ddh")
    df_clean_dms = clean_lat_long(df_lat_long_column, "messy_lat_long", output_format="dms")
    df_check_ddh = df_lat_long_column.copy()
    df_check_ddh["messy_lat_long_clean"] = [
        "41.5° N, 81.0° W",
        "41.5° N, 81.0° W",
        "41.5° N, 81.0° W",
        "41.5° N, 81.0° W",
        "41.5° N, 81.0° W",
        "41.5° S, 81.0° E",
        np.nan,
        "23.4394° N, 23.4583° E",
        "23.4394° N, 23.4583° E",
        "39.3333° N, 74.5833° W",
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check_dms = df_lat_long_column.copy()
    df_check_dms["messy_lat_long_clean"] = [
        "41° 30′ 0″ N, 81° 0′ 0″ W",
        "41° 30′ 0″ N, 81° 0′ 0″ W",
        "41° 30′ 0″ N, 81° 0′ 0″ W",
        "41° 30′ 0″ N, 81° 0′ 0″ W",
        "41° 30′ 0″ N, 81° 0′ 0″ W",
        "41° 30′ 0″ S, 81° 0′ 0″ E",
        np.nan,
        "23° 26′ 22″ N, 23° 27′ 30″ E",
        "23° 26′ 22″ N, 23° 27′ 30″ E",
        "39° 20′ 0″ N, 74° 34′ 60″ W",
        np.nan,
        np.nan,
        np.nan,
    ]
    assert df_check_ddh.equals(df_clean_ddh)
    assert df_check_dms.equals(df_clean_dms)


def test_clean_split(df_lat_long_column: pd.DataFrame) -> None:
    df_clean = clean_lat_long(df_lat_long_column, "messy_lat_long", split=True)
    df_check = df_lat_long_column.copy()
    df_check["latitude"] = [
        41.5,
        41.5,
        41.5,
        41.5,
        41.5,
        -41.5,
        np.nan,
        23.4394,
        23.4394,
        39.3333,
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["longitude"] = [
        -81.0,
        -81.0,
        -81.0,
        -81.0,
        -81.0,
        81.0,
        np.nan,
        23.4583,
        23.4583,
        -74.5833,
        np.nan,
        np.nan,
        np.nan,
    ]
    assert df_check.equals(df_clean)


def test_clean_output_format_split(df_lat_long_column: pd.DataFrame) -> None:
    df_clean = clean_lat_long(df_lat_long_column, "messy_lat_long", output_format="dm", split=True)
    df_check = df_lat_long_column.copy()
    df_check["latitude"] = [
        "41° 30′ N",
        "41° 30′ N",
        "41° 30′ N",
        "41° 30′ N",
        "41° 30′ N",
        "41° 30′ S",
        np.nan,
        "23° 26.3667′ N",
        "23° 26.3667′ N",
        "39° 20′ N",
        np.nan,
        np.nan,
        np.nan,
    ]
    df_check["longitude"] = [
        "81° 0′ W",
        "81° 0′ W",
        "81° 0′ W",
        "81° 0′ W",
        "81° 0′ W",
        "81° 0′ E",
        np.nan,
        "23° 27.5′ E",
        "23° 27.5′ E",
        "74° 35′ W",
        np.nan,
        np.nan,
        np.nan,
    ]
    assert df_check.equals(df_clean)


def test_clean_inplace(df_lat_long_column: pd.DataFrame) -> None:
    df_clean = clean_lat_long(df_lat_long_column, "messy_lat_long", inplace=True)
    df_check = pd.DataFrame(
        {
            "messy_lat_long_clean": [
                (41.5, -81.0),
                (41.5, -81.0),
                (41.5, -81.0),
                (41.5, -81.0),
                (41.5, -81.0),
                (-41.5, 81.0),
                np.nan,
                (23.4394, 23.4583),
                (23.4394, 23.4583),
                (39.3333, -74.5833),
                np.nan,
                np.nan,
                np.nan,
            ]
        }
    )
    assert df_check.equals(df_clean)


def test_clean_split_inplace(df_lat_long_column: pd.DataFrame) -> None:
    df_clean = clean_lat_long(df_lat_long_column, "messy_lat_long", split=True, inplace=True)
    df_check = pd.DataFrame(
        {
            "latitude": [
                41.5,
                41.5,
                41.5,
                41.5,
                41.5,
                -41.5,
                np.nan,
                23.4394,
                23.4394,
                39.3333,
                np.nan,
                np.nan,
                np.nan,
            ],
            "longitude": [
                -81.0,
                -81.0,
                -81.0,
                -81.0,
                -81.0,
                81.0,
                np.nan,
                23.4583,
                23.4583,
                -74.5833,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )
    assert df_check.equals(df_clean)


def test_clean_lat_long_separate_columns_split(df_separate_lat_long_columns: pd.DataFrame) -> None:
    df_clean = clean_lat_long(
        df_separate_lat_long_columns, lat_col="messy_lat", long_col="messy_long", split=True
    )
    df_check = df_separate_lat_long_columns.copy()
    df_check["messy_lat_clean"] = [np.nan, 41.5, -41.0, 80.0, np.nan, np.nan]
    df_check["messy_long_clean"] = [30.0, np.nan, -41.0, 80.0, np.nan, np.nan]
    assert df_check.equals(df_clean)


def test_validate_value() -> None:
    assert validate_lat_long("41° 30′ 0″ N") == False
    assert validate_lat_long("41.5 S;81.0 E") == True
    assert validate_lat_long("-41.5 S;81.0 E") == False
    assert validate_lat_long((41.5, 81)) == True
    assert validate_lat_long(41.5, lat_long=False, lat=True) == True


def test_validate_series_lat_long(df_lat_long_column: pd.DataFrame) -> None:
    srs_valid = validate_lat_long(df_lat_long_column["messy_lat_long"])
    srs_check = pd.Series(
        [
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
            False,
            False,
            False,
        ],
        name="messy_lat_long",
    )
    assert srs_check.equals(srs_valid)


def test_validate_series_lat(df_separate_lat_long_columns: pd.DataFrame) -> None:
    srs_valid = validate_lat_long(
        df_separate_lat_long_columns["messy_lat"], lat_long=False, lat=True
    )
    srs_check = pd.Series(
        [
            False,
            True,
            True,
            True,
            False,
            False,
        ],
        name="messy_lat_clean",
    )
    assert srs_check.equals(srs_valid)
