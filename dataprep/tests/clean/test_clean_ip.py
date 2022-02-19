"""
module for testing the functions `clean_ip()` and `validate_ip()`.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from ...clean import clean_ip, validate_ip

LOGGER = logging.getLogger(__name__)


@pytest.fixture(scope="module")  # type: ignore
def df_ips() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "messy_ip": [
                "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
                "12.3.4.5",
                "233.5.6.0",
                None,
                {},
                2982384756,
                "fdf8:f53b:82e4::53",
            ]
        }
    )
    return df


# input_formats tests
def test_clean_input_ipv4(df_ips: pd.DataFrame) -> None:
    df_clean = clean_ip(df_ips, column="messy_ip", input_format="ipv4")
    df_check = df_ips.copy()
    df_check["messy_ip_clean"] = [
        np.nan,
        "12.3.4.5",
        "233.5.6.0",
        np.nan,
        np.nan,
        "177.195.148.116",
        np.nan,
    ]

    assert df_check.equals(df_clean)


def test_clean_input_ipv6(df_ips: pd.DataFrame) -> None:
    df_clean = clean_ip(df_ips, column="messy_ip", input_format="ipv6")
    df_check = df_ips.copy()
    df_check["messy_ip_clean"] = [
        "2001:db8:85a3::8a2e:370:7334",
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        "fdf8:f53b:82e4::53",
    ]
    assert df_check.equals(df_clean)


def test_clean_default(df_ips: pd.DataFrame) -> None:
    df_clean = clean_ip(df_ips, column="messy_ip")
    df_check = df_ips.copy()
    df_check["messy_ip_clean"] = [
        "2001:db8:85a3::8a2e:370:7334",
        "12.3.4.5",
        "233.5.6.0",
        np.nan,
        np.nan,
        "177.195.148.116",
        "fdf8:f53b:82e4::53",
    ]
    assert df_check.equals(df_clean)


# output_format tests
def test_clean_output_full(df_ips: pd.DataFrame) -> None:
    df_clean = clean_ip(df_ips, column="messy_ip", output_format="full")
    df_check = df_ips.copy()
    df_check["messy_ip_clean"] = [
        "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
        "0012.0003.0004.0005",
        "0233.0005.0006.0000",
        np.nan,
        np.nan,
        "0177.0195.0148.0116",
        "fdf8:f53b:82e4:0000:0000:0000:0000:0053",
    ]
    assert df_check.equals(df_clean)


def test_clean_output_hexa(df_ips: pd.DataFrame) -> None:
    df_clean = clean_ip(df_ips, column="messy_ip", output_format="hexa")
    df_check = df_ips.copy()
    df_check["messy_ip_clean"] = [
        "0x20010db885a3000000008a2e03707334",
        "0xc030405",
        "0xe9050600",
        np.nan,
        np.nan,
        "0xb1c39474",
        "0xfdf8f53b82e400000000000000000053",
    ]
    assert df_check.equals(df_clean)


def test_clean_output_binary(df_ips: pd.DataFrame) -> None:
    df_clean = clean_ip(df_ips, column="messy_ip", output_format="integer")
    df_check = df_ips.copy()
    df_check["messy_ip_clean"] = [
        42540766452641154071740215577757643572,
        201524229,
        3909420544,
        np.nan,
        np.nan,
        2982384756,
        337587346459823522461035528822286450771,
    ]
    assert df_check.equals(df_clean)


def test_clean_output_packed(df_ips: pd.DataFrame) -> None:
    df_clean = clean_ip(df_ips, column="messy_ip", output_format="packed")
    df_check = df_ips.copy()
    df_check["messy_ip_clean"] = [
        b" \x01\r\xb8\x85\xa3\x00\x00\x00\x00\x8a.\x03ps4",
        b"\x0c\x03\x04\x05",
        b"\xe9\x05\x06\x00",
        np.nan,
        np.nan,
        b"\xb1\xc3\x94t",
        b"\xfd\xf8\xf5;\x82\xe4\x00\x00\x00\x00\x00\x00\x00\x00\x00S",
    ]
    assert df_check.equals(df_clean)


def test_validate_value() -> None:
    assert validate_ip("2001:0db8:85a3:0000:0000:8a2e:0370:7334") == True
    assert validate_ip("") == False
    assert validate_ip("233.5.6.0") == True
    assert validate_ip(np.nan) == False
    assert validate_ip("873.234.1.0") == False


def test_validate_series(df_ips: pd.DataFrame) -> None:
    df_valid = validate_ip(df_ips["messy_ip"])
    df_check = pd.Series([True, True, True, False, False, True, True])
    assert df_check.equals(df_valid)
