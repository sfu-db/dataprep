"""
Clean and validate a DataFrame column containing IP addresses.
"""

from ipaddress import ip_address
from operator import itemgetter
from typing import Any, Union

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..progress_bar import ProgressBar
from .utils import NULL_VALUES, create_report_new, to_dask


def clean_ip(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    input_format: str = "auto",
    output_format: str = "compressed",
    inplace: bool = False,
    errors: str = "coerce",
    report: bool = True,
    progress: bool = True,
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Clean and standardize IP addresses.

    Read more in the :ref:`User Guide <ip_userguide>`.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing IP addresses.
    input_format
        The input format of the IP addresses.
            - 'auto': parse both ipv4 and ipv6 addresses.
            - 'ipv4': only parse ipv4 addresses.
            - 'ipv6': only parse ipv6 addresses.

        (default: 'auto')
    output_format
        The desired output format of the IP addresses.
            - 'compressed': compressed representation ('12.3.4.5')
            - 'full': full representation ('0012.0003.0004.0005')
            - 'binary': binary representation ('00001100000000110000010000000101')
            - 'hexa': hexadecimal representation ('0xc030405')
            - 'integer': integer representation (201524229)
            - 'packed': packed binary representation (big-endian, a bytes object)

        (default: 'compressed')
    inplace
        If True, delete the column containing the data that was cleaned. Otherwise,
        keep the original column.

        (default: False)
    errors
        How to handle parsing errors.
            - ‘coerce’: invalid parsing will be set to NaN.
            - ‘ignore’: invalid parsing will return the input.
            - ‘raise’: invalid parsing will raise an exception.

        (default: 'coerce')
    report
        If True, output the summary report. Otherwise, no report is outputted.

        (default: True)
    progress
        If True, display a progress bar.

        (default: True)

    Examples
    --------

    >>> df = pd.DataFrame({'ip': ['2001:0db8:85a3:0000:0000:8a2e:0370:7334', '233.5.6.000']})
    >>> clean_ip(df, 'ip')
    IP Cleaning Report:
        2 values cleaned (100.0%)
    Result contains 2 (100.0%) values in the correct format and 0 null values (0.0%)
                                            ip                      ip_clean
    0  2001:0db8:85a3:0000:0000:8a2e:0370:7334  2001:db8:85a3::8a2e:370:7334
    1                              233.5.6.000                     233.5.6.0
    """
    # pylint: disable=too-many-arguments

    # check if the parameters are of correct processing types and values
    if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
        raise ValueError("df is invalid, it needs to be a pandas or Dask DataFrame")

    if not isinstance(column, str):
        raise ValueError(f"column {column} is invalid")

    if input_format not in {"ipv4", "ipv6", "auto"}:
        raise ValueError(
            f'input_format {input_format} is invalid, it needs to be "ipv4", "ipv6" or "auto"'
        )

    if output_format not in {"compressed", "full", "binary", "hexa", "integer", "packed"}:
        raise ValueError(
            f'output_format {output_format} is invalid, it needs to be "compressed", "full", '
            '"binary", "hexa", "integer" or "packed"'
        )

    if not isinstance(inplace, bool):
        raise ValueError(f"inplace {inplace} is invalid, it needs to be True or False")

    if not isinstance(report, bool):
        raise ValueError(f"report {report} is invalid, it needs to be True or False")

    if errors not in {"coerce", "ignore", "raise"}:
        raise ValueError(f'errors {errors} is invalid, it needs to be "coerce", "ignore", "raise"')

    # convert to dask
    df = to_dask(df)

    # To clean, create a new column "clean_code_tup" which contains
    # the cleaned values and code indicating how the initial value was
    # changed in a tuple. Then split the column of tuples and count the
    # amount of different codes to produce the report
    df["clean_code_tup"] = df[column].map_partitions(
        lambda srs: [_format_ip(x, input_format, output_format, errors) for x in srs],
        meta=object,
    )
    df = df.assign(
        _temp_=df["clean_code_tup"].map(itemgetter(0)),
        _code_=df["clean_code_tup"].map(itemgetter(1)),
    )
    df = df.rename(columns={"_temp_": f"{column}_clean"})

    # counts of codes indicating how values were changed
    stats = df["_code_"].value_counts(sort=False)
    df = df.drop(columns=["clean_code_tup", "_code_"])

    if inplace:
        df = df.drop(columns=column)

    with ProgressBar(minimum=1, disable=not progress):
        df, stats = dask.compute(df, stats)

    # output a report describing the result of clean_ip
    if report:
        create_report_new("IP", stats, errors)

    return df


def validate_ip(x: Union[str, pd.Series], input_format: str = "auto") -> Union[bool, pd.Series]:
    """
    Validate IP addresses.

    Read more in the :ref:`User Guide <ip_userguide>`.

    Parameters
    ----------
    x
        pandas Series of IP addresses or a str ip address value
    input_format
        The IP address format to validate.
            - 'auto': validate both ipv4 and ipv6 addresses.
            - 'ipv4': only validate ipv4 addresses.
            - 'ipv6': only validate ipv6 addresses.

        (default: 'auto')

    Examples
    --------

    >>> validate_ip('fdf8:f53b:82e4::53')
    True
    >>> df = pd.DataFrame({'ip': ['fdf8:f53b:82e4::53', None]})
    >>> validate_ip(df['ip'])
    0     True
    1    False
    Name: ip, dtype: bool
    """
    if isinstance(x, pd.Series):
        return x.apply(_check_ip, args=(input_format, False))
    return _check_ip(x, input_format, False)


def _format_ip(val: Any, input_format: str, output_format: str, errors: str) -> Any:
    """
    This function transforms the value val into the desired ip format if possible

    The last component of the returned tuple contains a code indicating how the
    input value was changed:
        0 := the value is null
        1 := the value could not be parsed
        2 := the value is cleaned and the cleaned value is DIFFERENT than the input value
        3 := the value is cleaned and is THE SAME as the input value (no transformation)
    """
    # pylint: disable=too-many-branches
    address, status = _check_ip(val, input_format, True)

    if status == "null":
        return np.nan, 0
    if status == "unknown":
        if errors == "raise":
            raise ValueError(f"Unable to parse value {val}")
        return val if errors == "ignore" else np.nan, 1

    # compressed version without the leading zeros (for ipv6 double colon for zeros)
    if output_format == "compressed":
        result = address.compressed

    # Converts the integer repesentation of the ip address to its hexadecimal
    # form. Does not contain any dots or colons.
    elif output_format == "hexa":
        result = hex(int(address))

    # converts the ip address to its binary representation
    elif output_format == "binary":
        if address.version == 4:
            result = "{0:032b}".format(int(address))
        else:
            result = "{0:0128b}".format(int(address))

    # converts to integer format
    elif output_format == "integer":
        result = int(address)

    # converts to packed binary format (big-endian)
    elif output_format == "packed":
        result = address.packed

    # convert to full representation
    else:
        dlm = "." if address.version == 4 else ":"  # delimiter
        result = dlm.join(f"{'0' * (4 - len(x))}{x}" for x in address.exploded.split(dlm))

    return result, 2 if result != val else 3


def _check_ip(val: Any, input_format: str, clean: bool) -> Any:
    """
    Function to check whether a value is valid ip address
    """
    try:
        if val in NULL_VALUES:
            return (None, "null") if clean else False

        address = ip_address(val)
        vers = address.version

        if vers == 4 and input_format != "ipv6" or vers == 6 and input_format != "ipv4":
            return (address, "success") if clean else True
        return (None, "unknown") if clean else False

    except (TypeError, ValueError):
        return (None, "unknown") if clean else False


# def report_ip(nrows: int, errors: str, column: str) -> None:
#     """
#     This function displays the stats report
#     """
#     correct_format = (
#         STATS["correct_format"] - 1 if (STATS["first_val"] == 100) else STATS["correct_format"]
#     )
#     correct_format_percentage = (correct_format / nrows) * 100

#     incorrect_format = (
#     STATS["incorrect_format"] - 1 if (STATS["first_val"] == 200) else STATS["incorrect_format"]
#     )
#     incorrect_format_percentage = (incorrect_format / nrows) * 100

#     set_to = "NaN" if (errors == "coerce") else "their original values"
#     result_null = "null values" if (errors == "coerce") else "null / not parsable values"
#     result = (
#         f"Result contains {correct_format} "
#         f"({(correct_format / nrows) * 100 :.2f} %) rows  in correct format(stored in column "\
#         f"`{column}_transformed`) and {incorrect_format} {result_null}"
#         f"({(incorrect_format / nrows) * 100:.2f} %)."
#     )

#     print(
#         f"""
# IP address cleaning report:
#         {correct_format} values parsed ({correct_format_percentage:.2f} %)
#         {incorrect_format} values unable to be parsed ({incorrect_format_percentage:.2f} %), "\
#         f"set to {set_to}
# {result}
#         """
#     )
