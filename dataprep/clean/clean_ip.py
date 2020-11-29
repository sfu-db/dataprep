"""
Implement clean_ip functionality
"""
from ipaddress import ip_address
from operator import itemgetter
from typing import Any, Union

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..eda.progress_bar import ProgressBar
from .utils import NULL_VALUES, create_report_new, to_dask


def clean_ip(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    input_format: str = "auto",
    output_format: str = "compressed",
    inplace: bool = False,
    report: bool = True,
    errors: str = "coerce",
    progress: bool = True,
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    This function cleans a column of ip addresses in a Dataframe and formats them
    into the desired format

    Parameters
    ----------
    df
        Pandas or Dask DataFrame
    column
        Column name where the ip address are stored
    input_format
        Specify what format the data is in {'ipv4', 'ipv6', 'auto'}, default 'auto',
            'ipv4': will only parse ipv4 addresses
            'ipv6': will only parse ipv6 addresses
            'auto': will parse both ipv4 and ipv6 addresses
    output_format
        Desired output format,
        {'compressed', 'full', 'binary', 'hexa', 'integer'}, default is 'compressed'
            'compressed': provides a compressed version of the ip address,
            'full': provides full version of the ip address,
            'binary': provides binary representation of the ip address,
            'hexa': provides hexadecimal representation of the ip address,
            'integer': provides integer representation of the ip address.
    inplace
        If True, deletes the given column with dirty data, else, creates a new
        column with cleaned data.
        Default value is set to `False`
    report
        Displays the cleaning report for ip addresses
        Default value is set to `True`
    errors {‘ignore’, ‘raise’, ‘coerce’}, default 'coerce'
        * If ‘raise’, then invalid parsing will raise an exception.
        * If ‘coerce’, then invalid parsing will be set as NaN.
        * If ‘ignore’, then invalid parsing will return the input.
    progress
        If True, enable the progress bar

    Returns
    ----------
    A new Dataframe with the new relavant columns
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

    if output_format not in {"compressed", "full", "binary", "hexa", "integer"}:
        raise ValueError(
            f'output_format {output_format} is invalid, it needs to be "compressed", "full", '
            '"binary", "hexa" or "integer"'
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
    This function validates ip address, can be a series or a single value

    Parameters
    ----------
    x
        pandas Series of ip addresses or an ip address value
    input_format
        Specify what format the data is in {'ipv4', 'ipv6', 'auto'}, default 'auto',
            'ipv4': will only parse ipv4 addresses
            'ipv6': will only parse ipv6 addresses
            'auto': will parse both ipv4 and ipv6 addresses
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

    # convert to full representation
    else:
        dlm = "." if address.version == 4 else ":"  # delimiter
        result = "".join(f"{'0' * (4 - len(x))}{x}{dlm}" for x in address.exploded.split(dlm))[:-1]

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
