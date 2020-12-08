# pylint: disable=too-many-arguments, global-statement, line-too-long, too-many-locals, too-many-branches, too-many-return-statements
"""
implement clean_ip functionality
"""
import ipaddress
from typing import Any, Union

import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd

from .utils import to_dask

STATS = {"correct_format": 0, "incorrect_format": 0, "first_val": 0}


def clean_ip(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    input_format: str = "auto",
    output_format: str = "compressed",
    inplace: bool = False,
    report: bool = True,
    errors: str = "coerce",
) -> Union[pd.DataFrame, dd.DataFrame]:
    """

    This function cleans a column of ip addresses in a Dataframe and formats them into the desired format
    Parameters
    ----------
    df
        Pandas or Dask DataFrame
    column
        Column name where the ip address are stored
    input_format
        Specify what format the data is in {'ipv4', 'ipv6', 'auto'}, default 'auto',
            'ipv4' - will only parse ipv4 addresses
            'ipv6' - will only parse ipv6 addresses
            'auto' - will parse both ipv4 and ipv6 addresses
        Defaut value is set to "auto"
    output_format
        Desired output format,
        {'compressed', 'full', 'binary', 'hexa', 'integer'}, default is 'compressed'
            'compressed' - provides a compressed version of the ip address,
            'full' - provides full version of the ip address,
            'binary' - provides binary representation of the ip address,
            'hexa' - provides hexadecimal representation of the ip address,
            'integer' - provides integer representation of the ip address.
    inplace
        If True, deletes the given column with dirty data, else, creates a new
        column with cleaned data.
        Default value is set to `False`
    report
        Displays the cleaning report for ip addresses
        Default value is set to `True`
    errors
        Specify ways to deal with broken value
        {'ignore', 'coerce', 'raise'}, default 'coerce'
        'raise': raise an exception when there is broken value
        'coerce': set invalid value to NaN
        'ignore': just return the initial input

    Returns
    ----------
    A new Dataframe with the new relavant columns

    """
    # check if the parameters are of correct processing types and values

    if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
        raise ValueError("invalid `df` processing type.")

    if not isinstance(column, str):
        raise ValueError("invalid `column` processing type.")

    if input_format not in {"ipv4", "ipv6", "auto"}:
        raise ValueError("invalid `input_format` processing type.")

    if output_format not in {"compressed", "full", "binary", "hexa", "integer"}:
        raise ValueError("invalid `output_format` processing type.")

    if not isinstance(inplace, bool):
        raise ValueError("invalid `inplace` processing type.")

    if not isinstance(report, bool):
        raise ValueError("invalid `report` processing type")

    if errors not in {"coerce", "ignore", "raise"}:
        raise ValueError("invalid `errors` processing type.")

    reset_ip()
    df = to_dask(df)

    # specify the metadata for dask apply
    meta = df.dtypes.to_dict()

    meta[f"{column}_transformed"] = str

    df = df.apply(format_ip, args=(column, input_format, output_format, errors), axis=1, meta=meta)

    df, nrows = dask.compute(df, df.shape[0])

    if inplace:
        df = df.drop(columns=[column])

    if report:
        report_ip(nrows=nrows, errors=errors, column=column)

    return df


def format_ip(
    row: pd.Series, column: str, input_format: str, output_format: str, errors: str
) -> pd.Series:
    """
    This function formats each row of a pd.Series containing the url column
    """
    val_dict = transform_ip(
        row[column],
        input_format=input_format,
        output_format=output_format,
        errors=errors,
    )
    row[f"{column}_transformed"] = val_dict

    return row


def transform_ip(ip_address: str, input_format: str, output_format: str, errors: str) -> Any:
    """
    cleans and formats individual ip address entries
    """

    if check_ip(ip_address):
        return convert_ip(
            ip_address=ip_address, input_format=input_format, output_format=output_format
        )

    else:
        if check_first():
            STATS["first_val"] = 200
        STATS["incorrect_format"] += 1
        if errors == "coerce":
            return np.nan
        elif errors == "raise":
            raise ValueError("Cannot parse this value" + ip_address)
        else:
            return ip_address


def convert_ip(ip_address: str, input_format: str, output_format: str) -> Any:
    """
    This function converts individual ips to the desired output format
    """
    ip_address = ipaddress.ip_address(ip_address)
    version = ip_address.version  # type: ignore
    ip_version = "ipv" + str(version)

    if input_format not in ("auto", ip_version):
        if check_first():
            STATS["first_val"] = 200
        STATS["incorrect_format"] += 1
        return np.NaN
    else:
        if check_first():
            STATS["first_val"] = 100
        STATS["correct_format"] += 1

    # compressed version without the leading zeros (for ipv6 double colon for zeros)
    if output_format == "compressed":
        return ip_address.compressed  # type: ignore

    # converts the integer repesentation of the ip address to its hexadecimal form. Does not contain any dots or colon
    elif output_format == "hexa":
        return hex(int(ip_address))

    # converts the ip address to its binary representation
    elif output_format == "binary":  # type: ignore
        if version == 4:
            return "{0:032b}".format(int(ip_address))  # type: ignore
        else:
            return "{0:0128b}".format(int(ip_address))  # type: ignore

    # converts to integer format
    elif output_format == "integer":  # type: ignore
        return int(ip_address)  # type: ignore

    final = ""
    if ip_address.version == 4:  # type: ignore
        for x in ip_address.exploded.split("."):  # type: ignore
            diff = 4 - len(x)
            final += ("0" * diff) + x + "."
    else:
        for x in ip_address.exploded.split(":"):  # type: ignore
            diff = 4 - len(x)
            final += ("0" * diff) + x + ":"
    return final[:-1]


def validate_ip(x: Union[str, pd.Series]) -> Union[bool, pd.Series]:
    """
    This function validates ip address, can be a series or a single value

    Parameters
    ----------
    x
        pandas Series of ip addresses or an ip address value
    """
    if isinstance(x, pd.Series):
        verfied_series = x.apply(check_ip)
        return verfied_series
    else:
        return check_ip(x)


def check_ip(val: Union[str, Any]) -> Any:
    """
    Function to check whether a value is valid ip address
    """
    try:
        return bool(ipaddress.ip_address(val))
    except ValueError:
        return False


def report_ip(nrows: int, errors: str, column: str) -> None:
    """
    This function displays the stats report
    """
    correct_format = (
        STATS["correct_format"] - 1 if (STATS["first_val"] == 100) else STATS["correct_format"]
    )
    correct_format_percentage = (correct_format / nrows) * 100

    incorrect_format = (
        STATS["incorrect_format"] - 1 if (STATS["first_val"] == 200) else STATS["incorrect_format"]
    )
    incorrect_format_percentage = (incorrect_format / nrows) * 100

    set_to = "NaN" if (errors == "coerce") else "their original values"
    result_null = "null values" if (errors == "coerce") else "null / not parsable values"
    result = (
        f"Result contains {correct_format} "
        f"({(correct_format / nrows) * 100 :.2f} %) rows  in correct format(stored in column `{column}_transformed`) and {incorrect_format} {result_null}"
        f"({(incorrect_format / nrows) * 100:.2f} %)."
    )

    print(
        f"""
IP address cleaning report:
        {correct_format} values parsed ({correct_format_percentage:.2f} %)
        {incorrect_format} values unable to be parsed ({incorrect_format_percentage:.2f} %), set to {set_to}
{result}
        """
    )


def reset_ip() -> None:
    """
    This function resets the STATs
    """
    STATS["correct_format"] = 0
    STATS["incorrect_format"] = 0
    STATS["first_val"] = 0


def check_first() -> bool:
    """
    Dask runs 2 times for the first value (hence the first value is counted twice),
    this function checks whether the value we are parsing is the first value,
    after we find the first value we check whether it is in the correct form or incorrect form
    we then set STATS["first_val"] according to the following convention
    100 is the code for correct form and 200 is the code for the incorrect form
    we will use this value (STATS["first_val"] == 100 or STATS["first_val"] == 200) in our stats
    report to compensate for the overcounting of the first value by reducing the value.
    """
    return STATS["correct_format"] == 0 and STATS["incorrect_format"] == 0
