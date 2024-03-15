"""
Clean and validate a DataFrame column containing currencies
"""

# pylint: disable=too-many-arguments, line-too-long, too-many-locals, too-many-branches, too-many-statements, too-many-return-statements,

import json
from os import path
from operator import itemgetter
from typing import Any, List, Union, Optional


import dask
import dask.dataframe as dd

import pandas as pd
import numpy as np

from ..progress_bar import ProgressBar
from .utils import (
    NULL_VALUES,
    create_report_new,
    to_dask,
    _get_data,
    _get_crypto_symbol_and_id,
    _get_rate,
    _get_rate_crypto,
)


CURRENCIES_DATA_FILE = path.join(path.split(path.abspath(__file__))[0], "currencies.json")
CRYPTOCURRENCIES_DATA_FILE = path.join(path.split(path.abspath(__file__))[0], "cryptocurrency.json")

FIAT_BASE_URL = "api.ratesapi.io"
CRYPTO_BASE_URL = "api.coingecko.com"


def clean_currency(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    n_round: int = 2,
    input_currency: str = "usd",
    target_currency: Optional[str] = None,
    target_representation: str = "decimal",
    user_rate: Optional[float] = None,
    ignore_symbol: Optional[Union[List[str], str]] = None,
    errors: str = "coerce",
    split: bool = False,
    inplace: bool = False,
    report: bool = True,
    progress: bool = False,
) -> Union[pd.DataFrame, dd.DataFrame]:
    """
    Clean, standardize and convert currencies.

    Read more in the :ref:`User Guide <ip_userguide>`.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing currencies
    n_round
        Round a float to given number of decimals
    input_currency
        Name of the input currency
            - For fiat currencies use the 3 letter abbreviation. Eg. "usd", "inr", etc.
            - For cyrptocurrencies use the full name of the coin. Eg "Binance Coin", "ethereum", etc.
    target_currency
        Optional Parameter - Name of the output currency.
                - For fiat currencies use the 3 letter abbreviation. Eg. "usd", "inr", etc.
                - For cyrptocurrencies use the full name of the coin. Eg "Binance Coin", "ethereum", etc.
        (default : None)
    target_representation:
        The desired format in which the result would be displayed.
            - 'decimal' (floating point number)
            - 'abbreviation' (string with comma seprated values)
    user_rate:
        Optional parameter - user defined exchange rate

        (default : None)
    symbol:
        Optional parameter - A list of string provided by user that the parser will ignore

        (default : None)
    errors
        How to handle parsing errors.
            - ‘coerce’: invalid parsing will be set to NaN.
            - ‘ignore’: invalid parsing will return the input.
            - ‘raise’: invalid parsing will raise an exception.

        (default: 'coerce')
    split
        If True, split a column containing a cleaned or converted currency into different
        columns containing individual components.

        (default: False)
    inplace
        If True, delete the column containing the data that was cleaned. Otherwise,
        keep the original column.

        (default: False)
    report
        If True, output the summary report. Else, no report is outputted.

        (default: True)
    progress
        If True, enable the progress bar.

        (default: True)


    Examples
    --------

    >>> df = pd.DataFrame({'currencies': [234.56, 18790, '234,456.45']})
    >>> clean_currency(df, 'currencies', input_currency='INR', target_currency='USD')
    Currency Cleaning Report:
            3 values cleaned (100.0%)
    Result contains 3 (100.0%) values in the correct format and 0 null values (0.0%)
    currencies  currencies_clean
    0      234.56              3.12
    1       18790            249.91
    2  234,456.45           3118.27

    """

    _check_params(
        df=df,
        column=column,
        n_round=n_round,
        input_currency=input_currency,
        target_currency=target_currency,
        target_representation=target_representation,
        user_rate=user_rate,
        ignore_symbol=ignore_symbol,
        errors=errors,
        split=split,
        inplace=inplace,
        report=report,
        progress=progress,
    )

    # defining some variables here because of scoping issues
    target_symbol = ""
    input_symbol = ""
    conversion_rate = 1.0
    conversion_type = ""

    if target_currency is not None:
        # cryptocurrency data takes lower case values
        input_currency = input_currency.lower()
        target_currency = target_currency.lower()

        conversion_type = _detect_conversion_type(input_currency, target_currency)

        input_symbol, target_symbol, conversion_rate = _get_conversion_rates_and_symbols(
            input_currency=input_currency,
            target_currency=target_currency,
            conversion_type=conversion_type,
        )
    else:
        input_symbol = _get_conversion_rates_and_symbols(input_currency=input_currency)

    # finally if `user_rate` is defined, set it to conversion rate
    if user_rate is not None:
        conversion_rate = user_rate

    df = to_dask(df)

    # To clean, create a new column "clean_code_tup" which contains
    # the cleaned values and code indicating how the initial value was
    # changed in a tuple. Then split the column of tuples and count the
    # amount of different codes to produce the report
    df["clean_code_tup"] = df[column].map_partitions(
        lambda srs: [
            _format_currency(
                val=x,
                ignore_symbol=ignore_symbol,
                n_round=n_round,
                errors=errors,
                target_representation=target_representation,
                target_symbol=target_symbol,
                input_symbol=input_symbol,
                conversion_rate=conversion_rate,
                conversion_type=conversion_type,
                split=split,
            )
            for x in srs
        ],
        meta=object,
    )

    if split:
        if target_currency is not None:
            df = df.assign(
                input_currency_symbol=df["clean_code_tup"].map(itemgetter(0)),
                _temp_=df["clean_code_tup"].map(itemgetter(1)),
                conversion_rate=df["clean_code_tup"].map(itemgetter(2)),
                target_symbol=df["clean_code_tup"].map(
                    itemgetter(3), meta=("target_symbol", object)
                ),
                target_val=df["clean_code_tup"].map(itemgetter(4), meta=("target_val", object)),
                _code_=df["clean_code_tup"].map(itemgetter(5), meta=("_code_", object)),
            )
        else:
            df = df.assign(
                input_currency_symbol=df["clean_code_tup"].map(itemgetter(0)),
                _temp_=df["clean_code_tup"].map(itemgetter(1)),
                _code_=df["clean_code_tup"].map(itemgetter(2), meta=("_code_", object)),
            )

    else:
        # TODO - apply the condition where no split and target currency is specified.
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
        create_report_new("Currency", stats, errors)

    return df


def _format_currency(
    val: Any,
    ignore_symbol: Optional[Union[List[str], str]],
    n_round: int,
    errors: str,
    target_representation: str,
    input_symbol: str,
    split: bool,
    conversion_rate: float,
    conversion_type: str,
    target_symbol: str,
) -> Any:
    """
    Formats each values and returns it the specified format
    """

    # check whether valid format or not
    # if not return according to errors paramter
    if isinstance(ignore_symbol, str):
        ignore_symbol = [ignore_symbol]
    val, status = _check_currency(val, ignore_symbol, True)

    input_symbol = input_symbol.upper()
    target_symbol = target_symbol.upper()

    if val is not None:
        val, val_new = _get_values_target_representation(
            val=val,
            target_representation=target_representation,
            conversion_type=conversion_type,
            conversion_rate=conversion_rate,
            n_round=n_round,
            split=split,
            input_symbol=input_symbol,
            target_symbol=target_symbol,
        )

    if split:
        if status == "null":
            return (np.nan, np.nan, np.nan, np.nan, np.nan, 0)
        elif status == "unknown":
            if errors == "raise":
                raise ValueError(f"Unable to parse value {val}")
            return (
                (np.nan, val, np.nan, np.nan, np.nan, 1)
                if errors == "ignore"
                else (np.nan, np.nan, np.nan, np.nan, np.nan, 1)
            )
        else:
            if target_symbol != "null":
                return (input_symbol, val, conversion_rate, target_symbol, val_new, 2)
            else:
                return (input_symbol, val, 2)

    else:
        if status == "null":
            return np.nan, 0

        if status == "unknown":
            if errors == "raise":
                raise ValueError(f"Unable to parse value {val}")
            return val if errors == "ignore" else np.nan, 1
        else:
            if target_symbol != "null":
                return val_new, 2
            else:
                return val, 2


def validate_currency(
    x: Union[str, pd.Series], symbol: Optional[Union[List[str], str]] = None
) -> Union[bool, pd.Series]:
    """
    This function validates currencies
    Parameters
    ----------
    x
        pandas Series of currencies or currency instance
    """

    if isinstance(x, pd.Series):
        return x.apply(_check_currency, args=[symbol, False])
    else:
        return _check_currency(x, symbol, False)


def _check_currency(
    val: Union[str, Any], symbol: Optional[Union[List[str], str]], clean: bool
) -> Any:
    """
    Function to check whether a value is a valid currency
    """

    if symbol is None:
        symbol = []

    try:
        if val in NULL_VALUES:
            return (None, "null") if clean else False
        val = str(val)
        val = val.replace(",", "")

        ## strip symbols
        for s_s in symbol:
            val = val.strip(s_s)

        val = "".join(c for c in val if (c.isdigit() or c == "." or c not in symbol))

        if float(val):
            return (val, "success") if clean else True
        else:
            return (None, "unknown") if clean else False
    except (TypeError, ValueError):
        return (None, "unknown") if clean else False


def _get_values_target_representation(
    val: Union[str, Any],
    target_representation: str,
    conversion_type: str,
    conversion_rate: float,
    n_round: int,
    split: bool,
    input_symbol: str,
    target_symbol: str,
) -> Any:
    """
    Returns the value of the converted currency in the specified format.
    The two formats specified are "abbr", "decimal".
    """

    val_new = 0.0
    val = float(val)

    #   1. for fiat-to-fiat and crypto-to-fiat we multiply
    #   2. for fiat-to-crypto we divide

    if conversion_type in ("fiat_to_fiat", "crypto_to_fiat"):
        val_new = val * conversion_rate
    else:
        val_new = val / conversion_rate

    if target_representation == "abbr":
        val = "{:,.{a}f}".format(val, a=n_round)
        target_val = "{:,.{a}f}".format(val_new, a=n_round)
        if split:
            return val, target_val
        else:
            return input_symbol.upper() + str(val), target_symbol.upper() + str(target_val)
    else:
        return np.round(val, n_round), np.round(val_new, n_round)


def _check_params(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    n_round: int = 2,
    input_currency: str = "usd",
    target_currency: Optional[str] = None,
    target_representation: str = "decimal",
    user_rate: Optional[float] = None,
    ignore_symbol: Optional[Union[List[str], str]] = None,
    errors: str = "coerce",
    split: bool = False,
    inplace: bool = False,
    report: bool = True,
    progress: bool = False,
) -> None:
    """
    Checks whether the params passed by the end user is of the correct data type
    """
    if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
        raise ValueError("df is invalid, it needs to be a pandas or Dask DataFrame")

    if not isinstance(column, str):
        raise ValueError(f"column name {column} is invalid, it name needs to a string")

    if not isinstance(n_round, int):
        raise ValueError(f"n_round {n_round} is invalid, it needs to be an integer")

    if not isinstance(input_currency, str):
        raise ValueError(f"input currency {input_currency} is invalid, it needs to be a string")

    if target_currency is not None and not isinstance(target_currency, str):
        raise ValueError(f"target_currency {target_currency} is invalid, it needs to be a string")

    if not isinstance(target_representation, str):
        raise ValueError(
            f"target_representation {target_representation} is invalid, it needs to be a string"
        )

    if ignore_symbol is not None and not isinstance(ignore_symbol, (list, str)):
        raise ValueError(
            f"symbol {ignore_symbol} is invalid, it needs to be either a string or list of strings"
        )

    if not isinstance(split, bool):
        raise ValueError(f"split {split} is invalid, it needs to be a boolean")

    if not isinstance(inplace, bool):
        raise ValueError(f"inplace {inplace} is invalid, it needs to be to be a boolean")

    if not isinstance(report, bool):
        raise ValueError(f"report {report} is invalid, it needs to be a boolean")

    if not isinstance(progress, bool):
        raise ValueError(f"progress {progress} is invalid, it needs to be a boolean")

    if user_rate is not None and not isinstance(user_rate, (int, float)):
        raise ValueError(f"user_rate {user_rate} is invalid, it needs to be a boolean")

    if errors not in {"coerce", "ignore", "raise"}:
        raise ValueError(f'errors {errors} is invalid, it needs to be "coerce", "ignore", "raise"')


def _get_dictionary_from_json(path: str) -> Any:
    """
    Utility function to load a json file and return it as a dictionary
    """
    with open(path) as f:
        return json.loads(f.read())


def _detect_conversion_type(input_currency: str, target_currency: str) -> str:
    """
    detects the conversion type based upon the input and target currencies

    conversion type can be of 4 types:
       1. fiat to fiat
       2. fiat to cypto
       3. crypto  to fiat
       4. crypto to crypto (need to confirm whether is supported or not)
    """

    cryptocurrency_dict = _get_dictionary_from_json(CRYPTOCURRENCIES_DATA_FILE)
    currency_dict = _get_dictionary_from_json(CURRENCIES_DATA_FILE)

    currency_list = [x["cc"].lower() for x in currency_dict]

    if (input_currency in currency_list) and (target_currency in currency_list):
        conversion_type = "fiat_to_fiat"

    elif (input_currency in currency_list) and (target_currency in cryptocurrency_dict):
        conversion_type = "fiat_to_crypto"

    elif (input_currency in cryptocurrency_dict) and (target_currency in currency_list):
        conversion_type = "crypto_to_fiat"

    elif (input_currency in cryptocurrency_dict) and (target_currency in cryptocurrency_dict):
        raise ValueError("Currently we do not support crypto to crypto conversion")

    else:
        raise ValueError("Please check your input and target currencies")

    return conversion_type


def _get_conversion_rates_and_symbols(
    input_currency: str,
    target_currency: Optional[str] = None,
    conversion_type: Optional[str] = None,
) -> Any:
    """
    The function that returns the symbols and conversion rates as following:
        1. if `target_currency` is specified then it returns `input_symbol`, `target_symbol` and `conversion`
        2. if no `target_currency` is specified then it returns only `input symbol`
    """

    target_symbol = ""
    input_symbol = ""

    cryptocurrency_data = _get_dictionary_from_json(CRYPTOCURRENCIES_DATA_FILE)

    if target_currency is not None:
        if conversion_type == "fiat_to_fiat":
            input_currency = input_currency.upper()
            target_currency = target_currency.upper()
            input_symbol, _ = _get_data(input_currency, CURRENCIES_DATA_FILE)
            target_symbol, _ = _get_data(target_currency, CURRENCIES_DATA_FILE)
            conversion_rate = _get_rate(input_currency, target_currency, FIAT_BASE_URL)

        elif conversion_type == "fiat_to_crypto":
            input_symbol, _ = _get_data(input_currency.upper(), CURRENCIES_DATA_FILE)
            crypto_id, target_symbol = _get_crypto_symbol_and_id(
                crypto_name=target_currency, file_path=CRYPTOCURRENCIES_DATA_FILE
            )
            conversion_rate = _get_rate_crypto(crypto_id, input_currency, CRYPTO_BASE_URL)

        elif conversion_type == "crypto_to_fiat":
            target_symbol, _ = _get_data(target_currency.upper(), CURRENCIES_DATA_FILE)
            crypto_id, input_symbol = _get_crypto_symbol_and_id(
                crypto_name=input_currency, file_path=CRYPTOCURRENCIES_DATA_FILE
            )
            conversion_rate = _get_rate_crypto(crypto_id, target_currency, CRYPTO_BASE_URL)
        else:
            # TODO - Find out whether crypto to crypto conversion in supported??
            crypto_id, target_symbol = _get_crypto_symbol_and_id(
                crypto_name=target_currency, file_path=CRYPTOCURRENCIES_DATA_FILE
            )
            conversion_rate = _get_rate_crypto(crypto_id, input_currency, CRYPTO_BASE_URL)

        return input_symbol, target_symbol, conversion_rate

    if input_currency in cryptocurrency_data:
        _, input_symbol = _get_crypto_symbol_and_id(input_currency, CRYPTOCURRENCIES_DATA_FILE)
    else:
        input_currency = input_currency.upper()

        input_symbol, _ = _get_data(input_currency, CURRENCIES_DATA_FILE)

    return input_symbol
