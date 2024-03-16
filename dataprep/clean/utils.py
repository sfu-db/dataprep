"""Common functions"""

import http.client
import json
from math import ceil
from typing import Any, Dict, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd

NULL_VALUES = {
    np.nan,
    float("NaN"),
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "n/a",
    "nan",
    "null",
    "",
    None,
}

NEARBYKEYS = {
    "a": ["q", "w", "s", "x", "z"],
    "b": ["v", "g", "h", "n"],
    "c": ["x", "d", "f", "v"],
    "d": ["s", "e", "r", "f", "c", "x"],
    "e": ["w", "s", "d", "r"],
    "f": ["d", "r", "t", "g", "v", "c"],
    "g": ["f", "t", "y", "h", "b", "v"],
    "h": ["g", "y", "u", "j", "n", "b"],
    "i": ["u", "j", "k", "o"],
    "j": ["h", "u", "i", "k", "n", "m"],
    "k": ["j", "i", "o", "l", "m"],
    "l": ["k", "o", "p"],
    "m": ["n", "j", "k", "l"],
    "n": ["b", "h", "j", "m"],
    "o": ["i", "k", "l", "p"],
    "p": ["o", "l"],
    "q": ["w", "a", "s"],
    "r": ["e", "d", "f", "t"],
    "s": ["w", "e", "d", "x", "z", "a"],
    "t": ["r", "f", "g", "y"],
    "u": ["y", "h", "j", "i"],
    "v": ["c", "f", "g", "v", "b"],
    "w": ["q", "a", "s", "e"],
    "x": ["z", "s", "d", "c"],
    "y": ["t", "g", "h", "u"],
    "z": ["a", "s", "x"],
    " ": ["c", "v", "b", "n", "m"],
}


def to_dask(df: Union[pd.DataFrame, dd.DataFrame]) -> dd.DataFrame:
    """Convert a dataframe to a dask dataframe."""
    if isinstance(df, dd.DataFrame):
        return df

    df_size = df.memory_usage(deep=True).sum()
    npartitions = ceil(df_size / 128 / 1024 / 1024)  # 128 MB partition size
    return dd.from_pandas(df, npartitions=npartitions)


def create_report(type_cleaned: str, stats: Dict[str, int], nrows: int) -> None:
    """
    Describe what was done in the cleaning process
    """
    # pylint: disable=line-too-long
    print(f"{type_cleaned} Cleaning Report:")
    if stats["cleaned"] > 0:
        nclnd = stats["cleaned"]
        pclnd = round(nclnd / nrows * 100, 2)
        print(f"\t{nclnd} values cleaned ({pclnd}%)")
    if stats["unknown"] > 0:
        nunknown = stats["unknown"]
        punknown = round(nunknown / nrows * 100, 2)
        print(f"\t{nunknown} values unable to be parsed ({punknown}%), set to NaN")
    nnull = stats["null"] + stats["unknown"]
    pnull = round(nnull / nrows * 100, 2)
    ncorrect = nrows - nnull
    pcorrect = round(100 - pnull, 2)
    print(
        f"""Result contains {ncorrect} ({pcorrect}%) values in the correct format and {nnull} null values ({pnull}%)"""
    )


def create_report_new(type_cleaned: str, stats: pd.Series, errors: str) -> None:
    """
    Describe what was done in the cleaning process

    The stats series contains the following codes in its index
        0 := the number of null values
        1 := the number of values that could not be parsed
        2 := the number of values that were transformed during cleaning
        3 := the number of values that were already in the correct format
    """
    print(f"{type_cleaned} Cleaning Report:")
    nrows = stats.sum()

    nclnd = stats.loc[2] if 2 in stats.index else 0
    if nclnd > 0:
        pclnd = round(nclnd / nrows * 100, 2)
        print(f"\t{nclnd} values cleaned ({pclnd}%)")

    nunknown = stats.loc[1] if 1 in stats.index else 0
    if nunknown > 0:
        punknown = round(nunknown / nrows * 100, 2)
        expl = "set to NaN" if errors == "coerce" else "left unchanged"
        print(f"\t{nunknown} values unable to be parsed ({punknown}%), {expl}")

    nnull = stats.loc[0] if 0 in stats.index else 0
    if errors == "coerce":
        nnull += stats.loc[1] if 1 in stats.index else 0
    pnull = round(nnull / nrows * 100, 2)

    ncorrect = nclnd + (stats.loc[3] if 3 in stats.index else 0)
    pcorrect = round(ncorrect / nrows * 100, 2)
    print(
        f"Result contains {ncorrect} ({pcorrect}%) values in the correct format "
        f"and {nnull} null values ({pnull}%)"
    )


def _get_data(currency_code: str, file_path: str) -> Any:
    """
    returns the fiat currency's symbol and full name
    """
    with open(file_path) as f:
        currency_data = json.loads(f.read())
    currency_dict = next((item for item in currency_data if item["cc"] == currency_code), None)
    if currency_dict:
        symbol = currency_dict.get("symbol")
        currency_name = currency_dict.get("name")
        return symbol, currency_name
    return None, None


def _get_rate(base_cur: str, dest_cur: str, url: str) -> Any:
    """
    returns the conversion rate between 2 fiat currencies
    """
    if base_cur == dest_cur:
        return 1.0
    conn = http.client.HTTPSConnection(url)
    conn.request("GET", f"/api/latest?base={base_cur}&symbols={dest_cur}&rtype=fpy")
    response = conn.getresponse()

    if response.status == 200:
        response_json = json.loads(response.read())
        rate = np.round(response_json["rates"][dest_cur], 4)
        if not rate:
            raise RatesNotAvailableError(
                f"Currency Rate {base_cur} => {dest_cur} not available latest"
            )
        return rate
    raise RatesNotAvailableError("Currency Rates Source Not Ready / Available")


def _get_rate_crypto(base_cur: str, dest_cur: str, url: str) -> Any:
    """
    returns the price of the cryptocurrecy in the specified base currency
    """
    if base_cur == dest_cur:
        return 1.0
    conn = http.client.HTTPSConnection(url)
    conn.request("GET", f"/api/v3/simple/price?ids={base_cur}&vs_currencies={dest_cur}")
    response = conn.getresponse()

    if response.status == 200:
        response_json = json.loads(response.read())
        rate = response_json[base_cur][dest_cur]
        if not rate:
            raise RatesNotAvailableError(
                f"Currency Rate {base_cur} => {dest_cur} not available latest"
            )
        return rate
    raise RatesNotAvailableError("Currency Rates Source Not Ready / Available")


def _get_crypto_symbol_and_id(crypto_name: str, file_path: str) -> Any:
    """
    gets the cryprocurrency ID and symbol from cryptocurries.json file
    """
    cryptocurrencies = {}
    with open(file_path, "r") as file:
        cryptocurrencies = json.load(file)
    try:
        crypto = cryptocurrencies[crypto_name]
        crypto_id = crypto[0]
        crypto_symbol = crypto[1]
        return crypto_id, crypto_symbol
    except Exception as key_error:
        raise KeyError(
            f"The target curency name `{crypto_name}` doesn't sound correct, please recheck your value"
        ) from key_error


class RatesNotAvailableError(Exception):
    """
    Custome Exception when https://ratesapi.io is Down and not available for currency rates
    """
