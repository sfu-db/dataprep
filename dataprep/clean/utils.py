"""Common functions"""
from typing import Dict, Union

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
    npartitions = np.ceil(df_size / 128 / 1024 / 1024)  # 128 MB partition size
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
