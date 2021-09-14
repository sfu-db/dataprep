"""
Clean and validate a DataFrame column containing language.
"""

# pylint: disable=too-many-arguments, global-statement

from os import path
from typing import Any, Union, Tuple, List, Optional, Dict
from operator import itemgetter

import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..progress_bar import ProgressBar
from .utils import NULL_VALUES, to_dask
from .clean_headers import clean_headers

DEFAULT_LANGUAGE_DATA_FILE = path.join(path.split(path.abspath(__file__))[0], "language_data.csv")

DATA = pd.read_csv(DEFAULT_LANGUAGE_DATA_FILE, encoding="utf-8", dtype=str)
ALPHA2: Dict[str, List[int]] = {}
ALPHA3: Dict[str, List[int]] = {}
NAME: Dict[str, List[int]] = {}


def clean_language(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    input_format: Union[str, Tuple[str, ...]] = "auto",
    output_format: str = "name",
    kb_path: str = "default",
    encode: Optional[str] = None,
    inplace: bool = False,
    errors: str = "coerce",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean language type data in a DataFrame column.

    Parameters
    ----------
        df
            A pandas or Dask DataFrame containing the data to be cleaned.
        column
            The name of the column containing data of language type.
        input_format
            The ISO 639 input format of the language.
                - 'auto': infer the input format
                - 'name': language name ('English')
                - 'alpha-2': alpha-2 code ('en')
                - 'alpha-3': alpha-3 code ('eng')

            Can also be a tuple containing any combination of input formats,
            for example to clean a column containing name and alpha-2
            codes set input_format to ('name', 'alpha-2').

            (default: 'auto')
        output_format
            The desired ISO 639 format of the language.
                - 'name': language name ('English')
                - 'alpha-2': alpha-2 code ('en')
                - 'alpha-3': alpha-3 code ('eng')

            (default: 'name')
        kb_path
            The path of user specified knowledge base.
            In current stage, it should be in the user's local directory
            following by the format we proposing.

            (default: 'default')
        encode
            The encoding of the knowledge base. It will be passed to `pd.read_csv`.

            (default: None)
        inplace
           If True, delete the column containing the data that was cleaned.
           Otherwise, keep the original column.

           (default: False)
        errors
            How to handle parsing errors.
            - 'coerce': invalid parsing will be set to NaN.
            - 'ignore': invalid parsing will return the input.
            - 'raise': invalid parsing will raise an exception.

            (default: 'coerce')
        progress
            If True, display a progress bar.

            (default: True)

    Examples
    --------
    Clean a column of language data.

    >>> df = pd.DataFrame({'language': ['eng', 'zh', 'japanese']})
    >>> clean_language(df, 'language')
        language    language_clean
    0       eng     English
    1        zh     Chinese
    2  japanese     Japanese
    """
    # load knowledge base
    _load_kb(kb_path, encode)

    valid_output_formats = {"name", "alpha-2", "alpha-3"}
    if output_format not in valid_output_formats:
        raise ValueError(
            f'output_format {output_format} is invalid, it needs to be "name", '
            '"alpha-2" or "alpha-3"'
        )

    valid_errors = {"coerce", "ignore", "raise"}
    if errors not in valid_errors:
        raise ValueError(
            f'errors {errors} is invalid, it needs to be "coerce", ' '"ignore" or "raise"'
        )

    input_formats = _convert_format_to_tuple(input_format)

    # convert to dask
    df = to_dask(df)

    df["clean_code_tup"] = df[column].map_partitions(
        lambda srs: [_format(x, input_formats, output_format, errors) for x in srs],
        meta=object,
    )

    df = df.assign(
        _temp_=df["clean_code_tup"].map(itemgetter(0), meta=("_temp_", object)),
    )

    df = df.rename(columns={"_temp_": f"{column}_clean"})

    df = df.drop(columns=["clean_code_tup"])

    if inplace:
        df[column] = df[f"{column}_clean"]
        df = df.drop(columns=f"{column}_clean")
        df = df.rename(columns={column: f"{column}_clean"})

    with ProgressBar(minimum=1, disable=not progress):
        df = df.compute()

    return df


def validate_language(
    x: Union[str, pd.Series, dd.Series, pd.DataFrame, dd.DataFrame],
    column: str = "",
    input_format: Union[str, Tuple[str, ...]] = "auto",
    kb_path: str = "default",
    encode: Optional[str] = None,
) -> Union[bool, pd.Series, pd.DataFrame]:
    """
    Validate language type data in a DataFrame column. For each cell, return True or False.

    Parameters
    ----------
    x
        Language data to be validated. It could be a single string, or
        a pandas or Dask DataFrame, or a pandas or Dask Series.
    column
        The name of the column to be validated.
        If x is not a pandas or Dask DataFrame, it would be ignored.
        If x is a pandas or Dask DataFrame but `column` is not specified,
        then the whole dataframe will be validated.

        (default: None)
    input_format
        The ISO 639 input format of the language.
            - 'auto': infer the input format
            - 'name': language name ('English')
            - 'alpha-2': alpha-2 code ('en')
            - 'alpha-3': alpha-3 code ('eng')

        Can also be a tuple containing any combination of input formats,
        for example to clean a column containing name and alpha-2
        codes set input_format to ('name', 'alpha-2').

        (default: 'auto')
    kb_path
        The path of user specified knowledge base.
        In current stage, it should be in the user's local directory
        following by the format we proposing.

        (default: "default")
    encode
        The encoding of the knowledge base. It will be passed to `pd.read_csv`.

        (default: None)
    """
    # load knowledge base
    _load_kb(kb_path, encode)

    input_formats = _convert_format_to_tuple(input_format)

    if isinstance(x, (pd.Series, dd.Series)):
        return x.apply(_check_language, args=(input_formats, False))
    elif isinstance(x, (pd.DataFrame, dd.DataFrame)):
        if column != "":
            return x[column].apply(_check_language, args=(input_formats, False))
        else:
            return x.applymap(lambda val: _check_language(val, input_formats, False))
    return _check_language(x, input_formats, False)


def _format(val: Any, input_formats: Tuple[str, ...], output_format: str, errors: str) -> Any:
    """
    Reformat a language string with proper output format.
    """
    result_index, status = _check_language(val, input_formats, True)

    if status == "null":
        return [np.nan]
    if status == "unknown":
        if errors == "raise":
            raise ValueError(f"unable to parse value {val}")
        return [val] if errors == "ignore" else [np.nan]

    formated_val = DATA.loc[result_index, output_format]
    if pd.isna(formated_val):
        # country doesn't have the required output format
        if errors == "raise":
            raise ValueError(f"unable to parse value {val}")
        return [val] if errors == "ignore" else [np.nan]

    return [formated_val.title()] if output_format == "name" else [formated_val]


def _check_language(val: Any, input_formats: Tuple[str, ...], clean: bool) -> Any:
    """
    Find the index of the given language string in the DATA dataframe.

    Parameters
    ----------
    val
        String containing the language value to be cleaned.
    input_formats
        Tuple containing potential ISO 639 input formats of the language.
    clean
        If True, a tuple (index, status) is returned. There are 3 status:
             - "null": val is a null value.
             - "unknown": val could not be parsed.
             - "success": a successful parse of the value.
        If False, the function returns True/False to be used by the validate function.
    """
    if val in NULL_VALUES:
        return (None, "null") if clean else False

    val = str(val).lower().strip()
    first_letter = val[0]

    # select possible formats from input_formats;
    possible_formats: Tuple[str, ...] = ()
    if len(val) > 1 and "name" in input_formats:
        # it is a potential valid language
        possible_formats = ("name",) + possible_formats

    if len(val) == 3 and "alpha-3" in input_formats:
        # alpha-3 or name, and alpha-3 is preferred
        possible_formats = ("alpha-3",) + possible_formats
    elif len(val) == 2 and "alpha-2" in input_formats:
        # alpha-2 or name, and alpha-2 is preferred
        possible_formats = ("alpha-2",) + possible_formats

    # search the value
    format_dicts = {"name": NAME, "alpha-2": ALPHA2, "alpha-3": ALPHA3}
    for fmt in possible_formats:
        format_dict = format_dicts[fmt]
        inds = format_dict.get(
            first_letter
        )  # get the indices of value that starts with the same letter
        if inds is None:  # no value starts with this letter
            continue
        df_temp = DATA.iloc[inds][fmt]  # extract these values
        res = df_temp[df_temp.str.lower() == val]  # search the input value within them
        if len(res) != 0:
            return (res.index[0], "success") if clean else True

    return (None, "unknown") if clean else False


def _load_kb(kb_path: str, encode: Optional[str] = None) -> Any:
    """
    Load knowledge base from a specified path.
    """
    global DATA, NAME, ALPHA2, ALPHA3

    if kb_path == "default":
        DATA = pd.read_csv(DEFAULT_LANGUAGE_DATA_FILE, encoding="utf-8", dtype=str)
    else:
        DATA = pd.read_csv(kb_path, encoding=encode, dtype=str)
        DATA = clean_headers(DATA, case="kebab", report=False)  # to lowercase
        # check whether the format of the knowledge base is valid
        valid_formats = {"name", "alpha-2", "alpha-3"}
        for fmt in valid_formats:
            if fmt not in DATA.columns:
                raise KeyError(
                    "knowledge base does not follow the format, "
                    'it needs to contain "name", "alpha-2", and "alpha-3"'
                )

    # divide the dataset according to the first letter of each value, store the indices
    # e.g. {'a': [12, 36, 39], 'b': [15, 89], ...}
    NAME, ALPHA2, ALPHA3 = {}, {}, {}
    format_dicts = {"name": NAME, "alpha-2": ALPHA2, "alpha-3": ALPHA3}
    for fmt, fmt_dict in format_dicts.items():
        first_letters = DATA[fmt].str.lower().dropna().apply(lambda x: x[0])
        grps = DATA.groupby(first_letters).groups
        fmt_dict.update({k: list(v) for k, v in grps.items()})


def _convert_format_to_tuple(input_format: Union[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    """
    Converts a string input format to a tuple of allowed input formats and
    raises an error if an input format is not valid.
    """
    if isinstance(input_format, str):
        if input_format == "auto":
            return ("name", "alpha-2", "alpha-3")
        else:
            input_format = (input_format,)

    valid_input_formats = {"auto", "name", "alpha-2", "alpha-3"}
    for fmt in input_format:
        if fmt not in valid_input_formats:
            raise ValueError(
                f'input_format {fmt} is invalid, it needs to be one of "auto", '
                '"name", "alpha-2" or "alpha-3"'
            )
    if "auto" in input_format:
        return ("name", "alpha-2", "alpha-3")

    return input_format
