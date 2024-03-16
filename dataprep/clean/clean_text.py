"""
Clean a DataFrame column containing text data.
"""

import re
import string
from functools import partial, update_wrapper
from typing import Any, Callable, Dict, List, Optional, Set, Union
from unicodedata import normalize

import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..assets.english_stopwords import english_stopwords
from .utils import NULL_VALUES, to_dask

REGEX_BRACKETS = {
    "angle": re.compile(r"(\<)[^<>]*(\>)"),
    "curly": re.compile(r"(\{)[^{}]*(\})"),
    "round": re.compile(r"(\()[^()]*(\))"),
    "square": re.compile(r"(\[)[^\[\]]*(\])"),
}
REGEX_DIGITS = re.compile(r"\d+")
REGEX_DIGITS_BLOCK = re.compile(r"\b\d+\b")
REGEX_HTML = re.compile(r"<[A-Za-z/][^>]*>|&(?:[a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
REGEX_PUNCTUATION = re.compile(rf"[{re.escape(string.punctuation)}]")
REGEX_URL = re.compile(r"(?:https?://|www\.)\S+")
REGEX_WHITESPACE = re.compile(r"[\n\t]|[ ]{2,}")


def clean_text(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    pipeline: Optional[List[Dict[str, Any]]] = None,
    stopwords: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Clean text data in a DataFrame column.

    Read more in the :ref:`User Guide <clean_text_user_guide>`.

    Parameters
    ----------
        df
            A pandas or Dask DataFrame containing the data to be cleaned.
        column
            The name of the column containing text data.
        pipeline
            A list of cleaning functions to be applied to the column. If None,
            use the default pipeline. See the :ref:`User Guide <clean_text_custom_pipeline>`
            for more information on customizing the pipeline.

            (default: None)
        stopwords
            A set of words to be removed from the column. If None, use NLTK's
            stopwords.

            (default: None)

    Examples
    --------
    Clean a column of text data using the default pipeline.

    >>> df = pd.DataFrame({"text": ["This show was an amazing, fresh & innovative idea in the \
70's when it first aired."]})
    >>> clean_text(df, 'text')
                                                 text
    0  show amazing fresh innovative idea first aired
    """
    df = to_dask(df)

    pipe = _get_default_pipeline(stopwords) if not pipeline else _get_custom_pipeline(pipeline)

    for func in pipe:
        df[column] = df[column].apply(func, meta=object)

    df = df.compute()

    return df


def default_text_pipeline() -> List[Dict[str, Any]]:
    """
    Return a list of dictionaries representing the functions in the default pipeline.
    Use as a template for creating a custom pipeline.

    Read more in the :ref:`User Guide <clean_text_user_guide>`.

    Examples
    --------
    >>> default_text_pipeline()
    [{'operator': 'fillna'}, {'operator': 'lowercase'}, {'operator': 'remove_digits'},
    {'operator': 'remove_html'}, {'operator': 'remove_urls'}, {'operator': 'remove_punctuation'},
    {'operator': 'remove_accents'}, {'operator': 'remove_stopwords', 'parameters':
    {'stopwords': None}}, {'operator': 'remove_whitespace'}]
    """
    return [
        {"operator": "fillna"},
        {"operator": "lowercase"},
        {"operator": "remove_digits"},
        {"operator": "remove_html"},
        {"operator": "remove_urls"},
        {"operator": "remove_punctuation"},
        {"operator": "remove_accents"},
        {"operator": "remove_stopwords", "parameters": {"stopwords": None}},
        {"operator": "remove_whitespace"},
    ]


def _get_default_pipeline(
    stopwords: Optional[Set[str]] = None,
) -> List[Callable[..., Any]]:
    """
    Return a list of functions defining the default pipeline.
    """
    return [
        _fillna,
        _lowercase,
        _remove_digits,
        _remove_html,
        _remove_urls,
        _remove_punctuation,
        _remove_accents,
        lambda x: _remove_stopwords(x, stopwords),
        _remove_whitespace,
    ]


def _get_custom_pipeline(pipeline: List[Dict[str, Any]]) -> List[Callable[..., Any]]:
    """
    Return a list of functions defining a custom pipeline.
    """
    func_dict = _get_func_dict()
    custom_pipeline: List[Callable[..., Any]] = []

    for component in pipeline:
        # Check whether function is built in or user defined
        operator = (
            func_dict[component["operator"]]
            if isinstance(component["operator"], str)
            else component["operator"]
        )
        # Append the function to the pipeline
        # If parameters are specified, create a partial function to lock in
        # the values and prevent them from being overwritten in subsequent loops
        if "parameters" in component:
            custom_pipeline.append(_wrapped_partial(operator, component["parameters"]))
        else:
            custom_pipeline.append(operator)

    return custom_pipeline


def _get_func_dict() -> Dict[str, Callable[..., Any]]:
    """
    Return a mapping of strings to function names.
    """
    return {
        "fillna": _fillna,
        "lowercase": _lowercase,
        "sentence_case": _sentence_case,
        "title_case": _title_case,
        "uppercase": _uppercase,
        "remove_accents": _remove_accents,
        "remove_bracketed": _remove_bracketed,
        "remove_digits": _remove_digits,
        "remove_html": _remove_html,
        "remove_prefixed": _remove_prefixed,
        "remove_punctuation": _remove_punctuation,
        "remove_stopwords": _remove_stopwords,
        "remove_urls": _remove_urls,
        "remove_whitespace": _remove_whitespace,
        "replace_bracketed": _replace_bracketed,
        "replace_digits": _replace_digits,
        "replace_prefixed": _replace_prefixed,
        "replace_punctuation": _replace_punctuation,
        "replace_stopwords": _replace_stopwords,
        "replace_text": _replace_text,
        "replace_urls": _replace_urls,
    }


def _fillna(text: Any, value: Any = np.nan) -> Any:
    """
    Replace all null values with NaN (default) or the supplied value.
    """
    return value if text in NULL_VALUES else str(text)


def _lowercase(text: Any) -> Any:
    """
    Convert all characters to lowercase.
    """
    return str(text).lower() if pd.notna(text) else text


def _sentence_case(text: Any) -> Any:
    """
    Convert first character to uppercase and remaining to lowercase.
    """
    return str(text).capitalize() if pd.notna(text) else text


def _title_case(text: Any) -> Any:
    """
    Convert first character of each word to uppercase and remaining to lowercase.
    """
    return str(text).title() if pd.notna(text) else text


def _uppercase(text: Any) -> Any:
    """
    Convert all characters to uppercase.
    """
    return str(text).upper() if pd.notna(text) else text


def _remove_accents(text: Any) -> Any:
    """
    Remove accents (diacritic marks).
    """
    return (
        normalize("NFD", str(text)).encode("ascii", "ignore").decode("ascii")
        if pd.notna(text)
        else text
    )


def _remove_bracketed(text: Any, brackets: Union[str, Set[str]], inclusive: bool = True) -> Any:
    """
    Remove text between brackets.

    Parameters
    ----------
    brackets
        The bracket style.
            - "angle": <>
            - "curly": {}
            - "round": ()
            - "square": []

    inclusive
        If True (default), remove the brackets along with the text in between.
        Otherwise, keep the brackets.
    """
    if pd.isna(text):
        return text

    text = str(text)
    value = "" if inclusive else r"\g<1>\g<2>"
    if isinstance(brackets, set):
        for bracket in brackets:
            text = re.sub(REGEX_BRACKETS[bracket], value, text)
    else:
        text = re.sub(REGEX_BRACKETS[brackets], value, text)

    return text


def _remove_digits(text: Any) -> Any:
    """
    Remove all digits.
    """
    return re.sub(REGEX_DIGITS, "", str(text)) if pd.notna(text) else text


def _remove_html(text: Any) -> Any:
    """
    Remove HTML tags.
    """
    return re.sub(REGEX_HTML, "", str(text)) if pd.notna(text) else text


def _remove_prefixed(text: Any, prefix: Union[str, Set[str]]) -> Any:
    """
    Remove substrings that start with the prefix(es).
    """
    if pd.isna(text):
        return text

    text = str(text)
    if isinstance(prefix, set):
        for pre in prefix:
            text = re.sub(rf"{pre}\S+", "", text)
    else:
        text = re.sub(rf"{prefix}\S+", "", text)

    return text


def _remove_punctuation(text: Any) -> Any:
    """
    Remove punctuation marks.
    """
    return re.sub(REGEX_PUNCTUATION, " ", str(text)) if pd.notna(text) else text


def _remove_stopwords(text: Any, stopwords: Optional[Set[str]] = None) -> Any:
    """
    Remove a set of words from the text.
    If `stopwords` is None (default), use NLTK's stopwords.
    """
    if pd.isna(text):
        return text

    stopwords = english_stopwords if not stopwords else stopwords
    return " ".join(word for word in str(text).split() if word.lower() not in stopwords)


def _remove_urls(text: Any) -> Any:
    """
    Remove URLS.
    """
    return re.sub(REGEX_URL, "", str(text)) if pd.notna(text) else text


def _remove_whitespace(text: Any) -> Any:
    """
    Remove extra spaces along with tabs and newlines.
    """
    return re.sub(REGEX_WHITESPACE, " ", str(text)).strip() if pd.notna(text) else text


def _replace_bracketed(
    text: Any, brackets: Union[str, Set[str]], value: str, inclusive: bool = True
) -> Any:
    """
    Replace text between brackets with the value.

    Parameters
    ----------
    brackets
        The bracket style.
            - "angle": <>
            - "curly": {}
            - "round": ()
            - "square": []

    value
        The value to replace the text between the brackets.

    inclusive
        If True (default), replace the brackets with the new text as well.
        Otherwise, keep the brackets.
    """
    if pd.isna(text):
        return text

    text = str(text)
    value = value if inclusive else rf"\g<1>{value}\g<2>"
    if isinstance(brackets, set):
        for bracket in brackets:
            text = re.sub(REGEX_BRACKETS[bracket], value, text)
    else:
        text = re.sub(REGEX_BRACKETS[brackets], value, text)

    return text


def _replace_digits(text: Any, value: str, block: Optional[bool] = True) -> Any:
    """
    Replace all digits with the value. If `block` is True (default),
    only replace blocks of digits.
    """
    if pd.isna(text):
        return text

    return (
        re.sub(REGEX_DIGITS_BLOCK, value, str(text))
        if block
        else re.sub(REGEX_DIGITS, value, str(text))
    )


def _replace_prefixed(text: Any, prefix: Union[str, Set[str]], value: str) -> Any:
    """
    Replace all substrings starting with the prefix(es) with the value.
    """
    if pd.isna(text):
        return text

    text = str(text)
    if isinstance(prefix, set):
        for pre in prefix:
            text = re.sub(rf"{pre}\S+", value, text)
    else:
        text = re.sub(rf"{prefix}\S+", value, text)

    return text


def _replace_punctuation(text: Any, value: str) -> Any:
    """
    Replace all punctuation marks with the value.
    """
    return re.sub(REGEX_PUNCTUATION, value, str(text)) if pd.notna(text) else text


def _replace_stopwords(text: Any, value: str, stopwords: Optional[Set[str]] = None) -> Any:
    """
    Replace a set of words in the text with the value.
    If `stopwords` is None (default), use NLTK's stopwords.
    """
    if pd.isna(text):
        return text

    stopwords = english_stopwords if not stopwords else stopwords
    return " ".join(word if word.lower() not in stopwords else value for word in str(text).split())


def _replace_text(text: Any, value: Dict[str, str], block: Optional[bool] = True) -> Any:
    """
    Replace a sequence of characters with another according to the value mapping.
    If `block` is True (default), only replace standalone blocks of the sequence.
    """
    if pd.isna(text):
        return text

    text = str(text)
    for old_value, new_value in value.items():
        text = (
            re.sub(rf"\b{old_value}\b", new_value, text, flags=re.IGNORECASE)
            if block
            else re.sub(rf"{old_value}", new_value, text, flags=re.IGNORECASE)
        )

    return text


def _replace_urls(text: Any, value: str) -> Any:
    """
    Replace all URLs with the value.
    """
    return re.sub(REGEX_URL, value, str(text)) if pd.notna(text) else text


def _wrapped_partial(
    func: Callable[..., Callable[..., Any]], params: Dict[str, Any]
) -> Callable[..., Callable[..., Any]]:
    """
    Return a partial function with a name and a doc attribute.
    """
    partial_func = partial(func, **params)
    update_wrapper(partial_func, func)
    return partial_func
