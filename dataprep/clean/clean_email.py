"""
Clean and validate a DataFrame column containing email addresses.
"""

import re
from operator import itemgetter
from typing import Any, Union

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..progress_bar import ProgressBar
from .utils import NEARBYKEYS, NULL_VALUES, create_report_new, to_dask

USER_REGEX = re.compile(
    # dot-atom
    r"(^[-!#$%&'*+/=?^_`{}|~0-9A-Z]+" r"(\.[-!#$%&'*+/=?^_`{}|~0-9A-Z]+)*$"
    # quoted-string
    r'|^"([\001-\010\013\014\016-\037!#-\[\]-\177]|' r"""\\[\001-\011\013\014\016-\177])*"$)""",
    re.IGNORECASE,
)

DOMAIN_REGEX = re.compile(
    # domain
    r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+" r"(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?$)"
    # literal form, ipv4 address (SMTP 4.1.3)
    r"|^\[(25[0-5]|2[0-4]\d|[0-1]?\d?\d)" r"(\.(25[0-5]|2[0-4]\d|[0-1]?\d?\d)){3}\]$",
    re.IGNORECASE,
)

DOMAIN_WHITELIST = {"localhost"}

DOMAINS = {
    # Default domains included#
    "aol.com",
    "att.net",
    "comcast.net",
    "facebook.com",
    "gmail.com",
    "gmx.com",
    "googlemail.com",
    "google.com",
    "hotmail.com",
    "hotmail.co.uk",
    "mac.com",
    "me.com",
    "mail.com",
    "msn.com",
    "live.com",
    "sbcglobal.net",
    "verizon.net",
    "yahoo.com",
    "yahoo.co.uk",
    # Other global domains
    "email.com",
    "fastmail.fm",
    "games.com",
    "gmx.net",
    "hush.com",
    "hushmail.com",
    "icloud.com",
    "iname.com",
    "inbox.com",
    "lavabit.com",
    "love.com",
    "outlook.com",
    "pobox.com",
    "protonmail.ch",
    "protonmail.com",
    "tutanota.de",
    "tutanota.com",
    "tutamail.com",
    "tuta.io",
    "keemail.me",
    "rocketmail.com",
    "safe-mail.net",
    "wow.com",
    "ygm.com",
    "ymail.com",
    "zoho.com",
    "yandex.com",
    # United States ISP domains
    "bellsouth.net",
    "charter.net",
    "cox.net",
    "earthlink.net",
    "juno.com",
    # British ISP domains
    "btinternet.com",
    "virginmedia.com",
    "blueyonder.co.uk",
    "freeserve.co.uk",
    "live.co.uk",
    "ntlworld.com",
    "o2.co.uk",
    "orange.net",
    "sky.com",
    "talktalk.co.uk",
    "tiscali.co.uk",
    "virgin.net",
    "wanadoo.co.uk",
    "bt.com",
    # Domains used in Asia
    "sina.com",
    "sina.cn",
    "qq.com",
    "naver.com",
    "hanmail.net",
    "daum.net",
    "nate.com",
    "yahoo.co.jp",
    "yahoo.co.kr",
    "yahoo.co.id",
    "yahoo.co.in",
    "yahoo.com.sg",
    "yahoo.com.ph",
    "163.com",
    "yeah.net",
    "126.com",
    "21cn.com",
    "aliyun.com",
    "foxmail.com",
    # French ISP domains
    "hotmail.fr",
    "live.fr",
    "laposte.net",
    "yahoo.fr",
    "wanadoo.fr",
    "orange.fr",
    "gmx.fr",
    "sfr.fr",
    "neuf.fr",
    "free.fr",
    # German ISP domains
    "gmx.de",
    "hotmail.de",
    "live.de",
    "online.de",
    "t-online.de",
    "web.de",
    "yahoo.de",
    # Italian ISP domains
    "libero.it",
    "virgilio.it",
    "hotmail.it",
    "aol.it",
    "tiscali.it",
    "alice.it",
    "live.it",
    "yahoo.it",
    "email.it",
    "tin.it",
    "poste.it",
    "teletu.it",
    # Russian ISP domains
    "mail.ru",
    "rambler.ru",
    "yandex.ru",
    "ya.ru",
    "list.ru",
    # Belgian ISP domains
    "hotmail.be",
    "live.be",
    "skynet.be",
    "voo.be",
    "tvcablenet.be",
    "telenet.be",
    # Argentinian ISP domains
    "hotmail.com.ar",
    "live.com.ar",
    "yahoo.com.ar",
    "fibertel.com.ar",
    "speedy.com.ar",
    "arnet.com.ar",
    # Domains used in Mexico
    "yahoo.com.mx",
    "live.com.mx",
    "hotmail.es",
    "hotmail.com.mx",
    "prodigy.net.mx",
    # Domains used in Canada
    "yahoo.ca",
    "hotmail.ca",
    "bell.net",
    "shaw.ca",
    "sympatico.ca",
    "rogers.com",
    # Domains used in Brazil
    "yahoo.com.br",
    "hotmail.com.br",
    "outlook.com.br",
    "uol.com.br",
    "bol.com.br",
    "terra.com.br",
    "ig.com.br",
    "itelefonica.com.br",
    "r7.com",
    "zipmail.com.br",
    "globo.com",
    "globomail.com",
    "oi.com.br",
}


def clean_email(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    remove_whitespace: bool = False,
    fix_domain: bool = False,
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    report: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean and standardize email address.

    Read more in the :ref:`User Guide <email_userguide>`.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing email addresses.
    remove_whitespace
        If True, remove all whitespace from the input value before
        verifying and cleaning it.

        (default: False)
    fix_domain
        If True, for invalid email domains, try to fix it using 4 strategies:
            - Swap neighboring characters.
            - Add a single character.
            - Remove a single character.
            - Swap each character with its nearby keys on the qwerty keyboard.

        The first valid domain found will be returned.

        (default: False)
    split
        If True, split a column containing email addresses into one column
        for the usernames and another column for the domains.

        (default: False)
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
        If True, output the summary report. Else, no report is outputted.

        (default: True)

    progress
        If True, display a progress bar.

        (default: True)

    Examples
    --------

    >>> df = pd.DataFrame({'email': ['Abc.example.com', 'Abc@example.com', 'H ELLO@hotmal.COM']})
    >>> clean_email(df, 'email')
    Email Cleaning Report:
        2 values with bad format (66.67%)
    Result contains 1 (33.33%) values in the correct format and 2 null values (66.67%)
                email      email_clean
    0    Abc.example.com              NaN
    1    Abc@example.com  abc@example.com
    2  H ELLO@hotmal.COM              NaN
    """
    # pylint: disable=too-many-arguments

    # check if the parameters are of correct processing types and values
    if not isinstance(df, (pd.DataFrame, dd.DataFrame)):
        raise ValueError("df is invalid, it needs to be a pandas or Dask DataFrame.")

    if not isinstance(column, str):
        raise ValueError(f"column {column} is invalid.")

    if not isinstance(remove_whitespace, bool):
        raise ValueError(
            f"remove_whitespace {remove_whitespace} is invalid, it needs to be True or False."
        )

    if not isinstance(fix_domain, bool):
        raise ValueError(f"fix_domain {fix_domain} is invalid, it needs to be True or False.")

    if not isinstance(inplace, bool):
        raise ValueError(f"inplace {inplace} is invalid, it needs to be True or False.")

    if not isinstance(report, bool):
        raise ValueError(f"report {report} is invalid, it needs to be True or False.")

    if errors not in {"coerce", "ignore", "raise"}:
        raise ValueError(
            f'errors {errors} is invalid, it needs to be "coerce", "ignore", or "raise".'
        )

    # convert to dask
    df = to_dask(df)

    # To clean, create a new column "clean_code_tup" which contains
    # the cleaned values and code indicating how the initial value was
    # changed in a tuple. Then split the column of tuples and count the
    # amount of different codes to produce the report
    df["clean_code_tup"] = df[column].map_partitions(
        lambda srs: [_format_email(x, split, remove_whitespace, fix_domain, errors) for x in srs],
        meta=object,
    )

    if split:
        df = df.assign(
            username=df["clean_code_tup"].map(itemgetter(0)),
            domain=df["clean_code_tup"].map(itemgetter(1)),
            _code_=df["clean_code_tup"].map(itemgetter(2), meta=("_code_", object)),
        )
    else:
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

    # output a report describing the result of clean_email
    if report:
        create_report_new("email", stats, errors)

    return df


def validate_email(x: Union[str, pd.Series]) -> Union[bool, pd.Series]:
    """
    Validate email addresses.

    Read more in the :ref:`User Guide <email_userguide>`.

    Parameters
    ----------
    x
        pandas Series of emails or a string containing an email.

    Examples
    --------

    >>> validate_email('Abc.example@com')
    False
    >>> df = pd.DataFrame({'email': ['abc.example.com', 'HELLO@HOTMAIL.COM']})
    >>> validate_email(df['email'])
    0    False
    1     True
    Name: email, dtype: bool
    """

    if isinstance(x, pd.Series):
        return x.apply(_check_email, clean=False)
    return _check_email(x, False)


def _format_email(
    val: Any, split: bool, remove_whitespace: bool, fix_domain: bool, errors: str
) -> Any:
    """
    Function to transform an email address into a clean format.
    """
    # pre-cleaning email text by removing all whitespaces
    if remove_whitespace:
        val = re.sub(r"(\s|\u180B|\u200B|\u200C|\u200D|\u2060|\uFEFF)+", "", str(val))

    valid_type = _check_email(val, True)

    if valid_type != "valid":
        return _not_email(val, split, valid_type, errors)

    user_part, domain_part = str(val).lower().rsplit("@", 1)

    # fix domain by detecting minor typos in advance
    if fix_domain:
        domain_part = _fix_domain_name(domain_part)

    code = 3 if f"{user_part}@{domain_part}" == val else 2

    if split:
        return user_part, domain_part, code

    return f"{user_part}@{domain_part}", code


def _check_email(val: Any, clean: bool) -> Any:
    """
    Function to check whether a value is a valid email.
    """
    # pylint: disable=too-many-return-statements
    if val in NULL_VALUES:
        return "null" if clean else False

    val = str(val)
    if "@" not in val:
        return "bad_format" if clean else False

    user_part, domain_part = val.rsplit("@", 1)

    if not USER_REGEX.match(user_part):
        return "bad_format" if clean else False

    if len(user_part.encode("utf-8")) > 64:
        return "overflow" if clean else False

    if domain_part not in DOMAIN_WHITELIST and not DOMAIN_REGEX.match(domain_part):
        # Try for possible IDN domain-part
        try:
            domain_part = domain_part.encode("idna").decode("ascii")
            if DOMAIN_REGEX.match(domain_part):
                return "valid" if clean else True
            return "unknown" if clean else False
        except UnicodeError:
            return "unknown" if clean else False

    return "valid" if clean else True


def _fix_domain_name(dom: str) -> str:
    """
    Function to fix domain name with frequent typo
    """
    if dom not in DOMAINS:
        for i, curr_c in enumerate(dom):
            # two neighbor chars in reverse order
            if dom[:i] + dom[i + 1 :] in DOMAINS:
                return dom[:i] + dom[i + 1 :]

            # missing single char
            for new_c in "abcdefghijklmnopqrstuvwxyz":
                if dom[: i + 1] + new_c + dom[i + 1 :] in DOMAINS:
                    return dom[: i + 1] + new_c + dom[i + 1 :]

            # redundant single char
            if i < len(dom) - 1:
                if dom[:i] + dom[i + 1] + curr_c + dom[i + 2 :] in DOMAINS:
                    return dom[:i] + dom[i + 1] + curr_c + dom[i + 2 :]

            # misspelled single char
            if curr_c in NEARBYKEYS:
                for c_p in NEARBYKEYS[curr_c]:
                    if dom[:i] + c_p + dom[i + 1 :] in DOMAINS:
                        return dom[:i] + c_p + dom[i + 1 :]
    return dom


def _not_email(val: Any, split: bool, errtype: str, processtype: str) -> Any:
    """
    Return result when value unable to be parsed.
    """
    if processtype == "coerce":
        if split:
            return (np.nan, np.nan, 0) if errtype == "null" else (np.nan, np.nan, 1)
        return (np.nan, 0) if errtype == "null" else (np.nan, 1)
    elif processtype == "ignore":
        if split:
            return (val, np.nan, 0) if errtype == "null" else (val, np.nan, 1)
        return (val, 0) if errtype == "null" else (val, 1)
    elif processtype == "raise":
        raise ValueError(f"unable to parse value {val}")
    else:
        raise ValueError("invalid error processing type")
