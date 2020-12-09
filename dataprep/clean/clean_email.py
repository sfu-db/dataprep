"""
Implement clean_email function
"""
# pylint: disable=too-many-boolean-expressions
import re
from typing import Any, Union

import dask.dataframe as dd
import dask
import pandas as pd

from .utils import NULL_VALUES, to_dask, NEARBYKEYS


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

DOMAIN_WHITELIST = ["localhost"]

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
    # rgentinian ISP domains
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

STATS = {
    "cleaned": 0,
    "null": 0,
    "overflow": 0,
    "bad_format": 0,
    "unknown": 0,
    "valid": 0,
}


def clean_email(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    split: bool = False,
    inplace: bool = False,
    pre_clean: bool = False,
    fix_domain: bool = False,
    report: bool = True,
    errors: str = "coerce",
) -> pd.DataFrame:
    """
    This function cleans emails
    Parameters
    ----------
    df
        pandas or Dask DataFrame
    column
        column name
    split
        If True, split a column containing username and domain name
        into one column for username and one column for domain name
    inplace
        If True, delete the given column with dirty data, else, create a new
        column with cleaned data.
    pre_clean
        If True, apply basic text clean(like removing whitespaces) before
        verifying and clean values.
    fix_domain
        If True, fix small typos in domain input
    report
        If True, generate cleaning report for Emails
    errors
        Specify ways to deal with broken value
        {'ignore', 'coerce', 'raise'}, default 'coerce'
        'raise': raise an exception when there is broken value
        'coerce': set invalid value to NaN
        'ignore': just return the initial input
    """
    # pylint: disable=too-many-arguments

    reset_stats()

    df = to_dask(df)

    # specify the metadata for dask apply
    meta = df.dtypes.to_dict()
    if split:
        meta.update(zip(("username", "domain"), (str, str)))
    else:
        meta[f"{column}_clean"] = str

    df = df.apply(
        format_email,
        args=(column, split, pre_clean, fix_domain, errors),
        axis=1,
        meta=meta,
    )

    if inplace:
        df = df.drop(columns=[column])

    df, nrows = dask.compute(df, df.shape[0])

    if report:
        report_email(nrows)

    return df


def fix_domain_name(dom: str) -> str:
    """
    Function to fix domain name with frequent typo
    """
    if dom.lower() not in DOMAINS:
        for i, curr_c in enumerate(dom):
            # two neighbor chars in reverse order
            if (dom[:i] + dom[i + 1 :]).lower() in DOMAINS:
                dom = (dom[:i] + dom[i + 1 :]).lower()
                break

            # missing single char
            for new_c in "abcdefghijklmnopqrstuvwxyz":
                if (dom[0 : i + 1] + new_c + dom[i + 1 :]).lower() in DOMAINS:
                    dom = (dom[0 : i + 1] + new_c + dom[i + 1 :]).lower()
                    break

            # redundant single char
            if i < len(dom) - 1:
                if (dom[:i] + dom[i + 1] + curr_c + dom[i + 2 :]).lower() in DOMAINS:
                    dom = (dom[:i] + dom[i + 1] + curr_c + dom[i + 2 :]).lower()
                    break

            # misspelled single char
            if curr_c in NEARBYKEYS:
                for c_p in NEARBYKEYS[curr_c]:
                    if (dom[:i] + c_p + dom[i + 1 :]).lower() in DOMAINS:
                        dom = (dom[:i] + c_p + dom[i + 1 :]).lower()
                        break
    return dom


def format_email(
    row: pd.Series,
    col: str,
    split: bool,
    pre_clean: bool,
    fix_domain: bool,
    errors: str,
) -> pd.Series:
    """
    Function to transform an email address into clean format
    """
    # pylint: disable=too-many-nested-blocks, too-many-locals,too-many-branches,too-many-statements,too-many-return-statements, too-many-arguments

    # pre-cleaning email text by removing all whitespaces
    if pre_clean:
        row[col] = re.sub(r"(\s|\u180B|\u200B|\u200C|\u200D|\u2060|\uFEFF)+", "", str(row[col]))

    valid_type = check_email(row[col], True)

    if valid_type != "valid":
        return not_email(row, col, split, valid_type, errors)

    user_part, domain_part = str(row[col]).rsplit("@", 1)

    # fix domain by detecting minor typos in advance
    if fix_domain:
        domain_part = fix_domain_name(domain_part)

    if split:
        row["username"], row["domain"] = (
            str(user_part).lower(),
            str(domain_part).lower(),
        )
    else:
        row[f"{col}_clean"] = str(user_part).lower() + "@" + str(domain_part).lower()

    return row


def validate_email(x: Union[str, pd.Series]) -> Union[bool, pd.Series]:
    """
    This function validates emails
    Parameters
    ----------
    x
        pandas Series of emails or email instance
    """

    if isinstance(x, pd.Series):
        return x.apply(check_email, clean=False)
    else:
        return check_email(x, False)


def check_email(val: Union[str, Any], clean: bool) -> Any:
    """
    Function to check whether a value is a valid email
    """
    # pylint: disable=too-many-return-statements, too-many-branches
    if val in NULL_VALUES:
        return "null" if clean else False

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


def not_email(row: pd.Series, col: str, split: bool, errtype: str, processtype: str) -> pd.Series:
    """
    Return result when value unable to be parsed
    """

    if processtype == "coerce":
        STATS[errtype] += 1
        if split:
            row["username"], row["domain"] = "None", "None"
        else:
            row[f"{col}_clean"] = "None"
    elif processtype == "ignore":
        STATS[errtype] += 1
        if split:
            row["username"], row["domain"] = row[col], "None"
        else:
            row[f"{col}_clean"] = row[col]
    elif processtype == "raise":
        raise ValueError(f"unable to parse value {row[col]}")
    else:
        raise ValueError("invalid error processing type")
    return row


def reset_stats() -> None:
    """
    Reset global statistics dictionary
    """
    STATS["cleaned"] = 0
    STATS["null"] = 0
    STATS["unknown"] = 0
    STATS["bad_format"] = 0
    STATS["overflow"] = 0


def report_email(nrows: int) -> None:
    """
    Describe what was done in the cleaning process
    """
    print("Email Cleaning Report:")
    if STATS["cleaned"] > 0:
        nclnd = STATS["cleaned"]
        pclnd = round(nclnd / nrows * 100, 2)
        print(f"\t{nclnd} values cleaned ({pclnd}%)")
    if STATS["bad_format"] > 0:
        n_bf = STATS["bad_format"]
        p_bf = round(n_bf / nrows * 100, 2)
        print(f"\t{n_bf} values with bad format ({p_bf}%)")
    if STATS["overflow"] > 0:
        n_of = STATS["overflow"]
        p_of = round(n_of / nrows * 100, 2)
        print(f"\t{n_of} values with too long username ({p_of}%)")
    if STATS["unknown"] > 0:
        n_uk = STATS["unknown"]
        p_uk = round(n_uk / nrows * 100, 2)
        print(f"\t{n_uk} values unable to be parsed ({p_uk}%)")
    nnull = STATS["null"] + STATS["unknown"] + STATS["bad_format"] + STATS["overflow"]
    pnull = round(nnull / nrows * 100, 2)
    ncorrect = nrows - nnull
    pcorrect = round(100 - pnull, 2)
    print(
        f"""Result contains {ncorrect} ({pcorrect}%) values in \
the correct format and {nnull} null values ({pnull}%)"""
    )
