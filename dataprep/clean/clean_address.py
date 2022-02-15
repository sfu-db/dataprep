"""
Clean and validate a DataFrame column containing US street addresses.
"""

import re
from operator import itemgetter
from typing import Any, Dict, List, Tuple, Union

import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd

from ..progress_bar import ProgressBar
from .address_utils import (
    ABBR_STATES,
    FULL_PREFIX,
    FULL_STATES,
    KEYWORDS,
    PREFIXES,
    SUFFIXES,
    TAG_MAPPING,
    tag,
    RepeatedLabelError,
)
from .utils import NULL_VALUES, create_report_new, to_dask


def clean_address(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    output_format: str = "(building) house_number street_prefix_abbr "
    "street_name street_suffix_abbr, apartment, city, state_abbr zipcode",
    must_contain: Tuple[str, ...] = ("house_number", "street_name"),
    split: bool = False,
    inplace: bool = False,
    errors: str = "coerce",
    report: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Clean and standardize US street addresses.

    Read more in the :ref:`User Guide <address_userguide>`.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing addresses.
    output_format
        The output format can be specified using the following keywords.
            - 'house_number': '1234'
            - 'street_prefix_abbr': 'N', 'S', 'E', or 'W'
            - 'street_prefix_full': 'North', 'South', 'East', or 'West'
            - 'street_name': 'Main'
            - 'street_suffix_abbr': 'St', 'Ave'
            - 'street_suffix_full': 'Street', 'Avenue'
            - 'apartment': 'Apt 1'
            - 'building': 'Staples Center'
            - 'city': 'Los Angeles'
            - 'state_abbr': 'CA'
            - 'state_full': 'California'
            - 'zipcode': '57903'

        The output_format can contain '\\\\t' characters to specify how to split the output into
        columns.

        (default: '(building) house_number street_prefix_abbr street_name street_suffix_abbr,
        apartment, city, state_abbr zipcode')
    must_contain
        A tuple containing parts of the address that must be included for the address to be
        successfully cleaned.

            - 'house_number': '1234'
            - 'street_prefix': 'N', 'North'
            - 'street_name': 'Main'
            - 'street_suffix': 'St', 'Avenue'
            - 'apartment': 'Apt 1'
            - 'building': 'Staples Center'
            - 'city': 'Los Angeles'
            - 'state': 'CA', 'California'
            - 'zipcode': '57903'

        (default: ('house_number', 'street_name'))
    split
        If True, each component of the address specified by the output_format parameter will be put
        into it's own column.

        For example if output_format = "house_number street_name" and split = True, then there
        will be one column for house_number and another for street_name.

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
        If True, output the summary report. Otherwise, no report is outputted.

        (default: True)
    progress
        If True, display a progress bar.

        (default: True)

    Examples
    --------
    Clean addresses and add the house number and street name to separate columns.

    >>> df = pd.DataFrame({'address': ['123 pine avenue', '1234 w main st 57033']})
    >>> clean_address(df, 'address', output_format='house_number \\t street_name')
    Address Cleaning Report:
            2 values cleaned (100.0%)
    Result contains 2 (100.0%) values in the correct format and 0 null values (0.0%)
        address                house_number      street_name
    0    123 pine avenue           123             Pine
    1   1234 w main st 57033       1234            Main
    """
    # pylint: disable=too-many-arguments

    df = to_dask(df)

    df["clean_code_tup"] = df[column].map_partitions(
        lambda srs: [_format_address(x, output_format, must_contain, split, errors) for x in srs],
        meta=object,
    )

    headers = _get_column_names(output_format, split)

    # if there's only one column in the output, name it f"{column}_clean". Otherwise,
    # get names from the output_format
    if len(headers) == 1:
        df = df.assign(
            _temp_=df["clean_code_tup"].map(itemgetter(0)),
            _code_=df["clean_code_tup"].map(itemgetter(1)),
        )
        df = df.rename(columns={"_temp_": f"{column}_clean"})
    else:
        assignments = {
            headers[i]: df["clean_code_tup"].map(itemgetter(i), meta=(headers[i], str))
            for i in range(len(headers))
        }
        assignments["_code_"] = df["clean_code_tup"].map(
            itemgetter(len(headers)), meta=("_code_", int)
        )
        df = df.assign(**assignments)

    stats = df["_code_"].value_counts(sort=False)
    df = df.drop(columns=["clean_code_tup", "_code_"])

    if inplace:
        df = df.drop(columns=column)

    with ProgressBar(minimum=1, disable=not progress):
        df, stats = dask.compute(df, stats)

    if report:
        create_report_new("Address", stats, errors)
    return df


def validate_address(
    x: Union[str, pd.Series], must_contain: Tuple[str, ...] = ("house_number", "street_name")
) -> Union[bool, pd.Series]:
    """
    Validate US street addresses.

    Read more in the :ref:`User Guide <address_userguide>`.

    Parameters
    ----------
    x
        pandas Series of addresses or a string containing an address.
    must_contain
        A tuple containing parts of the address that must be included for the
        address to be successfully cleaned.

            - 'house_number': '1234'
            - 'street_prefix': 'N', 'North'
            - 'street_name': 'Main'
            - 'street_suffix': 'St', 'Avenue'
            - 'apartment': 'Apt 1'
            - 'building': 'Staples Center'
            - 'city': 'Los Angeles'
            - 'state': 'CA', 'California'
            - 'zipcode': '57903'

        (default: ('house_number', 'street_name'))

    Examples
    --------

    >>> df = pd.DataFrame({'address': ['123 pine avenue', 'NULL']})
    >>> validate_address(df['address'])
    0    True
    1    False
    Name: address, dtype: bool
    """

    if isinstance(x, pd.Series):
        return x.apply(_check_address, args=(must_contain, False))

    return _check_address(x, must_contain, False)


def _format_address(
    address: Any, output_format: str, must_contain: Tuple[str, ...], split: bool, errors: str
) -> Any:
    """
    Function to transform an address instance into the desired format

    The last component of the returned tuple contains a code indicating how the
    input value was changed:
        0 := the value is null
        1 := the value could not be parsed
        2 := the value is cleaned and the cleaned value is DIFFERENT than the input value
        3 := the value is cleaned and is THE SAME as the input value (no transformation)
    """
    address_dict, status = _check_address(address, must_contain, True)
    outputs = _address_dict_to_string(address_dict, output_format, split)

    if status == "null":
        return (np.nan,) * len(_get_column_names(output_format, split)) + (0,)

    elif status == "unknown":
        if errors == "raise":
            raise ValueError(f"unable to parse value {address}")
        return tuple(
            np.nan if not value else value if errors == "ignore" else np.nan for value in outputs
        ) + (1,)

    if len(outputs) == 1 and address == outputs[0]:
        code = 3
    else:
        code = 2
    return tuple(np.nan if not value else value for value in outputs) + (code,)


def _check_address(address: Any, must_contain: Tuple[str, ...], clean: bool) -> Any:
    """
    Finds the index of the given country in the DATA dataframe.

    Parameters
    ----------
    address_str
        address value to be cleaned
    must_contain
        A tuple containing parts of the address that must be included for the
         address to be successfully cleaned
    clean
        If True, a tuple (index, status) is returned.
        If False, the function returns True/False to be used by the validate address function.
    """
    if address in NULL_VALUES:
        return (None, "null") if clean else False

    address = re.sub(r"[().]", "", str(address))

    try:
        address, _ = tag(address, TAG_MAPPING)

    except RepeatedLabelError:
        return (None, "unknown") if clean else False

    status = _check_status(address, must_contain)

    if status:
        return (address, "success") if clean else True

    return (address, "unknown") if clean else False


def _check_status(address_dict: Dict[str, str], must_contain: Tuple[str, ...]) -> bool:
    """
    Returns True if all address attributes in must_contain are present in
    address_dict, otherwise returns False.
    """
    return all(address_part in address_dict for address_part in must_contain)


def _address_dict_to_string(address: Dict[str, str], output_format: str, split: bool) -> List[str]:
    """
    Returns a list of address parts, in a format specified by output_format.
    Each item in the list will be added to the final dataframe in it's own column.
    """

    address_items = _clean_address_parts(address)

    # add tabs between each attribute if split is True
    if split:
        output_format = "\t".join(output_format.split())

    # add a comma after the street name if there is no street suffix
    # in address_items
    if "street_suffix_abbr" not in address_items and not split:
        output_format = output_format.replace("street_name", "street_name,")

    # first split output_format into each column of the final output
    # for each column split it into attributes and add the corresponding
    # cleaned part of the address to the output for each attribute
    output = []
    columns = output_format.split("\t")
    current_part = ""

    for column in columns:
        for output_attr in column.split():
            for address_attr, address_val in address_items.items():
                idx = output_attr.find(address_attr)
                if idx != -1 and address_val is not None:
                    # include parts at the beginning and end ie. include parens
                    # if (building) is in output_str. Only if split is False
                    end = idx + len(address_attr)
                    if split:
                        current_part += f" {address_val}"
                    else:
                        current_part += f" {output_attr[:idx]}{address_val}{output_attr[end:]}"
        output.append(current_part.strip(" ,").replace(" # ", " "))
        current_part = ""

    return output


def _clean_address_parts(address_dict: Dict[str, str]) -> Dict[str, str]:
    """
    Apply basic cleaning functions to parts of the address.
    """
    if not address_dict:
        return {}

    result_dict: Dict[str, str] = {}

    cleaning_funcs = {
        "house_number": _clean_house_number,
        "street_prefix": _clean_prefix,
        "street_name": _clean_street,
        "street_suffix": _clean_suffix,
        "state": _clean_state,
        "city": _clean_city,
        "building": _clean_building,
        "apartment": _clean_apartment,
        "zipcode": _clean_zip,
    }
    for address_attr, value in address_dict.items():
        if address_attr in cleaning_funcs:
            cleaning_funcs[address_attr](result_dict, value)

    return result_dict


def _get_column_names(output_format: str, split: bool) -> List[str]:
    """
    returns the column names that will be present in the final dataframe,
    based on the output_format
    """
    if not split:
        return [name.strip() for name in output_format.split("\t")]

    output_tokens = output_format.split()
    headers = []
    for output_part in output_tokens:
        for attr in KEYWORDS:
            if attr in output_part:
                headers.append(attr)
                break
    return headers


def _clean_prefix(result_dict: Dict[str, str], prefix: str) -> None:
    """
    Adds a cleaned full prefix and cleaned abbreviated prefix to result_dict,
    based on the value of street prefix
    """
    prefix_abbr = PREFIXES.get(prefix.lower())
    if prefix_abbr:
        result_dict["street_prefix_abbr"] = prefix_abbr
        result_dict["street_prefix_full"] = FULL_PREFIX[prefix_abbr]


def _clean_suffix(result_dict: Dict[str, str], suffix: str) -> None:
    """
    Adds a cleaned full suffix and cleaned abbreviated suffix to result_dict,
    based on the value of the street suffix
    """
    suffix_tuple = SUFFIXES.get(suffix.upper())
    if suffix_tuple:
        result_dict["street_suffix_abbr"] = suffix_tuple[0].capitalize() + "."
        result_dict["street_suffix_full"] = suffix_tuple[1].capitalize()


def _clean_state(result_dict: Dict[str, str], state: str) -> None:
    """
    Adds a cleaned full state and cleaned abbreviated state to result_dict,
    based on the value of the state
    """
    if state.title() in FULL_STATES:
        result_dict["state_abbr"] = FULL_STATES[state.title()]
        result_dict["state_full"] = state.title()
    if state.upper() in ABBR_STATES:
        result_dict["state_abbr"] = state.upper()
        result_dict["state_full"] = ABBR_STATES[state.upper()]


def _clean_city(result_dict: Dict[str, str], city: str) -> None:
    """
    capitalize each word of city and add it to result_dict
    """
    result_dict["city"] = city.title()


def _clean_house_number(result_dict: Dict[str, str], house_number: str) -> None:
    """
    adds house_number to result_dict
    """
    result_dict["house_number"] = house_number


def _clean_building(result_dict: Dict[str, str], building: str) -> None:
    """
    capitalize each word of building and add it to result_dict
    """
    result_dict["building"] = building.title()


def _clean_zip(result_dict: Dict[str, str], zipcode: str) -> None:
    """
    adds zipcode to result_dict
    """
    result_dict["zipcode"] = zipcode


def _clean_street(result_dict: Dict[str, str], street: str) -> None:
    """
    capitalize each word of the street name and add it to result_dict,
    except keep the number suffixes 'st', 'nd', 'rd', 'th' lower case
    """
    if re.match(r"\d+[st|nd|rd|th]", street, flags=re.IGNORECASE):
        result_dict["street_name"] = street.lower()
    else:
        result_dict["street_name"] = street.title()


def _clean_apartment(result_dict: Dict[str, str], apartment: str) -> None:
    """
    capitalize each word of the apartment and add it to result_dict
    """
    result_dict["apartment"] = apartment.title()
