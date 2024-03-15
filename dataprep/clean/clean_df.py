"""
Conduct a set of operations that would be useful for
cleaning and standardizing a full Pandas DataFrame.
"""

# pylint: disable-msg=relative-beyond-top-level
# pylint: disable-msg=cyclic-import

import operator
from typing import Dict, Any, Union, Tuple

import numpy as np
import pandas as pd

from pandas.api.types import infer_dtype

# from ..eda.progress_bar import ProgressBar

from .clean_email import validate_email
from .clean_country import validate_country
from .clean_phone import validate_phone
from .clean_url import validate_url
from .clean_lat_long import validate_lat_long
from .clean_ip import validate_ip
from .clean_address import validate_address
from .clean_headers import clean_headers
from .utils import NULL_VALUES


def clean_df(
    df: pd.DataFrame,
    clean_header: bool = True,
    data_type_detection: str = "semantic",
    standardize_missing_values: str = "fill",
    downcast_memory: bool = True,
    remove_duplicate_entries: bool = False,
    report: bool = True,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
    """
    This function cleans the whole DataFrame with a sequence of useful operations.

    Parameters
    ----------
    df
        A Pandas DataFrame containing the data to be cleaned.
    clean_header
        If True, call the clean_headers() function to clean the column names.

        (default: True)
    data_type_detection
        The desired way to check data types. Must from: {'semantic', 'atomic', 'none'}.
        * If ’semantic’, then perform a column-wise semantic and atomic type detection.
        * If 'atomic', then return the column data types from Python atomic ones.
        * If 'none', then no results will be returned.

        (default: 'semantic')
    standardize_missing_values
        The desired way to standardize missing values. Must from: {‘fill’, ‘remove’, ’ignore’}.
        * If ’fill’, then fill detected missing values using np.NaN or pd.NaT (for "Date" type).
        * If ‘remove’, then rows with any missing values will be removed.
        * If ‘ignore’, then no action will be taken.

        (default: 'fill')
    remove_duplicate_entries
        If True, remove the repetitive data entries (rows) and report how many entries are removed.

        (default: False)
    downcast_memory
        If True, downcast the memory of the DataFrame by using subtypes in numerical columns;
        for categorical types, downcast from `object` to `category`.

        (default: True)
    report
        If True, output the summary report. Otherwise, no report is outputted.

        (default: True)

    Examples
    --------
    >>> df = pd.DataFrame({"Name": ["Abby", None], "Age": [12, np.nan], "wEIGht": [32.5, np.nan],
                   "Email Address": ["abby@gmail.com", None], "Country of Birth": ["CA", None]})
    >>> clean_df(df)
    Data Type Detection Report:
        These data types are supported by DataPrep to clean: ['email', 'country']
    Column Headers Cleaning Report:
        5 values cleaned (100.0%)
    Downcast Memory Report:
        Memory reducted from 475 to 451. New size: (94.95%)
    (                 semantic_data_type atomic_data_type
    Name                         string           string
    Age                        floating         floating
    wEIGht                     floating         floating
    Email Address                 email           string
    Country of Birth            country           string,
        name   age  w_ei_ght   email_address country_of_birth
    0  Abby    12      32.5  abby@gmail.com               CA
    1  <NA>  <NA>       NaN            <NA>             <NA>)
    """
    # pylint: disable=too-many-arguments
    # pylint: disable-msg=too-many-locals
    # pylint:disable=too-many-branches
    # type: ignore

    if data_type_detection not in {"semantic", "atomic", "none"}:
        raise ValueError(
            f"data_type_detection {data_type_detection} is invalid, "
            'it needs to be "semantic", "atomic" or "none"'
        )

    if standardize_missing_values not in {"fill", "remove", "ignore"}:
        raise ValueError(
            f"standardize_missing_values {standardize_missing_values} is invalid, "
            'it needs to be "fill", "remove" or "ignore"'
        )

    # Data type detection
    if data_type_detection != "none":
        df_datatype_detection = _infer_data_type_df(df, data_type_detection)

    # Clean headers
    if clean_header:
        orig_columns = df.columns.astype(str).tolist()
        df = clean_headers(df, report=False)

    # Clean missing values and duplicate rows; keep track of how many rows are removed.
    orig_nrows = df.shape[0]

    df = _standardize_missing_values_df(df, standardize_missing_values)
    if remove_duplicate_entries:
        df = df.drop_duplicates()

    cleaned_nrows = df.shape[0]

    # Downcast memory; keep track of how much memory is saved
    orig_size = df.memory_usage(deep=True).sum()
    if downcast_memory:
        df = _downcast_memory(df)

    # Prepare for reports depending on which cleaning steps are performed
    if report:
        # Output which semantic data types are supported by Dataprep to further clean
        if data_type_detection == "semantic":
            semantic_data_type_values_list = df_datatype_detection[
                "semantic_data_type"
            ].values.tolist()
            stats_datatype = {
                "email": "email" in semantic_data_type_values_list,
                "phone": "phone" in semantic_data_type_values_list,
                "country": "country" in semantic_data_type_values_list,
                "coordinate": "coordinate" in semantic_data_type_values_list,
                "ip": "ip" in semantic_data_type_values_list,
                "URL": "URL" in semantic_data_type_values_list,
                "address": "address" in semantic_data_type_values_list,
            }

            _create_report_df(stats_datatype, len(df.columns), "datatype")
        if clean_header:
            new_columns = df.columns.astype(str).tolist()
            cleaned = [
                1 if new_columns[i] != orig_columns[i] else 0 for i in range(len(orig_columns))
            ]
            stats_header = {"cleaned": sum(cleaned)}

            _create_report_df(stats_header, len(df.columns), "header")
        if standardize_missing_values == "remove" or remove_duplicate_entries:
            stats_duplicate = {"cleaned": orig_nrows - cleaned_nrows}

            _create_report_df(stats_duplicate, orig_nrows, "duplicate")
        if downcast_memory:
            cleaned_size = df.memory_usage(deep=True).sum()
            stats_memory = {"cleaned": cleaned_size}

            _create_report_df(stats_memory, orig_size, "memory")

    df = df.reset_index(drop=True)

    if data_type_detection != "none":
        return df_datatype_detection, df
    else:
        return df


def _check_valid_values(data: str) -> bool:
    """
    Function to check if data is valid or belongs to NULL_VALUES.
    if valid, return True. Otherwise return False.

    """
    not_valid = data in NULL_VALUES or pd.isna(data)
    return not not_valid


def _check_null_values(data: str) -> bool:
    """
    Reversed version of function _check_valid_values() to check if data is invalid.

    """
    not_valid = data in NULL_VALUES or pd.isna(data)
    return bool(not_valid)


def _infer_semantic_data_type(column: pd.Series) -> Any:
    """
    Function to infer the semantic data type for a column.

    We extract non-NA values and use a subset to infer. A valid semantic data type will be returned
    if at least half of the entries in the subset match. The order of determination is:
        1. if the header contains "lat" or "long", try to infer the "coordinate" data type.
        2. numeric ("integer" or "floating") or "boolean".
        3. string (email, country, phone, etc.) for which we desinged validating functions.
        4. other types ("date" or "string") from the best infer by Pandas.

    Parameters
    ----------
    column
        A column of a Pandas DataFrame to be checked.
    """

    # extract non-NA values and a subset to infer the data type
    column_not_na = column[column.apply(_check_valid_values, 0)]
    sample_size = (
        column_not_na.size if column_not_na.size <= 100 else min(int(0.1 * column_not_na.size), 500)
    )
    column_not_na_subset = column_not_na.sample(n=sample_size, random_state=1)

    # 1. For geographic coordinates: lat and long
    lat_lon = ["lat", "Lat", "LAT", "lon", "Lon", "LON"]
    if any(x in column.name for x in lat_lon):
        lat_long_infer_count = sum(
            pd.Series(validate_lat_long(column_not_na_subset, lat_long=False, lat=True)).tolist()
        )
        # return if at least half of the entries match
        if lat_long_infer_count > (column_not_na_subset.size / 2):
            return "coordinate"

    # 2. For numeric ("integer", "floating") or boolean data types
    if infer_dtype(column_not_na_subset) != "string":
        return infer_dtype(column_not_na_subset)

    # 3. For string and semantic data types (email, country, phone, etc.)
    default_infer_dtype = infer_dtype(column_not_na_subset)

    semantic_data_type_dic = {"email": 0, "country": 0, "phone": 0, "ip": 0, "URL": 0, "address": 0}

    semantic_data_type_dic["email"] = sum(pd.Series(validate_email(column_not_na_subset)).tolist())
    semantic_data_type_dic["country"] = sum(
        pd.Series(validate_country(column_not_na_subset)).tolist()
    )
    semantic_data_type_dic["phone"] = sum(pd.Series(validate_phone(column_not_na_subset)).tolist())
    semantic_data_type_dic["ip"] = sum(pd.Series(validate_ip(column_not_na_subset)).tolist())
    semantic_data_type_dic["URL"] = sum(pd.Series(validate_url(column_not_na_subset)).tolist())
    semantic_data_type_dic["address"] = sum(
        pd.Series(validate_address(column_not_na_subset)).tolist()
    )

    if all(value == 0 for value in semantic_data_type_dic.values()):
        # no semantic data types match, return the default one
        return default_infer_dtype
    # else, return the best inferred semantic data type
    best_infer = max(semantic_data_type_dic.items(), key=operator.itemgetter(1))[0]
    best_infer_count = max(semantic_data_type_dic.items(), key=operator.itemgetter(1))[1]
    return best_infer if best_infer_count > (column_not_na_subset.size / 2) else default_infer_dtype


def _infer_atomic_data_type(column: pd.Series) -> Any:
    """
    Function to infer the atomic data type for a column.

    Parameters
    ----------
    column
        A Column of a Pandas DataFrame to be checked.
    """

    return infer_dtype(column[column.apply(_check_valid_values, 0)])


def _infer_data_type_df(df: pd.DataFrame, data_type_detection: str) -> pd.DataFrame:
    """
    Function to infer the best data type for a whole data frame.

    Parameters
    ----------
    df
        A Pandas dataframe containing the data.
    data_type_detection
        The desired way to check data types.
        * If ’semantic’, then perform a column-wise semantic and atomic type detection.
        * If 'atomic', then return the best inferred atomic data type from Python default ones.
        * If 'none', then no results will be returned.
    """

    if data_type_detection == "semantic":
        atomic_data_types_df = df.apply(_infer_atomic_data_type, 0).to_frame()
        semantic_data_types_df = df.apply(_infer_semantic_data_type, 0).to_frame()
        combined_data_types_df = pd.concat([semantic_data_types_df, atomic_data_types_df], axis=1)
        combined_data_types_df.columns = ["semantic_data_type", "atomic_data_type"]
        return combined_data_types_df
    # else, perform the atomic detection only
    atomic_data_types_df = df.apply(_infer_atomic_data_type, 0).to_frame()
    atomic_data_types_df.columns = ["atomic_data_type"]
    return atomic_data_types_df


def _fill_missing_values_column(column: pd.Series) -> None:
    """
    Function to standardize the missing values for a column, depending on its data type detection:
    1. For date, change to pd.NaT
    3. For any other data type, change to np.nan

    Parameters
    ----------
    column
        Columns of a Pandas DataFrame to be standardized.
    """

    if infer_dtype(column) == "date":
        column[column.apply(_check_null_values, 0)] = pd.NaT
        column = pd.to_datetime(column)
    else:
        column[column.apply(_check_null_values, 0)] = np.nan


def _standardize_missing_values_df(
    df: pd.DataFrame, standardize_missing_values: str
) -> pd.DataFrame:
    """
    Function to standardize the missing values for a whole dataframe.

    Parameters
    ----------
    df
        Pandas Dataframe to be standarized.
    standardize_missing_values {‘fill’, ‘remove’, ’ignore’}
        * If ’fill’, then all detected missing values will be set as np.NaN.
        * If ‘remove’, then any rows with missing values will be deleted.
        * If ‘ignore’, then no action will be taken.
    """
    if standardize_missing_values == "fill":
        df.apply(_fill_missing_values_column, 0)
        df = df.convert_dtypes()
    elif standardize_missing_values == "remove":
        df.apply(_fill_missing_values_column, 0)
        df = df.convert_dtypes()
        df.dropna(how="any", inplace=True)
    elif standardize_missing_values == "ignore":
        pass

    return df


def _create_report_df(stats: Dict[str, Any], old_stat: Any, option: str) -> None:
    """
    Describe what was done in the cleaning process.
    """
    if option not in {"datatype", "header", "duplicate", "memory"}:
        raise ValueError(
            f"clean_df report option {option} is invalid, "
            'it needs to be "datatype", "header", "duplicate", or "memory"'
        )

    if option == "datatype":
        print("Data Type Detection Report:")
        print(
            "\tThese data types are supported by DataPrep to clean:",
            [k for k, v in stats.items() if v],
        )
    if option == "header":
        print("Column Headers Cleaning Report:")
        if stats["cleaned"] > 0:
            nclnd = stats["cleaned"]
            pclnd = round(nclnd / old_stat * 100, 2)
            print(f"\t{nclnd} values cleaned ({pclnd}%)")
        else:
            print("No Headers Cleaned.")
    if option == "duplicate":
        print("Number of Entries Cleaning Report:")
        if stats["cleaned"] > 0:
            nclnd = stats["cleaned"]
            pclnd = round(nclnd / old_stat * 100, 2)
            print(f"\t{nclnd} entries dropped ({pclnd}%)")
        else:
            print("No Duplicated Entries Cleaned.")
    if option == "memory":
        print("Downcast Memory Report:")
        if stats["cleaned"] > 0:
            nclnd = stats["cleaned"]
            pclnd = round(nclnd / old_stat * 100, 2)
            print(f"\tMemory reducted from {old_stat} to {nclnd}. New size: ({pclnd}%)")
        else:
            print("Downcast Memory not Performed.")


def _downcast_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to downcast the memory size of the DataFrame by using subtypes in
    numerical columns; for categorical types, downcast from `object` to `category`.
    """
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    for i, j in enumerate(types):
        if "Int" in str(j) or "int" in str(j):
            if (
                df[cols[i]].min() > np.iinfo(np.int8).min
                and df[cols[i]].max() < np.iinfo(np.int8).max
            ):
                df[cols[i]] = df[cols[i]].astype("Int8")
            elif (
                df[cols[i]].min() > np.iinfo(np.int16).min
                and df[cols[i]].max() < np.iinfo(np.int16).max
            ):
                df[cols[i]] = df[cols[i]].astype("Int16")
            elif (
                df[cols[i]].min() > np.iinfo(np.int32).min
                and df[cols[i]].max() < np.iinfo(np.int32).max
            ):
                df[cols[i]] = df[cols[i]].astype("Int32")
            else:
                df[cols[i]] = df[cols[i]].astype("Int64")
        # Avoid forcing "float16" because it loses precision and is fragile
        elif "Float" in str(j) or "float" in str(j):
            if (
                df[cols[i]].min() > np.finfo(np.float16).min
                and df[cols[i]].max() < np.finfo(np.float32).max
            ):
                df[cols[i]] = pd.to_numeric(df[cols[i]], downcast="float")
            else:
                df[cols[i]] = df[cols[i]].astype("Float64")
        elif "Object" in str(j) or "object" in str(j):
            df[cols[i]] = df[cols[i]].astype("category")

    return df
