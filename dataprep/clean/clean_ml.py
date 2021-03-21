"""
Implement clean_ml function
"""

# pylint: disable=too-many-arguments, too-many-locals, too-many-branches
from typing import Union, Dict, List, Tuple, Optional, Any

import dask.dataframe as dd
import pandas as pd
from .pipeline import Pipeline
from .utils import to_dask, NULL_VALUES


def clean_ml(
    training_df: Union[pd.DataFrame, dd.DataFrame],
    test_df: Union[pd.DataFrame, dd.DataFrame],
    target: str = "target",
    cat_imputation: str = "constant",
    cat_null_value: Optional[List[Any]] = None,
    fill_val: str = "missing_value",
    num_imputation: str = "mean",
    num_null_value: Optional[List[Any]] = None,
    cat_encoding: str = "one_hot",
    variance_threshold: bool = False,
    variance: float = 0.0,
    num_scaling: str = "standardize",
    include_operators: Optional[List[str]] = None,
    exclude_operators: Optional[List[str]] = None,
    customized_cat_pipeline: Optional[List[Dict[str, Any]]] = None,
    customized_num_pipeline: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function transforms an arbitrary tabular dataset
    into a format that's suitable for a typical ML application.

    Parameters
    ----------
    training_df
        Training dataframe. Pandas or Dask DataFrame.
    test_df
        Test dataframe. Pandas or Dask DataFrame.
    target
        Name of target column. String.
    cat_imputation
        The mode of imputation for categorical columns.
        If it equals to "constant",
            then all missing values are filled with `fill_val`.
        If it equals to "most_frequent",
            then all missing values are filled with most frequent value.
        If it equals to "drop",
            then all categorical columns with missing values will be dropped.
    cat_null_value
        Specified categorical null values which should be recognized.
    fill_val
        When cat_imputation = "constant",
            then all missing values are filled with `fill_val`.
    num_imputation
        The mode of imputation for numerical columns.
        If it equals to "mean",
            then all missing values are filled with mean value.
        If it equals to "median",
            then all missing values are filled with median value.
        If it equals to "most_frequent",
            then all missing values are filled with most frequent value.
        If it equals to "drop",
            then all numerical columns with missing values will be dropped.
    num_null_value
        Specified numerical null values which should be recognized.
    cat_encoding
        The mode of encoding categorical columns.
        If it equals to "one_hot", do one-hot encoding.
        If it equals to "no_encoding", nothing will be done.
    variance_threshold
        If it is True,
            then dropping numerical columns with variance less than `variance`.
    variance
        Variance value when variance_threshold = True.
    num_scaling
        The mode of scaling for numerical columns.
        If it equals to "standardize", do standardize for all numerical columns.
        If it equals to "minmax", do minmax scaling for all numerical columns.
        If it equals to "maxabs", do maxabs scaling for all numerical columns.
        If it equals to "no_scaling", nothing will be done.
    include_operators
        Components included for `clean_ml`, like "one_hot", "standardize", etc.
    exclude_operators
        Components excluded for `clean_ml`, like "one_hot", "standardize", etc.
    """
    if cat_null_value is None:
        cat_null_value = list(NULL_VALUES)
    if num_null_value is None:
        num_null_value = list(NULL_VALUES)
    training_df = to_dask(training_df)
    test_df = to_dask(test_df)
    col_names = []
    for label, _ in training_df.items():  # doctest: +SKIP
        col_names.append(label)
    for col_name in col_names:
        if col_name == target:
            continue
        if not customized_cat_pipeline is None and customized_num_pipeline is None:
            temp_training_df, temp_test_df = format_data_with_customized_cat(
                training_df[col_name].compute(),
                test_df[col_name].compute(),
                num_imputation,
                num_null_value,
                variance_threshold,
                variance,
                num_scaling,
                include_operators,
                exclude_operators,
                customized_cat_pipeline,
            )
        elif customized_cat_pipeline is None and not customized_num_pipeline is None:
            temp_training_df, temp_test_df = format_data_with_customized_num(
                training_df[col_name].compute(),
                test_df[col_name].compute(),
                cat_imputation,
                cat_null_value,
                fill_val,
                cat_encoding,
                include_operators,
                exclude_operators,
                customized_num_pipeline,
            )
        elif customized_cat_pipeline is None and customized_num_pipeline is None:
            temp_training_df, temp_test_df = format_data_with_default(
                training_df[col_name].compute(),
                test_df[col_name].compute(),
                cat_imputation,
                cat_null_value,
                fill_val,
                num_imputation,
                num_null_value,
                cat_encoding,
                variance_threshold,
                variance,
                num_scaling,
                include_operators,
                exclude_operators,
            )
        elif not customized_cat_pipeline is None and not customized_num_pipeline is None:
            temp_training_df, temp_test_df = format_data_with_customized_cat_and_num(
                training_df[col_name].compute(),
                test_df[col_name].compute(),
                include_operators,
                exclude_operators,
                customized_cat_pipeline,
                customized_num_pipeline,
            )
        if temp_training_df.values.size > 0:
            training_df[col_name] = temp_training_df
            test_df[col_name] = temp_test_df
        else:
            training_df = training_df.drop(columns=[col_name])
            test_df = test_df.drop(columns=[col_name])
    return training_df.compute(), test_df.compute()


def format_data_with_customized_cat(
    training_row: dd.Series,
    test_row: dd.Series,
    num_imputation: str = "mean",
    num_null_value: Optional[List[Any]] = None,
    variance_threshold: bool = False,
    variance: float = 0.0,
    num_scaling: str = "standardize",
    include_operators: Optional[List[str]] = None,
    exclude_operators: Optional[List[str]] = None,
    customized_cat_pipeline: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[dd.Series, dd.Series]:
    """
    This function transforms an arbitrary tabular dataset
    into a format that's suitable for a typical ML application.
    Customized categorical pipeline and related parameters should be provided by users

    Parameters
    ----------
    training_row
        One column of training dataset. Dask Series.
    test_row
        One column of test dataset. Dask Series.
    num_imputation
        The mode of imputation for numerical columns.
        If it equals to "mean",
            then all missing values are filled with mean value.
        If it equals to "median",
            then all missing values are filled with median value.
        If it equals to "most_frequent",
            then all missing values are filled with most frequent value.
        If it equals to "drop",
            then all numerical columns with missing values will be dropped.
    num_null_value
        Specified numerical null values which should be recognized.
    variance_threshold
        If it is True, then dropping numerical columns with variance less than `variance`.
    variance
        Variance value when variance_threshold = True.
    num_scaling
        The mode of scaling for numerical columns.
        If it equals to "standardize", do standardize for all numerical columns.
        If it equals to "minmax", do minmax scaling for all numerical columns.
        If it equals to "maxabs", do maxabs scaling for all numerical columns.
        If it equals to "no_scaling", nothing will be done.
    include_operators
        Components included for `clean_ml`, like "one_hot", "standardize", etc.
    exclude_operators
        Components excluded for `clean_ml`, like "one_hot", "standardize", etc.
    customized_cat_pipeline
        User-specified pipeline managing categorical columns.
    """

    cat_pipe_info: Dict[str, Any] = {}
    cat_pipeline = []

    if not customized_cat_pipeline is None:
        for item in customized_cat_pipeline:
            (component_key,) = item
            cat_pipeline.append(component_key)
        cat_pipe_info["cat_pipeline"] = cat_pipeline
        for item in customized_cat_pipeline:
            (component_key,) = item
            if (
                not exclude_operators is None
                and item[component_key]["operator"] in exclude_operators
            ) or (
                not include_operators is None
                and item[component_key]["operator"] not in include_operators
            ):
                cat_pipe_info[component_key] = None
                continue
            for key in item[component_key]:
                if key == "operator":
                    cat_pipe_info[component_key] = item[component_key][key]
                else:
                    cat_pipe_info[key] = item[component_key][key]

    num_pipe_info: Dict[str, Any] = {}
    if variance_threshold:
        num_pipe_info["num_pipeline"] = [
            "num_imputation",
            "variance_threshold",
            "num_scaling",
        ]
        num_pipe_info["variance_threshold"] = variance_threshold
        num_pipe_info["variance"] = variance
    else:
        num_pipe_info["num_pipeline"] = ["num_imputation", "num_scaling"]
    if (not exclude_operators is None and num_imputation in exclude_operators) or (
        not include_operators is None and num_imputation not in include_operators
    ):
        num_pipe_info["num_imputation"] = None
        num_pipe_info["num_null_value"] = None
    else:
        num_pipe_info["num_imputation"] = num_imputation
        num_pipe_info["num_null_value"] = num_null_value

    if (not exclude_operators is None and num_scaling in exclude_operators) or (
        not include_operators is None and num_scaling not in include_operators
    ):
        num_pipe_info["num_scaling"] = None
    else:
        num_pipe_info["num_scaling"] = num_scaling
    if num_scaling == "no_scaling":
        num_pipe_info["num_scaling"] = None
    else:
        num_pipe_info["num_scaling"] = num_scaling

    clean_pipeline = Pipeline(cat_pipe_info, num_pipe_info)
    training_result, test_result = clean_pipeline.fit_transform(training_row, test_row)
    return training_result, test_result


def format_data_with_customized_num(
    training_row: dd.Series,
    test_row: dd.Series,
    cat_imputation: str = "constant",
    cat_null_value: Optional[List[Any]] = None,
    fill_val: str = "missing_value",
    cat_encoding: str = "one_hot",
    include_operators: Optional[List[str]] = None,
    exclude_operators: Optional[List[str]] = None,
    customized_num_pipeline: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[dd.Series, dd.Series]:
    """
    This function transforms an arbitrary tabular dataset
    into a format that's suitable for a typical ML application.
    Customized numerical pipeline and related parameters should be provided by users

    Parameters
    ----------
    training_row
        One column of training dataset. Dask Series.
    test_row
        One column of test dataset. Dask Series.
    cat_imputation
        The mode of imputation for categorical columns.
        If it equals to "constant",
            then all missing values are filled with `fill_val`.
        If it equals to "most_frequent",
            then all missing values are filled with most frequent value.
        If it equals to "drop",
            then all categorical columns with missing values will be dropped.
    cat_null_value
        Specified categorical null values which should be recognized.
    fill_val
        When cat_imputation = "constant", then all missing values are filled with `fill_val`.
    cat_encoding
        The mode of encoding categorical columns.
        If it equals to "one_hot", do one-hot encoding.
        If it equals to "no_encoding", nothing will be done.
    include_operators
        Components included for `clean_ml`, like "one_hot", "standardize", etc.
    exclude_operators
        Components excluded for `clean_ml`, like "one_hot", "standardize", etc.
    customized_num_pipeline
        User-specified pipeline managing numerical columns.
    """
    cat_pipe_info: Dict[str, Any] = {}
    cat_pipe_info["cat_pipeline"] = ["cat_imputation", "cat_encoding"]
    # cat_pipe_info['cat_pipeline'] = ['cat_imputation']
    if (not exclude_operators is None and cat_imputation in exclude_operators) or (
        not include_operators is None and cat_imputation not in include_operators
    ):
        cat_pipe_info["cat_imputation"] = None
        cat_pipe_info["cat_null_value"] = None
        cat_pipe_info["fill_val"] = None
    else:
        cat_pipe_info["cat_imputation"] = cat_imputation
        cat_pipe_info["cat_null_value"] = cat_null_value
        cat_pipe_info["fill_val"] = fill_val

    if (not exclude_operators is None and cat_encoding in exclude_operators) or (
        not include_operators is None and cat_encoding not in include_operators
    ):
        cat_pipe_info["cat_encoding"] = None
    else:
        cat_pipe_info["cat_encoding"] = cat_encoding
    if cat_encoding == "no_encoding":
        cat_pipe_info["cat_encoding"] = None
    else:
        cat_pipe_info["cat_encoding"] = cat_encoding

    num_pipe_info: Dict[str, Any] = {}
    num_pipeline = []
    if not customized_num_pipeline is None:
        for item in customized_num_pipeline:
            (component_key,) = item
            num_pipeline.append(component_key)
        num_pipe_info["num_pipeline"] = num_pipeline
        for item in customized_num_pipeline:
            (component_key,) = item
            if (
                not exclude_operators is None
                and item[component_key]["operator"] in exclude_operators
            ) or (
                not include_operators is None
                and item[component_key]["operator"] not in include_operators
            ):
                num_pipe_info[component_key] = None
                continue
            for key in item[component_key]:
                if key == "operator":
                    num_pipe_info[component_key] = item[component_key][key]
                else:
                    num_pipe_info[key] = item[component_key][key]

    clean_pipeline = Pipeline(cat_pipe_info, num_pipe_info)
    training_result, test_result = clean_pipeline.fit_transform(training_row, test_row)
    return training_result, test_result


def format_data_with_default(
    training_row: dd.Series,
    test_row: dd.Series,
    cat_imputation: str = "constant",
    cat_null_value: Optional[List[Any]] = None,
    fill_val: str = "missing_value",
    num_imputation: str = "mean",
    num_null_value: Optional[List[Any]] = None,
    cat_encoding: str = "one_hot",
    variance_threshold: bool = True,
    variance: float = 0.0,
    num_scaling: str = "standardize",
    include_operators: Optional[List[str]] = None,
    exclude_operators: Optional[List[str]] = None,
) -> Tuple[dd.Series, dd.Series]:
    """
    This function transforms an arbitrary tabular dataset
    into a format that's suitable for a typical ML application.
    No customized pipeline should be provided. Use default pipeline.

    Parameters
    ----------
    training_row
        One column of training dataset. Dask Series.
    test_row
        One column of test dataset. Dask Series.
    cat_imputation
        The mode of imputation for categorical columns.
        If it equals to "constant",
            then all missing values are filled with `fill_val`.
        If it equals to "most_frequent",
            then all missing values are filled with most frequent value.
        If it equals to "drop",
            then all categorical columns with missing values will be dropped.
    cat_null_value
        Specified categorical null values which should be recognized.
    fill_val
        When cat_imputation = "constant", then all missing values are filled with `fill_val`.
    num_imputation
        The mode of imputation for numerical columns.
        If it equals to "mean",
            then all missing values are filled with mean value.
        If it equals to "median",
            then all missing values are filled with median value.
        If it equals to "most_frequent",
            then all missing values are filled with most frequent value.
        If it equals to "drop",
            then all numerical columns with missing values will be dropped.
    num_null_value
        Specified numerical null values which should be recognized.
    cat_encoding
        The mode of encoding categorical columns.
        If it equals to "one_hot", do one-hot encoding.
        If it equals to "no_encoding", nothing will be done.
    variance_threshold
        If it is True, then dropping numerical columns with variance less than `variance`.
    variance
        Variance value when variance_threshold = True.
    num_scaling
        The mode of scaling for numerical columns.
        If it equals to "standardize", do standardize for all numerical columns.
        If it equals to "minmax", do minmax scaling for all numerical columns.
        If it equals to "maxabs", do maxabs scaling for all numerical columns.
        If it equals to "no_scaling", nothing will be done.
    include_operators
        Components included for `clean_ml`, like "one_hot", "standardize", etc.
    exclude_operators
        Components excluded for `clean_ml`, like "one_hot", "standardize", etc.
    """
    cat_pipe_info: Dict[str, Any] = {}
    cat_pipe_info["cat_pipeline"] = ["cat_imputation", "cat_encoding"]
    # cat_pipe_info['cat_pipeline'] = ['cat_imputation']
    if (not exclude_operators is None and cat_imputation in exclude_operators) or (
        not include_operators is None and cat_imputation not in include_operators
    ):
        cat_pipe_info["cat_imputation"] = None
        cat_pipe_info["cat_null_value"] = None
        cat_pipe_info["fill_val"] = None
    else:
        cat_pipe_info["cat_imputation"] = cat_imputation
        cat_pipe_info["cat_null_value"] = cat_null_value
        cat_pipe_info["fill_val"] = fill_val
    if (not exclude_operators is None and cat_encoding in exclude_operators) or (
        not include_operators is None and cat_encoding not in include_operators
    ):
        cat_pipe_info["cat_encoding"] = None
    else:
        cat_pipe_info["cat_encoding"] = cat_encoding
    if cat_encoding == "no_encoding":
        cat_pipe_info["cat_encoding"] = None
    else:
        cat_pipe_info["cat_encoding"] = cat_encoding

    num_pipe_info: Dict[str, Any] = {}
    if variance_threshold:
        num_pipe_info["num_pipeline"] = [
            "num_imputation",
            "variance_threshold",
            "num_scaling",
        ]
        num_pipe_info["variance_threshold"] = variance_threshold
        num_pipe_info["variance"] = variance
    else:
        num_pipe_info["num_pipeline"] = ["num_imputation", "num_scaling"]
    # num_pipe_info['num_pipeline'] = ['num_imputation', 'num_scaling']
    if (not exclude_operators is None and num_imputation in exclude_operators) or (
        not include_operators is None and num_imputation not in include_operators
    ):
        num_pipe_info["num_imputation"] = None
        num_pipe_info["num_null_value"] = None
    else:
        num_pipe_info["num_imputation"] = num_imputation
        num_pipe_info["num_null_value"] = num_null_value

    if (not exclude_operators is None and num_scaling in exclude_operators) or (
        not include_operators is None and num_scaling not in include_operators
    ):
        num_pipe_info["num_scaling"] = None
    else:
        num_pipe_info["num_scaling"] = num_scaling
    if num_scaling == "no_scaling":
        num_pipe_info["num_scaling"] = None
    else:
        num_pipe_info["num_scaling"] = num_scaling

    clean_pipeline = Pipeline(cat_pipe_info, num_pipe_info)
    training_result, test_result = clean_pipeline.fit_transform(training_row, test_row)
    return training_result, test_result


def format_data_with_customized_cat_and_num(
    training_row: dd.Series,
    test_row: dd.Series,
    include_operators: Optional[List[str]] = None,
    exclude_operators: Optional[List[str]] = None,
    customized_cat_pipeline: Optional[List[Dict[str, Any]]] = None,
    customized_num_pipeline: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[dd.Series, dd.Series]:
    """
    This function transforms an arbitrary tabular dataset
    into a format that's suitable for a typical ML application.
    Both customized pipeline managing categorical columns and numerical columns should be provided.

    Parameters
    ----------
    training_row
        One column of training dataset. Dask Series.
    test_row
        One column of test dataset. Dask Series.
    include_operators
        Components included for `clean_ml`, like "one_hot", "standardize", etc.
    exclude_operators
        Components excluded for `clean_ml`, like "one_hot", "standardize", etc.
    customized_cat_pipeline
        User-specified pipeline managing categorical columns.
    customized_num_pipeline
        User-specified pipeline managing numerical columns.
    """
    cat_pipe_info: Dict[str, Any] = {}
    cat_pipeline = []
    if not customized_cat_pipeline is None:
        for item in customized_cat_pipeline:
            (component_key,) = item
            cat_pipeline.append(component_key)
        cat_pipe_info["cat_pipeline"] = cat_pipeline
        for item in customized_cat_pipeline:
            (component_key,) = item
            if (
                not exclude_operators is None
                and item[component_key]["operator"] in exclude_operators
            ) or (
                not include_operators is None
                and item[component_key]["operator"] not in include_operators
            ):
                cat_pipe_info[component_key] = None
                continue
            for key in item[component_key]:
                if key == "operator":
                    cat_pipe_info[component_key] = item[component_key][key]
                else:
                    cat_pipe_info[key] = item[component_key][key]

    num_pipe_info: Dict[str, Any] = {}
    num_pipeline = []
    if not customized_num_pipeline is None:
        for item in customized_num_pipeline:
            (component_key,) = item
            num_pipeline.append(component_key)
        num_pipe_info["num_pipeline"] = num_pipeline
        for item in customized_num_pipeline:
            (component_key,) = item
            if (
                not exclude_operators is None
                and item[component_key]["operator"] in exclude_operators
            ) or (
                not include_operators is None
                and item[component_key]["operator"] not in include_operators
            ):
                num_pipe_info[component_key] = None
                continue
            for key in item[component_key]:
                if key == "operator":
                    num_pipe_info[component_key] = item[component_key][key]
                else:
                    num_pipe_info[key] = item[component_key][key]

    clean_pipeline = Pipeline(cat_pipe_info, num_pipe_info)
    training_result, test_result = clean_pipeline.fit_transform(training_row, test_row)
    return training_result, test_result
