"""
Module containing plot_outlier function.
"""


import dask.dataframe as dd

from ..intermediate import Intermediate

DEFAULT_PARTITIONS = 1


def _calc_num_outlier(df: dd.DataFrame, col_x: str) -> Intermediate:
    """
    calculate outliers based on the MAD method for numerical values.
    :param df: the input dataframe
    :param col_x: the column of df (univariate outlier detection)
    :return: dict(index: value) of outliers
    """
    data_df = dd.from_dask_array(df[col_x].to_dask_array(), columns=["data"])
    median = data_df["data"].quantile(0.5)
    MAD = abs(data_df["data"] - median).quantile(0.5)  # pylint: disable=invalid-name
    data_df["z_score"] = (0.6745 * (data_df["data"] - median)) / MAD
    res_df = data_df[data_df["z_score"] > 3.5].drop("z_score", axis=1)
    result = {"outliers_index": list(res_df["data"].index.compute())}
    raw_data = {"df": df, "col_x": col_x}
    return Intermediate(result, raw_data)


def _calc_cat_outlier(df: dd.DataFrame, col_x: str, threshold: int = 1) -> Intermediate:
    """
    calculate outliers based on the threshold for categorical values.
    :param df: the input dataframe
    :param col_x: the column of df (univariate outlier detection)
    :return: dict(index: value) of outliers
    """
    groups = df.groupby([col_x]).size()
    result = {"outlier_index": list(groups[groups <= threshold].index.compute())}
    raw_data = {"df": df, "col_x": col_x, "threshold": threshold}
    return Intermediate(result, raw_data)
