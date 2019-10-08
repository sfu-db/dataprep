"""
    This module implements the intermediates computation
    for plot_correlation(df) function.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import dask
import numpy as np
import pandas as pd
from bokeh.io import show
from bokeh.models.widgets import Tabs
from bokeh.plotting import Figure
from scipy.stats import kendalltau

from ...utils import DataType, _drop_non_numerical_columns, get_type
from ..common import Intermediate
from .visualization import (
    _vis_correlation_pd,
    _vis_correlation_pd_x_k,
    _vis_correlation_pd_x_y_k,
    _vis_cross_table,
)


def merge_dicts(*dict_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    :param dict_args: The dictionary we want to merge
    :return: merged dictionary

    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def _calc_kendall(data_a: np.ndarray, data_b: np.ndarray) -> Any:
    """
    :param data_a: the column of data frame
    :param data_b: the column of data frame
    :return: A float value which indicates the
    correlation of two numpy array
    """
    kendallta, _ = kendalltau(data_a, data_b)
    return kendallta


def _value_to_rank(array: np.ndarray) -> pd.Series:
    """
    :param array: an column of data frame whose
    type is numpy
    :return: translate value to order rank
    """
    array_ranks = pd.Series(array).rank()
    return array_ranks.values


def _calc_correlation_pd(df: pd.DataFrame,) -> Any:  # pylint: disable=too-many-locals
    """
    :param pd_data_frame: the pandas data_frame for which plots
    are calculated for each column.
    :return: An object to encapsulate the
    intermediate results.
    """
    method_list = ["pearson", "spearman", "kendall"]
    cal_matrix = df.values.T
    result = {}
    for method in method_list:
        if method == "pearson":
            cov_xy = np.cov(cal_matrix)
            std_xy = np.sqrt(np.diag(cov_xy))
            corr_matrix = cov_xy / std_xy[:, None] / std_xy[None, :]
            result["corr_p"] = corr_matrix
        elif method == "spearman":
            matrix_row, _ = np.shape(cal_matrix)
            for i in range(matrix_row):
                cal_matrix[i, :] = _value_to_rank(cal_matrix[i, :])
            cov_xy = np.cov(cal_matrix)
            std_xy = np.sqrt(np.diag(cov_xy))
            corr_matrix = cov_xy / std_xy[:, None] / std_xy[None, :]
            result["corr_s"] = corr_matrix
        elif method == "kendall":
            matrix_row, _ = np.shape(cal_matrix)
            corr_matrix = np.ones(shape=(matrix_row, matrix_row))
            corr_list = []
            for i in range(matrix_row):
                for j in range(i + 1, matrix_row):
                    tmp = dask.delayed(_calc_kendall)(
                        cal_matrix[i, :], cal_matrix[j, :]
                    )
                    corr_list.append(tmp)
            corr_comp = dask.compute(*corr_list)
            idx = 0
            for i in range(matrix_row):  # TODO: Optimize by using numpy api
                for j in range(i + 1, matrix_row):
                    corr_matrix[i][j] = corr_comp[idx]
                    corr_matrix[j][i] = corr_matrix[i][j]
                    idx = idx + 1
            result["corr_k"] = corr_matrix
        else:
            raise ValueError("Method Error")
    raw_data = {"df": df, "method_list": method_list}
    intermediate = Intermediate(result, raw_data)
    return intermediate


def _calc_correlation_pd_k(pd_data_frame: pd.DataFrame, k: int) -> Any:
    """
    :param pd_data_frame: the pandas data_frame for which plots
    are calculated for each column.
    :param k: choose top-k correlation value
    :return: An object to encapsulate the
    intermediate results.
    """
    result = {}
    intermediate_pd = _calc_correlation_pd(df=pd_data_frame)
    method_list = intermediate_pd.raw_data["method_list"]
    for method in method_list:
        corr_matrix = intermediate_pd.result["corr_" + method[0]]
        matrix_row, _ = np.shape(corr_matrix)
        corr_matrix_re = np.reshape(np.triu(corr_matrix, 1), (matrix_row * matrix_row,))
        idx = np.argsort(np.absolute(corr_matrix_re))
        mask = np.zeros(shape=(matrix_row * matrix_row,))
        for i in range(k):
            mask[idx[-i - 1]] = 1
        corr_matrix = np.multiply(corr_matrix_re, mask)
        corr_matrix = np.reshape(corr_matrix, (matrix_row, matrix_row))
        corr_matrix = corr_matrix.T
        result["corr_" + method[0]] = corr_matrix
        result["mask_" + method[0]] = mask
    raw_data = {"df": pd_data_frame, "method_list": method_list, "k": k}
    intermediate = Intermediate(result, raw_data)
    return intermediate


def _calc_correlation_pd_x_k_pearson(  # pylint: disable=too-many-locals
    # pylint: disable=invalid-unary-operand-type
    name_list: List[str],
    x_name: str,
    cal_matrix: np.ndarray,
    value_range: Optional[List[float]] = None,
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    :param name_list: name list of data frame
    :param x_name: a valid column name of the data frame
    :param cal_matrix: data frame numpy value
    :param value_range: a range which return correlation
    :param k: choose top-k or reverse top-k
    :return: An dictionary contains intermediate results
    """
    name_idx = name_list.index(x_name)
    cal_matrix_p = cal_matrix.copy()
    cov_xy = np.cov(cal_matrix_p)
    std_xy = np.sqrt(np.diag(cov_xy))
    corr_matrix_p = cov_xy / std_xy[:, None] / std_xy[None, :]

    if value_range is not None:
        value_start = value_range[0]
        value_end = value_range[1]
        row_p = corr_matrix_p[name_idx, :]
        idx_p = np.argsort(row_p)
        len_p = len(idx_p)
        start_p = len_p
        end_p = len_p
        for i, _ in enumerate(idx_p):
            if start_p == len_p and row_p[idx_p[i]] >= value_start:
                start_p = i
            if end_p == len_p and row_p[idx_p[i]] > value_end:
                end_p = i
        result = {"start_p": start_p, "end_p": end_p}
        if k is not None:
            if result["end_p"] - result["start_p"] > k:
                result["start_p"] = result["end_p"] - k
            start_p = result["start_p"]
            end_p = result["end_p"]
            result["pearson"] = row_p[idx_p[start_p:end_p]]
            result["col_p"] = np.array(name_list)[idx_p[start_p:end_p]]
        else:
            start_p = result["start_p"]
            end_p = result["end_p"]
            result["pearson"] = row_p[idx_p[start_p:end_p]]
            result["col_p"] = np.array(name_list)[idx_p[start_p:end_p]]
    else:
        if k is not None:
            row_p = corr_matrix_p[name_idx, :]
            row_p_abs = np.absolute(row_p)
            idx_p = np.argsort(-row_p_abs)
            col_p = np.array(name_list)[idx_p[:k]]
            result = {"pearson": row_p[idx_p[:k]], "col_p": col_p}
        else:
            row_p = corr_matrix_p[name_idx, :]
            row_p_abs = np.absolute(row_p)
            idx_p = np.argsort(-row_p_abs)
            col_p = np.array(name_list)[idx_p]
            result = {"pearson": row_p[idx_p], "col_p": col_p}
    return result


def _calc_correlation_pd_x_k_spearman(  # pylint: disable=too-many-locals
    # pylint: disable=invalid-unary-operand-type
    name_list: List[str],
    x_name: str,
    cal_matrix: np.ndarray,
    value_range: Optional[List[float]] = None,
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    :param name_list: name list of data frame
    :param x_name: a valid column name of the data frame
    :param cal_matrix: data frame numpy value
    :param value_range: a range which return correlation
    :param k: choose top-k or reverse top-k
    :return: An dictionary contains intermediate results
    """
    name_idx = name_list.index(x_name)
    cal_matrix_s = cal_matrix.copy()
    matrix_row, _ = np.shape(cal_matrix_s)
    for i in range(matrix_row):
        cal_matrix_s[i, :] = _value_to_rank(cal_matrix_s[i, :])
    cov_xy = np.cov(cal_matrix_s)
    std_xy = np.sqrt(np.diag(cov_xy))
    corr_matrix_s = cov_xy / std_xy[:, None] / std_xy[None, :]

    if value_range is not None:
        value_start = value_range[0]
        value_end = value_range[1]
        row_s = corr_matrix_s[name_idx, :]
        idx_s = np.argsort(row_s)
        len_s = len(idx_s)
        start_s = len_s
        end_s = len_s
        for i, _ in enumerate(idx_s):
            if start_s == len_s and row_s[idx_s[i]] >= value_start:
                start_s = i
            if end_s == len_s and row_s[idx_s[i]] > value_end:
                end_s = i
        result = {"start_s": start_s, "end_s": end_s}
        if k is not None:
            if result["end_s"] - result["start_s"] > k:
                result["start_s"] = result["end_s"] - k
            start_s = result["start_s"]
            end_s = result["end_s"]
            result["spearman"] = row_s[idx_s[start_s:end_s]]
            result["col_s"] = np.array(name_list)[idx_s[start_s:end_s]]
        else:
            start_s = result["start_s"]
            end_s = result["end_s"]
            result["spearman"] = row_s[idx_s[start_s:end_s]]
            result["col_s"] = np.array(name_list)[idx_s[start_s:end_s]]
    else:
        if k is not None:
            row_s = corr_matrix_s[name_idx, :]
            row_s_abs = np.absolute(row_s)
            idx_s = np.argsort(-row_s_abs)
            col_s = np.array(name_list)[idx_s[:k]]
            result = {"spearman": row_s[idx_s[:k]], "col_s": col_s}
        else:
            row_s = corr_matrix_s[name_idx, :]
            row_s_abs = np.absolute(row_s)
            idx_s = np.argsort(-row_s_abs)
            col_s = np.array(name_list)[idx_s]
            result = {"spearman": row_s[idx_s], "col_s": col_s}
    return result


def _calc_correlation_pd_x_k_kendall(  # pylint: disable=too-many-locals
    # pylint: disable=invalid-unary-operand-type
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    name_list: List[str],
    x_name: str,
    cal_matrix: np.ndarray,
    value_range: Optional[List[float]] = None,
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    :param name_list: name list of data frame
    :param x_name: a valid column name of the data frame
    :param cal_matrix: data frame numpy value
    :param value_range: a range which return correlation
    :param k: choose top-k or reverse top-k
    :return: An dictionary contains intermediate results
    """
    name_idx = name_list.index(x_name)
    cal_matrix_k = cal_matrix.copy()
    matrix_row, _ = np.shape(cal_matrix_k)
    corr_matrix_k = np.ones(shape=(matrix_row, matrix_row))
    corr_list = []
    for i in range(0, name_idx):
        tmp = dask.delayed(_calc_kendall)(cal_matrix_k[name_idx, :], cal_matrix_k[i, :])
        corr_list.append(tmp)
    for i in range(name_idx + 1, matrix_row):
        tmp = dask.delayed(_calc_kendall)(cal_matrix_k[name_idx, :], cal_matrix_k[i, :])
        corr_list.append(tmp)
    corr_comp = dask.compute(*corr_list)
    idx = 0
    for i in range(0, name_idx):
        corr_matrix_k[name_idx][i] = corr_comp[idx]
        corr_matrix_k[i][name_idx] = corr_matrix_k[name_idx][i]
        idx = idx + 1
    for i in range(name_idx + 1, matrix_row):
        corr_matrix_k[name_idx][i] = corr_comp[idx]
        corr_matrix_k[i][name_idx] = corr_matrix_k[name_idx][i]
        idx = idx + 1

    if value_range is not None:
        value_start = value_range[0]
        value_end = value_range[1]
        row_k = corr_matrix_k[name_idx, :]
        idx_k = np.argsort(row_k)
        len_k = len(idx_k)
        start_k = len_k
        end_k = len_k
        for i, _ in enumerate(idx_k):
            if start_k == len_k and row_k[idx_k[i]] >= value_start:
                start_k = i
            if end_k == len_k and row_k[idx_k[i]] > value_end:
                end_k = i
        result = {"start_k": start_k, "end_k": end_k}
        if k is not None:
            if result["end_k"] - result["start_k"] > k:
                result["start_k"] = result["end_k"] - k
            start_k = result["start_k"]
            end_k = result["end_k"]
            result["kendall"] = row_k[idx_k[start_k:end_k]]
            result["col_k"] = np.array(name_list)[idx_k[start_k:end_k]]
        else:
            start_k = result["start_k"]
            end_k = result["end_k"]
            result["kendall"] = row_k[idx_k[start_k:end_k]]
            result["col_k"] = np.array(name_list)[idx_k[start_k:end_k]]
    else:
        if k is not None:
            row_k = corr_matrix_k[name_idx, :]
            row_k_abs = np.absolute(row_k)
            idx_k = np.argsort(-row_k_abs)
            col_k = np.array(name_list)[idx_k[:k]]
            result = {"kendall": row_k[idx_k[:k]], "col_k": col_k}
        else:
            row_k = corr_matrix_k[name_idx, :]
            row_k_abs = np.absolute(row_k)
            idx_k = np.argsort(-row_k_abs)
            col_k = np.array(name_list)[idx_k]
            result = {"kendall": row_k[idx_k], "col_k": col_k}
    return result


def _calc_correlation_pd_x_k(  # pylint: disable=too-many-statements
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    pd_data_frame: pd.DataFrame,
    x_name: str,
    value_range: Optional[List[float]] = None,
    k: Optional[int] = None,
) -> Intermediate:
    """
    :param pd_data_frame: the pandas data_frame for which plots
    are calculated for each column.
    :param x_name: a valid column name of the data frame
    :param value_range: a range which return correlation
    :param k: choose top-k or reverse top-k
    :return: An object to encapsulate the
    intermediate results.
    """
    if k == 0:
        raise ValueError("k should be larger than 0")
    if k is not None and len(pd_data_frame.columns) < k:
        raise ValueError("k should be smaller than the number of columns")

    name_list = list(pd_data_frame.columns)
    cal_matrix = pd_data_frame.values.T
    raw_data = {
        "df": pd_data_frame,
        "x_name": x_name,
        "value_range": value_range,
        "k": k,
    }
    result_p = _calc_correlation_pd_x_k_pearson(
        name_list=name_list,
        x_name=x_name,
        cal_matrix=cal_matrix,
        value_range=value_range,
        k=k,
    )
    result_s = _calc_correlation_pd_x_k_spearman(
        name_list=name_list,
        x_name=x_name,
        cal_matrix=cal_matrix,
        value_range=value_range,
        k=k,
    )
    result_k = _calc_correlation_pd_x_k_kendall(
        name_list=name_list,
        x_name=x_name,
        cal_matrix=cal_matrix,
        value_range=value_range,
        k=k,
    )
    result = merge_dicts(result_p, result_s, result_k)
    intermediate = Intermediate(result, raw_data)
    return intermediate


def _calc_correlation_pd_x_y_k(  # pylint: disable=too-many-locals
    pd_data_frame: pd.DataFrame, x_name: str, y_name: str, k: Optional[int] = None
) -> Intermediate:
    """
    :param pd_data_frame: the pandas data_frame for which plots
    are calculated for each column.
    :param x_name: a valid column name of the data frame
    :param y_name: a valid column name of the data frame
    :param k: highlight k points which influence pearson correlation most
    :return: An object to encapsulate the
    intermediate results.
    """
    if k == 0:
        raise ValueError("k should be larger than 0")

    data_x = pd_data_frame[x_name].values
    data_y = pd_data_frame[y_name].values
    corr = np.corrcoef(data_x, data_y)[1, 0]
    line_a, line_b = np.linalg.lstsq(
        np.vstack([data_x, np.ones(len(data_x))]).T, data_y, rcond=None
    )[0]

    sample_array = np.random.choice(len(data_x), int(len(data_x) / 10))
    data_x_sample = data_x[sample_array]
    data_y_sample = data_y[sample_array]
    if k is None:
        result = {
            "corr": corr,
            "line_a": line_a,
            "line_b": line_b,
            "data_x_sample": data_x_sample,
            "data_y_sample": data_y_sample,
        }
        raw_data = {"df": pd_data_frame, "x_name": x_name, "y_name": y_name, "k": k}
        intermediate = Intermediate(result, raw_data)
        return intermediate

    diff_inc = []
    diff_dec = []
    inc_point_x = []
    inc_point_y = []
    dec_point_x = []
    dec_point_y = []
    data_mask = np.array([True for _, _ in enumerate(data_x_sample)], dtype=bool)
    for i in range(len(data_x_sample)):
        data_mask[i] = False
        data_x_sel = data_x_sample[data_mask]
        data_y_sel = data_y_sample[data_mask]
        corr_sel = np.corrcoef(data_x_sel, data_y_sel)[1, 0]
        diff_inc.append(corr_sel - corr)
        diff_dec.append(corr - corr_sel)
        data_mask[i] = True
    diff_inc_sort = np.argsort(diff_inc)
    diff_dec_sort = np.argsort(diff_dec)
    inc_point_x.append(
        data_x_sample[diff_inc_sort[-k:]]  # pylint: disable=invalid-unary-operand-type
    )
    inc_point_y.append(
        data_y_sample[diff_inc_sort[-k:]]  # pylint: disable=invalid-unary-operand-type
    )
    dec_point_x.append(
        data_x_sample[diff_dec_sort[-k:]]  # pylint: disable=invalid-unary-operand-type
    )
    dec_point_y.append(
        data_y_sample[diff_dec_sort[-k:]]  # pylint: disable=invalid-unary-operand-type
    )
    data_x_sample = np.delete(
        data_x_sample,  # pylint: disable=invalid-unary-operand-type
        np.append(
            diff_inc_sort[-k:],  # pylint: disable=invalid-unary-operand-type
            diff_dec_sort[-k:],  # pylint: disable=invalid-unary-operand-type
        ),
        None,
    )
    data_y_sample = np.delete(
        data_y_sample,  # pylint: disable=invalid-unary-operand-type
        np.append(
            diff_inc_sort[-k:],  # pylint: disable=invalid-unary-operand-type
            diff_dec_sort[-k:],  # pylint: disable=invalid-unary-operand-type
        ),
        None,
    )
    result = {
        "corr": corr,
        "line_a": line_a,
        "line_b": line_b,
        "data_x_sample": data_x_sample,
        "data_y_sample": data_y_sample,
        "dec_point_x": dec_point_x,
        "dec_point_y": dec_point_y,
        "inc_point_x": inc_point_x,
        "inc_point_y": inc_point_y,
    }
    raw_data = {"df": pd_data_frame, "x_name": x_name, "y_name": y_name, "k": k}
    intermediate = Intermediate(result, raw_data)
    return intermediate


def _calc_cross_table(  # pylint: disable=too-many-locals
    pd_data_frame: pd.DataFrame, x_name: str, y_name: str
) -> Intermediate:
    """
    :param pd_data_frame: the pandas data_frame for which plots
    are calculated for each column.
    :param x_name: a valid column name of the data frame
    :param y_name: a valid column name of the data frame
    :return: An object to encapsulate the
    intermediate results.
    """
    x_cat_list = list(pd_data_frame[x_name].cat.categories)
    y_cat_list = list(pd_data_frame[y_name].cat.categories)
    dict_x_cat = {x_cat_list[i]: i for i, _ in enumerate(x_cat_list)}
    dict_y_cat = {y_cat_list[i]: i for i, _ in enumerate(y_cat_list)}
    cross_matrix = np.zeros(shape=(len(x_cat_list), len(y_cat_list)))
    x_value_list = pd_data_frame[x_name].values
    y_value_list = pd_data_frame[y_name].values
    for i, _ in enumerate(x_value_list):
        x_pos = dict_x_cat[x_value_list[i]]
        y_pos = dict_y_cat[y_value_list[i]]
        cross_matrix[x_pos][y_pos] = cross_matrix[x_pos][y_pos] + 1
    result = {
        "cross_table": cross_matrix,
        "x_cat_list": x_cat_list,
        "y_cat_list": y_cat_list,
    }
    raw_data = {"df": pd_data_frame, "x_name": x_name, "y_name": y_name}
    intermediate = Intermediate(result, raw_data)
    return intermediate


def plot_correlation(  # pylint: disable=too-many-arguments
    pd_data_frame: pd.DataFrame,
    x_name: Optional[str] = None,
    y_name: Optional[str] = None,
    value_range: Optional[List[float]] = None,
    k: Optional[int] = None,
    return_intermediate: bool = False,
) -> Union[Union[Figure, Tabs], Tuple[Union[Figure, Tabs], Any]]:
    """
    :param pd_data_frame: the pandas data_frame for which plots are calculated for each
    column.
    :param x_name: a valid column name of the data frame
    :param y_name: a valid column name of the data frame
    :param value_range: range of value
    :param k: choose top-k element
    :param return_intermediate: whether show intermediate results to users
    :return: A (column: [array/dict]) dict to encapsulate the
    intermediate results.

    match (x_name, y_name, k)
        case (None, None, None) => heatmap
        case (Some, None, Some) => Top K columns for (pearson, spearman, kendall)
        case (Some, Some, _) => Scatter with regression line with/without top k outliers
        otherwise => error
    """
    params = {
        "width": 325,
        "alpha": 0.5,
        "plot_width": 400,
        "plot_height": 400,
        "size": 6,
    }
    if x_name is not None and y_name is not None:
        if (
            get_type(pd_data_frame[x_name]) == DataType.TYPE_CAT
            and get_type(pd_data_frame[y_name]) == DataType.TYPE_CAT
        ):
            intermediate = _calc_cross_table(
                pd_data_frame=pd_data_frame, x_name=x_name, y_name=y_name
            )
            fig = _vis_cross_table(intermediate=intermediate, params=params)
        elif (
            not get_type(pd_data_frame[x_name]) != DataType.TYPE_NUM
            and not get_type(pd_data_frame[y_name]) != DataType.TYPE_NUM
        ):
            intermediate = _calc_correlation_pd_x_y_k(
                pd_data_frame=pd_data_frame, x_name=x_name, y_name=y_name, k=k
            )
            fig = _vis_correlation_pd_x_y_k(intermediate=intermediate, params=params)
        else:
            raise ValueError(
                "Cannot calculate the correlation " "between two different dtype column"
            )
    elif x_name is not None:
        if get_type(pd_data_frame[x_name]) != DataType.TYPE_NUM:
            raise ValueError("The dtype of data frame column " "should be numerical")
        pd_data_frame = _drop_non_numerical_columns(pd_data_frame=pd_data_frame)
        intermediate = _calc_correlation_pd_x_k(
            pd_data_frame=pd_data_frame, x_name=x_name, value_range=value_range, k=k
        )
        fig = _vis_correlation_pd_x_k(intermediate=intermediate, params=params)
    elif x_name is None and y_name is not None:
        raise ValueError("Please give a value to x_name")
    elif k is not None:
        pd_data_frame = _drop_non_numerical_columns(pd_data_frame=pd_data_frame)
        intermediate = _calc_correlation_pd_k(pd_data_frame=pd_data_frame, k=k)
        fig = _vis_correlation_pd(intermediate=intermediate, params=params)
    else:
        pd_data_frame = _drop_non_numerical_columns(pd_data_frame=pd_data_frame)
        intermediate = _calc_correlation_pd(df=pd_data_frame)
        fig = _vis_correlation_pd(intermediate=intermediate, params=params)
    show(fig)
    if return_intermediate:
        return fig, intermediate
    return fig
