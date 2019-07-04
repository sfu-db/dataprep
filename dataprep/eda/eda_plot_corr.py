"""
    This module implements the plot_corr(df) function.
"""
from typing import Any, Dict, Optional, Tuple, Union

import math
import dask
import holoviews as hv
import numpy as np
import pandas as pd
import random
import string
import tempfile
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure
from scipy.stats import kendalltau
from ..utils import is_notebook


def _rand_str(
        str_length: int = 20
) -> Any:
    """
    :param str_length: The length of random string
    :return: A generated random string
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters)
                   for _ in range(str_length))


def _calc_kendall(
        data_a: np.ndarray,
        data_b: np.ndarray
) -> Any:
    """
    :param data_a: the input numpy array
    :param data_b: the input numpy array
    :return: A float value which indicates the
    correlation of two numpy array
    """
    kendallta, _ = kendalltau(data_a, data_b)
    return kendallta


def _value_to_rank(
        array: np.ndarray
) -> pd.Series:
    """
    :param array: the input numpy array
    :return: the output numpy array
    """
    array_ranks = pd.Series(array).rank()
    return array_ranks.values


def _line_fit(
        data_x: np.ndarray,
        data_y: np.ndarray
) -> Tuple[Union[float, Any],
           Union[float, Any],
           Union[float, Any]]:
    """
    :param data_x: the input numpy array
    :param data_y: the input numpy array
    :return: the parameter of line and the relationship
    """
    num = float(len(data_x))
    sum_x, sum_y, sum_xx, sum_yy, sum_xy = 0, 0, 0, 0, 0
    for i in range(0, int(num)):
        sum_x = sum_x + data_x[i]
        sum_y = sum_y + data_y[i]
        sum_xx = sum_xx + data_x[i] * data_x[i]
        sum_yy = sum_yy + data_y[i] * data_y[i]
        sum_xy = sum_xy + data_x[i] * data_y[i]
    line_a = (sum_y * sum_x / num - sum_xy) / (sum_x * sum_x / num - sum_xx)
    line_b = (sum_y - line_a * sum_x) / num
    line_r = abs(sum_y * sum_x / num - sum_xy) / \
        math.sqrt((sum_xx - sum_x * sum_x / num) *
                  (sum_yy - sum_y * sum_y / num))
    return line_a, line_b, line_r


def _is_categorical(
        df_column: pd.Series
) -> Any:
    """
    :param df_column: a column of data frame
    :return: a boolean value
    """
    return df_column.dtype.name == 'category'


def _is_not_numerical(
        df_column: pd.Series
) -> Any:
    """
    :param df_column: a column of data frame
    :return: a boolean value
    """
    return df_column.dtype.name == 'category' or \
           df_column.dtype.name == 'object' or \
           df_column.dtype.name == 'datetime64[ns]'


def _del_column(
        pd_data_frame: pd.DataFrame
) -> pd.DataFrame:
    """
    :param pd_data_frame: the pandas data_frame for
    which plots are calculated for each column.
    :return: the numerical pandas data_frame for
    which plots are calculated for each column.
    """
    drop_list = []
    for column_name in pd_data_frame.columns.values:
        if _is_not_numerical(pd_data_frame[column_name]):
            drop_list.append(column_name)
    pd_data_frame.drop(columns=drop_list)
    return pd_data_frame


def _vis_correlation_pd(  # pylint: disable=too-many-locals
        pd_data_frame: pd.DataFrame,
        result: Dict[str, Any],
        method: str = 'pearson'
) -> Any:
    """
    :param pd_data_frame: the pandas data_frame for
    which plots are calculated for each column.
    :param result: A dict to encapsulate the
    intermediate results.
    :param method: The method used for
    calculating correlation.
    :return: A figure object
    """
    hv.extension('bokeh')
    corr_matrix = result['corr']
    name_list = pd_data_frame.columns.values
    data = []
    for i, _ in enumerate(name_list):
        for j, _ in enumerate(name_list):
            data.append((name_list[i],
                         name_list[j],
                         corr_matrix[i, j]))
    heatmap = hv.HeatMap(data).opts(tools=['hover'],
                                    colorbar=True,
                                    width=325,
                                    toolbar='above',
                                    title="heatmap_" + method)
    fig = hv.render(heatmap, backend='bokeh')
    if is_notebook():
        output_notebook()
        show(fig, notebook_handle=True)
    else:
        output_file(filename=tempfile.gettempdir() + '/' +
                             _rand_str() + '.html',
                    title='heat_map')
        show(fig)
    return fig


def _vis_correlation_pd_x_k(  # pylint: disable=too-many-locals
        result: Dict[str, Any]
) -> Any:
    """
    :param result: A dict to encapsulate the
    intermediate results.
    :param k: Choose top-k correlation value
    :return: A figure object
    """
    hv.extension('bokeh')
    corr_matrix = np.array([result['pearson'],
                            result['spearman'],
                            result['kendall']])
    method_list = ['pearson', 'spearman', 'kendall']
    data = []
    for i, method_name in enumerate(method_list):
        for j, _ in enumerate(result['col_' + method_name[0]]):
            data.append((method_list[i],
                         result['col_' + method_name[0]][j],
                         corr_matrix[i, j]))
    heatmap = hv.HeatMap(data).opts(tools=['hover'],
                                    colorbar=True,
                                    width=325,
                                    toolbar='above',
                                    title="heatmap")
    fig = hv.render(heatmap, backend='bokeh')
    if is_notebook():
        output_notebook()
        show(fig, notebook_handle=True)
    else:
        output_file(filename=tempfile.gettempdir() + '/' +
                             _rand_str() + '.html',
                    title='heat_map')
        show(fig)
    return fig


def _vis_correlation_pd_x_y_k_zero(
        data_x: np.ndarray,
        data_y: np.ndarray,
        result: Dict[str, Any]
) -> Any:
    """
    :param data_x: The column of data frame
    :param data_y: The column of data frame
    :param result: A dict to encapsulate the
    intermediate results.
    :return: A figure object
    """
    fig = figure(plot_width=400, plot_height=400)
    sample_x = np.linspace(min(data_x), max(data_y), 100)
    sample_y = result['line_a'] * sample_x + result['line_b']
    fig.circle(data_x, data_y,
               legend='origin data', size=10,
               color='navy', alpha=0.5)
    fig.line(sample_x, sample_y, line_width=3)
    if is_notebook():
        output_notebook()
        show(fig, notebook_handle=True)
    else:
        output_file(filename=tempfile.gettempdir() + '/' +
                             _rand_str() + '.html',
                    title='scatter')
        show(fig)
    return fig


def _vis_correlation_pd_x_y_k(
        data_x: np.ndarray,
        data_y: np.ndarray,
        result: Dict[str, Any]
) -> Any:
    """
    :param data_x: The column of data frame
    :param data_y: The column of data frame
    :param result: A dict to encapsulate the
    intermediate results.
    :return: A figure object
    """
    fig = figure(plot_width=400, plot_height=400)
    sample_x = np.linspace(min(data_x), max(data_y), 100)
    sample_y = result['line_a'] * sample_x + result['line_b']
    fig.circle(data_x, data_y,
               legend='origin data', size=10,
               color='navy', alpha=0.5)
    fig.circle(result['dec_point_x'],
               result['dec_point_y'],
               legend='decrease points', size=10,
               color='yellow', alpha=0.5)
    fig.circle(result['inc_point_x'],
               result['inc_point_y'],
               legend='increase points', size=10,
               color='red', alpha=0.5)
    fig.line(sample_x, sample_y, line_width=3)
    if is_notebook():
        output_notebook()
        show(fig, notebook_handle=True)
    else:
        output_file(filename=tempfile.gettempdir() + '/' +
                             _rand_str() + '.html',
                    title='scatter')
        show(fig)
    return fig


def _vis_cross_table(
        result: Dict[str, Any]
) -> Any:
    """
    :param result: A dict to encapsulate the
    intermediate results.
    :return:
    """
    hv.extension('bokeh')
    cross_matrix = result['cross_table']
    x_cat_list = result['x_cat_list']
    y_cat_list = result['y_cat_list']
    data = []
    for i, _ in enumerate(x_cat_list):
        for j, _ in enumerate(y_cat_list):
            data.append((x_cat_list[i],
                         y_cat_list[j],
                         cross_matrix[i, j]))
    heatmap = hv.HeatMap(data).opts(tools=['hover'],
                                    colorbar=True,
                                    width=325,
                                    toolbar='above',
                                    title="heatmap")
    fig = hv.render(heatmap, backend='bokeh')
    if is_notebook():
        output_notebook()
        show(fig, notebook_handle=True)
    else:
        output_file(filename=tempfile.gettempdir() + '/' +
                             _rand_str() + '.html',
                    title='cross_table')
        show(fig)
    return fig


def _cal_correlation_pd(  # pylint: disable=too-many-locals
        pd_data_frame: pd.DataFrame,
        method: str = 'pearson'
) -> Dict[str, Any]:
    """
    :param pd_data_frame: the pandas data_frame for which plots
    are calculated for each column.
    :param method: Three method we can use to calculate
    the correlation matrix
    :return: A (column: [array/dict]) dict to encapsulate the
    intermediate results.
    """
    result = dict()
    if method == 'pearson':
        cal_matrix = pd_data_frame.values.T
        cov_xy = np.cov(cal_matrix)
        std_xy = np.sqrt(np.diag(cov_xy))
        corr_matrix = cov_xy / std_xy[:, None] / std_xy[None, :]
    elif method == 'spearman':
        cal_matrix = pd_data_frame.values.T
        matrix_row, _ = np.shape(cal_matrix)
        for i in range(matrix_row):
            cal_matrix[i, :] = _value_to_rank(cal_matrix[i, :])
        cov_xy = np.cov(cal_matrix)
        std_xy = np.sqrt(np.diag(cov_xy))
        corr_matrix = cov_xy / std_xy[:, None] / std_xy[None, :]
    elif method == 'kendall':
        cal_matrix = pd_data_frame.values.T
        matrix_row, _ = np.shape(cal_matrix)
        corr_matrix = np.ones(shape=(matrix_row, matrix_row))
        corr_list = []
        for i in range(matrix_row):
            for j in range(i + 1, matrix_row):
                tmp = dask.delayed(_calc_kendall)(
                    cal_matrix[i, :], cal_matrix[j, :])
                corr_list.append(tmp)
        corr_comp = dask.compute(*corr_list)
        idx = 0
        for i in range(matrix_row):
            for j in range(i + 1, matrix_row):
                corr_matrix[i][j] = corr_comp[idx]
                corr_matrix[j][i] = corr_matrix[i][j]
                idx = idx + 1
    else:
        raise ValueError("Method Error")
    result['corr'] = corr_matrix
    return result


def _cal_correlation_pd_k(
        pd_data_frame: pd.DataFrame,
        k: int = 0,
        method: str = 'pearson'
) -> Dict[str, Any]:
    """
    :param pd_data_frame: the pandas data_frame for which plots
    are calculated for each column.
    :param k: choose top-k correlation value
    :param method: Three method we can use to calculate
    the correlation matrix
    :return: A (column: [array/dict]) dict to encapsulate the
    intermediate results.
    """
    result = dict()
    result_pd = _cal_correlation_pd(pd_data_frame=pd_data_frame, method=method)
    corr_matrix = result_pd['corr']
    matrix_row, _ = np.shape(corr_matrix)
    corr_matrix_re = np.reshape(np.triu(corr_matrix), (matrix_row * matrix_row,))
    idx = np.argsort(corr_matrix_re)
    mask = np.zeros(shape=(matrix_row * matrix_row,))
    mask[idx[:k]] = 1
    mask[idx[-k:]] = 1
    corr_matrix = np.multiply(corr_matrix_re, mask)
    corr_matrix = np.reshape(corr_matrix,
                             (matrix_row, matrix_row))
    corr_matrix += corr_matrix.T - np.diag(corr_matrix.diagonal())
    result['corr'] = corr_matrix
    return result


def _cal_correlation_pd_x_k(  # pylint: disable=too-many-locals
        pd_data_frame: pd.DataFrame,
        x_name: Optional[str] = None,
        k: int = 0
) -> Dict[str, Any]:
    """
    :param pd_data_frame: the pandas data_frame for which plots
    are calculated for each column.
    :param x_name: a valid column name of the data frame
    :param k: choose top-k
    :return: A (column: [array/dict]) dict to encapsulate the
    intermediate results.
    """
    name_list = pd_data_frame.columns.values.tolist()

    col = len(name_list)
    if col < k:
        raise ValueError("k is not allowed to be "
                         "bigger than column size")

    name_idx = name_list.index(x_name)
    cal_matrix = pd_data_frame.values.T
    cal_matrix_p = cal_matrix.copy()
    cal_matrix_s = cal_matrix.copy()
    cal_matrix_k = cal_matrix.copy()

    cov_xy = np.cov(cal_matrix_p)
    std_xy = np.sqrt(np.diag(cov_xy))
    corr_matrix_p = cov_xy / std_xy[:, None] / std_xy[None, :]
    row_p = corr_matrix_p[name_idx, :]
    row_p[name_idx] = -1
    idx_p = np.argsort(row_p)
    col_p = np.array(name_list)[idx_p[-k:]]
    col_p = col_p[::-1]

    matrix_row, _ = np.shape(cal_matrix_s)
    for i in range(matrix_row):
        cal_matrix_s[i, :] = _value_to_rank(cal_matrix_s[i, :])
    cov_xy = np.cov(cal_matrix_s)
    std_xy = np.sqrt(np.diag(cov_xy))
    corr_matrix_s = cov_xy / std_xy[:, None] / std_xy[None, :]
    row_s = corr_matrix_s[name_idx, :]
    row_s[name_idx] = -1
    idx_s = np.argsort(row_s)
    col_s = np.array(name_list)[idx_s[-k:]]
    col_s = col_s[::-1]

    matrix_row, _ = np.shape(cal_matrix_k)
    corr_matrix_k = np.ones(shape=(matrix_row, matrix_row))
    corr_list = []
    for i in range(0, name_idx):
        tmp = dask.delayed(_calc_kendall)(
            cal_matrix_k[name_idx, :], cal_matrix_k[i, :])
        corr_list.append(tmp)
    for i in range(name_idx + 1, matrix_row):
        tmp = dask.delayed(_calc_kendall)(
            cal_matrix_k[name_idx, :], cal_matrix_k[i, :])
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
    row_k = corr_matrix_k[name_idx, :]
    row_k[name_idx] = -1
    idx_k = np.argsort(row_k)
    col_k = np.array(name_list)[idx_k[-k:]]
    col_k = col_k[::-1]

    result = {'pearson': sorted(row_p[idx_p[-k:]], reverse=True),
              'spearman': sorted(row_s[idx_s[-k:]], reverse=True),
              'kendall': sorted(row_k[idx_k[-k:]], reverse=True),
              'col_p': col_p,
              'col_s': col_s,
              'col_k': col_k}
    return result


def _cal_correlation_pd_x_y_k(  # pylint: disable=too-many-locals
        pd_data_frame: pd.DataFrame,
        x_name: Optional[str] = None,
        y_name: Optional[str] = None,
        k: int = 0
) -> Tuple[Dict[str, Any],
           np.ndarray, np.ndarray]:
    """
    :param pd_data_frame: the pandas data_frame for which plots
    are calculated for each column.
    :param x_name: a valid column name of the data frame
    :param y_name: a valid column name of the data frame
    :param k: highlight k points which influence pearson correlation most
    :return: A (column: [array/dict]) dict to encapsulate the
    intermediate results.
    """
    data_x = pd_data_frame[x_name].values
    data_y = pd_data_frame[y_name].values
    corr = np.corrcoef(data_x, data_y)[1, 0]
    line_a, line_b, _ = _line_fit(data_x=data_x, data_y=data_y)
    if k == 0:
        result = {'corr': corr,
                  'line_a': line_a,
                  'line_b': line_b}
    else:
        inc_point_x = []
        inc_point_y = []
        data_x_copy = data_x.copy()
        data_y_copy = data_y.copy()
        for _ in range(k):
            max_diff = 0
            max_idx = 0
            for j in range(len(data_x_copy)):
                data_x_sel = np.append(data_x_copy[:j], data_x_copy[j + 1:])
                data_y_sel = np.append(data_y_copy[:j], data_y_copy[j + 1:])
                corr_sel = np.corrcoef(data_x_sel, data_y_sel)[1, 0]
                if corr_sel - corr > max_diff:
                    max_diff = corr_sel - corr
                    max_idx = j
            inc_point_x.append(data_x_copy[max_idx])
            inc_point_y.append(data_y_copy[max_idx])
            data_x_copy = np.delete(data_x_copy, max_idx)
            data_y_copy = np.delete(data_y_copy, max_idx)
        dec_point_x = []
        dec_point_y = []
        data_x_copy = data_x.copy()
        data_y_copy = data_y.copy()
        for _ in range(k):
            max_diff = 0
            max_idx = 0
            for j in range(len(data_x_copy)):
                data_x_sel = np.append(data_x_copy[:j], data_x_copy[j + 1:])
                data_y_sel = np.append(data_y_copy[:j], data_y_copy[j + 1:])
                corr_sel = np.corrcoef(data_x_sel, data_y_sel)[1, 0]
                if corr - corr_sel > max_diff:
                    max_diff = corr - corr_sel
                    max_idx = j
            dec_point_x.append(data_x_copy[max_idx])
            dec_point_y.append(data_y_copy[max_idx])
            data_x_copy = np.delete(data_x_copy, max_idx)
            data_y_copy = np.delete(data_y_copy, max_idx)
        result = {'corr': corr,
                  'dec_point_x': dec_point_x,
                  'dec_point_y': dec_point_y,
                  'inc_point_x': inc_point_x,
                  'inc_point_y': inc_point_y,
                  'line_a': line_a,
                  'line_b': line_b}
    return result, data_x, data_y


def _cal_cross_table(
        pd_data_frame: pd.DataFrame,
        x_name: str,
        y_name: str
) -> Dict[str, Any]:
    """
    :param pd_data_frame: the pandas data_frame for which plots are calculated for each
    column.
    :param x_name: a valid column name of the data frame
    :param y_name: a valid column name of the data frame
    :return: A dict to encapsulate the
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
    return {'cross_table': cross_matrix,
            'x_cat_list': x_cat_list,
            'y_cat_list': y_cat_list}


def plot_correlation(
        pd_data_frame: pd.DataFrame,
        x_name: Optional[str] = None,
        y_name: Optional[str] = None,
        k: int = 0,
        method: str = 'pearson',
        show_intermediate: bool = False
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    :param pd_data_frame: the pandas data_frame for which plots are calculated for each
    column.
    :param x_name: a valid column name of the data frame
    :param y_name: a valid column name of the data frame
    :param k: choose top-k element
    :param method: Three method we can use to calculate the correlation matrix
    :param show_intermediate: whether show intermediate results to users
    :return: A (column: [array/dict]) dict to encapsulate the
    intermediate results.
    """
    if x_name is not None and y_name is not None:
        if _is_not_numerical(pd_data_frame[x_name]) or \
                _is_not_numerical(pd_data_frame[y_name]):
            if _is_categorical(pd_data_frame[x_name]) and \
                    _is_categorical(pd_data_frame[y_name]):
                result = _cal_cross_table(pd_data_frame=pd_data_frame,
                                          x_name=x_name,
                                          y_name=y_name)
                fig = _vis_cross_table(result=result)
            else:
                raise ValueError("Cannot calculate the correlation "
                                 "between two different dtype column")
        else:
            result, data_x, data_y = _cal_correlation_pd_x_y_k(
                pd_data_frame=pd_data_frame,
                x_name=x_name, y_name=y_name, k=k)
            if k == 0:
                fig = _vis_correlation_pd_x_y_k_zero(data_x=data_x,
                                               data_y=data_y,
                                               result=result)
            else:
                fig = _vis_correlation_pd_x_y_k(data_x=data_x,
                                          data_y=data_y,
                                          result=result)
    elif x_name is not None:
        if _is_not_numerical(pd_data_frame[x_name]):
            raise ValueError("The dtype of data frame column "
                             "should be numerical")
        pd_data_frame = _del_column(pd_data_frame=pd_data_frame)
        result = _cal_correlation_pd_x_k(
            pd_data_frame=pd_data_frame,
            x_name=x_name, k=k)
        fig = _vis_correlation_pd_x_k(result=result)
    elif k != 0:
        pd_data_frame = _del_column(pd_data_frame=pd_data_frame)
        result = _cal_correlation_pd_k(pd_data_frame=pd_data_frame,
                                       method=method, k=k)
        fig = _vis_correlation_pd(pd_data_frame=pd_data_frame,
                                  result=result, method=method)
    else:
        pd_data_frame = _del_column(pd_data_frame=pd_data_frame)
        result = _cal_correlation_pd(pd_data_frame=pd_data_frame,
                                     method=method)
        fig = _vis_correlation_pd(pd_data_frame=pd_data_frame,
                                  result=result, method=method)
    if show_intermediate:
        return fig, result
    else:
        return fig
