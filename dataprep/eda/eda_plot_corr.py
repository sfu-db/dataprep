"""
    This module implements the plot_corr(df) function.
"""
from typing import Any, Dict, Optional, Tuple, Union

import math
import dask
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
from scipy.stats import kendalltau


def _calc_kendall(
        data_a: np.ndarray,
        data_b: np.ndarray) -> Any:
    """
    :param data_a: the input numpy array
    :param data_b: the input numpy array
    :return: A float value which indicates the
    correlation of two numpy array
    """
    kendallta, _ = kendalltau(data_a, data_b)
    return kendallta


def _value_to_rank(array: np.ndarray) -> pd.Series:
    """
    :param array: the input numpy array
    :return: the output numpy array
    """
    array_ranks = pd.Series(array).rank()
    return array_ranks.values


def _line_fit(
        data_x: np.ndarray,
        data_y: np.ndarray
) -> Tuple[Union[float, Any], Union[float, Any], Union[float, Any]]:
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


def vis_correlation_pd_x_k(
        result: Dict[str, Any],
        k: int
) -> None:
    """
    :param result: A dict to encapsulate the
    intermediate results.
    :param k: choose top-k correlation value
    :return:
    """
    corr_matrix = np.array([result['pearson'],
                            result['spearman'],
                            result['kendall']])
    name_list = [str(i) for i in range(1, k + 1)]
    method_list = ['pearson', 'spearman', 'kendall']
    trace = go.Heatmap(z=corr_matrix,
                       x=name_list,
                       y=method_list,
                       colorscale='Blues',
                       reversescale=True)
    data = [trace]
    py.plot(data,
            filename='heatmap' + '_corr_pd_x_k',
            auto_open=True)


def vis_correlation_pd_x_y_k_zero(
        data_x: np.ndarray,
        data_y: np.ndarray,
        result: Dict[str, Any]
) -> None:
    """
    :param data_x: The column of dataframe
    :param data_y: The column of dataframe
    :param result: A dict to encapsulate the
    intermediate results.
    :return:
    """
    trace_zero = go.Scatter(
        x=data_x,
        y=data_y,
        mode='markers',
        name='origin data',
        marker=dict(
            size=10,
            color='rgba(152, 0, 0, 0.8)',
        )
    )
    sample_x = np.linspace(min(data_x), max(data_x), 100)
    sample_y = result['line_a'] * sample_x + result['line_b']
    trace_one = go.Scatter(
        x=sample_x,
        y=sample_y,
        mode='lines',
        name='regression line',
        line=dict(
            color='rgba(205, 12, 24, 0.6)',
            width=4,
        )
    )
    data = [trace_zero, trace_one]
    layout = dict(title='plot_correlation_pd_x_y_k',
                  xaxis=dict(zeroline=False, title='X'),
                  yaxis=dict(zeroline=False, title='Y'))
    fig = dict(data=data, layout=layout)
    py.plot(fig, filename='plot_correlation_pd_x_y_k')


def vis_correlation_pd_x_y_k(
        data_x: np.ndarray,
        data_y: np.ndarray,
        result: Dict[str, Any]
) -> None:
    """
    :param data_x: The column of dataframe
    :param data_y: The column of dataframe
    :param result: A dict to encapsulate the
    intermediate results.
    :return:
    """
    trace_zero = go.Scatter(
        x=data_x,
        y=data_y,
        mode='markers',
        name='origin data',
        marker=dict(
            size=10,
            color='rgba(152, 0, 0, 0.8)',
        )
    )
    trace_one = go.Scatter(
        x=result['dec_point_x'],
        y=result['dec_point_y'],
        mode='markers',
        name='decrease points',
        marker=dict(
            size=10,
        )
    )
    trace_two = go.Scatter(
        x=result['inc_point_x'],
        y=result['inc_point_y'],
        mode='markers',
        name='increase points',
        marker=dict(
            size=10,
        )
    )
    sample_x = np.linspace(min(data_x), max(data_x), 100)
    sample_y = result['line_a'] * sample_x + result['line_b']
    trace_three = go.Scatter(
        x=sample_x,
        y=sample_y,
        mode='lines',
        name='regression line',
        line=dict(
            color='rgba(205, 12, 24, 0.6)',
            width=4,
        )
    )
    data = [trace_zero, trace_one, trace_two, trace_three]
    layout = dict(title='plot_correlation_pd_x_y_k',
                  xaxis=dict(zeroline=False, title='X'),
                  yaxis=dict(zeroline=False, title='Y')
                  )
    fig = dict(data=data, layout=layout)
    py.plot(fig, filename='plot_correlation_pd_x_y_k')


def cal_correlation_pd(  # pylint: disable=too-many-locals
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
    name_list = pd_data_frame.columns.values
    trace = go.Heatmap(z=corr_matrix,
                       x=name_list,
                       y=name_list,
                       colorscale='Blues',
                       reversescale=True)
    data = [trace]
    py.plot(data,
            filename='heatmap_' + method + '_corr_pd',
            auto_open=True)
    return result


def cal_correlation_pd_k(
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
    result_pd = cal_correlation_pd(pd_data_frame=pd_data_frame, method=method)
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
    name_list = pd_data_frame.columns.values
    trace = go.Heatmap(z=corr_matrix,
                       x=name_list,
                       y=name_list,
                       colorscale='Blues',
                       reversescale=True)
    data = [trace]
    py.plot(data,
            filename='heatmap_' + method + '_corr_pd_k',
            auto_open=True)
    return result


def cal_correlation_pd_x_k(  # pylint: disable=too-many-locals
        pd_data_frame: pd.DataFrame,
        x_name: Optional[str] = None,
        k: int = 0
) -> Dict[str, Any]:
    """
    :param pd_data_frame: the pandas data_frame for which plots
    are calculated for each column.
    :param x_name: a valid column name of the dataframe
    :param k: choose top-k
    :return: A (column: [array/dict]) dict to encapsulate the
    intermediate results.
    """
    name_list = pd_data_frame.columns.values.tolist()
    name_idx = name_list.index(x_name)
    cal_matrix = pd_data_frame.values.T

    cov_xy = np.cov(cal_matrix)
    std_xy = np.sqrt(np.diag(cov_xy))
    corr_matrix_p = cov_xy / std_xy[:, None] / std_xy[None, :]
    row_p = corr_matrix_p[name_idx, :]
    row_p[name_idx] = -1
    idx_p = np.argsort(row_p)

    matrix_row, _ = np.shape(cal_matrix)
    for i in range(matrix_row):
        cal_matrix[i, :] = _value_to_rank(cal_matrix[i, :])
    cov_xy = np.cov(cal_matrix)
    std_xy = np.sqrt(np.diag(cov_xy))
    corr_matrix_s = cov_xy / std_xy[:, None] / std_xy[None, :]
    row_s = corr_matrix_s[name_idx, :]
    row_s[name_idx] = -1
    idx_s = np.argsort(row_s)

    matrix_row, _ = np.shape(cal_matrix)
    corr_matrix_k = np.ones(shape=(matrix_row, matrix_row))
    corr_list = []
    for i in range(0, name_idx):
        tmp = dask.delayed(_calc_kendall)(
            cal_matrix[name_idx, :], cal_matrix[i, :])
        corr_list.append(tmp)
    for i in range(name_idx + 1, matrix_row):
        tmp = dask.delayed(_calc_kendall)(
            cal_matrix[name_idx, :], cal_matrix[i, :])
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
    result = {'pearson': sorted(row_p[idx_p[-k:]], reverse=True),
              'spearman': sorted(row_s[idx_s[-k:]], reverse=True),
              'kendall': sorted(row_k[idx_k[-k:]], reverse=True)}
    return result


def cal_correlation_pd_x_y_k(  # pylint: disable=too-many-locals
        pd_data_frame: pd.DataFrame,
        x_name: Optional[str] = None,
        y_name: Optional[str] = None,
        k: int = 0
) -> Tuple[Union[Dict[str, Union[float, Any]],
                 Dict[str, Union[Union[list, float], Any]]],
           Any, Any]:
    """
    :param pd_data_frame: the pandas data_frame for which plots
    are calculated for each column.
    :param x_name: a valid column name of the dataframe
    :param y_name: a valid column name of the dataframe
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


def plot_correlation(
        pd_data_frame: pd.DataFrame,
        x_name: Optional[str] = None,
        y_name: Optional[str] = None,
        k: int = 0,
        method: str = 'pearson'
) -> Dict[str, Any]:
    """
    :param pd_data_frame: the pandas data_frame for which plots are calculated for each
    column.
    :param x_name: a valid column name of the dataframe
    :param y_name: a valid column name of the dataframe
    :param k: choose top-k element
    :param method: Three method we can use to calculate the correlation matrix
    :return: A (column: [array/dict]) dict to encapsulate the
    intermediate results.
    """
    if x_name is not None and y_name is not None:
        result, data_x, data_y = cal_correlation_pd_x_y_k(
            pd_data_frame=pd_data_frame,
            x_name=x_name, y_name=y_name, k=k)
        if k == 0:
            vis_correlation_pd_x_y_k_zero(data_x=data_x,
                                          data_y=data_y,
                                          result=result)
        else:
            vis_correlation_pd_x_y_k(data_x=data_x,
                                     data_y=data_y,
                                     result=result)
    elif x_name is not None:
        result, data_x, data_y = cal_correlation_pd_x_k(
            pd_data_frame=pd_data_frame,
            x_name=x_name, k=k)
        vis_correlation_pd_x_k(result=result, k=k)
    elif k != 0:
        result = cal_correlation_pd_k(pd_data_frame=pd_data_frame,
                                      method=method, k=k)
    else:
        result = cal_correlation_pd(pd_data_frame=pd_data_frame,
                                    method=method)
    return result
