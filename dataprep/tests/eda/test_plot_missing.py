"""
    This module for testing plot_missing(df, x, y) function.
"""
from typing import Any

import random
from time import time
import dask
import numpy as np
import pandas as pd

from ...eda import plot_missing


def _calc_none_sum(
        series: pd.Series,
        length: int
) -> Any:
    """
    :param series: A column of data frame
    :param length: The length of array
    :return: The count of None, Nan, Null
    """
    return series.isna().sum() / length


def test_plot_missing_df() -> None:
    """
    :return:
    """
    numbers_nan = [1, 2, 3, 4, 5, 6, 7, 8, 9, None, np.nan]
    df_data = pd.DataFrame({'a': [random.choice(numbers_nan) for _ in range(100)]})
    df_data['b'] = [random.choice(numbers_nan) for _ in range(100)]
    df_data['c'] = [random.choice(numbers_nan) for _ in range(100)]
    df_data['d'] = [random.choice(numbers_nan) for _ in range(100)]
    start1 = time()
    column_name_list = list(df_data.columns.values)
    count_none_list = []
    for column_name in column_name_list:
        count_none_list.append(
            dask.delayed(_calc_none_sum)(
                df_data[column_name], 100
            )
        )
    count_none_comp = dask.compute(*count_none_list)
    end1 = time()
    print("Pandas time used: ", end1 - start1)
    start2 = time()
    _, intermediate = plot_missing(
        pd_data_frame=df_data,
        return_intermediate=True
    )
    end2 = time()
    print("Numpy time used: ", end2 - start2)
    assert intermediate.result['count'] == count_none_comp


def test_plot_missing_df_x() -> None:
    """
    :return:
    """
    numbers_nan = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, None, np.nan]
    numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    letters = ['A', 'B', 'C', 'D', 'E']
    df_data = pd.DataFrame({'a': [random.choice(numbers_nan) for _ in range(100)]})
    df_data['b'] = [random.choice(numbers) for _ in range(100)]
    df_data['c'] = [random.choice(numbers) for _ in range(100)]
    df_data['d'] = [random.choice(numbers) for _ in range(100)]
    _, _ = plot_missing(
        pd_data_frame=df_data,
        x_name='a',
        return_intermediate=True
    )
    df_data = pd.DataFrame({'a': [random.choice(numbers_nan) for _ in range(100)]})
    df_data['b'] = [random.choice(numbers) for _ in range(100)]
    df_data['c'] = [random.choice(letters) for _ in range(100)]
    df_data['d'] = [random.choice(numbers) for _ in range(100)]
    df_data['c'] = df_data['c'].astype('category')
    _, _ = plot_missing(
        pd_data_frame=df_data,
        x_name='a',
        return_intermediate=True
    )


def test_plot_missing_df_x_y() -> None:
    """
    :return:
    """
    numbers_nan = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, None, np.nan]
    numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    letters = ['A', 'B', 'C', 'D', 'E']
    df_data = pd.DataFrame({'a': [random.choice(numbers_nan) for _ in range(100)]})
    df_data['b'] = [random.choice(numbers) for _ in range(100)]
    df_data['c'] = [random.choice(letters) for _ in range(100)]
    df_data['d'] = [random.choice(numbers) for _ in range(100)]
    _, _ = plot_missing(
        pd_data_frame=df_data,
        x_name='a',
        y_name='b',
        return_intermediate=True
    )
    df_data['c'] = df_data['c'].astype('category')
    _, _ = plot_missing(
        pd_data_frame=df_data,
        x_name='a',
        y_name='c',
        return_intermediate=True
    )
