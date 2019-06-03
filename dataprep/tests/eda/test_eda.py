"""
    module for testing plot(df, x, y) function.
"""
import datetime

import numpy as np
import pandas as pd
from pandas import Timestamp

from ...eda.eda_plot import plot  # dataprep.tests.eda.test_eda


def test_normal():
    """

    :return:
    """
    data_1 = {

        'id': [chr(97 + c) for c in range(1, 10)],

        'x': [50, 50, -10, 0, 0, 5, 15, -3, None],

        'y': [0.000001, 654.152, None, 15.984512, 3122, -3.1415926535, 111,
              15.9, 13.5],

        's1': np.ones(9),

        'somedate': [datetime.date(2011, 7, 4),
                     datetime.datetime(2022, 1, 1, 13, 57),
                     datetime.datetime(1990, 12, 9), np.nan,
                     datetime.datetime(1990, 12, 9),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1898, 1, 2),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1950, 12, 9)],

        'bool_tf': [True, True, False, True, False, True, True, False,
                    True],

        'bool_tf_with_nan': [True, False, False, False, False, True, True,
                             False, np.nan],

        'bool_01': [1, 1, 0, 1, 1, 0, 0, 0, 1],

        'bool_01_with_nan': [1, 0, 1, 0, 0, 1, 1, 0, np.nan],

        'mixed': [1, 2, "a", 4, 5, 6, 7, 8, 9]

    }

    df_1 = pd.DataFrame(data_1)

    df_1_expected = {'bool_01': {0: 4, 1: 5},
                     'bool_01_with_nan': {0.0: 4, 1.0: 4},
                     'bool_tf': {False: 3, True: 6},
                     'bool_tf_with_nan': {
                         False: 5, True: 3}, 's1': {1.0: 9},
                     'x': np.array([1, 3, 1, 0, 1, 0, 0, 0, 0, 2],
                                   dtype=np.int64),
                     'y': np.array([6, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                                   dtype=np.int64)}
    res = plot(df_1, force_cat=['bool_01', 'bool_01_with_nan', 's1'])

    assert res['bool_01'] == df_1_expected['bool_01']
    assert res['bool_01_with_nan'] == df_1_expected['bool_01_with_nan']
    assert res['bool_tf'] == df_1_expected['bool_tf']
    assert res['bool_tf_with_nan'] == df_1_expected['bool_tf_with_nan']
    assert res['s1'] == df_1_expected['s1']
    assert np.all(res['x'] == df_1_expected['x'])
    assert np.all(res['y'] == df_1_expected['y'])

    data = {

        'id': [chr(97 + c) for c in range(1, 21)],

        'x': ['d', 'c', 'b', 'a', 'b', 'd', 'c', 'a', 'a', 'a', 'c', 'b',
              'c', 'a', 'd', 'b', 'b', 'b', 'b', 'b'],

        'y': [794, 652, 158, 134, 448, 682, 135, 795, 353, 395, 403, 498,
              622, 80, 654, 772, 867, 676, 670, 736],

        's1': np.ones(20),

        'somedate': [datetime.date(2011, 7, 4),
                     datetime.datetime(1898, 1, 2),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1898, 1, 2),
                     datetime.datetime(1990, 12, 9), np.nan,
                     datetime.datetime(1990, 12, 9),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1898, 1, 2),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1950, 12, 9),
                     datetime.datetime(1950, 12, 9)],
    }

    df_data = pd.DataFrame(data)

    df_expected = {'a': {'25%': 134.0,
                         '50%': 353.0,
                         '75%': 395.0,
                         'iqr': 261.0,
                         'max': 395,
                         'min': 80,
                         'outliers': [795]},
                   'b': {'25%': 485.5,
                         '50%': 673.0,
                         '75%': 745.0,
                         'iqr': 259.5,
                         'max': 867,
                         'min': 158,
                         'outliers': []},
                   'c': {'25%': 336.0,
                         '50%': 512.5,
                         '75%': 629.5,
                         'iqr': 293.5,
                         'max': 652,
                         'min': 135,
                         'outliers': []},
                   'd': {'25%': 668.0,
                         '50%': 682.0,
                         '75%': 738.0,
                         'iqr': 70.0,
                         'max': 794,
                         'min': 654,
                         'outliers': []}}
    res = plot(df_data, 'y', 'x')

    assert res['a'] == df_expected['a']
    assert res['b'] == df_expected['b']
    assert res['c'] == df_expected['c']
    assert res['d'] == df_expected['d']

    df_expected_2 = {('a', Timestamp('1898-01-02 00:00:00')): 1,
                     ('a', Timestamp('1950-12-09 00:00:00')): 3,
                     ('a', Timestamp('1990-12-09 00:00:00')): 1,
                     ('b', Timestamp('1898-01-02 00:00:00')): 1,
                     ('b', Timestamp('1950-12-09 00:00:00')): 7,
                     ('c', Timestamp('1898-01-02 00:00:00')): 1,
                     ('c', Timestamp('1950-12-09 00:00:00')): 2,
                     ('d', Timestamp('1950-12-09 00:00:00')): 1,
                     ('d', Timestamp('1990-12-09 00:00:00')): 1,
                     ('d', Timestamp('2011-07-04 00:00:00')): 1
                    }

    res_2 = plot(df_data, 'x', 'somedate')
    assert df_expected_2 == res_2

def test_corner():
    """

    :return:
    """
    df_2 = pd.DataFrame(
        {'all_nan': [np.nan for _ in range(10)], 'all_one': np.ones(10),
         'all_zeros': np.zeros(10), 'random': np.array(
             [0.38538395, 0.13609054, 0.15973238, 0.96192966, 0.03708882,
              0.03633855, 0.25260128, 0.72139843, 0.74553949,
              0.41102021])})

    df_1_expected = {'all_one': {1.0: 10},
                     'all_zeros': {0.0: 10},
                     'random': np.array([2, 2, 1, 1, 1, 0, 0, 2, 0, 1],
                                        dtype=np.int64)}

    res = plot(df_2, force_cat=['all_one', 'all_zeros'])

    # assert np.all(res['all_nan'] ==  .df_1_expected['all_nan'])
    assert res['all_one'] == df_1_expected['all_one']
    assert res['all_zeros'] == df_1_expected['all_zeros']

    df_2 = pd.DataFrame({
        'empty': [],
        'another_empty': []
    })

    df_2_expected = dict()

    res = plot(df_2, 'empty', 'another_empty')
    assert res == df_2_expected
