import datetime

import numpy as np
import pandas as pd

from ...eda.EDA_plot import give_count, give_hist, plot


class TestClass(object):

    def test_normal(self):
        self.data_1 = {

            'id': [chr(97 + c) for c in range(1, 10)],

            'x': [50, 50, -10, 0, 0, 5, 15, -3, None],

            'y': [0.000001, 654.152, None, 15.984512, 3122, -3.1415926535, 111, 15.9, 13.5],

            's1': np.ones(9),

            'somedate': [datetime.date(2011, 7, 4), datetime.datetime(2022, 1, 1, 13, 57), datetime.datetime(1990, 12, 9), np.nan, datetime.datetime(1990, 12, 9), datetime.datetime(1950, 12, 9), datetime.datetime(1898, 1, 2), datetime.datetime(1950, 12, 9), datetime.datetime(1950, 12, 9)],

            'bool_tf': [True, True, False, True, False, True, True, False, True],

            'bool_tf_with_nan': [True, False, False, False, False, True, True, False, np.nan],

            'bool_01': [1, 1, 0, 1, 1, 0, 0, 0, 1],

            'bool_01_with_nan': [1, 0, 1, 0, 0, 1, 1, 0, np.nan],

            'mixed': [1, 2, "a", 4, 5, 6, 7, 8, 9]

        }

        self.df_1 = pd.DataFrame(self.data_1)

        self.df_1_expected = {'bool_01': {0: 4, 1: 5}, 'bool_01_with_nan': {0.0: 4, 1.0: 4}, 'bool_tf': {False: 3, True: 6}, 'bool_tf_with_nan': {
            False: 5, True: 3}, 's1': {1.0: 9}, 'x': np.array([1, 3, 1, 0, 1, 0, 0, 0, 0, 2], dtype=np.int64), 'y': np.array([6, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.int64)}
        res = plot(self.df_1, force_cat=['bool_01', 'bool_01_with_nan', 's1'])
        assert res['bool_01'] == self.df_1_expected['bool_01']
        assert res['bool_01_with_nan'] == self.df_1_expected['bool_01_with_nan']
        assert res['bool_tf'] == self.df_1_expected['bool_tf']
        assert res['bool_tf_with_nan'] == self.df_1_expected['bool_tf_with_nan']
        assert res['s1'] == self.df_1_expected['s1']
        assert np.all(res['x'] == self.df_1_expected['x'])
        assert np.all(res['y'] == self.df_1_expected['y'])

    def test_corner(self):

        self.df_2 = pd.DataFrame({'all_nan': [np.nan for _ in range(10)], 'all_one': np.ones(10), 'all_zeros': np.zeros(10), 'random': np.array([0.38538395,  0.13609054,  0.15973238,  0.96192966,  0.03708882,
                                                                                                                                                 0.03633855,  0.25260128,  0.72139843,  0.74553949,  0.41102021])})

        self.df_1_expected = {'all_one': {1.0: 10},
                              'all_zeros': {0.0: 10},
                              'random': np.array([2, 2, 1, 1, 1, 0, 0, 2, 0, 1], dtype=np.int64)}

        res = plot(self.df_2, force_cat=['all_one', 'all_zeros'])

        #assert np.all(res['all_nan'] == self.df_1_expected['all_nan'])
        assert res['all_one'] == self.df_1_expected['all_one']
        assert res['all_zeros'] == self.df_1_expected['all_zeros']
