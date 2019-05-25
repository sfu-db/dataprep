import datetime
import numpy as np
import pandas as pd
from EDA_plot_2 import plot
from EDA_plot_2 import boxplot
from EDA_plot_2 import stackedplot
from EDA_plot_2 import scatterplot
from pandas import Timestamp


class TestClass_EDA_2(object):
    
    def test_normal(self):
        self.data = {

            'id': [chr(97 + c) for c in range(1, 21)],

            'x' : ['d', 'c', 'b', 'a', 'b', 'd', 'c', 'a', 'a', 'a', 'c', 'b', 'c', 'a', 'd', 'b', 'b', 'b', 'b', 'b'],

            'y' : [794, 652, 158, 134, 448, 682, 135, 795, 353, 395, 403, 498, 622, 80, 654, 772, 867, 676, 670, 736],

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

        self.df = pd.DataFrame(self.data)
        
        self.df_expected = {'a': {  '25%': 134.0,
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
        res = plot(self.df, 'y', 'x') 
       
        assert res['a'] == self.df_expected['a']
        assert res['b'] == self.df_expected['b']
        assert res['c'] == self.df_expected['c']
        assert res['d'] == self.df_expected['d']
        
        self.df_expected_2 = {   ('a', Timestamp('1898-01-02 00:00:00')): 1,
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

        res_2 = plot(self.df, 'x', 'somedate')
        assert self.df_expected_2 == res_2
        
    
    def test_corner(self):
        
        self.df_2 = pd.DataFrame({
                    'empty' : [],
                    'another_empty' : []
            })
        
        self.df_2_expected = dict()
        
        res = plot(self.df_2, 'empty', 'another_empty')
        assert res == self.df_2_expected
        