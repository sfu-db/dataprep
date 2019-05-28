from EDA_plot import get_type
from typing import Dict
from typing import List
from typing import NewType
from typing import Union
from typing import Tuple
import dask.dataframe as dd
import dask.array as da
import pandas as pd
import numpy as np

DF = NewType('DF', pd.DataFrame)

def boxplot(dataframe: DF, col_x : str, col_y : str) -> Dict[str, dict]:
	""" Returns intermediate stats of the box plot
		of columns col_x and col_y.

	PARAMETERS
	__________
	dataframe: the input dataframe
	col_x : a valid column name of the dataframe
	col_y : a valid column name of the dataframe


	RETURNS
	__________
	a (column_name: data) dict storing the intermediate results
	"""
	cat_col = col_x if (get_type(dataframe[col_x])=='TYPE_CAT') else col_y
    num_col = col_y if (get_type(dataframe[col_x])=='TYPE_CAT') else col_x
    
    dx = dask.dataframe.from_pandas(dataframe, npartitions = 1)
    grp_object = dx.groupby(cat_col)
    #print (cat_col)
    groups = list(dataframe[cat_col].unique())

    res = dict()

    for group in groups:
        stats = dict()
        grp_series = grp_object.get_group(group)[num_col]
        quantiles = grp_series.quantile([.25, .50, .75]).compute()
        stats['25%'], stats['50%'], stats['75%'] = dask.delayed(quantiles[.25]), dask.delayed(quantiles[.50]), dask.delayed(quantiles[.75])
        stats['iqr'] = stats['75%'] - stats['25%']
        outliers = list()
        grp_series = grp_series.compute()
        for i in grp_series.index:
            if grp_series[i] not in numpy.arange(stats['25%'].compute()-1.5*iqr, stats['50%'].compute()+1.5*iqr):
                dask.delayed(outliers.append)(each)
                grp_series.drop(index=i, inplace=True)
        stats['outliers'] = outliers
        stats['min'] = grp_series.min()
        stats['max'] = grp_series.max()
        res[group] = stats
    res, = dask.compute(res)
    return res

def stackedplot(dataframe: DF, col_x : str, col_y : str) -> Dict[str, dict]:
	""" Returns intermediate stats of the stacked column plot
		of columns col_x and col_y.

	PARAMETERS
	__________
	dataframe: the input dataframe
	col_x : a valid column name of the dataframe
	col_y : a valid column name of the dataframe


	RETURNS
	__________
	a (column_name: data) dict storing the intermediate results
	"""
    
    dx = dask.dataframe.from_pandas(dataframe, npartitions = 1)
    grp_object = dx.groupby([col_x, col_y])
    #print (cat_col)
	grp_series = grp_object['b'].count().compute()
    return dict(grp_series)


def scatterplot(dataframe: DF, col_x : str, col_y : str) -> Tuple[Union[int, float], Union[int, float]]:
	""" 
	TODO: WARNING: For very large amount of points, implement Heat Map.

		Returns intermediate stats of the scattered plot
		of columns col_x and col_y.

	PARAMETERS
	__________
	dataframe: the input dataframe
	col_x : a valid column name of the dataframe
	col_y : a valid column name of the dataframe


	RETURNS
	__________
	a (column_name: data) dict storing the intermediate results
	"""
    
    dx = dask.dataframe.from_pandas(dataframe, npartitions = 1)
    series_x = dx[col_x].compute()
    series_y = dx[col_y].compute()

    res = []
    for each in zip(series_x, series_y):
    	res.append(each)

    return res


def plot(dataframe: DF, col_x : str, col_y : str) -> Dict[str, dict]:
	""" Returns intermediate stats of the bi-variate plots 
		of columns col_x and col_y.

	PARAMETERS
	__________
	dataframe: the input dataframe
	col_x : a valid column name of the dataframe
	col_y : a valid column name of the dataframe


	RETURNS
	__________
	a (column_name: data) dict storing the intermediate results
	"""

	type_x = get_type(dataframe[col_x])
	type_y = get_type(dataframe[col_y])
	result = None

	try:
		if (type_y=='TYPE_CAT' and type_x=='TYPE_NUM' or type_y=='TYPE_NUM' and type_x=='TYPE_CAT'):
			result = boxplot(dataframe, col_x, col_y)
		
		elif (type_x=='TYPE_CAT' and type_y=='TYPE_NUM'):
			result = stackplot(dataframe, col_x, col_y)

		elif (type_x=='TYPE_NUM' and type_y=='TYPE_NUM'):
			result = scatterplot(dataframe, col_x, col_y)
		else:
			pass 
			##WARNING: TODO
	except:
		result = dict()

	return result
