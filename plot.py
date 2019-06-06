import dask
from dask import delayed
import dask.dataframe as dd
import dask.array as da


def f1(dataframe, col):
    x = dataframe.groupby(col)[col].count()
    return dict(x)


def f2(dataframe, col):
    minv = dataframe[col].min()
    maxv = dataframe[col].max()
    # print ('min = ',  minv, 'maxv = ', maxv)
    # print (dataframe[col])
    dframe = dd.from_array(dataframe[col]).dropna()
    h, b = da.histogram(dframe.values, range=[minv, maxv], bins=10)
    return h


def plot(df, unique_threshold):
    # df = pd.read_csv('C:/Users/sladdha/Desktop/DataPrep/Datasets/Normal.csv')

    ls = list()
    for col in df.columns:
        ls.append(delayed(df[col].nunique)())

    x, = dask.compute(ls)
    y, = dask.compute(x)
    result = list()

    test = []

    for i, col in enumerate(df.columns):
        if (df[col].count() == 0):
            continue

        if (y[i] / df[col].count() < unique_threshold):  # categorial
            cnt_series = delayed(f1)(df, col)
            # grp_cnt = delayed(dict)(cnt_series)
            # print('cat')
            result.append(cnt_series)
            test.append(col)
        elif (df[col].dtype == 'float64' or df[col].dtype == 'int64'):  # numeric
            # print('num')
            hist = f2(df, col)
            result.append(hist)
            test.append(col)

    computed_res, = dask.compute(result)
    column_dict = dict()

    for i, res in enumerate(computed_res):
        column_dict[test[i]] = res

    return (column_dict)

