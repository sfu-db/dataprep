import dask
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt


def cov(matrix):
    return np.cov(matrix)


def std(matrix):
    return np.sqrt(np.diag(matrix))


def divide(matrix, arr):
    return matrix / arr[:, None] / arr[None, :]


def plot_corr(df):
    start1 = time()
    df.corr()
    end1 = time()
    print("pandas:" + str(end1 - start1) + " s")
    # print(df.corr())

    a = np.random.randint(0, 50, 10000000)
    b = a + np.random.normal(0, 10, 10000000)
    c = a + np.random.normal(0, 10, 10000000)
    d = a + np.random.normal(0, 10, 10000000)
    res = np.vstack((a, b, c, d))
    start2 = time()
    np.corrcoef(res)
    end2 = time()
    print("numpy:" + str(end2 - start2) + " s")
    # print(np.corrcoef(res))

    start3 = time()
    cal_matrix = df.values
    # cov_xy = np.cov(cal_matrix.T)
    # std_xy = np.sqrt(np.diag(cov_xy))
    # result = cov_xy / std_xy[:, None] / std_xy[None, :]
    cov_xy = dask.delayed(cov)(cal_matrix.T)
    std_xy = dask.delayed(std)(cov_xy)
    result = dask.delayed(divide)(cov_xy, std_xy).compute()
    end3 = time()
    print(str(end3 - start3) + " s")
    # print(result)

    return result


if __name__ == '__main__':
    # df = pd.DataFrame({'a': np.random.randint(0, 50, 10000000)})
    # df['b'] = df['a'] + np.random.normal(0, 10, 10000000)
    # df['c'] = df['a'] + np.random.normal(0, 10, 10000000)
    # df['d'] = df['a'] + np.random.normal(0, 10, 10000000)
    df = pd.read_csv('normal.csv')
    res = plot_corr(df)
    plt.matshow(res)
    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(len(df.columns)), df.columns)
    plt.colorbar()
    plt.show()
