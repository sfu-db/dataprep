"""
    module for testing plot_corr(df, x, y) function.
"""
from time import time
import random
import numpy as np
import pandas as pd

from ...eda import plot_correlation


def test_plot_corr_df(  # pylint: disable=too-many-locals
) -> None:
    """
    :return:
    """
    data = np.random.rand(100, 20)
    df_data = pd.DataFrame(data)

    start_p_pd = time()
    res = df_data.corr(method='pearson')
    end_p_pd = time()
    print("pd pearson time: ", str(end_p_pd - start_p_pd) + " s")

    start_p = time()
    _, intermediate = plot_correlation(
        pd_data_frame=df_data,
        return_intermediate=True
    )
    end_p = time()
    print("our pearson time: ", str(end_p - start_p) + " s")
    assert np.isclose(res, intermediate.result['corr_p']).all()

    start_s_pd = time()
    res = df_data.corr(method='spearman')
    end_s_pd = time()
    print("pd spearman time: ", str(end_s_pd - start_s_pd) + " s")

    start_s = time()
    _, intermediate = plot_correlation(
        pd_data_frame=df_data,
        return_intermediate=True
    )
    end_s = time()
    print("our spearman time: ", str(end_s - start_s) + " s")
    assert np.isclose(res, intermediate.result['corr_s']).all()

    start_k_pd = time()
    res = df_data.corr(method='kendall')
    end_k_pd = time()
    print("pd kendall time: ", str(end_k_pd - start_k_pd) + " s")

    start_k = time()
    _, intermediate = plot_correlation(
        pd_data_frame=df_data,
        return_intermediate=True
    )
    end_k = time()
    print("our kendall time: ", str(end_k - start_k) + " s")
    assert np.isclose(res, intermediate.result['corr_k']).all()


def test_plot_corr_df_k() -> None:
    """
    :return:
    """
    data = np.random.rand(100, 20)
    df_data = pd.DataFrame(data)
    k = 5
    res = df_data.corr(method='pearson')
    row, _ = np.shape(res)
    res_re = np.reshape(
        np.triu(res, 1),
        (row * row,)
    )
    idx = np.argsort(res_re)
    mask = np.zeros(
        shape=(row * row,)
    )
    for i in range(k):
        if res_re[idx[i]] < 0:
            mask[idx[i]] = 1
        if res_re[idx[-i - 1]] > 0:
            mask[idx[-i - 1]] = 1
    res = np.multiply(res_re, mask)
    res = np.reshape(
        res,
        (row, row)
    )
    res += res.T - np.diag(res.diagonal())
    _, intermediate = plot_correlation(
        pd_data_frame=df_data,
        return_intermediate=True,
        k=k
    )
    assert np.isclose(intermediate.result['corr_p'], res).all()
    assert np.isclose(intermediate.result['mask_p'], mask).all()


def test_plot_corr_df_x_k() -> None:
    """
    :return:
    """
    df_data = pd.DataFrame({'a': np.random.normal(0, 10, 100)})
    df_data['b'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['c'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['d'] = df_data['a'] + np.random.normal(0, 10, 100)
    x_name = 'b'
    k = 3
    name_list = list(df_data.columns.values)
    idx_name = name_list.index(x_name)
    res_p = df_data.corr(method='pearson').values
    res_p[idx_name][idx_name] = -1
    res_s = df_data.corr(method='spearman').values
    res_s[idx_name][idx_name] = -1
    res_k = df_data.corr(method='kendall').values
    res_k[idx_name][idx_name] = -1
    _, _ = plot_correlation(
        pd_data_frame=df_data,
        x_name=x_name,
        return_intermediate=True,
        k=k
    )


def test_plot_corr_df_x_y_k() -> None:
    """
    :return:
    """
    df_data = pd.DataFrame({'a': np.random.normal(0, 10, 100)})
    df_data['b'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['c'] = df_data['a'] + np.random.normal(0, 10, 100)
    df_data['d'] = df_data['a'] + np.random.normal(0, 10, 100)
    x_name = 'b'
    y_name = 'c'
    k = 3
    _ = plot_correlation(
        pd_data_frame=df_data,
        x_name=x_name,
        y_name=y_name,
        return_intermediate=False,
        k=k
    )

    letters = ['a', 'b', 'c']
    df_data_cat = pd.DataFrame({'a': np.random.normal(0, 10, 100)})
    df_data_cat['b'] = pd.Categorical([random.choice(letters) for _ in range(100)])
    df_data_cat['c'] = pd.Categorical([random.choice(letters) for _ in range(100)])
    _, intermediate = plot_correlation(
        pd_data_frame=df_data_cat,
        x_name='b',
        y_name='c',
        return_intermediate=True
    )
    assert np.isclose(pd.crosstab(df_data_cat['b'], df_data_cat['c']).values,
                      intermediate.result['cross_table']).all()
