=============================================================
`plot_correlation`: analyzing the correlation between columns
=============================================================

.. toctree::
   :maxdepth: 2

 
Overview
========

The goal of `plot_correlation` is to analyze the correlation between columns. It provides the following functionalities:

1. `plot_correlation(df)`: plot the correlation matrix of all columns.
2. `plot_correlation(df, x)`: plot the most correlated columns to column x.
3. `plot_correlation(df, x, y)`: plot the scatter plot between column x and column y, as well as the regression line. Besides, the point that has most impact on the correlation value could be identified by passing a parameter.
4. `plot_correlation(df, x, y, k, value_range)`: filter the result by correlation value or by top-k.

..
    The following table summarizes the output plots for different setting of x and y.

    +-------------+-------------------------------+--------------------------------------------------------------------------------------------------------+
    |             | **plot_correlation(df,x, y)** |                                                                                                        |
    +-------------+-------------------------------+--------------------------------------------------------------------------------------------------------+
    | **x**       | **y**                         | **output plots**                                                                                       |
    +-------------+-------------------------------+--------------------------------------------------------------------------------------------------------+
    | None        | None                          | n*n correlation matrix for Person, Spearman and KendallTau correlation, where n is max(50, df.columns) |
    +-------------+-------------------------------+--------------------------------------------------------------------------------------------------------+
    | Numerical   | None                          | n*1 correlation matrix for Person, Spearman and KendallTau correlation                                 |
    +-------------+-------------------------------+--------------------------------------------------------------------------------------------------------+
    | Categorical | None                          | TODO                                                                                                   |
    +-------------+-------------------------------+--------------------------------------------------------------------------------------------------------+
    | Numerical   | Numerical                     | `scatter plot  <https://www.wikiwand.com/en/Scatter_plot>`_ with regression line                       |
    +-------------+-------------------------------+--------------------------------------------------------------------------------------------------------+
    | Numerical   | Categorical                   | TODO                                                                                                   |
    +-------------+-------------------------------+--------------------------------------------------------------------------------------------------------+
    | Categorical | Numerical                     | TODO                                                                                                   |
    +-------------+-------------------------------+--------------------------------------------------------------------------------------------------------+
    | Categorical | Categorical                   | TODO                                                                                                   |
    +-------------+-------------------------------+--------------------------------------------------------------------------------------------------------+

In the following, we use several examples to demonstrate the functionalities.


Loading dataset
===============
We support two types of dataframe: pandas dataframe and dask dataframe. Here we load the well known `wine quality` dataset into a pandas dataframe and use it to demonstrate our functionality::

    import pandas as pd
    df = pd.read_csv("https://www.openml.org/data/get_csv/4965268/wine-quality-red.arff")


Plotting correlation matrix via `plot_correlation(df)`
======================================================

After getting a dataset, we could plot the correlation matrix of all columns by calling `plot_correlation(df)`. We will compute three types of correlations (`Person  <https://www.wikiwand.com/en/Pearson_correlation_coefficient>`_, `Spearman  <https://www.wikiwand.com/en/Spearman%27s_rank_correlation_coefficient>`_ and `KendallTau  <https://www.wikiwand.com/en/Kendall_rank_correlation_coefficient>`_) and for each of them generating a correlation matrix.  In the matrix, each cell represents the correlation value of two columns. The following shows an example::

    from dataprep.eda import plot_correlation
    plot_correlation(df)


.. raw:: html

   <iframe src="../_static/images/plot_correlation/df.html" height="550" width="100%"></iframe>


Finding the most correlated columns via `plot_correlation(df, x)`
=================================================================

After getting the correlation matrix, user may zoom into a column and explore how other columns correlated to it. To achieve this goal, we provide `plot_correlation(df, x)`. It computes the correlations (`Person  <https://www.wikiwand.com/en/Pearson_correlation_coefficient>`_, `Spearman  <https://www.wikiwand.com/en/Spearman%27s_rank_correlation_coefficient>`_ and `KendallTau  <https://www.wikiwand.com/en/Kendall_rank_correlation_coefficient>`_) of the interested column x to all other columns and sorting them based on the correlation values. In this case, user could know which column is most correlated or un-correlated to column x. The following shows an example::
    
    plot_correlation(df, "alcohol")

.. raw:: html

   <iframe src="../_static/images/plot_correlation/df_num.html" height="550" width="100%"></iframe>


Exploring the correlation between two columns via `plot_correlation(df, x, y)`
===============================================================================

Furthermore, we provide `plot_correlation(df, x, y)` to allow user analyze the correlation between two columns. It plots a scatter plot of column x and y, along with a regression line. The following shows an example::

    plot_correlation(df, "alcohol", "pH")

.. raw:: html

   <iframe src="../_static/images/plot_correlation/df_num_num.html" height="550" width="100%"></iframe>
  

Besides, when user passes the parameter k, it could identify the k points that have the largest impact on the correlation value. The impact means that after removing the k points, the correlation value will increase most (positive influence) or decrease most (negative influence). The following shows an example::

    plot_correlation(df, "alcohol", "pH", k = 2)

.. raw:: html

   <iframe src="../_static/images/plot_correlation/df_num_num_k.html" height="550" width="100%"></iframe>


Filtering the result by top-k and value range filter
====================================================

We provide two types of filters to filter the result: top-k and value range. They could be applied to `plot_correlation(df)` and `plot_correlation(df, x)` by passing parameter `k` and `value_range`. After applying top-k filter, only the top-k correlation values will be shown. For value range filter, only the the correlation value in a given range will be shown.

The following shows an example of applying top-k filter in `plot_correlation(df)`::

    plot_correlation(df, k = 3)

.. raw:: html

   <iframe src="../_static/images/plot_correlation/df_k.html" height="550" width="100%"></iframe>


The following shows an example of applying value range filter in `plot_correlation(df, x)`::

    plot_correlation(df, "alcohol", value_range=[0.1, 1])

.. raw:: html

   <iframe src="../_static/images/plot_correlation/df_num_valueRange.html" height="550" width="100%"></iframe>
