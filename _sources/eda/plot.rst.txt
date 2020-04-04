
==================================================
`plot`: analyzing basic characteristics of dataset 
==================================================

.. toctree::
   :maxdepth: 2

Overview
========

The goal of `plot` is to explore basic characteristics of the dataset. It can generate different plots to reveal different characteristics of interested columns. It mainly provides the following functionalities:

1. plot(df): plot basic characteristics (the histogram and the bar chart) for all columns.
2. plot(df, x): zoom into column x and plot more refined characteristics.
3. plot(df, x, y): zoom into column x and column y, and plot more refined characteristics to explore their relationship.


The generated plots of `plot` function are different for numerical column and categorical column. The following table summarizes the output plots for different setting of x and y.

+-------------+------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|             | **plot(df,x,y)** |                                                                                                                                                                                                                                                  |
+-------------+------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **x**       | **y**            | **output plots**                                                                                                                                                                                                                                 |
+-------------+------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| None        | None             | `histgram  <https://www.wikiwand.com/en/Histogram>`_ or `bar chart  <https://www.wikiwand.com/en/Bar_chart>`_ for each column                                                                                                                    |
+-------------+------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Numerical   | None             | `histgram  <https://www.wikiwand.com/en/Histogram>`_, `kde plot  <https://www.wikiwand.com/en/Kernel_density_estimation>`_, `box plot  <https://www.wikiwand.com/en/Box_plot>`_, `qq-norm plot  <https://www.wikiwand.com/en/Q%E2%80%93Q_plot>`_ |
+-------------+------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Categorical | None             | `bar chart  <https://www.wikiwand.com/en/Bar_chart>`_, `pie chart  <https://www.wikiwand.com/en/Pie_chart>`_                                                                                                                                     |
+-------------+------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Numerical   | Numerical        | `scatter plot  <https://www.wikiwand.com/en/Scatter_plot>`_, `hexbin plot <https://www.data-to-viz.com/graph/hexbinmap.html>`_, `box plot  <https://www.wikiwand.com/en/Box_plot>`_                                                              |
+-------------+------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Numerical   | Categorical      | `box plot  <https://www.wikiwand.com/en/Box_plot>`_, `line plot <https://www.wikiwand.com/en/Line_chart>`_                                                                                                                                       |
+-------------+------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Categorical | Numerical        | `box plot  <https://www.wikiwand.com/en/Box_plot>`_, `line plot <https://www.wikiwand.com/en/Line_chart>`_                                                                                                                                       |
+-------------+------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Categorical | Categorical      | `nested bar chart <https://www.wikiwand.com/en/Bar_chart#/Grouped_and_stacked>`_, `stacked bar chart <https://www.wikiwand.com/en/Bar_chart#/Grouped_and_stacked>`_, `heat map <https://www.wikiwand.com/en/Heat_map>`_                          |
+-------------+------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Next, we use several examples to demonstrate the functionalities.


Loading dataset
===============
We support two types of dataframe: pandas dataframe and dask dataframe. Here we load the well known `adult` dataset into a pandas dataframe and use it to demonstrate our functionality::

    import pandas as pd
    df = pd.read_csv("https://www.openml.org/data/get_csv/1595261/phpMawTba", na_values = [' ?'])

Basic exploration for all columns via `plot(df)`
================================================

After getting a dataset, we could do a rough exploration by calling `plot(df)`. It will plot a histogram for each numeric column and a bar chart for each categorical column. The shown bin number (of histogram) and shown category number (of categorical column) are both customizable. Besides, if a column contains missing values, it ignores them when generating the plot but shows the percentage of missing values in the title. The following shows an example of `plot(df)`::

    from dataprep.eda import plot
    plot(df)


.. raw:: html

   <iframe src="../_static/images/plot/plot_df.html" height="520" width="100%"></iframe>


Zooming into a column via `plot(df, x)`
=======================================

After we get the basic information of the dataset, we could zoom into an interested column to explore it more by calling `plot(df, x)`, where x is the interested column.  The output is of `plot(df, x)` is different for numerical column and categorical column.

When x is a numeric column, it plots the histogram, kde plot, box plot and qq-norm plot. The following shows an example::
    
    plot(df, "age")

.. raw:: html

   <iframe src="../_static/images/plot/plot_df_age.html" height="450" width="100%"></iframe>


When x is a categorical column, it plots bar chart and pie chart. The following shows an example::

    plot(df, "education")

.. raw:: html

   <iframe src="../_static/images/plot/plot_df_education.html" height="450" width="100%"></iframe>


Zooming into two columns via `plot(df, x, y)`
=============================================

Furthermore, we provide `plot(df, x, y)` to explore the relationship between interested columns x and y. The output is based on the column types of x and y.

When x and y are both numerical columns, it plots `scatter plot  <https://www.wikiwand.com/en/Scatter_plot>`_, `hexbin plot <https://www.data-to-viz.com/graph/hexbinmap.html>`_ and `box plot  <https://www.wikiwand.com/en/Box_plot>`_. The following shows an example::

    plot(df, "age", "hours-per-week")

.. raw:: html

   <iframe src="../_static/images/plot/plot_df_age_hours.html" height="450" width="100%"></iframe>
  

When x and y are both categorical columns, it plots `nested bar chart <https://www.wikiwand.com/en/Bar_chart#/Grouped_and_stacked>`_, `stacked bar chart <https://www.wikiwand.com/en/Bar_chart#/Grouped_and_stacked>`_ and `heat map <https://www.wikiwand.com/en/Heat_map>`_ . The following shows an example::

    plot(df, "education", "marital-status")

.. raw:: html

   <iframe src="../_static/images/plot/plot_df_education_marital.html" height="400" width="100%"></iframe>


When one of x and y is a numerical column and the other is categorical column, it plots `box plot  <https://www.wikiwand.com/en/Box_plot>`_ and `line plot <https://www.wikiwand.com/en/Line_chart>`_. The following shows an example::

    plot(df, "age", "education")
    # or plot(df, "education", "age")

.. raw:: html

   <iframe src="../_static/images/plot/plot_df_education_age.html" height="450" width="100%"></iframe>