
==================================================================
`create_report`: generate profile reports from a pandas DataFrame
==================================================================

.. toctree::
   :maxdepth: 2

Overview
========

The goal of `create_report` is to generate profile reports from a pandas DataFrame. `create_report` utilizes the functionalities and formats the plots from `dataprep`. It provides the following information:

1. Overview: detect the types of columns in a dataframe
2. Variables: variable type, unique values, distint count, missing values
3. Quantile statistics like minimum value, Q1, median, Q3, maximum, range, interquartile range
4. Descriptive statistics like mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness
5. Text analysis for length, sample and letter
6. Correlations: highlighting of highly correlated variables, Spearman, Pearson and Kendall matrices
7. Missing Values: bar chart, heatmap and spectrum of missing values

In the following, we break down the report into different sections to demonstrate each section of the report.

Loading dataset
===============
Here we load the `titanic` dataset into a pandas dataframe and use it to demonstrate our functionality::

    import pandas as pd
    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

Generate report via `create_report(df)`
=======================================

After getting a dataset, we could generate the report by calling `create_report(df)`. The following shows an example::

    from dataprep.eda import create_report
    create_report(df)

You can see the full report :download:`here <../../_static/images/create_report/titanic_dp.html>`

`Overview` section
==================

In this section, we can see the types of columns and the statistics of the dataset.

.. raw:: html

   <iframe src="../_static/images/create_report/overview.html" height="340" width="100%"></iframe>

`Variables` section
===================

In this section, we can see the statistics and plots for each of variable in the dataset.

For numerical variable, the report shows quantile statistics, descriptive statistics, histogram, KDE plot, QQ norm plot and box plot.

For categorical variable, the report shows text analysis, bar chart, pie chart, word cloud, word frequencies and word length.

For datetime variable, the report shows line chart

.. raw:: html

    <iframe src="../_static/images/create_report/variables_num.html" height="270" width="100%"></iframe>

.. raw:: html

    <iframe src="../_static/images/create_report/variables_cat.html" height="270" width="100%"></iframe>

`Interactions` section
======================

In this section, the report will show an interactive plot, user can use the dropdown menu above the plot to select which two variables user wants to compare.

The plot has scatter plot and the regression line regarding to the two variabes.

.. raw:: html

    <iframe src="../_static/images/create_report/interactions.html" height="620" width="100%"></iframe>


`Correlations` section
======================

In this section, we can see the correlations bewteen variables in Spearman, Pearson and Kendall matrices.

.. raw:: html

    <iframe src="../_static/images/create_report/correlations.html" height="530" width="100%"></iframe>


`Missing Values` section
========================

In this section, we can see the missing values in the dataset through bar chart, spectrum and heatmap.

.. raw:: html

    <iframe src="../_static/images/create_report/missing.html" height="530" width="100%"></iframe>
