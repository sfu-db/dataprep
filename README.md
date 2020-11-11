<div align="center"><img width="100%" src="https://github.com/sfu-db/dataprep/raw/develop/assets/logo.png"/></div>

-----------------
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/dataprep?style=flat-square"/></a>
  <a href="https://sfu-db.github.io/dataprep/"><img src="https://img.shields.io/badge/dynamic/json?color=blue&label=docs&prefix=v&query=%24.info.version&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fdataprep%2Fjson&style=flat-square"/></a>
  <a href="https://pypi.org/project/dataprep/"><img src="https://img.shields.io/pypi/pyversions/dataprep?style=flat-square"/></a>
  <a href="https://www.codacy.com/gh/sfu-db/dataprep?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=sfu-db/dataprep&amp;utm_campaign=Badge_Coverage"><img src="https://app.codacy.com/project/badge/Coverage/ed658f08dcce4f088c850253475540ba"/></a>
<!--   <a href="https://codecov.io/gh/sfu-db/dataprep"><img src="https://img.shields.io/codecov/c/github/sfu-db/dataprep?style=flat-square"/></a> -->
  <a href="https://www.codacy.com/gh/sfu-db/dataprep?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=sfu-db/dataprep&amp;utm_campaign=Badge_Grade"><img src="https://app.codacy.com/project/badge/Grade/ed658f08dcce4f088c850253475540ba"/></a>
  <a href="https://discord.gg/xwbkFNk"><img src="https://img.shields.io/discord/702765817154109472?style=flat-square"/></a>
</p>


<p align="center">
  <a href="https://sfu-db.github.io/dataprep/">Documentation</a>
  |
  <a href="https://discord.gg/xwbkFNk">Forum</a>
  | 
  <a href="https://groups.google.com/forum/#!forum/dataprep">Mail List</a>
</p>

Dataprep lets you prepare your data using a single library with a few lines of code.

Currently, you can use `dataprep` to:
* Collect data from common data sources (through `dataprep.connector`)
* Do your exploratory data analysis (through `dataprep.eda`)
* ...more modules are coming

## Releases

<div align="center">
  <table>
    <tr>
      <th>Repo</th>
      <th>Version</th>
      <th>Downloads</th>
    </tr>
    <tr>
      <td>PyPI</td>
      <td><a href="https://pypi.org/project/dataprep/"><img src="https://img.shields.io/pypi/v/dataprep?style=flat-square"/></a></td>
      <td><a href="https://pepy.tech/project/dataprep"><img src="https://pepy.tech/badge/dataprep"/></a></td>
    </tr>
    <tr> 
      <td>conda-forge</td>
      <td><a href="https://anaconda.org/conda-forge/dataprep"><img src="https://img.shields.io/conda/vn/conda-forge/dataprep.svg"/></a></td>
      <td><a href="https://anaconda.org/conda-forge/dataprep"><img src="https://img.shields.io/conda/dn/conda-forge/dataprep.svg"/></a></td>
    </tr>
  </table>
</div>


## Installation

```bash
pip install -U dataprep
```

## Examples & Usages

The following examples can give you an impression of what dataprep can do:

* [Documentation: Connector](https://sfu-db.github.io/dataprep/user_guide/connector/connector.html)
* [Documentation: EDA](https://sfu-db.github.io/dataprep/user_guide/eda/introduction.html)
* [EDA Case Study: Titanic](https://sfu-db.github.io/dataprep/user_guide/eda/titanic.html)
* [EDA Case Study: House Price](https://sfu-db.github.io/dataprep/user_guide/eda/house_price.html)

### EDA

There are common tasks during the exploratory data analysis stage, 
like a quick look at the columnar distribution, or understanding the correlations
between columns. 

The EDA module categorizes these EDA tasks into functions helping you finish EDA
tasks with a single function call.

* Want to understand the distributions for each DataFrame column? Use `plot`.

<a href="https://sfu-db.github.io/dataprep/user_guide/eda/plot.html#Get-an-overview-of-the-dataset-with-plot(df)"><img src="https://github.com/sfu-db/dataprep/raw/develop/assets/plot(df).gif"/></a>

* Want to understand the correlation between columns? Use `plot_correlation`.

<a href="https://sfu-db.github.io/dataprep/user_guide/eda/plot_correlation.html#Get-an-overview-of-the-correlations-with-plot_correlation(df)"><img src="https://github.com/sfu-db/dataprep/raw/develop/assets/plot_correlation(df).gif"/></a>

* Or, if you want to understand the impact of the missing values for each column, use `plot_missing`.

<a href="https://sfu-db.github.io/dataprep/user_guide/eda/plot_missing.html#Get-an-overview-of-the-missing-values-with-plot_missing(df)"><img src="https://github.com/sfu-db/dataprep/raw/develop/assets/plot_missing(df).gif"/></a>

You can drill down to get more information by given `plot`, `plot_correlation` and `plot_missing` a column name.: E.g. for `plot_missing`

<a href="https://sfu-db.github.io/dataprep/user_guide/eda/plot_missing.html#Understand-the-impact-of-the-missing-values-in-column-x-with-plot_missing(df,-x)"><img src="https://github.com/sfu-db/dataprep/raw/develop/assets/plot_missing(df, x).gif"/></a>

&nbsp;&nbsp;&nbsp;&nbsp;for numerical column using`plot`:

<a href="https://sfu-db.github.io/dataprep/user_guide/eda/plot.html#Understand-a-column-with-plot(df,-x)"><img src="https://github.com/sfu-db/dataprep/raw/develop/assets/plot(df,x)_num.gif"/></a>

&nbsp;&nbsp;&nbsp;&nbsp;for categorical column using`plot`:

<a href="https://sfu-db.github.io/dataprep/user_guide/eda/plot.html#Understand-a-column-with-plot(df,-x)"><img src="https://github.com/sfu-db/dataprep/raw/develop/assets/plot(df,x)_cat.gif"/></a>

Don't forget to checkout the [examples] folder for detailed demonstration!

### Connector

Connector provides a simple programming interface to collect structured data from different Web APIs (e.g., Twitter API, Yelp Fusion API, Spotify API, DBLP API), making web data collection easy and efficient, without requiring advanced programming skills.


Do you want to leverage the growing number of websites that are opening their data through public APIs?.


Connector is for you!  

Let's check out the several benefits that Connector offers:

* A unified API: you can fetch data using one or two lines of code to get data from many websites.
<img src="https://github.com/sfu-db/dataprep/raw/develop/assets/connector_main.gif"/></a>
* Auto Pagination: it automatically does the pagination for you so that you can specify the desired count of the returned results without even considering the count-per-request restriction from the API.
<img src="https://github.com/sfu-db/dataprep/raw/develop/assets/connector_pagination.gif"/></a>
* Smart API request strategy: it can issue API requests in parallel while respecting the rate limit policy.
<img src="https://github.com/sfu-db/dataprep/raw/develop/assets/connector_concurrency.gif"/></a>

In the following examples, you can download the Yelp business search result into a pandas DataFrame, 
using only two lines of code, without taking deep looking into the Yelp documentation!
More examples can be found here:
[Examples](https://github.com/sfu-db/dataprep/tree/develop/examples)

<center><a href="https://sfu-db.github.io/dataprep/connector.html#getting-web-data-with-connector-query"><img src="https://github.com/sfu-db/dataprep/raw/develop/assets/connector.png"/></a></center>

### Clean

DataPrep.Clean contains simple functions designed for cleaning and standardizing a column in a DataFrame. It provides
* A unified API: each function follows the syntax `clean_{type}(df, "column name")` (see an example below)
* Python Data Science Support: its design for cleaning pandas and Dask DataFrames enables seamless integration into the Python data science workflow
* Transparency: a report is generated that summarizes the alterations to the data that occured during cleaning

The following example shows how to clean a column containing messy emails:
<center><img src="https://github.com/sfu-db/dataprep/blob/develop/assets/clean_example_1.jpg"/></center>
<center><img src="https://github.com/sfu-db/dataprep/blob/develop/assets/clean_example_2.jpg"/></center>

Type validation is also supported:
<center><img src="https://github.com/sfu-db/dataprep/blob/develop/assets/clean_example_3.jpg"/></center>

Below are the supported semantic types (more are currently being developed).
<table>
    <tr>
      <th>Semantic Types</th>
    </tr>
    <tr>
      <td>longitude/latitude</td>
    </tr>
    <tr>
      <td>country</td>
    </tr>
    <tr>
      <td>email</td>
    </tr>
    <tr>
      <td>url</td>
    </tr>
    <tr>
      <td>phone</td>
    </tr>
  </table>

For more information, refer to the [User Guide](https://sfu-db.github.io/dataprep/user_guide/clean/introduction.html).

## Contribute

There are many ways to contribute to Dataprep.

* Submit bugs and help us verify fixes as they are checked in.
* Review the source code changes.
* Engage with other Dataprep users and developers on StackOverflow.
* Help each other in the [Dataprep Community Discord](https://discord.gg/xwbkFNk) and [Mail list & Forum].
* [![Twitter]](https://twitter.com/sfu_db)
* Contribute bug fixes.
* Providing use cases and writing down your user experience.

Please take a look at our [wiki] for development documentations!


[Build Status]: https://img.shields.io/circleci/build/github/sfu-db/dataprep/master?style=flat-square&token=f68e38757f5c98771f46d1c7e700f285a0b9784d
[Mail list & Forum]: https://groups.google.com/forum/#!forum/dataprep
[wiki]: https://github.com/sfu-db/dataprep/wiki
[examples]: https://github.com/sfu-db/dataprep/tree/master/examples
[Twitter]: https://img.shields.io/twitter/follow/sfu_db?style=social

## Acknowledgement

  Some functionalities of DataPrep are inspired by the following packages.
  
- [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling)

  Inspired the report functionality and insights provided in DataPrep.eda.

- [missingno](https://github.com/ResidentMario/missingno) 

  Inspired the missing value analysis in DataPrep.eda.
