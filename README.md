<div align="center"><img width="100%" src="https://github.com/sfu-db/dataprep/raw/develop/assets/logo.png"/></div>

---

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
  <a href="https://discord.gg/xwbkFNk">Discord</a>
  | 
  <a href="https://github.com/sfu-db/dataprep/discussions">Forum</a>
</p>

DataPrep lets you prepare your data using a single library with a few lines of code.

Currently, you can use DataPrep to:

- Collect data from common data sources (through [`dataprep.connector`](#connector))
- Do your exploratory data analysis (through [`dataprep.eda`](#eda))
- Clean and standardize data (through [`dataprep.clean`](#clean))
- ...more modules are coming

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

## Connector

Connector is an intuitive, open-source API wrapper that speeds up development by standardizing calls to multiple APIs as a simple workflow.

Connector provides a simple wrapper to collect structured data from different Web APIs (e.g., Twitter, Spotify), making web data collection easy and efficient, without requiring advanced programming skills.

Do you want to leverage the growing number of websites that are opening their data through public APIs? Connector is for you!

Let's check out the several benefits that Connector offers:

- **A unified API:** You can fetch data using one or two lines of code to get data from [tens of popular websites](https://github.com/sfu-db/DataConnectorConfigs).
- **Auto Pagination:** Do you want to invoke a Web API that could return a large result set and need to handle it through pagination? Connector automatically does the pagination for you! Just specify the desired number of returned results (argument `_count`) without getting into unnecessary detail about a specific pagination scheme.
- **Smart API request strategy:** Do you want to fetch results more quickly by making concurrent requests to Web APIs? Through the `_concurrency` argument, Connector simplifies concurrency, issuing API requests in parallel while respecting the API's rate limit policy.
#### How to fetch all publications of Andrew Y. Ng?

```python
from dataprep.connector import connect
conn_dblp = connect("dblp", _concurrency = 5)
df = await conn_dblp.query("publication", author = "Andrew Y. Ng", _count = 2000)
```

Here, you can find detailed [Examples.](https://github.com/sfu-db/dataprep/tree/develop/examples)

Connector is designed to be easy to extend. If you want to connect with your own web API, you just have to write a simple [configuration file](https://github.com/sfu-db/DataConnectorConfigs/blob/develop/CONTRIBUTING.md#add-configuration-files) to support it. This configuration file describes the API's main attributes like the URL, query parameters, authorization method, pagination properties, etc.

## EDA

DataPrep.EDA is the fastest and the easiest EDA (Exploratory Data Analysis) tool in Python. It allows you to understand a Pandas/Dask DataFrame with a few lines of code in seconds.

#### Create Profile Reports, Fast

You can create a beautiful profile report from a Pandas/Dask DataFrame with the `create_report` function. DataPrep.EDA has the following advantages compared to other tools:

- **10-100X Faster**: DataPrep.EDA is 10-100X faster than Pandas-based profiling tools due to its highly optimized Dask-based computing module.
- **Interactive Visualization**: DataPrep.EDA generates interactive visualizations in a report, which makes the report look more appealing to end users.
- **Big Data Support**: DataPrep.EDA naturally supports big data stored in a Dask cluster by accepting a Dask dataframe as input.

The following code demonstrates how to use DataPrep.EDA to create a profile report for the titanic dataset.

```python
from dataprep.datasets import load_dataset
from dataprep.eda import create_report
df = load_dataset("titanic")
create_report(df).show_browser()
```

Click [here](https://sfu-db.github.io/dataprep/_downloads/c9bf292ac949ebcf9b65bb2a2bc5a149/titanic_dp.html) to see the generated report of the above code.

#### Innovative System Design

DataPrep.EDA is the **_only_** task-centric EDA system in Python. It is carefully designed to improve usability.

- **Task-Centric API Design**: You can declaratively specify a wide range of EDA tasks in different granularities with a single function call. All needed visualizations will be automatically and intelligently generated for you.
- **Auto-Insights**: DataPrep.EDA automatically detects and highlights the insights (e.g., a column has many outliers) to facilitate pattern discovery about the data.
- **[How-to Guide](https://sfu-db.github.io/dataprep/user_guide/eda/how_to_guide.html)** : A how-to guide is provided to show the configuration of each plot function. With this feature, you can easily customize the generated visualizations.

#### Understand the Titanic dataset with Task-Centric API:

<a href="assets/eda_demo.gif"><img src="assets/eda_demo.gif"/></a>

Click [here](https://sfu-db.github.io/dataprep/user_guide/eda/introduction.html) to check all the supported tasks.

Check [plot](https://sfu-db.github.io/dataprep/user_guide/eda/plot.html), [plot_correlation](https://sfu-db.github.io/dataprep/user_guide/eda/plot_correlation.html), [plot_missing](https://sfu-db.github.io/dataprep/user_guide/eda/plot_missing.html) and [create_report](https://sfu-db.github.io/dataprep/user_guide/eda/create_report.html) to see how each function works.

## Clean

DataPrep.Clean contains simple functions designed for cleaning and validating data in a DataFrame. It provides

- **A Unified API**: each function follows the syntax `clean_{type}(df, 'column name')` (see an example below).
- **Speed**: the computations are parallelized using Dask. It can clean **50K rows per second** on a dual-core laptop (that means cleaning 1 million rows in only 20 seconds).
- **Transparency**: a report is generated that summarizes the alterations to the data that occured during cleaning.

The following example shows how to clean and standardize a column of country names.

``` python
from dataprep.clean import clean_country
import pandas as pd
df = pd.DataFrame({'country': ['USA', 'country: Canada', '233', ' tr ', 'NA']})
df2 = clean_country(df, 'country')
df2
           country  country_clean
0              USA  United States
1  country: Canada         Canada
2              233        Estonia
3              tr          Turkey
4               NA            NaN
```

Type validation is also supported:

``` python
from dataprep.clean import validate_country
series = validate_country(df['country'])
series
0     True
1    False
2     True
3     True
4    False
Name: country, dtype: bool
```

**Currently supports functions for:** Column Headers | Country Names | Dates and Times | Duplicate Strings | Email Addresses | Geographic Coordinates | IP Addresses | Phone Numbers | URLs | US Street Addresses

## Documentation

The following documentation can give you an impression of what DataPrep can do:

- [Connector](https://sfu-db.github.io/dataprep/user_guide/connector/connector.html)
- [EDA](https://sfu-db.github.io/dataprep/user_guide/eda/introduction.html)
- [Clean](https://sfu-db.github.io/dataprep/user_guide/clean/introduction.html)

## Contribute

There are many ways to contribute to DataPrep.

- Submit bugs and help us verify fixes as they are checked in.
- Review the source code changes.
- Engage with other DataPrep users and developers on StackOverflow.
- Help each other in the [DataPrep Community Discord](https://discord.gg/xwbkFNk) and [Forum].
- [![Twitter]](https://twitter.com/sfu_db)
- Contribute bug fixes.
- Providing use cases and writing down your user experience.

Please take a look at our [wiki] for development documentations!

[build status]: https://img.shields.io/circleci/build/github/sfu-db/dataprep/master?style=flat-square&token=f68e38757f5c98771f46d1c7e700f285a0b9784d
[Forum]: https://github.com/sfu-db/dataprep/discussions
[wiki]: https://github.com/sfu-db/dataprep/wiki
[examples]: https://github.com/sfu-db/dataprep/tree/master/examples
[twitter]: https://img.shields.io/twitter/follow/sfu_db?style=social

## Acknowledgement

Some functionalities of DataPrep are inspired by the following packages.

- [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling)

  Inspired the report functionality and insights provided in `dataprep.eda`.

- [missingno](https://github.com/ResidentMario/missingno)

  Inspired the missing value analysis in `dataprep.eda`.
