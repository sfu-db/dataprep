<div align="center"><img width="100%" src="https://github.com/sfu-db/dataprep/raw/develop/assets/logo.png"/></div>

---

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/dataprep?style=flat-square"/></a>
  <a href="https://sfu-db.github.io/dataprep/"><img src="https://img.shields.io/badge/dynamic/json?color=blue&label=docs&prefix=v&query=%24.info.version&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fdataprep%2Fjson&style=flat-square"/></a>
  <a href="https://pypi.org/project/dataprep/"><img src="https://img.shields.io/pypi/pyversions/dataprep?style=flat-square"/></a>
  <!-- <a href="https://www.codacy.com/gh/sfu-db/dataprep?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=sfu-db/dataprep&amp;utm_campaign=Badge_Coverage"><img src="https://app.codacy.com/project/badge/Coverage/ed658f08dcce4f088c850253475540ba"/></a> -->
<!--   <a href="https://codecov.io/gh/sfu-db/dataprep"><img src="https://img.shields.io/codecov/c/github/sfu-db/dataprep?style=flat-square"/></a> -->
  <!-- <a href="https://www.codacy.com/gh/sfu-db/dataprep?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=sfu-db/dataprep&amp;utm_campaign=Badge_Grade"><img src="https://app.codacy.com/project/badge/Grade/ed658f08dcce4f088c850253475540ba"/></a> -->
</p>

<p align="center">
  <a href="https://sfu-db.github.io/dataprep/">Documentation</a>
  | 
  <a href="https://github.com/sfu-db/dataprep/discussions">Forum</a>
</p>

<p align="center"><b>Low code data preparation</b></p>

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

## EDA

DataPrep.EDA is the fastest and the easiest EDA (Exploratory Data Analysis) tool in Python. It allows you to understand a Pandas/Dask DataFrame with a few lines of code in seconds.

#### Create Profile Reports, Fast

You can create a beautiful profile report from a Pandas/Dask DataFrame with the `create_report` function. DataPrep.EDA has the following advantages compared to other tools:

- **[10X Faster](https://arxiv.org/abs/2104.00841)**: DataPrep.EDA can be 10X faster than Pandas-based profiling tools due to its highly optimized Dask-based computing module.
- **Interactive Visualization**: DataPrep.EDA generates interactive visualizations in a report, which makes the report look more appealing to end users.
- **Big Data Support**: DataPrep.EDA naturally supports big data stored in a Dask cluster by accepting a Dask dataframe as input.

The following code demonstrates how to use DataPrep.EDA to create a profile report for the titanic dataset.

```python
from dataprep.datasets import load_dataset
from dataprep.eda import create_report
df = load_dataset("titanic")
create_report(df).show_browser()
```

Click [here](https://docs.dataprep.ai/_downloads/1a61c6aebb3ecbe9dc9742bd6ca78ddb/titanic_dp.html) to see the generated report of the above code.

Click [here](https://docs.dataprep.ai/dev/bench/index.html) to see the benchmark result.

#### Try DataPrep.EDA Online: [DataPrep.EDA Demo in Colab](https://colab.research.google.com/drive/1U_-pAMcne3hK1HbMB3kuEt-093Np_7Uk?usp=sharing)

#### Innovative System Design

DataPrep.EDA is the **_only_** task-centric EDA system in Python. It is carefully designed to improve usability.

- **Task-Centric API Design**: You can declaratively specify a wide range of EDA tasks in different granularity with a single function call. All needed visualizations will be automatically and intelligently generated for you.
- **Auto-Insights**: DataPrep.EDA automatically detects and highlights the insights (e.g., a column has many outliers) to facilitate pattern discovery about the data.
- **How-to Guide**: A how-to guide is provided to show the configuration of each plot function. With this feature, you can easily customize the generated visualizations.

#### Learn DataPrep.EDA in 2 minutes:

<a href="https://youtu.be/nSkQy3ew3EI"><img src="assets/eda_video_cover.png"/></a>

Click [here](https://sfu-db.github.io/dataprep/user_guide/eda/introduction.html) to check all the supported tasks.

Check [plot](https://sfu-db.github.io/dataprep/user_guide/eda/plot.html), [plot_correlation](https://sfu-db.github.io/dataprep/user_guide/eda/plot_correlation.html), [plot_missing](https://sfu-db.github.io/dataprep/user_guide/eda/plot_missing.html) and [create_report](https://sfu-db.github.io/dataprep/user_guide/eda/create_report.html) to see how each function works.

## Clean

DataPrep.Clean contains about **140+** functions designed for cleaning and validating data in a DataFrame. It provides

- **A Convenient GUI**: incorporated into Jupyter Notebook, users can clean their own DataFrame without any coding (see the video below).
- **A Unified API**: each function follows the syntax `clean_{type}(df, 'column name')` (see an example below).
- **Speed**: the computations are parallelized using Dask. It can clean **50K rows per second** on a dual-core laptop (that means cleaning 1 million rows in only 20 seconds).
- **Transparency**: a report is generated that summarizes the alterations to the data that occured during cleaning.

The following video shows how to use GUI of Dataprep.Clean
<a href="https://youtu.be/WtJaVBIVoxQ"><img src="assets/clean_video_cover.png"/></a>

The following example shows how to clean and standardize a column of country names.

```python
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

```python
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

Check [Documentation of Dataprep.Clean](https://docs.dataprep.ai/user_guide/clean/introduction.html) to see how each function works.

## Connector

Connector now supports loading data from both web API and databases.

### Web API

Connector is an intuitive, open-source API wrapper that speeds up development by standardizing calls to multiple APIs as a simple workflow.

Connector provides a simple wrapper to collect structured data from different Web APIs (e.g., Twitter, Spotify), making web data collection easy and efficient, without requiring advanced programming skills.

Do you want to leverage the growing number of websites that are opening their data through public APIs? Connector is for you!

Let's check out the several benefits that Connector offers:

- **A unified API:** You can fetch data using one or two lines of code to get data from [tens of popular websites](https://github.com/sfu-db/DataConnectorConfigs).
- **Auto Pagination:** Do you want to invoke a Web API that could return a large result set and need to handle it through pagination? Connector automatically does the pagination for you! Just specify the desired number of returned results (argument `_count`) without getting into unnecessary detail about a specific pagination scheme.
- **Speed:** Do you want to fetch results more quickly by making concurrent requests to Web APIs? Through the `_concurrency` argument, Connector simplifies concurrency, issuing API requests in parallel while respecting the API's rate limit policy.

#### How to fetch all publications of Andrew Y. Ng?

```python
from dataprep.connector import connect
conn_dblp = connect("dblp", _concurrency = 5)
df = await conn_dblp.query("publication", author = "Andrew Y. Ng", _count = 2000)
```

Here, you can find detailed [Examples.](https://github.com/sfu-db/dataprep/tree/develop/examples)

Connector is designed to be easy to extend. If you want to connect with your own web API, you just have to write a simple [configuration file](https://github.com/sfu-db/DataConnectorConfigs/blob/develop/CONTRIBUTING.md#add-configuration-files) to support it. This configuration file describes the API's main attributes like the URL, query parameters, authorization method, pagination properties, etc.

### Database

Connector now has adopted [connectorx](https://github.com/sfu-db/connector-x) in order to enable loading data from databases (Postgres, Mysql, SQLServer, etc.) into Python dataframes (pandas, dask, modin, arrow, polars) in the fastest and most memory efficient way. [[Benchmark]](https://github.com/sfu-db/connector-x/blob/main/Benchmark.md#benchmark-result-on-aws-r54xlarge)

What you need to do is just install `connectorx` (`pip install -U connectorx`) and run one line of code:

```python
from dataprep.connector import read_sql
read_sql("postgresql://username:password@server:port/database", "SELECT * FROM lineitem")
```

Check out [here](https://github.com/sfu-db/connector-x#supported-sources--destinations) for supported databases and dataframes and more examples usages.

## Documentation

The following documentation can give you an impression of what DataPrep can do:

- [Connector](https://docs.dataprep.ai/user_guide/connector/introduction.html)
- [EDA](https://docs.dataprep.ai/user_guide/eda/introduction.html)
- [Clean](https://docs.dataprep.ai/user_guide/clean/introduction.html)

## Contribute

There are many ways to contribute to DataPrep.

- Submit bugs and help us verify fixes as they are checked in.
- Review the source code changes.
- Engage with other DataPrep users and developers on StackOverflow.
- Ask questions & propose new ideas in our [Forum].
- [![Twitter]](https://twitter.com/dataprepai)
- Contribute bug fixes.
- Providing use cases and writing down your user experience.

Please take a look at our [wiki] for development documentations!

[build status]: https://img.shields.io/circleci/build/github/sfu-db/dataprep/master?style=flat-square&token=f68e38757f5c98771f46d1c7e700f285a0b9784d
[forum]: https://github.com/sfu-db/dataprep/discussions
[wiki]: https://github.com/sfu-db/dataprep/wiki
[examples]: https://github.com/sfu-db/dataprep/tree/master/examples
[twitter]: https://img.shields.io/twitter/follow/dataprepai?style=social

## Acknowledgement

Some functionalities of DataPrep are inspired by the following packages.

- [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling)

  Inspired the report functionality and insights provided in `dataprep.eda`.

- [missingno](https://github.com/ResidentMario/missingno)

  Inspired the missing value analysis in `dataprep.eda`.

## Citing DataPrep

If you use DataPrep, please consider citing the following paper:

Jinglin Peng, Weiyuan Wu, Brandon Lockhart, Song Bian, Jing Nathan Yan, Linghao Xu, Zhixuan Chi, Jeffrey M. Rzeszotarski, and Jiannan Wang. [DataPrep.EDA: Task-Centric Exploratory Data Analysis for Statistical
Modeling in Python.](https://arxiv.org/abs/2104.00841) _SIGMOD 2021_.

BibTeX entry:

```bibtex
@inproceedings{dataprepeda2021,
  author    = {Jinglin Peng and Weiyuan Wu and Brandon Lockhart and Song Bian and Jing Nathan Yan and Linghao Xu and Zhixuan Chi and Jeffrey M. Rzeszotarski and Jiannan Wang},
  title     = {DataPrep.EDA: Task-Centric Exploratory Data Analysis for Statistical Modeling in Python},
  booktitle = {Proceedings of the 2021 International Conference on Management of Data (SIGMOD '21), June 20--25, 2021, Virtual Event, China},
  year      = {2021}
}
```
