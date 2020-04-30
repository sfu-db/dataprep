# Dataprep

[![License]](LICENSE) [![Doc Badge]](https://sfu-db.github.io/dataprep/) [![Version]](https://pypi.org/project/dataprep/) [![Python Version]](https://pypi.org/project/dataprep/)  [![Downloads]](https://pepy.tech/project/dataprep) [![Codecov]](https://codecov.io/gh/sfu-db/dataprep) ![Build Status]  [![Chat]](https://discord.gg/xwbkFNk) 

[Documentation] | [Mail List & Forum] 

Dataprep let you prepare your data using a single library with a few lines of code.

Currently, you can use `dataprep` to:
* Collect data from common data sources (through `dataprep.data_connector`)
* Do your exploratory data analysis (through `dataprep.eda`)
* ...more modules are coming

## Installation

```bash
pip install dataprep
```

## Examples & Usages

The following examples can give you an impression of what dataprep can do:

* [Documentation: Data Connector](https://sfu-db.github.io/dataprep/data_connector.html)
* [Documentation: EDA](https://sfu-db.github.io/dataprep/eda/introduction.html)
* [EDA Case Study: Titanic](https://sfu-db.github.io/dataprep/case_study/titanic.html)
* [EDA Case Study: House Price](https://sfu-db.github.io/dataprep/case_study/house_price.html)

### EDA

There are common tasks during the exploratory data analysis stage, 
like a quick look at the columnar distribution, or understanding the correlations
between columns. 

The EDA module categorizes these EDA tasks into functions helping you finish EDA
tasks with a single function call.

* Want to understand the distributions for each DataFrame column? Use `plot`.

<center><a href="https://sfu-db.github.io/dataprep/eda/introduction.html#analyzing-basic-characteristics-via-plot"><img src="https://github.com/sfu-db/dataprep/raw/master/assets/plot(df).png"/></a></center>

* Want to understand the correlation between columns? Use `plot_correlation`.

<center><a href="https://sfu-db.github.io/dataprep/eda/introduction.html#analyzing-correlation-via-plot-correlation"><img src="https://github.com/sfu-db/dataprep/raw/master/assets/plot_correlation(df).png"/></a></center>

* Or, if you want to understand the impact of the missing values for each column, use `plot_missing`.

<center><a href="https://sfu-db.github.io/dataprep/eda/plot_missing.html#plotting-the-position-of-missing-values-via-plot-missing-df"><img src="https://github.com/sfu-db/dataprep/raw/master/assets/plot_missing(df).png"/></a></center>

* You can drill down to get more information by given `plot`, `plot_correlation` and `plot_missing` a column name. E.g. for `plot_missing`:

<center><a href="https://sfu-db.github.io/dataprep/eda/plot_missing.html#the-impact-on-basic-characteristics-of-missing-values-in-column-x-via-plot-missing-df-x"><img src="https://github.com/sfu-db/dataprep/raw/master/assets/plot_missing(df,x).png"/></a></center>

Don't forget to checkout the [examples] folder for detailed demonstration!

### Data Connector

You can download Yelp business search result into a pandas DataFrame, 
using two lines of code, without taking deep looking into the Yelp documentation!

```python
from dataprep.data_connector import Connector

dc = Connector("yelp", auth_params={"access_token":"<Your yelp access token>"})
df = dc.query("businesses", term="korean", location="seattle")
```
<center><a href="https://sfu-db.github.io/dataprep/data_connector.html#getting-web-data-with-connector-query"><img src="https://github.com/sfu-db/dataprep/raw/master/assets/data_connector.png"/></a></center>


## Contribute

There are many ways to contribute to Dataprep.

* Submit bugs and help us verify fixes as they are checked in.
* Review the source code changes.
* Engage with other Dataprep users and developers on StackOverflow.
* Help each other in the [Dataprep Community Discord](https://discord.gg/FXsK2P) and [Mail list & Forum].
* [![Twitter]](https://twitter.com/sfu_db)
* Contribute bug fixes.
* Providing use cases and writing down your user experience.

Please take a look at our [wiki] for development documentations!


[Build Status]: https://img.shields.io/circleci/build/github/sfu-db/dataprep/master?style=flat-square&token=f68e38757f5c98771f46d1c7e700f285a0b9784d
[Documentation]: https://sfu-db.github.io/dataprep/
[Mail list & Forum]: https://groups.google.com/forum/#!forum/dataprep
[wiki]: https://github.com/sfu-db/dataprep/wiki
[examples]: https://github.com/sfu-db/dataprep/tree/master/examples
[Chat]: https://img.shields.io/discord/702765817154109472?style=flat-square
[License]: https://img.shields.io/pypi/l/dataprep?style=flat-square
[Downloads]: https://pepy.tech/badge/dataprep
[Python Version]: https://img.shields.io/pypi/pyversions/dataprep?style=flat-square
[Version]: https://img.shields.io/pypi/v/dataprep?style=flat-square
[Codecov]: https://img.shields.io/codecov/c/github/sfu-db/dataprep?style=flat-square
[Twitter]: https://img.shields.io/twitter/follow/sfu_db?style=social
[Doc Badge]: https://img.shields.io/badge/dynamic/json?color=blue&label=docs&prefix=v&query=%24.info.version&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fdataprep%2Fjson&style=flat-square
