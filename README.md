# Dataprep ![Build Status]
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

Detailed examples can be found in the [examples] folder.

### EDA

There are common tasks during the exploratory data analysis stage, 
like a quick look at the columnar distribution, or understanding the correlations
between columns. 

The EDA module categorizes these EDA tasks into functions helping you finish EDA
tasks with a single function call.

* Want to understand the distributions for each DataFrame column? Use `plot`.
```python
from dataprep.eda import plot

df = ...

plot(df)
```
<center><img src="https://github.com/sfu-db/dataprep/raw/master/assets/plot(df).png"/></center>

* Want to understand the correlation between columns? Use `plot_correlation`.

```python
from dataprep.eda import plot_correlation

df = ...

plot_correlation(df)
```
<center><img src="https://github.com/sfu-db/dataprep/raw/master/assets/plot_correlation(df).png" width="50%" height="50%"/></center>

* Or, if you want to understand the impact of the missing values for each column, use `plot_missing`.

```python
from dataprep.eda import plot_missing

df = ...

plot_missing(df)
```
<center><img src="https://github.com/sfu-db/dataprep/raw/master/assets/plot_missing(df).png" width="50%" height="50%"/></center>

* You can even drill down to get more information by given `plot`, `plot_correlation` and `plot_missing` a column name.

```python
df = ...

plot_missing(df, x="some_column_name")
```

<center><img src="https://github.com/sfu-db/dataprep/raw/master/assets/plot_missing(df,x).png" width="50%"/></center>

Don't forget to checkout the [examples] folder for detailed demonstration!

### Data Connector

You can download Yelp business search result into a pandas DataFrame, 
using two lines of code, without taking deep looking into the Yelp documentation!

```python
from dataprep.data_connector import Connector

dc = Connector("yelp", auth_params={"access_token":"<Your yelp access token>"})
df = dc.query("businesses", term="ramen", location="vancouver")
```
![DataConnectorResult]


## Contribution

Dataprep is in its early stage. Any contribution including:
* Filing an issue
* Providing use cases
* Writing down your user experience
* Submitting a PR
* ...

are greatly appreciated!

Please take a look at our [wiki] for development documentations!


[Build Status]: https://img.shields.io/circleci/build/github/sfu-db/dataprep/master?style=flat-square&token=f68e38757f5c98771f46d1c7e700f285a0b9784d
[Documentation]: https://sfu-db.github.io/dataprep/
[Mail list & Forum]: https://groups.google.com/forum/#!forum/dataprep
[wiki]: https://github.com/sfu-db/dataprep/wiki
[examples]: https://github.com/sfu-db/dataprep/tree/master/examples
[DataConnectorResult]: https://github.com/sfu-db/dataprep/raw/master/assets/data_connector.png
