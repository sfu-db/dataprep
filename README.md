# DataPrep ![Build Status]
[Documentation] | [Mail List & Forum] 

Dataprep is a collection of functions that 
helps you accomplish tasks before you build a predictive model.


## Implementation Status

Currently, you can use `dataprep` to:
* Collect data from common data sources (through `dataprep.data_connector`)
* Do your exploratory data analysis (through `dataprep.eda`)
* ...


## Installation

```bash
pip install dataprep
```

`dataprep` is in its alpha stage now, so please manually specific the version number.


## Examples & Usages

More detailed examples can be found at the [examples] folder.

### Data Connector

You can download Yelp business search result into a pandas DataFrame, 
using two lines of code, without taking deep looking into the Yelp documentation!

```python
from dataprep.data_connector import Connector

dc = Connector("yelp", auth_params={"access_token":"<Your yelp access token>"})
df = dc.query("businesses", term="ramen", location="vancouver")
```
![DataConnectorResult]


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

## Contribution
Dataprep is in its very early stage. Any contribution including:
* Filing an issue
* Providing use cases
* Write down your user experience
* Submit a PR
* ...

are greatly appreciated!

If you want to make code contribution to dataprep, be sure to read the [contribution guidelines].





[Build Status]: https://img.shields.io/circleci/build/github/sfu-db/dataprep/master?style=flat-square&token=f68e38757f5c98771f46d1c7e700f285a0b9784d
[Documentation]: https://sfu-db.github.io/dataprep/
[Mail list & Forum]: https://groups.google.com/forum/#!forum/dataprep
[contribution guidelines]: https://github.com/sfu-db/dataprep/blob/master/CONTRIBUTING.md
[examples]: https://github.com/sfu-db/dataprep/tree/master/examples
[DataConnectorResult]: https://github.com/sfu-db/dataprep/raw/master/assets/data_connector.png
