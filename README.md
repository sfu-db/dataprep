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

`pip install dataprep==0.1.0a2`

`dataprep` is in its alpha stage now, so please manually specific the version number.


## Examples & Usages

More detailed examples can be found at the [examples](examples) folder.

### Data Connector

You can download Yelp business search result into a pandas DataFrame, 
using two lines of code, without taking deep looking into the Yelp documentation!

```python
from dataprep.data_connector import Connector
# Put "yelp" as the first parameter to indicate we want to download some data from Yelp.
# You also need the Yelp access token for programmably access. 
dc = Connector("yelp", auth_params={"access_token":"<Your yelp access token>"})
# Here we want to download data from the "restaurant" endpoint 
# with term "ramen" and location "vancouver"
df = dc.query("businesses", term="ramen", location="vancouver")
# df will be a pandas dataframe.
```

### EDA

There are common tasks during the exploratory data analysis stage, 
like a quick look at the columnar distribution, or understanding the correlations
between columns. 

The EDA module categorizes these EDA tasks into functions helping you finish EDA
tasks with a single function call.

Want to understand the distributions for each DataFrame column? Use `plot`.
```python
from dataprep.eda import plot

df = ...

plot(df)
```

Want to understand the correlation between columns? Use `plot_correlation`.

```python
from dataprep.eda import plot_correlation

df = ...

plot_correlation(df)
```

Or, if you want to understand the impact of the missing values for each column, use `plot_missing`.

```python
from dataprep.eda import plot_missing

df = ...

plot_missing(df)
```

You can even drill down to get more information by given `plot`, `plot_correlation` and `plot_missing` a column name.

```python
df = ...

plot_missing(df, x="some_column_name")
```

Don't forget to checkout the [examples](examples) folder for detailed demostration!

## Contribution
Contribution is always welcome. 
If you want to contribute to dataprep, be sure to read the [contribution guidelines](CONTRIBUTING.md).



[Build Status]: https://img.shields.io/circleci/build/github/sfu-db/dataprep/master?style=flat-square&token=f68e38757f5c98771f46d1c7e700f285a0b9784d
[Documentation]: https://sfu-db.github.io/dataprep/
[Mail list & Forum]: https://groups.google.com/forum/#!forum/dataprep
