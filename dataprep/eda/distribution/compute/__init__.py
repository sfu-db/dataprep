"""Computations for plot(df, ...)."""

from typing import Optional, Tuple, Union, cast

import dask.dataframe as dd
import pandas as pd

from ...dtypes import DTypeDef, string_dtype_to_object
from ...intermediate import Intermediate
from ...utils import to_dask
from .bivariate import compute_bivariate
from .overview import compute_overview
from .trivariate import compute_trivariate
from .univariate import compute_univariate

__all__ = ["compute"]


def compute(
    df: Union[pd.DataFrame, dd.DataFrame],
    x: Optional[str] = None,
    y: Optional[str] = None,
    z: Optional[str] = None,
    *,
    bins: int = 10,
    ngroups: int = 10,
    largest: bool = True,
    nsubgroups: int = 5,
    timeunit: str = "auto",
    agg: str = "mean",
    sample_size: int = 1000,
    top_words: int = 30,
    stopword: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
    value_range: Optional[Tuple[float, float]] = None,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """All in one compute function.

    Parameters
    ----------
    df
        Dataframe from which plots are to be generated
    x: Optional[str], default None
        A valid column name from the dataframe
    y: Optional[str], default None
        A valid column name from the dataframe
    z: Optional[str], default None
        A valid column name from the dataframe
    bins: int, default 10
        For a histogram or box plot with numerical x axis, it defines
        the number of equal-width bins to use when grouping.
    ngroups: int, default 10
        When grouping over a categorical column, it defines the
        number of groups to show in the plot. Ie, the number of
        bars to show in a bar chart.
    largest: bool, default True
        If true, when grouping over a categorical column, the groups
        with the largest count will be output. If false, the groups
        with the smallest count will be output.
    nsubgroups: int, default 5
        If x and y are categorical columns, ngroups refers to
        how many groups to show from column x, and nsubgroups refers to
        how many subgroups to show from column y in each group in column x.
    timeunit: str, default "auto"
        Defines the time unit to group values over for a datetime column.
        It can be "year", "quarter", "month", "week", "day", "hour",
        "minute", "second". With default value "auto", it will use the
        time unit such that the resulting number of groups is closest to 15.
    agg: str, default "mean"
        Specify the aggregate to use when aggregating over a numeric column
    sample_size: int, default 1000
        Sample size for the scatter plot
    top_words: int, default 30
        Specify the amount of words to show in the wordcloud and
        word frequency bar chart
    stopword: bool, default True
        Eliminate the stopwords in the text data for plotting wordcloud and
        word frequency bar chart
    lemmatize: bool, default False
        Lemmatize the words in the text data for plotting wordcloud and
        word frequency bar chart
    stem: bool, default False
        Apply Potter Stem on the text data for plotting wordcloud and
        word frequency bar chart
    value_range: Optional[Tuple[float, float]], default None
        The lower and upper bounds on the range of a numerical column.
        Applies when column x is specified and column y is unspecified.
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """  # pylint: disable=too-many-locals
    df = to_dask(df)
    df.columns = df.columns.astype(str)
    df = string_dtype_to_object(df)

    if not any((x, y, z)):
        return compute_overview(df, bins, ngroups, largest, timeunit, dtype)

    if sum(v is None for v in (x, y, z)) == 2:
        col: str = cast(str, x or y or z)
        return compute_univariate(
            df,
            col,
            bins,
            ngroups,
            largest,
            timeunit,
            top_words,
            stopword,
            lemmatize,
            stem,
            value_range,
            dtype,
        )

    if sum(v is None for v in (x, y, z)) == 1:
        x, y = (v for v in (x, y, z) if v is not None)
        return compute_bivariate(
            df,
            x,
            y,
            bins,
            ngroups,
            largest,
            nsubgroups,
            timeunit,
            agg,
            sample_size,
            dtype,
        )

    if x is not None and y is not None and z is not None:
        return compute_trivariate(df, x, y, z, ngroups, largest, timeunit, agg, dtype)

    raise ValueError("not possible")
