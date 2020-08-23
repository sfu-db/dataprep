"""Computations for plot(df, x)."""

from typing import Any, Dict, List, Optional, Tuple, Union

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
from dask.array.stats import kurtosis, skew
from nltk.stem import PorterStemmer, WordNetLemmatizer
from scipy.stats import gaussian_kde

from ....assets.english_stopwords import english_stopwords
from ....errors import UnreachableError
from ...dtypes import (
    Continuous,
    DateTime,
    DTypeDef,
    Nominal,
    detect_dtype,
    drop_null,
    is_dtype,
)
from ...intermediate import Intermediate
from .common import _calc_line_dt


def compute_univariate(
    df: dd.DataFrame,
    x: str,
    bins: int,
    ngroups: int,
    largest: bool,
    timeunit: str,
    top_words: int,
    stopword: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
    value_range: Optional[Tuple[float, float]] = None,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """Compute functions for plot(df, x).

    Parameters
    ----------
    df
        Dataframe from which plots are to be generated
    x
        A valid column name from the dataframe
    bins
        For a histogram or box plot with numerical x axis, it defines
        the number of equal-width bins to use when grouping.
    ngroups
        When grouping over a categorical column, it defines the
        number of groups to show in the plot. Ie, the number of
        bars to show in a bar chart.
    largest
        If true, when grouping over a categorical column, the groups
        with the largest count will be output. If false, the groups
        with the smallest count will be output.
    timeunit
        Defines the time unit to group values over for a datetime column.
        It can be "year", "quarter", "month", "week", "day", "hour",
        "minute", "second". With default value "auto", it will use the
        time unit such that the resulting number of groups is closest to 15.
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
    value_range
        The lower and upper bounds on the range of a numerical column.
        Applies when column x is specified and column y is unspecified.
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # pylint: disable=too-many-locals, too-many-arguments

    col_dtype = detect_dtype(df[x], dtype)
    if is_dtype(col_dtype, Nominal()):
        # extract the column
        df_x = df[x]
        # calculate the total rows
        nrows = df_x.shape[0]
        # cast the column as string type if it contains a mutable type
        if df_x.head().apply(lambda x: hasattr(x, "__hash__")).any():
            # drop_null() will not work if the column conatains a mutable type
            df_x = df_x.dropna().astype(str)

        # drop null values
        df_x = drop_null(df_x)

        # calc_word_freq() returns the frequency of words (for the word cloud and word
        # frequency bar chart) and the total number of words
        word_data = calc_word_freq(df_x, top_words, stopword, lemmatize, stem)

        # calc_cat_stats() computes all the categorical stats including the length
        # histogram. calc_bar_pie() does the calculations for the bar and pie charts
        # NOTE this dictionary could be returned to create_report without
        # calling the subsequent compute
        cat_data = {
            "stats": calc_cat_stats(df_x, nrows, bins),
            "bar_pie": calc_bar_pie(df_x, ngroups, largest),
            "word_data": word_data,
        }
        cat_data = dask.compute(cat_data)[0]

        return Intermediate(
            col=x,
            stats=cat_data["stats"],
            bar_pie=cat_data["bar_pie"],
            word_data=cat_data["word_data"],
            visual_type="categorical_column",
        )
    elif is_dtype(col_dtype, Continuous()):

        # calculate the total number of rows then drop the missing values
        nrows = df.shape[0]
        df_x = drop_null(df[x])

        if value_range is not None:
            df_x = df_x[df_x.between(*value_range)]

        # TODO perhaps we should not use to_dask() on the entire
        # initial dataframe and instead only use the column of data
        # df_x = df_x.repartition(partition_size="100MB")

        # calculate numerical statistics and extract the min and max
        num_stats = calc_num_stats(df_x, nrows)
        minv, maxv = num_stats["min"], num_stats["max"]

        # NOTE this dictionary could be returned to create_report without
        # calling the subsequent compute
        num_data = {
            "hist": da.histogram(df_x, bins=bins, range=[minv, maxv]),
            "kde": calc_kde(df_x, bins, minv, maxv),
            "box_data": calc_box_new(df_x, num_stats["qntls"]),
            "stats": num_stats,
        }
        num_data = dask.compute(num_data)[0]

        return Intermediate(
            col=x,
            hist=num_data["hist"],
            kde=num_data["kde"],
            box_data=num_data["box_data"],
            stats=num_data["stats"],
            visual_type="numerical_column",
        )
    elif is_dtype(col_dtype, DateTime()):
        data_dt: List[Any] = []
        # line chart
        data_dt.append(dask.delayed(_calc_line_dt)(df[[x]], timeunit))
        # stats
        data_dt.append(dask.delayed(calc_stats_dt)(df[x]))
        data, statsdata_dt = dask.compute(*data_dt)
        return Intermediate(
            col=x, data=data, stats=statsdata_dt, visual_type="datetime_column",
        )
    else:
        raise UnreachableError


def calc_bar_pie(
    srs: dd.Series, ngroups: int, largest: bool
) -> Tuple[dd.DataFrame, dd.core.Scalar]:
    """
    Calculates the counts of categorical values and the total number of
    categorical values required for the bar and pie charts in plot(df, x).

    Parameters
    ----------
    srs
        One categorical column
    ngroups
        Number of groups to return
    largest
        If true, show the groups with the largest count,
        else show the groups with the smallest count
    """
    # counts of unique values in the series
    grps = srs.value_counts(sort=False)
    # total number of groups
    ttl_grps = grps.shape[0]
    # select the largest or smallest groups
    fnl_grp_cnts = grps.nlargest(ngroups) if largest else grps.nsmallest(ngroups)

    return fnl_grp_cnts.to_frame(), ttl_grps


def calc_word_freq(
    srs: dd.Series,
    top_words: int = 30,
    stopword: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
) -> Tuple[dd.Series, dd.core.Scalar]:
    """
    Parse a categorical column of text data into words, and then
    compute the frequency distribution of words and the total
    number of words.

    Parameters
    ----------
    srs
        One categorical column
    top_words
        Number of highest frequency words to show in the
        wordcloud and word frequency bar chart
    stopword
        If True, remove stop words, else keep them
    lemmatize
        If True, lemmatize the words before computing
        the word frequencies, else don't
    stem
        If True, extract the stem of the words before
        computing the word frequencies, else don't
    """
    # pylint: disable=unnecessary-lambda
    if stopword:
        # use a regex to replace stop words with empty string
        srs = srs.str.replace(r"\b(?:{})\b".format("|".join(english_stopwords)), "")
    # replace all non-alphanumeric characters with an empty string, and convert to lowercase
    srs = srs.str.replace(r"[^\w+ ]", "").str.lower()

    # split each string on whitespace into words then apply "explode()" to "stack" all
    # the words into a series
    # NOTE this is slow. One possibly better solution: after .split(), count the words
    # immediately rather than create a new series with .explode() and apply
    # .value_counts()
    srs = srs.str.split().explode()

    # lemmatize and stem
    if lemmatize or stem:
        srs = srs.dropna()
    if lemmatize:
        lem = WordNetLemmatizer()
        srs = srs.apply(lambda x: lem.lemmatize(x), meta=(srs.name, "object"))
    if stem:
        porter = PorterStemmer()
        srs = srs.apply(lambda x: porter.stem(x), meta=(srs.name, "object"))

    # counts of words, excludes null values
    word_cnts = srs.value_counts(sort=False)
    # total number of words
    nwords = word_cnts.sum()
    # words with the highest frequency
    fnl_word_cnts = word_cnts.nlargest(n=top_words)

    return fnl_word_cnts, nwords


def calc_kde(
    srs: dd.Series, bins: int, minv: float, maxv: float,
) -> Tuple[Tuple[da.core.Array, da.core.Array], np.ndarray]:
    """
    Calculate a density histogram and its corresponding kernel density
    estimate over a given series. The kernel is Gaussian.

    Parameters
    ----------
    data
        One numerical column over which to compute the histogram and kde
    bins
        Number of bins to use in the histogram
    """
    # compute the density histogram
    hist = da.histogram(srs, bins=bins, range=[minv, maxv], density=True)
    # probability density function for the series
    # NOTE gaussian_kde triggers a .compute()
    try:
        kde = gaussian_kde(
            srs.map_partitions(lambda x: x.sample(min(1000, x.shape[0])), meta=srs)
        )
    except np.linalg.LinAlgError:
        kde = None

    return hist, kde


def calc_box_new(srs: dd.Series, qntls: dd.Series) -> Dict[str, Any]:
    """
    Calculate the data required for a box plot

    Parameters
    ----------
    srs
        One numerical column from which to compute the box plot data
    qntls
        Quantiles from the normal Q-Q plot
    """
    # box plot stats
    # inter-quartile range
    # TODO figure out how to extract a scalar from a Dask series without using a function like sum()
    qrtl1 = qntls.loc[0.25].sum()
    qrtl3 = qntls.loc[0.75].sum()
    iqr = qrtl3 - qrtl1
    srs_iqr = srs[srs.between(qrtl1 - 1.5 * iqr, qrtl3 + 1.5 * iqr)]
    # outliers
    otlrs = srs[~srs.between(qrtl1 - 1.5 * iqr, qrtl3 + 1.5 * iqr)]
    # randomly sample at most 100 outliers from each partition without replacement
    otlrs = otlrs.map_partitions(lambda x: x.sample(min(100, x.shape[0])), meta=otlrs)

    box_data = {
        "grp": srs.name,
        "q1": qrtl1,
        "q2": qntls.loc[0.5].sum(),
        "q3": qrtl3,
        "lw": srs_iqr.min(),
        "uw": srs_iqr.max(),
        "otlrs": otlrs.values,
        "x": 1,  # x, x0, and x1 are for plotting the box plot with bokeh
        "x0": 0.2,
        "x1": 0.8,
    }

    return box_data


def calc_num_stats(srs: dd.Series, nrows: dd.core.Scalar) -> Dict[str, Any]:
    """
    Calculate statistics for a numerical column

    Parameters
    ----------
    srs
        a numerical column
    nrows
        number of rows in the column before dropping null values
    """
    stats = {
        "nrows": nrows,
        "npresent": srs.shape[0],
        "nunique": srs.nunique(),
        "ninfinite": ((srs == np.inf) | (srs == -np.inf)).sum(),
        "nzero": (srs == 0).sum(),
        "min": srs.min(),
        "max": srs.max(),
        "qntls": srs.quantile(np.linspace(0.01, 0.99, 99)),
        "mean": srs.mean(),
        "std": srs.std(),
        "skew": skew(srs),
        "kurt": kurtosis(srs),
        "mem_use": srs.memory_usage(),
    }
    return stats


def calc_cat_stats(
    srs: dd.Series, nrows: int, bins: int
) -> Union[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]], Dict[str, Any]]:
    """
    Calculate stats for a categorical column

    Parameters
    ----------
    srs
        a categorical column
    nrows
        number of rows before dropping null values
    bins
        number of bins for the category length frequency histogram
    """
    # overview stats
    stats = {
        "nrows": nrows,
        "npresent": srs.shape[0],
        "nunique": srs.nunique(),
        "mem_use": srs.memory_usage(),
        "first_rows": srs.loc[:4],
    }
    # length stats
    lengths = srs.str.len()
    minv, maxv = lengths.min(), lengths.max()
    hist = da.histogram(lengths.values, bins=bins, range=[minv, maxv])
    length_stats = {
        "Mean": lengths.mean(),
        "Median": lengths.quantile(0.5),
        "Minimum": minv,
        "Maximum": maxv,
        "hist": hist,
    }
    # letter stats
    letter_stats = {
        "Count": srs.str.count(r"[a-zA-Z]").sum(),
        "Lowercase Letter": srs.str.count(r"[a-z]").sum(),
        "Space Separator": srs.str.count(r"[ ]").sum(),
        "Uppercase Letter": srs.str.count(r"[A-Z]").sum(),
        "Dash Punctuation": srs.str.count(r"[-]").sum(),
        "Decimal Number": srs.str.count(r"[0-9]").sum(),
    }

    return stats, length_stats, letter_stats


def calc_stats_dt(srs: dd.Series) -> Dict[str, str]:
    """
    Calculate stats from a datetime column

    Parameters
    ----------
    srs
        a datetime column
    Returns
    -------
    Dict[str, str]
        Dictionary that contains Overview
    """
    size = len(srs)  # include nan
    count = srs.count()  # exclude nan
    uniq_count = srs.nunique()
    overview_dict = {
        "Distinct Count": uniq_count,
        "Unique (%)": uniq_count / count,
        "Missing": size - count,
        "Missing (%)": 1 - (count / size),
        "Memory Size": srs.memory_usage(),
        "Minimum": srs.min(),
        "Maximum": srs.max(),
    }

    return overview_dict
