"""Computations for plot(df, x)."""

from typing import Any, Dict, List, Optional, Tuple

import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.array.stats import chisquare, kurtosis, skew
from nltk.stem import PorterStemmer, WordNetLemmatizer

from ....assets.english_stopwords import english_stopwords as ess
from ....errors import UnreachableError
from ...dtypes import (
    Continuous,
    DateTime,
    DTypeDef,
    Nominal,
    detect_dtype,
    is_dtype,
)
from ...intermediate import Intermediate
from .common import _calc_line_dt, gaussian_kde, normaltest


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
        first_rows = df[x].head()  # dd.Series.head() triggers a (small) data read
        # cast the column as string type if it contains a mutable type
        try:
            first_rows.apply(hash)
        except TypeError:
            df[x] = df[x].astype(str)
        # all computations for plot(df, Nominal())
        data = nom_comps(
            df[x],
            first_rows,
            ngroups,
            largest,
            bins,
            top_words,
            stopword,
            lemmatize,
            stem,
        )
        (data,) = dask.compute(data)

        return Intermediate(
            col=x,
            data=data,
            visual_type="categorical_column",
        )

    elif is_dtype(col_dtype, Continuous()):
        # extract the column
        srs = df[x]
        # select values in the user defined range
        if value_range is not None:
            srs = srs[srs.between(*value_range)]

        # all computations for plot(df, Continuous())
        (data,) = dask.compute(cont_comps(srs, bins))

        return Intermediate(
            col=x,
            data=data,
            visual_type="numerical_column",
        )

    elif is_dtype(col_dtype, DateTime()):
        data_dt: List[Any] = []
        # stats
        data_dt.append(dask.delayed(calc_stats_dt)(df[x]))
        # line chart
        data_dt.append(dask.delayed(_calc_line_dt)(df[[x]], timeunit))
        data, line = dask.compute(*data_dt)
        return Intermediate(
            col=x,
            data=data,
            line=line,
            visual_type="datetime_column",
        )
    else:
        raise UnreachableError


def nom_comps(
    srs: dd.Series,
    first_rows: pd.Series,
    ngroups: int,
    largest: bool,
    bins: int,
    top_words: int,
    stopword: bool,
    lemmatize: bool,
    stem: bool,
) -> Dict[str, Any]:
    """
    This function aggregates all of the computations required for plot(df, Nominal())

    Parameters
    ----------
    srs
        one categorical column
    ngroups
        Number of groups to return
    largest
        If true, show the groups with the largest count,
        else show the groups with the smallest count
    bins
        number of bins for the category length frequency histogram
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
    """  # pylint: disable=too-many-arguments

    data: Dict[str, Any] = {}

    # total rows
    data["nrows"] = srs.shape[0]
    # drop null values
    srs = srs.dropna()

    ## if cfg.bar_enable or cfg.pie_enable
    # counts of unique values in the series
    grps = srs.value_counts(sort=False)
    # total number of groups
    data["nuniq"] = grps.shape[0]
    # select the largest or smallest groups
    data["bar"] = grps.nlargest(ngroups) if largest else grps.nsmallest(ngroups)
    ##     if cfg.barchart_bars == cfg.piechart_slices:
    data["pie"] = data["bar"]
    ##     else
    ##     data["pie"] = grps.nlargest(ngroups) if largest else grps.nsmallest(ngroups)
    ##     if cfg.insights.evenness_enable
    data["chisq"] = chisquare(grps.values)

    ## if cfg.stats_enable
    df = grps.reset_index()
    ## if cfg.stats_enable or cfg.word_freq_enable
    if not first_rows.apply(lambda x: isinstance(x, str)).all():
        srs = srs.astype(str)  # srs must be a string to compute the value lengths
        df[df.columns[0]] = df[df.columns[0]].astype(str)
    data.update(calc_cat_stats(srs, df, bins, data["nrows"], data["nuniq"]))
    # ## if cfg.word_freq_enable
    data.update(calc_word_freq(df, top_words, stopword, lemmatize, stem))

    return data


def cont_comps(srs: dd.Series, bins: int) -> Dict[str, Any]:
    """
    This function aggregates all of the computations required for plot(df, Continuous())

    Parameters
    ----------
    srs
        one numerical column
    bins
        the number of bins in the histogram
    """

    data: Dict[str, Any] = {}

    ## if cfg.stats_enable or cfg.hist_enable or
    # calculate the total number of rows then drop the missing values
    data["nrows"] = srs.shape[0]
    srs = srs.dropna()
    ## if cfg.stats_enable
    # number of not null (present) values
    data["npres"] = srs.shape[0]
    # remove infinite values
    srs = srs[~srs.isin({np.inf, -np.inf})]

    # shared computations
    ## if cfg.stats_enable or cfg.hist_enable or cfg.qqplot_enable and cfg.insights_enable:
    data["min"], data["max"] = srs.min(), srs.max()
    ## if cfg.hist_enable or cfg.qqplot_enable and cfg.ingsights_enable:
    data["hist"] = da.histogram(srs, bins=bins, range=[data["min"], data["max"]])
    ## if cfg.insights_enable and (cfg.qqplot_enable or cfg.hist_enable):
    data["norm"] = normaltest(data["hist"][0])
    ## if cfg.qqplot_enable
    data["qntls"] = srs.quantile(np.linspace(0.01, 0.99, 99))
    ## elif cfg.stats_enable
    ## data["qntls"] = srs.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    ## elif cfg.boxplot_enable
    ## data["qntls"] = srs.quantile([0.25, 0.5, 0.75])
    ## if cfg.stats_enable or cfg.hist_enable and cfg.insights_enable:
    data["skew"] = skew(srs)

    # if cfg.stats_enable
    data["nuniq"] = srs.nunique()
    data["nreals"] = srs.shape[0]
    data["nzero"] = (srs == 0).sum()
    data["nneg"] = (srs < 0).sum()
    data["mean"] = srs.mean()
    data["std"] = srs.std()
    data["kurt"] = kurtosis(srs)
    data["mem_use"] = srs.memory_usage(deep=True)

    ## if cfg.hist_enable and cfg.insight_enable
    data["chisq"] = chisquare(data["hist"][0])

    # compute the density histogram
    data["dens"] = da.histogram(srs, bins=bins, range=[data["min"], data["max"]], density=True)
    # gaussian kernel density estimate
    data["kde"] = gaussian_kde(
        srs.map_partitions(lambda x: x.sample(min(1000, x.shape[0])), meta=srs)
    )

    ## if cfg.box_enable
    data.update(calc_box(srs, data["qntls"]))

    return data


def calc_box(srs: dd.Series, qntls: da.Array) -> Dict[str, Any]:
    """
    Box plot calculations

    Parameters
    ----------
    srs
        one numerical column
    qntls
        quantiles of the column
    """
    data: Dict[str, Any] = {}

    # quartiles
    data["qrtl1"] = qntls.loc[0.25].sum()
    data["qrtl2"] = qntls.loc[0.5].sum()
    data["qrtl3"] = qntls.loc[0.75].sum()
    iqr = data["qrtl3"] - data["qrtl1"]
    srs_iqr = srs[srs.between(data["qrtl1"] - 1.5 * iqr, data["qrtl3"] + 1.5 * iqr)]
    # outliers
    otlrs = srs[~srs.between(data["qrtl1"] - 1.5 * iqr, data["qrtl3"] + 1.5 * iqr)]
    # randomly sample at most 100 outliers from each partition without replacement
    smp_otlrs = otlrs.map_partitions(lambda x: x.sample(min(100, x.shape[0])), meta=otlrs)
    data["lw"] = srs_iqr.min()
    data["uw"] = srs_iqr.max()
    data["otlrs"] = smp_otlrs.values
    ##    if cfg.insights_enable
    data["notlrs"] = otlrs.shape[0]

    return data


def calc_word_freq(
    df: dd.DataFrame,
    top_words: int = 30,
    stopword: bool = True,
    lemmatize: bool = False,
    stem: bool = False,
) -> Dict[str, Any]:
    """
    Parse a categorical column of text data into words, and then
    compute the frequency distribution of words and the total
    number of words.

    Parameters
    ----------
    df
        Groupby-count on the categorical column as a dataframe
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
    col = df.columns[0]
    if stopword:
        # use a regex to replace stop words and non-alphanumeric characters with empty string
        df[col] = df[col].str.replace(fr"\b(?:{'|'.join(ess)})\b|[^\w+ ]", "")
    else:
        df[col] = df[col].str.replace(r"[^\w+ ]", "")
    # convert to lowercase and split
    df[col] = df[col].str.lower().str.split()
    # "explode()" to "stack" all the words in a list into a new column
    df = df.explode(col)

    # lemmatize and stem
    if lemmatize or stem:
        df[col] = df[col].dropna()
    if lemmatize:
        lem = WordNetLemmatizer()
        df[col] = df[col].apply(lem.lemmatize, meta="object")
    if stem:
        porter = PorterStemmer()
        df[col] = df[col].apply(porter.stem, meta="object")

    # counts of words, excludes null values
    word_cnts = df.groupby(col)[df.columns[1]].sum()
    # total number of words
    nwords = word_cnts.sum()
    # total uniq words
    nuniq_words = word_cnts.shape[0]
    # words with the highest frequency
    fnl_word_cnts = word_cnts.nlargest(n=top_words)

    return {"word_cnts": fnl_word_cnts, "nwords": nwords, "nuniq_words": nuniq_words}


def calc_cat_stats(
    srs: dd.Series,
    df: dd.DataFrame,
    bins: int,
    nrows: int,
    nuniq: Optional[dd.core.Scalar] = None,
) -> Dict[str, Any]:
    """
    Calculate stats for a categorical column

    Parameters
    ----------
    srs
        a categorical column
    df
        groupby-count on the categorical column as a dataframe
    bins
        number of bins for the category length frequency histogram
    nrows
        number of rows before dropping null values
    nuniq
        number of unique values in the column
    """
    # pylint: disable=too-many-locals
    # overview stats
    stats = {
        "nrows": nrows,
        "npres": srs.shape[0],
        "nuniq": nuniq,  # if cfg.bar_endable or cfg.pie_enable else srs.nunique(),
        "mem_use": srs.memory_usage(deep=True),
        "first_rows": srs.reset_index(drop=True).loc[:4],
    }
    # length stats
    lengths = srs.str.len()
    minv, maxv = lengths.min(), lengths.max()
    hist = da.histogram(lengths.values, bins=bins, range=[minv, maxv])
    leng = {
        "Mean": lengths.mean(),
        "Standard Deviation": lengths.std(),
        "Median": lengths.quantile(0.5),
        "Minimum": minv,
        "Maximum": maxv,
    }
    # letter stats
    # computed on groupby-count:
    # compute the statistic for each group then multiply by the count of the group
    grp, col = df.columns
    lc_cnt = (df[grp].str.count(r"[a-z]") * df[col]).sum()
    uc_cnt = (df[grp].str.count(r"[A-Z]") * df[col]).sum()
    letter = {
        "Count": lc_cnt + uc_cnt,
        "Lowercase Letter": lc_cnt,
        "Space Separator": (df[grp].str.count(r"[ ]") * df[col]).sum(),
        "Uppercase Letter": uc_cnt,
        "Dash Punctuation": (df[grp].str.count(r"[-]") * df[col]).sum(),
        "Decimal Number": (df[grp].str.count(r"[0-9]") * df[col]).sum(),
    }

    return {"stats": stats, "len_stats": leng, "letter_stats": letter, "len_hist": hist}


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
    size = srs.shape[0]  # include nan
    count = srs.count()  # exclude nan
    uniq_count = srs.nunique()
    overview_dict = {
        "Distinct Count": uniq_count,
        "Unique (%)": uniq_count / count,
        "Missing": size - count,
        "Missing (%)": 1 - (count / size),
        "Memory Size": srs.memory_usage(deep=True),
        "Minimum": srs.min(),
        "Maximum": srs.max(),
    }

    return overview_dict
