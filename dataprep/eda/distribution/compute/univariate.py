"""
Computations for plot(df, x)
"""

from typing import Any, Dict, List, Optional, Union

import math
import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.array.stats import chisquare, kurtosis, skew
from nltk.stem import PorterStemmer, WordNetLemmatizer

from ....assets.english_stopwords import english_stopwords as ess
from ...configs import Config
from ...dtypes_v2 import (
    Continuous,
    DateTime,
    DTypeDef,
    Nominal,
    SmallCardNum,
    GeoGraphy,
    GeoPoint,
    LatLong,
)
from ...intermediate import Intermediate
from ...eda_frame import EDAFrame
from .common import gen_new_df_with_used_cols
from ...utils import _calc_line_dt, gaussian_kde, normaltest


def compute_univariate(
    df: Union[dd.DataFrame, pd.DataFrame],
    col: Union[str, LatLong],
    cfg: Config,
    dtype: Optional[DTypeDef],
) -> Intermediate:
    """
    Compute functions for plot(df, x)

    Parameters
    ----------
    df
        DataFrame from which visualizations are generated
    x
        A column name from the DataFrame
    cfg
        Config instance
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """

    new_col_names, ndf = gen_new_df_with_used_cols(df, col, None, None)
    x = new_col_names[col]
    if x is None:
        raise ValueError

    frame = EDAFrame(ndf, dtype)
    col_dtype = frame.get_eda_dtype(x)

    if isinstance(col_dtype, (Nominal, GeoPoint, SmallCardNum)):
        srs = frame.get_col_as_str(x)
        (data,) = dask.compute(nom_comps(srs, cfg))
        return Intermediate(col=x, data=data, visual_type="categorical_column")

    elif isinstance(col_dtype, Continuous):
        (data,) = dask.compute(cont_comps(frame.frame[x], cfg))
        return Intermediate(col=x, data=data, visual_type="numerical_column")

    elif isinstance(col_dtype, DateTime):
        data_dt: List[Any] = []
        # stats
        data_dt.append(dask.delayed(calc_stats_dt)(frame.frame[x]))
        # line chart
        if cfg.line.enable:
            data_dt.append(dask.delayed(_calc_line_dt)(frame.frame[[x]], cfg.line.unit))
            data, line = dask.compute(*data_dt)
        else:
            data = dask.compute(*data_dt)[0]
            line = []
        return Intermediate(
            col=x,
            data=data,
            line=line,
            visual_type="datetime_column",
        )
    elif isinstance(col_dtype, GeoGraphy):
        (data,) = dask.compute(nom_comps(frame.frame[x], cfg))
        return Intermediate(col=x, data=data, visual_type="geography_column")
    else:
        raise ValueError(f"unprocessed type. col:{x}, type:{col_dtype}")


def nom_comps(srs: dd.Series, cfg: Config) -> Dict[str, Any]:
    """
    All computations required for plot(df, Nominal). Assume srs is string column.
    """
    # pylint: disable=too-many-branches
    data: Dict[str, Any] = dict()

    data["nrows"] = srs.shape[0]  # total rows
    srs = srs.dropna()  # drop null values
    grps = srs.value_counts(sort=False)  # counts of unique values in the series
    data["geo"] = grps
    data["nuniq"] = grps.shape[0]  # total number of groups

    # compute bar and pie together unless the parameters are different
    if cfg.bar.enable or cfg.pie.enable or cfg.value_table.enable:
        # select the largest or smallest groups
        data["bar"] = (
            grps.nlargest(cfg.bar.bars) if cfg.bar.sort_descending else grps.nsmallest(cfg.bar.bars)
        )

        if cfg.bar.bars == cfg.pie.slices and cfg.bar.sort_descending == cfg.pie.sort_descending:
            data["pie"] = data["bar"]
        else:
            data["pie"] = (
                grps.nlargest(cfg.pie.slices)
                if cfg.pie.sort_descending
                else grps.nsmallest(cfg.pie.slices)
            )

        if cfg.bar.bars == cfg.value_table.ngroups and cfg.bar.sort_descending:
            data["value_table"] = data["bar"]
        elif cfg.pie.slices == cfg.value_table.ngroups and cfg.pie.sort_descending:
            data["value_table"] = data["pie"]
        else:
            data["value_table"] = grps.nlargest(cfg.value_table.ngroups)

        if cfg.insight.enable:
            data["chisq"] = chisquare(grps.values)

    df = grps.reset_index()  # dataframe with group names and counts
    if cfg.stats.enable or cfg.value_table.enable:
        data.update(_calc_nom_stats(srs, df, data["nrows"], data["nuniq"]))
    elif cfg.wordfreq.enable and cfg.insight.enable:
        data["len_stats"] = {"Minimum": srs.str.len().min(), "Maximum": srs.str.len().max()}
    if cfg.wordlen.enable:
        lens = srs.str.len()
        data["len_hist"] = da.histogram(lens, cfg.wordlen.bins, (lens.min(), lens.max()))
    if cfg.wordcloud.enable or cfg.wordfreq.enable:
        if all(
            getattr(cfg.wordcloud, att) == getattr(cfg.wordfreq, att)
            for att in ("top_words", "stopword", "stem", "lemmatize")
        ):
            word_freqs = _calc_word_freq(
                df,
                cfg.wordfreq.top_words,
                cfg.wordfreq.stopword,
                cfg.wordfreq.lemmatize,
                cfg.wordfreq.stem,
            )
            data["word_cnts_cloud"] = word_freqs["word_cnts"]
            data["nuniq_words_cloud"] = word_freqs["nuniq_words"]
        else:
            word_freqs = _calc_word_freq(
                df.copy(),
                cfg.wordfreq.top_words,
                cfg.wordfreq.stopword,
                cfg.wordfreq.lemmatize,
                cfg.wordfreq.stem,
            )
            word_freqs_cloud = _calc_word_freq(
                df,
                cfg.wordcloud.top_words,
                cfg.wordcloud.stopword,
                cfg.wordcloud.lemmatize,
                cfg.wordcloud.stem,
            )
            data["word_cnts_cloud"] = word_freqs_cloud["word_cnts"]
            data["nuniq_words_cloud"] = word_freqs["nuniq_words"]

        data["word_cnts_freq"] = word_freqs["word_cnts"]
        data["nwords_freq"] = word_freqs["nwords"]

    return data


def cont_comps(srs: dd.Series, cfg: Config) -> Dict[str, Any]:
    """
    All computations required for plot(df, Continuous)
    """
    # pylint: disable=too-many-branches
    data: Dict[str, Any] = {}

    data["nrows"] = srs.shape[0]  # total rows
    srs = srs.dropna()
    data["npres"] = srs.shape[0]  # number of present (not null) values
    srs = srs[~srs.isin({np.inf, -np.inf})]  # remove infinite values
    if cfg.hist.enable or cfg.qqnorm.enable and cfg.insight.enable:
        data["hist"] = da.histogram(srs, cfg.hist.bins, (srs.min(), srs.max()))
        if cfg.insight.enable:
            data["norm"] = normaltest(data["hist"][0])
    if cfg.hist.enable and cfg.insight.enable:
        data["chisq"] = chisquare(data["hist"][0])
    # compute only the required amount of quantiles
    if cfg.qqnorm.enable:
        data["qntls"] = srs.quantile(np.linspace(0.01, 0.99, 99))
    elif cfg.stats.enable:
        data["qntls"] = srs.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    elif cfg.box.enable:
        data["qntls"] = srs.quantile([0.25, 0.5, 0.75])
    if cfg.stats.enable or cfg.hist.enable and cfg.insight.enable:
        data["skew"] = skew(srs)
    if cfg.stats.enable or cfg.qqnorm.enable:
        data["mean"] = srs.mean()
        data["std"] = srs.std()
    if cfg.stats.enable:
        data["min"] = srs.min()
        data["max"] = srs.max()
        data["nreals"] = srs.shape[0]
        data["nzero"] = (srs == 0).sum()
        data["nneg"] = (srs < 0).sum()
        data["kurt"] = kurtosis(srs)
        data["mem_use"] = srs.memory_usage(deep=True)
    # compute the density histogram
    if cfg.kde.enable:
        # To avoid the singular matrix problem, gaussian_kde needs a non-zero std.
        if not math.isclose(dask.compute(data["min"])[0], dask.compute(data["max"])[0]):
            data["dens"] = da.histogram(srs, cfg.kde.bins, (srs.min(), srs.max()), density=True)
            # gaussian kernel density estimate
            data["kde"] = gaussian_kde(
                srs.map_partitions(lambda x: x.sample(min(1000, x.shape[0])), meta=srs)
            )
        else:
            data["kde"] = None
    if cfg.box.enable:
        data.update(_calc_box(srs, data["qntls"], cfg))
    if cfg.value_table.enable:
        value_counts = srs.value_counts(sort=False)
        data["nuniq"] = value_counts.shape[0]
        data["value_table"] = value_counts.nlargest(cfg.value_table.ngroups)
    elif cfg.stats.enable:
        data["nuniq"] = srs.nunique_approx()

    return data


def _calc_box(srs: dd.Series, qntls: da.Array, cfg: Config) -> Dict[str, Any]:
    """
    Box plot calculations
    """
    # quartiles
    data = {f"qrtl{i + 1}": qntls.loc[qnt].sum() for i, qnt in enumerate((0.25, 0.5, 0.75))}

    # inter-quartile range
    iqr = data["qrtl3"] - data["qrtl1"]
    srs_iqr = srs[srs.between(data["qrtl1"] - 1.5 * iqr, data["qrtl3"] + 1.5 * iqr)]
    # lower and upper whiskers
    data["lw"], data["uw"] = srs_iqr.min(), srs_iqr.max()

    # outliers
    otlrs = srs[~srs.between(data["qrtl1"] - 1.5 * iqr, data["qrtl3"] + 1.5 * iqr)]
    # randomly sample at most 100 outliers from each partition without replacement
    smp_otlrs = otlrs.map_partitions(lambda x: x.sample(min(100, x.shape[0])), meta=otlrs)
    data["otlrs"] = smp_otlrs.values
    if cfg.insight.enable:
        data["notlrs"] = otlrs.shape[0]

    return data


def _calc_word_freq(
    df: dd.DataFrame,
    top_words: int,
    stopword: bool,
    lemmatize: bool,
    stem: bool,
) -> Dict[str, Any]:
    """
    Parse a categorical column of text data into words, then compute
    the frequency distribution of words and the total number of words.
    """
    col = df.columns[0]

    regex = rf"\b(?:{'|'.join(ess)})\b|[^\w+ ]" if stopword else r"[^\w+ ]"
    # use a regex to replace stop words and non-alphanumeric characters with empty string
    df[col] = df[col].str.replace(regex, "").str.lower().str.split()

    # ".explode()" to "stack" all the words in a list into a new column
    df = df.explode(col)

    # lemmatize and stem
    if lemmatize or stem:
        df[col] = df[col].dropna()
    if lemmatize:
        df[col] = df[col].apply(WordNetLemmatizer().lemmatize, meta=object)
    if stem:
        df[col] = df[col].apply(PorterStemmer().stem, meta=object)

    word_cnts = df.groupby(col)[df.columns[1]].sum()  # counts of words, excludes null values
    nwords = word_cnts.sum()  # total number of words
    nuniq_words = word_cnts.shape[0]  # total unique words
    fnl_word_cnts = word_cnts.nlargest(top_words)  # words with the highest frequency

    return {"word_cnts": fnl_word_cnts, "nwords": nwords, "nuniq_words": nuniq_words}


def _calc_nom_stats(
    srs: dd.Series,
    df: dd.DataFrame,
    nrows: int,
    nuniq: dd.core.Scalar,
) -> Dict[str, Any]:
    """
    Calculate statistics for a nominal column
    """
    # overview stats
    stats = {
        "nrows": nrows,
        "npres": srs.shape[0],
        "nuniq": nuniq,
        "mem_use": srs.memory_usage(deep=True),
        "first_rows": srs.reset_index(drop=True).loc[:4],
    }
    # length stats
    leng = {
        "Mean": srs.str.len().mean(),
        "Standard Deviation": srs.str.len().std(),
        "Median": srs.str.len().quantile(0.5),
        "Minimum": srs.str.len().min(),
        "Maximum": srs.str.len().max(),
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

    return {"stats": stats, "len_stats": leng, "letter_stats": letter}


def calc_stats_dt(srs: dd.Series) -> Dict[str, str]:
    """
    Calculate stats from a datetime column
    """
    size = srs.shape[0]  # include nan
    count = srs.count()  # exclude nan
    # nunique_approx() has error when type is datetime
    try:
        uniq_count = srs.nunique_approx()
    except:  # pylint: disable=W0702
        uniq_count = srs.nunique()
    overview_dict = {
        "Distinct Count": uniq_count,
        "Approximate Unique (%)": uniq_count / count,
        "Missing": size - count,
        "Missing (%)": 1 - (count / size),
        "Memory Size": srs.memory_usage(deep=True),
        "Minimum": srs.min(),
        "Maximum": srs.max(),
    }

    return overview_dict
