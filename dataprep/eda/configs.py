"""
Parameter configurations

This file contains configurations for stats, auto-insights and plots. There are mainly two settings,
"display" and "config". Display is a list of Tab names which control the Tabs to show.
Config is a dictionary that contains the customizable parameters and corresponding values.
There are two types of parameters, global and local. Local parameters are plot-specified and
the names are separated  by ".". The portion before the first "." is plot name and the portion
after the first "." is parameter name. e.g. "hist.bins". The "." is also used when the parameter
name contains more than one word. e.g. "insight.duplicates.threshold". However, in the codebase,
the "." is replaced with "__" for parameters with long names.e.g. "insight.duplicates__threshold".
Global parameter is single-word. It applies to all the plots which has that parameter.
e.g. "bins:50" applies to "hist.bins", "line.bins", "kde.bins", "wordlen.bins" and "box.bins".
In addition,when global parameter and local parameter are both entered by a user in config,
the global parameter will be overwrote by local parameters for specific plots.
"""

# pylint: disable=too-many-lines,no-self-use,blacklisted-name,no-else-raise,too-many-branches,no-name-in-module

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

# This dictionary map the plot names in display to their canonicalized names in Config
DISPLAY_MAP = {
    "Bar Chart": "bar",
    "Pie Chart": "pie",
    "Word Cloud": "wordcloud",
    "Word Frequency": "wordfreq",
    "Word Length": "wordlen",
    "N Gram": "ngram",
    "Histogram": "hist",
    "KDE Plot": "kde",
    "Normal Q-Q Plot": "qqnorm",
    "Box Plot": "box",
    "Line Chart": "line",
    "Stats": "stats",
    "Insights": "insight",
    "Scatter Plot": "scatter",
    "Hexbin Plot": "hexbin",
    "Nested Bar Chart": "nested",
    "Stacked Bar Chart": "stacked",
    "Heat Map": "heatmap",
    "Pearson": "pearson",
    "Spearman": "spearman",
    "KendallTau": "kendall",
    "Spectrum": "spectrum",
    "Dendrogram": "dendro",
    "PDF": "pdf",
    "CDF": "cdf",
}


class Plot(BaseModel):
    """
    Class containing global parameters for the plots
    """

    width: Union[int, None] = None
    height: Union[int, None] = None
    bins: Union[int, None] = None
    ngroups: Union[int, None] = None
    grid_column: int = 3
    report: bool = False


class Stats(BaseModel):
    """
    enable: bool, default True
        Whether to display the stats section
    """

    enable: bool = True


class Insight(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    duplicates__threshold: int, default 1
        The threshold for duplicated row counts
    similar_distribution__threshold:int, default 0.05
        The significance level for Kolmogorov–Smirnov test
    uniform__threshold: int, default 0.999
        The p-value threshold for chi-square test
    missing__threshold: int, default 1
        The threshold for missing values count
    skewed__threshold: int, default 1e-5
        The threshold for skewness statistics
    infinity__threshold: int, default 1
        The threshold for infinity count
    zeros__threshold: int, default 5
        The threshold for zeros count
    negatives__threshold: int, default 1
        The threshold for negatives count
    normal__threshold: int, default 0.99
        The p-value threshold for normaltest, it is based on D’Agostino and Pearson’s test that
        combines skew and kurtosis to produce an omnibus test of normality
    high_cardinality__threshold: int, default 50
        The threshold for unique values count, count larger than threshold yields high cardinality
    constant__threshold: int, default 1
        The threshold for unique values count, count equals to threshold yields constant value
    outstanding_no1__threshold: int, default 1.5
        The threshold for outstanding no1 insight, measures the ratio of the largest category count
        to the second-largest category count
    attribution__threshold: int, default 0.5
        The threshold for the attribution insight, measures the percentage of the top 2 categories
    high_word_cardinality__threshold: int, default 1000
        The threshold for the high word cardinality insight, which measures the number of words of
        that cateogory
    outstanding_no1_word__threshold: int, default 0
        The threshold for the outstanding no1 word threshold, which measures the ratio of the most
        frequent word count to the second most frequent word count
    outlier__threshold: int, default 0
        The threshold for the outlier count in the box plot, default 0
    """

    # pylint: disable=too-many-instance-attributes
    enable: bool = True
    duplicates__threshold: int = 1
    similar_distribution__threshold: float = 0.05
    uniform__threshold: float = 0.999
    missing__threshold: int = 1
    skewed__threshold: float = 1e-5
    infinity__threshold: int = 1
    zeros__threshold: int = 5
    negatives__threshold: int = 1
    normal__threshold: float = 0.99
    high_cardinality__threshold: int = 50
    constant__threshold: int = 1
    outstanding_no1__threshold: float = 1.5
    attribution__threshold: float = 0.5
    high_word_cardinality__threshold: int = 1000
    outstanding_no1_word__threshold: float = 1.5
    outlier__threshold: int = 0


class Hist(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    bins: int, default 50
        Number of bins in the histogram
    yscale: str, default "linear"
        Y-axis scale ("linear" or "log")
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    bins: int = 50
    yscale: str = "linear"
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.bins, self.yscale, height, width]
        names = ["hist.bins", "hist.yscale", "height", "width"]
        descs = [
            "Number of bins in the histogram",
            'Y-axis scale ("linear" or "log")',
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {_form(val)}", desc) for name, val, desc in zip(names, vals, descs)]

    def grid_how_to_guide(self) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df)
        """
        vals = [self.bins, self.yscale]
        names = ["hist.bins", "hist.yscale"]
        descs = ["Number of bins in the histogram", 'Y-axis scale ("linear" or "log")']
        return [(f"'{name}': {_form(val)}", desc) for name, val, desc in zip(names, vals, descs)]


class Bar(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    bars: int, default 10
        Maximum number of bars to display
    sort_descending: bool, default True
        Whether to sort the bars in descending order
    yscale: str, default "linear"
        Y-axis scale ("linear" or "log")
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    bars: int = 10
    sort_descending: bool = True
    yscale: str = "linear"
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.bars, self.sort_descending, self.yscale, height, width]
        names = ["bar.bars", "bar.sort_descending", "bar.yscale", "height", "width"]
        descs = [
            "Maximum number of bars to display",
            "Whether to sort the bars in descending order",
            'Y-axis scale ("linear" or "log")',
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {_form(val)}", desc) for name, val, desc in zip(names, vals, descs)]

    def grid_how_to_guide(self) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df)
        """
        vals = [self.bars, self.sort_descending, self.yscale]
        names = ["bar.bars", "bar.sort_descending", "bar.yscale"]
        descs = [
            "Maximum number of bars to display",
            "Whether to sort the bars in descending order",
            'Y-axis scale ("linear" or "log")',
        ]
        return [(f"'{name}': {_form(val)}", desc) for name, val, desc in zip(names, vals, descs)]

    def missing_how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot_missing(df, x, [y])
        """
        vals = [height, width]
        names = ["height", "width"]
        descs = ["Height of the plot", "Width of the plot"]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class KDE(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    bins: int, default 50
        Number of bins in the histogram
    yscale: str, default "linear"
        Y-axis scale ("linear" or "log")
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    bins: int = 50
    yscale: str = "linear"
    width: Union[int, None] = None
    height: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.bins, self.yscale, height, width]
        names = ["kde.bins", "hist.yscale", "height", "width"]
        descs = [
            "Number of bins in the histogram",
            'Y-axis scale ("linear" or "log")',
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {_form(val)}", desc) for name, val, desc in zip(names, vals, descs)]


class QQNorm(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [height, width]
        names = ["height", "width"]
        descs = ["Height of the plot", "Width of the plot"]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class Box(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    ngroups: int, default 15
        Maximum number of groups to display
    bins: int, default 50
        Number of bins
    unit: str, default "auto"
        Defines the time unit to group values over for a datetime column.
        It can be "year", "quarter", "month", "week", "day", "hour",
        "minute", "second". With default value "auto", it will use the
        time unit such that the resulting number of groups is closest to 15
    sort_descending: bool, default True
        Whether to sort the boxes in descending order of frequency
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    ngroups: int = 15
    bins: int = 50
    unit: str = "auto"
    sort_descending: bool = True
    width: Union[int, None] = None
    height: Union[int, None] = None

    def univar_how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [height, width]
        names = ["height", "width"]
        descs = ["Height of the plot", "Width of the plot"]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]

    def nom_cont_how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, nominal, continuous)
        """
        vals = [self.ngroups, self.sort_descending, height, width]
        names = ["box.ngroups", ".box.sort_descending", "height", "width"]
        descs = [
            "Maximum number of groups to display",
            "Whether to sort the boxes in descending order of frequency",
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]

    def two_cont_how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, continuous, continuous)
        """
        vals = [self.bins, height, width]
        names = ["box.bins", "height", "width"]
        descs = ["Number of bins", "Height of the plot", "Width of the plot"]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class Pie(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    slices: int, default 10
        Maximum number of pie slices to display
    sort_descending: bool, default True
        Whether to sort the slices in descending order of frequency
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    slices: int = 10
    sort_descending: bool = True
    width: Union[int, None] = None
    height: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.slices, self.sort_descending, height, width]
        names = ["pie.slices", "pie.sort_descending", "height", "width"]
        descs = [
            "Maximum number of pie slices to display",
            "Whether to sort the slices in descending order of frequency",
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class WordCloud(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    top_words: int, default 30
        Maximum number of most frequent words to display
    stopword: bool, default True
        Whether to remove stopwords
    lemmatize: bool, default False
        Whether to lemmatize the words
    stem: bool, default False
        Whether to apply Potter Stem on the words
    """

    enable: bool = True
    top_words: int = 30
    stopword: bool = True
    lemmatize: bool = False
    stem: bool = False
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.top_words, self.stopword, self.lemmatize, self.stem, height, width]
        names = [
            "wordcloud.top_words",
            "wordcloud.stopword",
            "wordcloud.lemmatize",
            "wordcloud.stem",
            "height",
            "width",
        ]
        descs = [
            "Maximum number of most frequent words to display",
            "Whether to remove stopwords",
            "Whether to lemmatize the words",
            "Whether to apply Potter Stem on the words",
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class WordFrequency(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    top_words: int, default 30
        Maximum number of most frequent words to display
    stopword: bool, default True
        Whether to remove stopwords
    lemmatize: bool, default False
        Whether to lemmatize the words
    stem: bool, default False
        Whether to apply Potter Stem on the words
    """

    enable: bool = True
    top_words: int = 30
    stopword: bool = True
    lemmatize: bool = False
    stem: bool = False
    width: Union[int, None] = None
    height: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.top_words, self.stopword, self.lemmatize, self.stem, height, width]
        names = [
            "wordfreq.top_words",
            "wordfreq.stopword",
            "wordfreq.lemmatize",
            "wordfreq.stem",
            "height",
            "width",
        ]
        descs = [
            "Maximum number of most frequent words to display",
            "Whether to remove stopwords",
            "Whether to lemmatize the words",
            "Whether to apply Potter Stem on the words",
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class NGram(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    top_grams: int, default 30
        Maximum number of most frequent words to display
    grams: int, default 3
        Number of grams in the histogram
    stopword: bool, default True
        Whether to remove stopwords
    lemmatize: bool, default False
        Whether to lemmatize the words
    stem: bool, default False
        Whether to apply Potter Stem on the words
    """

    enable: bool = True
    top_grams: int = 30
    grams: int = 3
    stopword: bool = True
    lemmatize: bool = False
    stem: bool = False
    width: Union[int, None] = None
    height: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.top_grams, self.grams, self.stopword, self.lemmatize, self.stem, height, width]
        names = [
            "ngram.top_grams",
            "ngram.grams",
            "ngram.stopword",
            "ngram.lemmatize",
            "ngram.stem",
            "height",
            "width",
        ]
        descs = [
            "Maximum number of most frequent words to display",
            "Number of grams in the histogram",
            "Whether to remove stopwords",
            "Whether to lemmatize the words",
            "Whether to apply Potter Stem on the words",
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class WordLength(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    bins: int, default 50
        Number of bins in the histogram
    yscale: str, default "linear"
        Y-axis scale ("linear" or "log")
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    bins: int = 50
    yscale: str = "linear"
    width: Union[int, None] = None
    height: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.bins, self.yscale, height, width]
        names = ["wordlen.bins", "wordlen.yscale", "height", "width"]
        descs = [
            "Number of bins in the histogram",
            'Y-axis scale ("linear" or "log")',
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {_form(val)}", desc) for name, val, desc in zip(names, vals, descs)]


class Line(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    bins: int, default 50
        Number of bins
    ngroups: int, default 10
        Maximum number of groups to display
    sort_descending: bool, default True
        Whether to sort the groups in descending order of frequency
    yscale: str, default "linear"
        The scale to show on the y axis. Can be "linear" or "log".
    unit: str, default "auto"
        Defines the time unit to group values over for a datetime column.
        It can be "year", "quarter", "month", "week", "day", "hour",
        "minute", "second". With default value "auto", it will use the
        time unit such that the resulting number of groups is closest to 15
    agg: str, default "mean"
        Specify the aggregate to use when aggregating over a numeric column
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    # pylint: disable=too-many-instance-attributes

    enable: bool = True
    bins: int = 50
    ngroups: int = 10
    sort_descending: bool = True
    yscale: str = "linear"
    unit: str = "auto"
    agg: str = "mean"
    height: Union[int, None] = None
    width: Union[int, None] = None

    def nom_cont_how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, nominal, continuous)
        """
        vals = [self.ngroups, self.sort_descending, height, width]
        names = ["line.ngroups", "line.sort_descending", "height", "width"]
        descs = [
            "Maximum number of groups to display",
            "Whether to sort the groups in descending order of frequency",
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class Scatter(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    sample_size: int, default 1000
        Number of points to randomly sample per partition
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    sample_size: int = 1000
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        vals = [self.sample_size, height, width]
        names = ["scatter.sample_size", "height", "width"]
        descs = [
            "Number of points to randomly sample per partition",
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class Hexbin(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    tile_size: float, default "auto"
        The size of the tile in the hexbin plot. Measured from the middle
        of a hexagon to its left or right corner.
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    tile_size: str = "auto"
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, tile_size: float, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        vals = [tile_size, height, width]
        names = ["hexbin.tile_size", "height", "width"]
        descs = [
            "Tile size, measured from the middle of the hexagon to the left or right corner",
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class Nested(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    ngroups: int, default 10
        Maximum number of most frequent values from the first column to display
    nsubgroups: int, default 5
        Maximum number of most frequent values from the second column to display (computed
        on the filtered data consisting of the most frequent values from the first column)
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    ngroups: int = 10
    nsubgroups: int = 5
    width: Union[int, None] = None
    height: Union[int, None] = None

    def how_to_guide(self, x: str, y: str, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        vals = [self.ngroups, self.nsubgroups, height, width]
        names = ["nested.ngroups", "nested.nsubgroups", "height", "width"]
        descs = [
            f"Maximum number of most frequent values in column {x} to display",
            f"""Maximum number of most frequent values in column {y} to display (computed
            on the filtered data consisting of the most frequent values in column {x})""",
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class Stacked(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    ngroups: int, default 10
        Maximum number of most frequent values from the first column to display
    nsubgroups: int, default 5
        Maximum number of most frequent values from the second column to display (computed
        on the filtered data consisting of the most frequent values from the first column)
    unit: str, default "auto"
        Defines the time unit to group values over for a datetime column.
        It can be "year", "quarter", "month", "week", "day", "hour",
        "minute", "second". With default value "auto", it will use the
        time unit such that the resulting number of groups is closest to 15
    sort_descending: bool, default True
        Whether to sort the groups in descending order of frequency
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    ngroups: int = 10
    nsubgroups: int = 5
    unit: str = "auto"
    sort_descending: bool = True
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, x: str, y: str, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        vals = [self.ngroups, self.nsubgroups, height, width]
        names = ["stacked.ngroups", "stacked.nsubgroups", "height", "width"]
        descs = [
            f"Maximum number of most frequent values in column {x} to display",
            f"""Maximum number of most frequent values in column {y} to display (computed
            on the filtered data consisting of the most frequent values in column {x})""",
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class Heatmap(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    ngroups: int, default 10
        Maximum number of most frequent values from the first column to display
    nsubgroups: int, default 5
        Maximum number of most frequent values from the second column to display (computed
        on the filtered data consisting of the most frequent values from the first column)
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    ngroups: int = 10
    nsubgroups: int = 5
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, x: str, y: str, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, nominal, nominal)
        """
        vals = [self.ngroups, self.nsubgroups, height, width]
        names = ["heatmap.ngroups", "heatmap.nsubgroups", "height", "width"]
        descs = [
            f"Maximum number of most frequent values in column {x} to display",
            f"""Maximum number of most frequent values in column {y} to display (computed
            on the filtered data consisting of the most frequent values in column {x})""",
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]

    def missing_how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot_missing(df)
        """
        vals = [height, width]
        names = ["height", "width"]
        descs = ["Height of the plot", "Width of the plot"]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class Pearson(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        vals = [height, width]
        names = ["height", "width"]
        descs = ["Height of the plot", "Width of the plot"]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class Spearman(BaseModel):
    """
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        vals = [height, width]
        names = ["height", "width"]
        descs = ["Height of the plot", "Width of the plot"]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class KendallTau(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        vals = [height, width]
        names = ["height", "width"]
        descs = ["Height of the plot", "Width of the plot"]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class Spectrum(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    bins: int, default 20
        Number of bins
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    bins: int = 20
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        vals = [self.bins, height, width]
        names = ["spectrum.bins", "height", "width"]
        descs = ["Number of bins", "Height of the plot", "Width of the plot"]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class Dendrogram(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        vals = [height, width]
        names = ["height", "width"]
        descs = ["Height of the plot", "Width of the plot"]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class PDF(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    sample_size:
        Number of evenly spaced samples between the minimum and maximum values to compute the pdf at
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    sample_size: int = 100
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        vals = [self.sample_size, height, width]
        names = ["pdf.sample_size", "height", "width"]
        descs = [
            """Number of evenly spaced samples between the minimum and maximum values to
            compute the pdf at""",
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class CDF(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    sample_size:
        Number of evenly spaced samples between the minimum and maximum values to compute the cdf at
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    sample_size: int = 100
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        vals = [self.sample_size, height, width]
        names = ["pdf.sample_size", "height", "width"]
        descs = [
            """Number of evenly spaced samples between the minimum and maximum values to
            compute the pdf at""",
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


def _form(val: Any) -> Any:
    """
    Format a value for the how-to guide
    """
    return f"'{val}'" if isinstance(val, str) else val


class Config(BaseModel):
    """
    Configuration class
    """

    # pylint: disable=too-many-instance-attributes
    hist: Hist = Field(default_factory=Hist)
    bar: Bar = Field(default_factory=Bar)
    pie: Pie = Field(default_factory=Pie)
    line: Line = Field(default_factory=Line)
    stats: Stats = Field(default_factory=Stats)
    wordcloud: WordCloud = Field(default_factory=WordCloud)
    wordfreq: WordFrequency = Field(default_factory=WordFrequency)
    wordlen: WordLength = Field(default_factory=WordLength)
    ngram: NGram = Field(default_factory=NGram)
    qqnorm: QQNorm = Field(default_factory=QQNorm)
    kde: KDE = Field(default_factory=KDE)
    box: Box = Field(default_factory=Box)
    scatter: Scatter = Field(default_factory=Scatter)
    nested: Nested = Field(default_factory=Nested)
    stacked: Stacked = Field(default_factory=Stacked)
    heatmap: Heatmap = Field(default_factory=Heatmap)
    insight: Insight = Field(default_factory=Insight)
    hexbin: Hexbin = Field(default_factory=Hexbin)
    pearson: Pearson = Field(default_factory=Pearson)
    spearman: Spearman = Field(default_factory=Spearman)
    kendall: KendallTau = Field(default_factory=KendallTau)
    spectrum: Spectrum = Field(default_factory=Spectrum)
    dendro: Dendrogram = Field(default_factory=Dendrogram)
    pdf: PDF = Field(default_factory=PDF)
    cdf: CDF = Field(default_factory=CDF)
    plot: Plot = Field(default_factory=Plot)

    @classmethod
    def from_dict(
        cls, display: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None
    ) -> Config:
        """
        Converts an dictionary instance into a config class
        """
        cfg = cls()

        if display:
            display = [DISPLAY_MAP[disp] for disp in display]
            # set all plots not in display list to enable=False except for Plot class
            for plot in set(vars(cfg).keys()) - set(display) - {"plot"}:
                setattr(getattr(cfg, plot), "enable", False)

        if config:
            # get the global parameters from config
            global_params = {key: config[key] for key in config if "." not in key}
            for param, val in global_params.items():
                # set the parameter to the specified value for each plot that
                # has this parameter
                if param not in vars(cfg.plot).keys():
                    raise Exception(param + " does not exist")
                else:
                    for plot in vars(cfg).keys():
                        if hasattr(getattr(cfg, plot), param):
                            setattr(getattr(cfg, plot), param, val)

                    # ngroups applies to "bars" and "slices" for the bar and pie charts
                    if param == "ngroups":
                        setattr(getattr(cfg, "bar"), "bars", val)
                        setattr(getattr(cfg, "pie"), "slices", val)

            # get the local parameters from config
            local_params = {key: config[key] for key in config if key not in global_params}
            for key, value in local_params.items():
                plot, rest = key.split(".", 1)
                param = rest.replace(".", "__")
                if plot not in vars(cfg).keys():
                    raise Exception(plot + " does not exist")
                elif not hasattr(getattr(cfg, plot), param):
                    raise Exception(key.replace(f"{plot}.", "") + " does not exist")
                else:
                    setattr(getattr(cfg, plot), param, value)

        return cfg
