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
# pylint: disable = protected-access
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field
from ..utils import is_notebook


# This dictionary map the plot names in display to their canonicalized names in Config
DISPLAY_MAP = {
    "Bar Chart": "bar",
    "Pie Chart": "pie",
    "Word Cloud": "wordcloud",
    "Word Frequency": "wordfreq",
    "Word Length": "wordlen",
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
    "Value Table": "value_table",
}

# This dictionary map is used for session control in create_report
DISPLAY_REPORT_MAP = {
    "Overview": "overview",
    "Variables": "variables",
    "Interactions": "interactions",
    "Correlations": "correlations",
    "Missing Values": "missingvalues",
}


class Plot(BaseModel):
    """
    Class containing global parameters for the plots
    """

    width: Union[int, None] = None
    height: Union[int, None] = None
    bins: Union[int, None] = None
    ngroups: Union[int, None] = None
    report: bool = False


class Diff(BaseModel):
    """
    Define the parameters in the plot_diff
    """

    label: Union[List[str], None] = None
    baseline: int = 0
    density: bool = False


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
        Warn if the percent of duplicated values is above this threshold
    similar_distribution__threshold:float, default 0.05
        The significance level for Kolmogorov–Smirnov test
    uniform__threshold: float, default 0.999
        The p-value threshold for chi-square test
    missing__threshold: int, default 1
         Warn if the percent of missing values is above this threshold
    skewed__threshold: float, default 1e-5
        The p-value for the scipy.skewtest which test whether the skew is
        different from the normal distributionin
    infinity__threshold: int, default 1
         Warn if the percent of infinites is above this threshold
    zeros__threshold: int, default 5
         Warn if the percent of zeros is above this threshold
    negatives__threshold: int, default 1
         Warn if the percent of negatives is above this threshold
    normal__threshold: float, default 0.99
        The p-value threshold for normal test, it is based on D’Agostino and Pearson’s test that
        combines skew and kurtosis to produce an omnibus test of normality
    high_cardinality__threshold: int, default 50
        The threshold for unique values count, count larger than threshold yields high cardinality
    constant__threshold: int, default 1
        The threshold for unique values count, count equals to threshold yields constant value
    outstanding_no1__threshold: float, default 1.5
        The threshold for outstanding no1 insight, measures the ratio of the largest category count
        to the second-largest category count
    attribution__threshold: float, default 0.5
        The threshold for the attribution insight, measures the percentage of the top 2 categories
    high_word_cardinality__threshold: int, default 1000
        The threshold for the high word cardinality insight, which measures the number of words of
        that cateogory
    outstanding_no1_word__threshold: int, default 0
        The threshold for the outstanding no1 word threshold, which measures the ratio of the most
        frequent word count to the second most frequent word count
    outlier__threshold: int, default 0
        The threshold for the outlier count in the box plot
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
    color: str, default "#aec7e8"
        Color of the histogram
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    bins: int = 50
    yscale: str = "linear"
    color: str = "#aec7e8"
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.bins, self.yscale, self.color, height, width]
        names = ["hist.bins", "hist.yscale", "hist.color", "height", "width"]
        descs = [
            "Number of bins in the histogram",
            'Y-axis scale ("linear" or "log")',
            "Color",
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {_form(val)}", desc) for name, val, desc in zip(names, vals, descs)]

    def grid_how_to_guide(self) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df)
        """
        vals = [self.bins, self.yscale, self.color]
        names = ["hist.bins", "hist.yscale", "hist.color"]
        descs = ["Number of bins in the histogram", 'Y-axis scale ("linear" or "log")', "Color"]
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
    color: str, default "#1f77b4"
        Color of the bar chart
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    bars: int = 10
    sort_descending: bool = True
    yscale: str = "linear"
    color: str = "#1f77b4"
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.bars, self.sort_descending, self.yscale, self.color, height, width]
        names = ["bar.bars", "bar.sort_descending", "bar.yscale", "bar.color", "height", "width"]
        descs = [
            "Maximum number of bars to display",
            "Whether to sort the bars in descending order",
            'Y-axis scale ("linear" or "log")',
            "Color",
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
    hist_color: str, default "#aec7e8"
        Color of the density histogram
    line_color: str, default "#d62728
        Color of the density line
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    bins: int = 50
    yscale: str = "linear"
    hist_color: str = "#aec7e8"
    line_color: str = "#d62728"
    width: Union[int, None] = None
    height: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.bins, self.yscale, self.hist_color, self.line_color, height, width]
        names = ["kde.bins", "kde.yscale", "kde.hist_color", "kde.line_color", "height", "width"]
        descs = [
            "Number of bins in the histogram",
            'Y-axis scale ("linear" or "log")',
            "Color of the density histogram",
            "Color of the density line",
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {_form(val)}", desc) for name, val, desc in zip(names, vals, descs)]


class QQNorm(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    point_color: str, default "#1f77b4"
        Color of the density histogram
    line_color: str, default "#d62728
        Color of the density line
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    point_color: str = "#1f77b4"
    line_color: str = "#d62728"
    height: Union[int, None] = None
    width: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.point_color, self.line_color, height, width]
        names = ["qqnorm.point_color", "qqnorm.line_color", "height", "width"]
        descs = [
            "Color of the points",
            "Color of the line",
            "Height of the plot",
            "Width of the plot",
        ]
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
    color: str, default "#d62728
        Color of the box_plot
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
    color: str = "#1f77b4"
    width: Union[int, None] = None
    height: Union[int, None] = None

    def univar_how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.color, height, width]
        names = ["box.color", "height", "width"]
        descs = ["Color", "Height of the plot", "Width of the plot"]
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
    colors: Optional[List[str]], default None
        List of colors
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    slices: int = 10
    sort_descending: bool = True
    colors: Optional[List[str]] = None
    width: Union[int, None] = None
    height: Union[int, None] = None

    def how_to_guide(self, color_list: List[str], height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.slices, self.sort_descending, color_list, height, width]
        names = ["pie.slices", "pie.sort_descending", "pie.colors", "height", "width"]
        descs = [
            "Maximum number of pie slices to display",
            "Whether to sort the slices in descending order of frequency",
            "List of colors",
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

    enable: bool = is_notebook()
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
    color: str, default "#1f77b4"
        Color of the bar chart
    """

    enable: bool = True
    top_words: int = 30
    stopword: bool = True
    lemmatize: bool = False
    stem: bool = False
    color: str = "#1f77b4"
    width: Union[int, None] = None
    height: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.top_words, self.stopword, self.lemmatize, self.stem, self.color, height, width]
        names = [
            "wordfreq.top_words",
            "wordfreq.stopword",
            "wordfreq.lemmatize",
            "wordfreq.stem",
            "wordfreq.color",
            "height",
            "width",
        ]
        descs = [
            "Maximum number of most frequent words to display",
            "Whether to remove stopwords",
            "Whether to lemmatize the words",
            "Whether to apply Potter Stem on the words",
            "Color",
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
    color: str, default "#aec7e8"
        Color of the histogram
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    bins: int = 50
    yscale: str = "linear"
    color: str = "#aec7e8"
    width: Union[int, None] = None
    height: Union[int, None] = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.bins, self.yscale, self.color, height, width]
        names = ["wordlen.bins", "wordlen.yscale", "wordlen.color", "height", "width"]
        descs = [
            "Number of bins in the histogram",
            'Y-axis scale ("linear" or "log")',
            "Color",
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
    sample_size: int, optional, default=1000
        Number of points to randomly sample per partition.
        Cannot be used with sample_rate.
    sample_rate: float, optional, default None
        sample rate per partition. Cannot be used with
        sample_size. Set it to 1.0 for no sampling.
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    enable: bool = True
    sample_size: Optional[int] = 1000
    sample_rate: Optional[float] = None
    height: Union[int, None] = None
    width: Union[int, None] = None
    # used internally for param checking. Seems
    # internal param will be treat as class attr. in pydantic,
    # hence we need to init. it in __init__.
    _user_input_params: Dict[str, Any]

    def __init__(self) -> None:
        super().__init__()
        object.__setattr__(self, "_user_input_params", {})

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        if self.sample_size is not None:
            para_val: Union[int, float, None] = self.sample_size
            para_name = "scatter.sample_size"
            para_desc = "Number of points to randomly sample per partition"
        else:
            para_val = self.sample_rate
            para_name = "scatter.sample_rate"
            para_desc = "Sample rate to randomly sample per partition"

        vals = [para_val, height, width]
        names = [para_name, "height", "width"]
        descs = [
            para_desc,
            "Height of the plot",
            "Width of the plot",
        ]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]

    def _check_and_correct_param(self) -> None:
        """Check whether the parameters are valid, and correct param when necessary"""
        user_set_sample_size = (
            self._user_input_params["sample_size"]
            if "sample_size" in self._user_input_params
            else None
        )
        user_set_sample_rate = (
            self._user_input_params["sample_rate"]
            if "sample_rate" in self._user_input_params
            else None
        )
        if (user_set_sample_size is not None) and (user_set_sample_rate is not None):
            raise AttributeError(
                f"Scatter plot set sample size {user_set_sample_size} and "
                + f"sample rate {user_set_sample_rate}, please only set one of them."
            )
        if user_set_sample_rate is not None:
            self.sample_size = None


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
    sample_size: int, default 100
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


class ValueTable(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    ngroups: int, default 10
        Number of values to show in the table
    """

    enable: bool = True
    ngroups: int = 10

    def how_to_guide(self) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [self.ngroups]
        names = ["value_table.ngroups"]
        descs = ["The number of distinct values to show"]
        return [(f"'{name}': {_form(val)}", desc) for name, val, desc in zip(names, vals, descs)]


def _form(val: Any) -> Any:
    """
    Format a value for the how-to guide
    """
    return f"'{val}'" if isinstance(val, str) else val


# The following five classes are for create_report
class Overview(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    """

    enable: bool = True


class Variables(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    """

    enable: bool = True


class Interactions(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    cat_enable: bool, default False
        where enable categorical column in interactions. By default it is False,
        which means only numerical interactions are computed. If set to True, cat-cat
        and cat-num interactions will be computed.
    """

    enable: bool = True
    cat_enable: bool = False


class Correlations(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    value_range
        If the correlation value is out of the range, don't show it.
    k
        Choose top-k element
    """

    enable: bool = True
    value_range: Optional[Tuple[float, float]] = None
    k: Optional[int] = None

    def how_to_guide(self) -> List[Tuple[str, str]]:
        """
        how-to guide for plot_correlation(df, x)
        """
        vals = [self.value_range, self.k]
        names = ["correlations.value_range", "correlations.k"]
        descs = [
            "If the correlation value is out of the range, don't show it",
            "Choose top-k element",
        ]
        return [(f"'{name}': {_form(val)}", desc) for name, val, desc in zip(names, vals, descs)]


class MissingValues(BaseModel):
    """
    enable: bool, default True
        Whether to create this element
    """

    enable: bool = True


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
    value_table: ValueTable = Field(default_factory=ValueTable)
    plot: Plot = Field(default_factory=Plot)
    overview: Overview = Field(default_factory=Overview)
    variables: Variables = Field(default_factory=Variables)
    interactions: Interactions = Field(default_factory=Interactions)
    correlations: Correlations = Field(default_factory=Correlations)
    missingvalues: MissingValues = Field(default_factory=MissingValues)
    diff: Diff = Field(default_factory=Diff)

    def _set_enable_for_plots(self, display: List[str]) -> None:
        """set the enable for all plots from display, used for 'from_dict' constructor"""
        all_plot_names = vars(self).keys()
        try:
            # set all plots not in display list to enable=False except for Plot and Diff class
            valid_display = [DISPLAY_MAP[disp] for disp in display]
            for plot_name in set(all_plot_names) - set(valid_display) - {"plot"} - {"diff"}:
                setattr(getattr(self, plot_name), "enable", False)
        except KeyError:
            # handle report config
            valid_display = [DISPLAY_REPORT_MAP[disp] for disp in display]
            for plot_name in set(DISPLAY_REPORT_MAP.values()) - set(valid_display):
                setattr(getattr(self, plot_name), "enable", False)

    def _set_param_for_plot(
        self, plot_name: str, param: str, val: Any, raise_error_if_not_exists: bool
    ) -> None:
        """set the parameter for a given plot, used when
        set global and local parameters for each plot"""
        if plot_name not in vars(self).keys():
            raise AttributeError(f"plot {plot_name} does not exist")
        plot = getattr(self, plot_name)
        if hasattr(plot, param):
            setattr(plot, param, val)
            if hasattr(plot, "_user_input_params"):
                plot._user_input_params[param] = val
        else:
            if raise_error_if_not_exists:
                raise AttributeError(f"{plot_name} plot does not have parameter {param}")

    def _set_global_param_for_plots(self, global_params: Dict[str, Any]) -> None:
        """set the global parameters for all plots, used for 'from_dict' constructor"""
        all_plot_names = vars(self).keys()
        valid_global_params = vars(self.plot).keys()
        for param, val in global_params.items():
            # set the parameter to the specified value for each plot that
            # has this parameter
            if param not in valid_global_params:
                raise AttributeError(f"{param} is not a global parameter")
            else:
                # ngroups applies to "bars" and "slices" for the bar and pie charts
                if param == "ngroups":
                    setattr(getattr(self, "bar"), "bars", val)
                    setattr(getattr(self, "pie"), "slices", val)
                for plot_name in all_plot_names:
                    self._set_param_for_plot(plot_name, param, val, raise_error_if_not_exists=False)

    def _set_local_param_for_plots(self, local_params: Dict[str, Any]) -> None:
        """set the local parameters for all plots, used for 'from_dict' constructor"""
        for key, value in local_params.items():
            plot_name, rest = key.split(".", 1)
            param = rest.replace(".", "__")
            self._set_param_for_plot(plot_name, param, value, raise_error_if_not_exists=True)

    def _check_and_correct_params_for_plots(self) -> None:
        """Call the '_check_and_correct_param' for some plots, used for 'from_dict' constructor.
        The '_check_and_correct_param' is used to check and correct parameter and handle the case
        when multiple parameters are not allowed set at the same time. E.g., the sample size and
        sample rate in scatter plot."""
        all_plot_names = vars(self).keys()
        for plot_name in all_plot_names:
            plot = getattr(self, plot_name)
            if hasattr(plot, "_check_and_correct_param"):
                plot._check_and_correct_param()

    @classmethod
    def from_dict(
        cls, display: Optional[List[str]] = None, config: Optional[Dict[str, Any]] = None
    ) -> Config:
        """
        Converts an dictionary instance into a config class
        """
        cfg = cls()
        if display is not None:
            cfg._set_enable_for_plots(display)

        if config is not None:
            # get the global and local parameters from config
            global_params = {key: config[key] for key in config if "." not in key}
            local_params = {key: config[key] for key in config if key not in global_params}

            cfg._set_global_param_for_plots(global_params)
            cfg._set_local_param_for_plots(local_params)
            cfg._check_and_correct_params_for_plots()
        return cfg
