"""
Parameter configurations
"""
# pylint: disable=too-many-lines,no-self-use,blacklisted-name

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class Config:
    """
    Configuration class
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self) -> None:
        self.hist = Hist()
        self.bar = Bar()
        self.pie = Pie()
        self.line = Line()
        self.stats = Stats()
        self.wordcloud = WordCloud()
        self.wordfreq = WordFrequency()
        self.wordlen = WordLength()
        self.qqnorm = QQNorm()
        self.kde = KDE()
        self.box = Box()
        self.scatter = Scatter()
        self.nested = Nested()
        self.stacked = Stacked()
        self.heatmap = Heatmap()
        self.insight = Insight()
        self.hexbin = Hexbin()
        self.pearson = Pearson()
        self.spearman = Spearman()
        self.kendall = KendallTau()
        self.spectrum = Spectrum()
        self.heatmap = Heatmap()
        self.dendro = Dendrogram()
        self.pdf = PDF()
        self.cdf = CDF()
        self.plot = Plot()

    @classmethod
    def from_dict(cls, display: Optional[List[str]], config: Optional[Dict[str, Any]]) -> Config:
        """
        Converts an dictionary instance into a config class
        """
        cfg = cls()
        display_map = {
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
        }

        if display:
            display = [display_map[disp] for disp in display]
            # set all plots not in display list to enable=False
            for plot in set(vars(cfg).keys()) - set(display):
                setattr(getattr(cfg, plot), "enable", False)

        if config:
            # get the global parameters from config
            global_params = {key: config[key] for key in config if "." not in key}
            for param, val in global_params.items():
                # set the parameter to the specified value for each plot that
                # has the parameter
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
                plot = key.split(".")[0]
                param = key.replace(f"{plot}.", "").replace(".", "__")
                setattr(getattr(cfg, plot), param, value)

        return cfg


class Plot:
    """
    Class containing global parameters for the plots
    """

    def __init__(self) -> None:
        self.width = None
        self.height = None
        self.bins = None
        self.ngroups = None
        self.grid_column = 3
        self.report = False


class Stats:
    """
    enable: bool, default True
        Whether to display the stats section
    """

    def __init__(self) -> None:
        self.enable = True


class Insight:
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

    def __init__(self) -> None:
        self.enable = True
        self.duplicates__threshold = 1
        self.similar_distribution__threshold = 0.05
        self.uniform__threshold = 0.999
        self.missing__threshold = 1
        self.skewed__threshold = 1e-5
        self.infinity__threshold = 1
        self.zeros__threshold = 5
        self.negatives__threshold = 1
        self.normal__threshold = 0.99
        self.high_cardinality__threshold = 50
        self.constant__threshold = 1
        self.outstanding_no1__threshold = 1.5
        self.attribution__threshold = 0.5
        self.high_word_cardinality__threshold = 1000
        self.outstanding_no1_word__threshold = 1.5
        self.outlier__threshold = 0


class Hist:
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

    def __init__(self) -> None:
        self.enable = True
        self.bins = 50
        self.yscale = "linear"
        self.height = None
        self.width = None

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


class Bar:
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

    def __init__(self) -> None:
        self.enable = True
        self.bars = 10
        self.sort_descending = True
        self.yscale = "linear"
        self.height = None
        self.width = None

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


class KDE:
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

    def __init__(self) -> None:
        self.enable = True
        self.bins = 50
        self.yscale = "linear"
        self.width = None
        self.height = None

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


class QQNorm:
    """
    enable: bool, default True
        Whether to create this element
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    def __init__(self) -> None:
        self.enable = True
        self.height = None
        self.width = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide for plot(df, x)
        """
        vals = [height, width]
        names = ["height", "width"]
        descs = ["Height of the plot", "Width of the plot"]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class Box:
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

    def __init__(self) -> None:
        self.enable = True
        self.ngroups = 15
        self.bins = 50
        self.unit = "auto"
        self.sort_descending = True
        self.width = None
        self.height = None

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


class Pie:
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

    def __init__(self) -> None:
        self.enable = True
        self.slices = 10
        self.sort_descending = True
        self.width = None
        self.height = None

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


class WordCloud:
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

    def __init__(self) -> None:
        self.enable = True
        self.top_words = 30
        self.stopword = True
        self.lemmatize = False
        self.stem = False
        self.height = None
        self.width = None

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


class WordFrequency:
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

    def __init__(self) -> None:
        self.enable = True
        self.top_words = 30
        self.stopword = True
        self.lemmatize = False
        self.stem = False
        self.width = None
        self.height = None

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


class WordLength:
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

    def __init__(self) -> None:
        self.enable = True
        self.bins = 50
        self.yscale = "linear"
        self.width = None
        self.height = None

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


class Line:
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

    def __init__(self) -> None:
        self.enable = True
        self.bins = 50
        self.ngroups = 10
        self.sort_descending = True
        self.yscale = "linear"
        self.unit = "auto"
        self.agg = "mean"
        self.height = None
        self.width = None

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


class Scatter:
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

    def __init__(self) -> None:
        self.enable = True
        self.sample_size = 1000
        self.height = None
        self.width = None

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


class Hexbin:
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

    def __init__(self) -> None:
        self.enable = True
        self.tile_size = "auto"
        self.height = None
        self.width = None

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


class Nested:
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

    def __init__(self) -> None:
        self.enable = True
        self.ngroups = 10
        self.nsubgroups = 5
        self.width = None
        self.height = None

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


class Stacked:
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

    def __init__(self) -> None:
        self.enable = True
        self.ngroups = 10
        self.nsubgroups = 5
        self.unit = "auto"
        self.sort_descending = True
        self.height = None
        self.width = None

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


class Heatmap:
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

    def __init__(self) -> None:
        self.enable = True
        self.ngroups = 10
        self.nsubgroups = 5
        self.height = None
        self.width = None

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


class Pearson:
    """
    enable: bool, default True
        Whether to create this element
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    def __init__(self) -> None:
        self.enable = True
        self.height = None
        self.width = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        vals = [height, width]
        names = ["height", "width"]
        descs = ["Height of the plot", "Width of the plot"]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class Spearman:
    """
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    def __init__(self) -> None:
        self.enable = True
        self.height = None
        self.width = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        vals = [height, width]
        names = ["height", "width"]
        descs = ["Height of the plot", "Width of the plot"]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class KendallTau:
    """
    enable: bool, default True
        Whether to create this element
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    def __init__(self) -> None:
        self.enable = True
        self.height = None
        self.width = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        vals = [height, width]
        names = ["height", "width"]
        descs = ["Height of the plot", "Width of the plot"]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class Spectrum:
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

    def __init__(self) -> None:
        self.enable = True
        self.bins = 20
        self.height = None
        self.width = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        vals = [self.bins, height, width]
        names = ["spectrum.bins", "height", "width"]
        descs = ["Number of bins", "Height of the plot", "Width of the plot"]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class Dendrogram:
    """
    enable: bool, default True
        Whether to create this element
    height: int, default "auto"
        Height of the plot
    width: int, default "auto"
        Width of the plot
    """

    def __init__(self) -> None:
        self.enable = True
        self.height = None
        self.width = None

    def how_to_guide(self, height: int, width: int) -> List[Tuple[str, str]]:
        """
        how-to guide
        """
        vals = [height, width]
        names = ["height", "width"]
        descs = ["Height of the plot", "Width of the plot"]
        return [(f"'{name}': {val}", desc) for name, val, desc in zip(names, vals, descs)]


class PDF:
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

    def __init__(self) -> None:
        self.enable = True
        self.sample_size = 100
        self.height = None
        self.width = None

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


class CDF:
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

    def __init__(self) -> None:
        self.enable = True
        self.sample_size = 100
        self.height = None
        self.width = None

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
