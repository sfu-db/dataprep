"""
This file defines config classes for plots
"""
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class ConfigBase:
    """
    Base class for config
    """

    name: str = "Base"

    @classmethod
    def from_dict(cls, _: Dict[str, Any]) -> "ConfigBase":
        """Create a config from a dict of values"""


@dataclass
class Histogram(ConfigBase):
    """
    Histogram config
    """

    name: str = "Histogram"

    bins: int = 10
    agg: str = "mean"
    value_range: Optional[Tuple[float, float]] = None

    @classmethod
    def from_dict(cls, kwargs: Dict[str, Any]) -> "Histogram":
        if "agg" in kwargs.keys():
            if kwargs["agg"] != "mean" and kwargs["agg"] != "freq":
                raise ValueError("agg must be mean or freq")

        return cls(**kwargs)


# pylint: disable=too-many-instance-attributes
@dataclass
class BarChart(ConfigBase):
    """
    Bar Chart config
    """

    name: str = "BarChart"

    ngroups: int = 10
    largest: bool = False
    nsubgroups: int = 5
    top_words: int = 30
    stopword: bool = True
    lemmatize: bool = False
    stem: bool = False
    sort_by: str = "alphabet"
    sort_ascending: bool = True

    @classmethod
    def from_dict(cls, kwargs: Dict[str, Any]) -> "BarChart":
        if "sort_by" in kwargs.keys():
            if kwargs["sort_by"] != "alphabet" and kwargs["sort_by"] != "frequency":
                raise ValueError("sort_by must be alphabet or frequency")

        return cls(**kwargs)


@dataclass
class WordCloud(ConfigBase):
    """
    Word Cloud config
    """

    name: str = "WordCloud"

    top_words: int = 30
    stopword: bool = True
    lemmatize: bool = False
    stem: bool = False

    @classmethod
    def from_dict(cls, kwargs: Dict[str, Any]) -> "WordCloud":
        return cls(**kwargs)
