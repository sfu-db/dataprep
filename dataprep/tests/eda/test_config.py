"""
    This module for testing config parameter
"""

from ...eda.basic.configs import BarChart, Histogram, WordCloud


def test_hist() -> None:
    dict_data = {"bins": 20, "agg": "mean", "value_range": [0.1, 0.5]}
    histogram = Histogram()
    histogram = histogram.from_dict(dict_data)
    for key in dict_data.keys():
        assert histogram.__dict__[key] == dict_data[key]


def test_bar() -> None:
    dict_data = {
        "ngroups": 20,
        "largest": False,
        "nsubgroups": 10,
        "top_words": 50,
        "stopword": True,
        "lemmatize": False,
        "stem": False,
        "sort_by": "alphabet",
        "sort_ascending": False,
    }
    barchart = BarChart()
    barchart = barchart.from_dict(dict_data)
    for key in dict_data.keys():
        assert barchart.__dict__[key] == dict_data[key]


def test_word() -> None:
    dict_data = {"top_words": 20, "stopword": False, "lemmatize": False, "stem": False}
    wordcloud = WordCloud()
    wordcloud = wordcloud.from_dict(dict_data)
    for key in dict_data.keys():
        assert wordcloud.__dict__[key] == dict_data[key]
