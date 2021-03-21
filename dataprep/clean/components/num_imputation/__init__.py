"""
Initialize dictionary of numerical imputers.
"""

from .mean_imputer import MeanImputer
from .median_imputer import MedianImputer
from .most_frequent_imputer import MostFrequentImputer
from .drop_imputer import DropImputer

operator_dic = {
    "mean": MeanImputer,
    "median": MedianImputer,
    "most_frequent": MostFrequentImputer,
    "drop": DropImputer,
}
