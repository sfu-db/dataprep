"""
Initialize dictionary of categorical imputers.
"""

from .constant_imputer import ConstantImputer
from .most_frequent_imputer import MostFrequentImputer
from .drop_imputer import DropImputer

operator_dic = {
    "constant": ConstantImputer,
    "most_frequent": MostFrequentImputer,
    "drop": DropImputer,
}
