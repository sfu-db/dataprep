"""
Initialize dictionary of categorical encoders.
"""

from .one_hot_encoding import OneHotEncoder

operator_dic = {
    "one_hot": OneHotEncoder,
}
