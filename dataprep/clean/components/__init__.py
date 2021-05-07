"""
Initialize component dictionary.
"""

from .cat_encoder import CatEncoder
from .cat_imputer import CatImputer
from .num_imputer import NumImputer
from .num_scaler import NumScaler
from .variance_thresholder import VarianceThresholder

component_dic = {
    "cat_encoding": CatEncoder,
    "cat_imputation": CatImputer,
    "num_imputation": NumImputer,
    "num_scaling": NumScaler,
    "variance_threshold": VarianceThresholder,
}
