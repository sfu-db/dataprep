"""
Initialize dictionary of numerical scalers.
"""

from .standard_scaler import StandardScaler
from .minmax_scaler import MinmaxScaler
from .maxabs_scaler import MaxAbsScaler

operator_dic = {
    "standardize": StandardScaler,
    "minmax": MinmaxScaler,
    "maxabs": MaxAbsScaler,
}
