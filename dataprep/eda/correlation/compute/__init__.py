"""This module implements the intermediates computation
for plot_correlation(df) function."""

from typing import Optional, Tuple, List, Dict, Union, Any
from warnings import catch_warnings, filterwarnings

from ...configs import Config
from ...intermediate import Intermediate
from ...eda_frame import EDAFrame, DataFrame
from .bivariate import _calc_bivariate
from .overview import _calc_overview
from .univariate import _calc_univariate

__all__ = ["compute_correlation"]


def compute_correlation(
    df: DataFrame,
    col1: Optional[str] = None,
    col2: Optional[str] = None,
    *,
    cfg: Union[Config, Dict[str, Any], None] = None,
    display: Optional[List[str]] = None,
    value_range: Optional[Tuple[float, float]] = None,
    k: Optional[int] = None,
) -> Intermediate:
    # pylint: disable=too-many-arguments
    """
    Parameters
    ----------
    df
        The pandas dataframe for which plots are calculated for each column.
    cfg
        Config instance
    col1
        A valid column name of the dataframe
    col2
        A valid column name of the dataframe
    value_range
        If the correlation value is out of the range, don't show it.
    cfg: Union[Config, Dict[str, Any], None], default None
        When a user call plot_correlation(), the created Config object will be passed to
        compute_correlation().
        When a user call compute_correlation() directly, if he/she wants to customize the output,
        cfg is a dictionary for configuring. If not, cfg is None and
        default values will be used for parameters.
    display: Optional[List[str]], default None
        A list containing the names of the visualizations to display. Only exist when
        a user call compute_correlation() directly and want to customize the output
    k
        Choose top-k element
    """
    if isinstance(cfg, dict):
        cfg = Config.from_dict(display, cfg)
    elif not cfg:
        cfg = Config()

    x, y = col1, col2
    frame = EDAFrame(df)
    if x is None and y is None:  # pylint: disable=no-else-return
        with catch_warnings():
            filterwarnings(
                "ignore",
                "overflow encountered in long_scalars",
                category=RuntimeWarning,
            )
            return _calc_overview(frame, cfg, value_range=value_range, k=k)
    elif x is not None and y is None:
        with catch_warnings():
            filterwarnings(
                "ignore",
                "overflow encountered in long_scalars",
                category=RuntimeWarning,
            )
            return _calc_univariate(frame, x, cfg, value_range=value_range, k=k)
    elif x is None and y is not None:
        raise ValueError("Please give the column name to x instead of y")
    elif x is not None and y is not None:
        return _calc_bivariate(frame, cfg, x, y, k=k)

    raise ValueError("Not Possible")
