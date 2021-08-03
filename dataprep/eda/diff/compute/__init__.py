"""Computations for plot_diff([df...])."""

from typing import Optional, Union, List, Dict, Any
import dask.dataframe as dd
import pandas as pd
from ....errors import DataprepError
from ...intermediate import Intermediate
from ...utils import to_dask
from ...dtypes import DTypeDef
from ...configs import Config
from .multiple_df import compare_multiple_df  # type: ignore
from .multiple_column import compare_multiple_col  # type: ignore

__all__ = ["compute_diff"]


def compute_diff(
    df: Union[List[Union[pd.DataFrame, dd.DataFrame]], Union[pd.DataFrame, dd.DataFrame]],
    x: Optional[str] = None,
    *,
    cfg: Union[Config, Dict[str, Any], None] = None,
    display: Optional[List[str]] = None,
    dtype: Optional[DTypeDef] = None,
) -> Intermediate:
    """
    All in one compute function.

    Parameters
    ----------
    df
        DataFrame from which visualizations are generated
    cfg: Union[Config, Dict[str, Any], None], default None
        When a user call plot(), the created Config object will be passed to compute().
        When a user call compute() directly, if he/she wants to customize the output,
        cfg is a dictionary for configuring. If not, cfg is None and
        default values will be used for parameters.
    display: Optional[List[str]], default None
        A list containing the names of the visualizations to display. Only exist when
        a user call compute() directly and want to customize the output
    x: Optional[str], default None
        A valid column name from the dataframe
    dtype: str or DType or dict of str or dict of DType, default None
        Specify Data Types for designated column or all columns.
        E.g.  dtype = {"a": Continuous, "b": "Nominal"} or
        dtype = {"a": Continuous(), "b": "nominal"}
        or dtype = Continuous() or dtype = "Continuous" or dtype = Continuous()
    """
    # pylint:disable = too-many-branches
    if isinstance(cfg, dict):
        cfg = Config.from_dict(display, cfg)
    elif not cfg:
        cfg = Config()

    if isinstance(df, list):

        if len(df) < 2:
            raise DataprepError("plot_diff needs at least 2 DataFrames.")
        if len(df) > 5:
            raise DataprepError("Too many DataFrames, max: 5.")
        label = cfg.diff.label
        if not label:
            cfg.diff.label = [f"df{i+1}" for i in range(len(df))]
        elif len(df) != len(label):
            raise ValueError("Number of the given label doesn't match the number of DataFrames.")

        if cfg.diff.baseline > len(df) - 1:
            raise ValueError("Baseline is out of the boundary of the input.")

        df_list = list(map(to_dask, df))
        for i, _ in enumerate(df_list):
            df_list[i].columns = df_list[i].columns.astype(str)

        if x:
            if [col for dfs in df for col in dfs.columns].count(x) < 2:
                raise DataprepError("x must exist in at least two DataFrames")
            # return compare_multiple_on_column(df_list, x)
            return compare_multiple_col(df_list, x, cfg)  # type: ignore
        else:
            return compare_multiple_df(df_list, cfg, dtype)  # type: ignore

    else:
        raise TypeError(f"Invalid input type: {type(df)}")
