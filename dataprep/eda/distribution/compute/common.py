"""Common used methods for plot function"""

from typing import Union, Optional, Tuple, Dict
import pandas as pd
import dask.dataframe as dd
from ...dtypes_v2 import LatLong


def _gen_latlong(df: Union[pd.DataFrame, dd.DataFrame], x: LatLong) -> dd.Series:
    """
    Merge Latlong into one new column.
    """

    # Make sure the new column name is not contained
    columns = df.columns
    name = x.lat + "_" + x.long
    i = 0
    while name in columns:
        name = f"{name}_{i}"
        i += 1

    lat_long = pd.Series(zip(df[x.lat], df[x.long]), name=name)
    return lat_long


def gen_new_df_with_used_cols(
    df: Union[pd.DataFrame, dd.DataFrame],
    x: Optional[Union[str, LatLong]] = None,
    y: Optional[Union[str, LatLong]] = None,
    z: Optional[str] = None,
) -> Tuple[Dict[Optional[Union[str, LatLong]], Optional[str]], Union[pd.DataFrame, dd.DataFrame]]:
    """
    Keep only used columns in x, y, z, and gen new Latlong columns if any x, y, z is LatLong.
    """

    # The key of new_names is the old name, and value is
    # new name for x, y, z. For non-LatLong type, it's the original name,
    # Otherwise it's the generated column name.
    new_names: Dict[Optional[Union[str, LatLong]], Optional[str]] = {}

    if x is None and y is None and z is None:
        return {}, df

    used_org_cols = set()
    for col in (x, y, z):
        if col is not None:
            if isinstance(col, LatLong):
                used_org_cols.add(col.lat)
                used_org_cols.add(col.long)
            else:
                used_org_cols.add(col)

    if isinstance(df, dd.DataFrame):
        pd_df = df[list(used_org_cols)].compute()
    else:
        pd_df = df[list(used_org_cols)]

    # new_srss is the used columns. It's dict because of avoid duplicate names
    new_srss: Dict[Optional[Union[str, LatLong]], pd.Series] = {}
    for col in (x, y, z):
        if isinstance(col, LatLong):
            lat_long = _gen_latlong(pd_df, col)
            new_names[col] = lat_long.name
            new_srss[lat_long.name] = lat_long
            new_srss[col.lat] = pd_df[col.lat]
            new_srss[col.long] = pd_df[col.long]
        else:
            new_names[col] = col
            if col is not None:
                new_srss[col] = pd_df[col]

    new_df = pd.concat(new_srss.values(), axis="columns")

    return new_names, new_df
