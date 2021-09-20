"""
Implement numerical maxabs scaler.
"""

from typing import Any, Union
import dask.dataframe as dd


class MaxAbsScaler:
    """Max Absolute Value Scaler for scaling numerical values
    Attributes:
        name
            Name of scaler
        maxabs
            Max absolute value of provided data column
    """

    def __init__(self) -> None:
        """
        This function initiate numerical scaler.
        """

        self.name = "maxabsScaler"
        self.maxabs = 0

    def fit(self, col_df: dd.Series) -> Any:
        """
        Extract max absolute value for MaxAbs Scaler according to the provided column.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        self.maxabs = max(abs(col_df.drop_duplicates().values.tolist()))
        return self

    def transform(self, col_df: dd.Series) -> dd.Series:
        """
        Transform the provided data column with the extracted max absolute value.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        result = col_df.map(self.compute_val)
        return result

    def fit_transform(self, col_df: dd.Series) -> dd.Series:
        """
        Extract max absolute value for MaxAbs Scaler according to the provided column.
        Transform the provided data column with the extracted max absolute value.

        Parameters
        ----------
        col_df
            Data column.
        """

        return self.fit(col_df).transform(col_df)

    def compute_val(self, val: Union[int, float]) -> Union[int, float]:
        """
        Compute scaling value of provided value with fitted max absolute value.

        Parameters
        ----------
        val
            Value should be scaled.
        """

        return val / self.maxabs
