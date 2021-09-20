"""
Implement numerical standard scaler.
"""

from typing import Any, Union
import dask.dataframe as dd


class StandardScaler:
    """Standard Scaler for scaling numerical values
    Attributes:
        name
            Name of scaler
        mean
            Mean value of provided data column
        std
            Std value of provided data column
    """

    def __init__(self) -> None:
        """
        This function initiate numerical scaler.
        """

        self.name = "standardScaler"
        self.mean = 0
        self.std = 0

    def fit(self, col_df: dd.Series) -> Any:
        """
        Extract mean value and std value for Standard Scaler according to the provided column.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        self.mean = col_df.mean()
        self.std = col_df.std()
        return self

    def transform(self, col_df: dd.Series) -> dd.Series:
        """
        Transform the provided data column with the extracted mean value and std value.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        result = col_df.map(self.compute_val)
        return result

    def fit_transform(self, col_df: dd.Series) -> dd.Series:
        """ "
        Extract mean value and std value for Standard Scaler according to the provided column.
        Transform the provided data column with the extracted mean value and std value.

        Parameters
        ----------
        col_df
            Data column.
        """

        return self.fit(col_df).transform(col_df)

    def compute_val(self, val: Union[int, float]) -> Union[int, float]:
        """
        Compute scaling value of provided value with fitted mean value and std value.

        Parameters
        ----------
        val
            Value should be scaled.
        """

        return (val - self.mean) / self.std
