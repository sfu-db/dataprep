"""
Implement numerical mean imputer.
"""

from typing import Any, Union, List, Optional
import math
import dask.dataframe as dd


class MeanImputer:
    """Mean imputer for imputing numerical values
    Attributes:
        null_values
            Specified null values which should be recognized
        mean
            Mean value
    """

    def __init__(self, null_values: Optional[List[Any]]) -> None:
        """
        This function initiate mean imputer.

        Parameters
        ----------
        null_values
            Specified null values which should be recognized
        """

        self.null_values = null_values
        self.mean = 0

    def fit(self, col_df: dd.Series) -> Any:
        """
        Find the mean value for mean imputer according to the provided column.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        self.mean = col_df.mean()
        return self

    def transform(self, col_df: dd.Series) -> dd.Series:
        """
        Impute missing values with the fitted mean value.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        result = col_df.map(self.fillna)
        return result

    def fit_transform(self, col_df: dd.Series) -> dd.Series:
        """
        Extract the mean value from provided column.
        Impute with extracted mean value.

        Parameters
        ----------
        col_df
            Data column.
        """

        return self.fit(col_df).transform(col_df)

    def fillna(self, val: Union[int, float]) -> Union[int, float]:
        """
        Check if the value is in the list of null value.
        If yes, impute with extracted mean value.
        If no, just return the value.

        Parameters
        ----------
        val
            Each value in dask's Series
        """

        if isinstance(val, str) and val in self.null_values:
            return self.mean
        if math.isnan(float(val)):
            return self.mean
        return val
