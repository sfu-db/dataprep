"""
Implement numerical median imputer.
"""

from typing import Any, Union, List, Optional
import math
import dask.dataframe as dd


class MedianImputer:
    """Median imputer for imputing numerical values
    Attributes:
        null_values
            Specified null values which should be recognized
        median
            Median value
    """

    def __init__(self, null_values: Optional[List[Any]]) -> None:
        """
        This function initiate median imputer.

        Parameters
        ----------
        null_values
            Specified null values which should be recognized
        """

        self.null_values = null_values
        self.median = 0

    def fit(self, col_df: dd.Series) -> Any:
        """
        Find the median value for median imputer according to the provided column.

        Parameters
        ----------
        col_df
            Provided data column.
        """
        self.median = col_df.values.median()
        return self

    def transform(self, col_df: dd.Series) -> dd.Series:
        """
        Impute the provided data column with the fitted median value.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        result = col_df.map(self.fillna)
        return result

    def fit_transform(self, col_df: dd.Series) -> dd.Series:
        """
        Extract the median value from provided column.
        Impute the data column with extracted median value.

        Parameters
        ----------
        col_df
            Data column.
        """

        return self.fit(col_df).transform(col_df)

    def fillna(self, val: Union[int, float]) -> Union[int, float]:
        """
        Check if the value is in the list of null value.
        If yes, impute the data column with extracted median value.
        If no, just return the value.

        Parameters
        ----------
        val
            Each value in dask's Series
        """

        if isinstance(val, str) and val in self.null_values:
            return self.median
        if math.isnan(float(val)):
            return self.median
        return val
