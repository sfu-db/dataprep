"""
Implement categorical most-frequent imputer.
"""

from typing import Any, List, Optional
import dask.dataframe as dd


class MostFrequentImputer:
    """Most frequent imputer for imputing categorical values
    Attributes:
        null_values
            Specified null values which should be recognized
        fill_value
            Value used for imputing missing values.
    """

    def __init__(self, null_values: Optional[List[Any]], fill_value: str = "") -> None:
        """
        This function initiate most frequent imputer.

        Parameters
        ----------
        null_values
            Specified null values which should be recognized
        fill_value
            Value used for imputing missing values.
        """

        self.null_values = null_values
        self.fill_value = fill_value

    def fit(self, col_df: dd.Series) -> Any:
        """
        Find the most frequent value for most frequent imputer according to the provided column.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        self.fill_value = col_df.value_counts().index[0]
        return self

    def transform(self, col_df: dd.Series) -> dd.Series:
        """
        Impute the provided data column with the most frequent categorical element.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        result = col_df.map(self.fillna)
        return result

    def fit_transform(self, col_df: dd.Series) -> dd.Series:
        """
        Get most frequent categorical element.
        Impute the data column with most frequent categorical element.

        Parameters
        ----------
        col_df
            Data column.
        """

        return self.fit(col_df).transform(col_df)

    def fillna(self, val: str) -> str:
        """
        Check if the value is in the list of null value.
        If yes, impute the data column with most frequent categorical element.
        If no, just return the value.

        Parameters
        ----------
        val
            Each value in dask's Series
        """

        if not self.null_values is None:
            if val in self.null_values:
                return self.fill_value
        return val
