"""
Implement one-hot encoder.
"""

from typing import Any, List
import dask.dataframe as dd
import numpy as np


class OneHotEncoder:
    """One-hot encoder for encoding categorical values
    Attributes:
        name
            Name of encoder
        unique_list
            Unique categorical values in provided data columns
        unique_num
            Number of unique categorical values in provided data columns
    """

    def __init__(self) -> None:
        """
        This function initiate numerical scaler.
        """

        self.name = "OneHotEncoder"
        self.unique_list = np.zeros(1)
        self.unique_num = 0

    def fit(self, col_df: dd.Series) -> Any:
        """
        Extract unique categorical values for one-hot encoder according to the provided column.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        self.unique_list = col_df.drop_duplicates().values
        self.unique_num = col_df.drop_duplicates().count()
        return self

    def transform(self, col_df: dd.Series) -> dd.Series:
        """
        Transform the provided data column with the extracted unique values.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        result = col_df.map(self.compute_val)
        return result

    def fit_transform(self, col_df: dd.Series) -> dd.Series:
        """
        Extract unique categorical values for one-hot encoder according to the data column.
        Transform the data column with the extracted unique values.

        Parameters
        ----------
        col_df
            Data column.
        """

        return self.fit(col_df).transform(col_df)

    def compute_val(self, val: str) -> List[float]:
        """
        Compute one-hot encoding of provided value.

        Parameters
        ----------
        val
            Value should be transferred to one-hot encoding.
        """
        temp_result = np.zeros(len(self.unique_list))
        idx = self.unique_list.tolist().index(val)
        temp_result[idx] = 1
        result: List[float] = temp_result.tolist()
        return result
