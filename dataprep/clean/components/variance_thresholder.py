"""
Implement variance thresholder component.
"""

from typing import Any, Tuple, Dict
import dask.dataframe as dd

from .variance_threshold import operator_dic


class VarianceThresholder:
    """Variance thresholder for filtering non-informative numerical columns
    Attributes:
        variance
            Variance threshold provided
        thresholder
            Thresholder
    """

    def __init__(self, num_pipe_info: Dict[str, Any]) -> None:
        """
        This function initiate variance thresholder.

        Parameters
        ----------
        num_pipe_info
            Information of pipeline managing numerical columns,
            including the arrangement of components, name of operators
            and other information should be provided, such as filling value for imputation.
        """

        if num_pipe_info["variance_threshold"]:
            self.variance = num_pipe_info["variance"]
            self.thresholder = operator_dic["variance_threshold"](self.variance)

    def fit(self, col_df: dd.Series) -> Any:
        """
        Fit the parameters for thresholder according to the provided column.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        self.thresholder.fit(col_df)
        return self

    def transform(self, col_df: dd.Series) -> dd.Series:
        """
        Transform the provided data column with the fitted parameters.

        Parameters
        ----------
        df
            Provided data column.
        """

        return self.thresholder.transform(col_df)

    def fit_transform(
        self, training_df: dd.Series, test_df: dd.Series
    ) -> Tuple[dd.Series, dd.Series]:
        """
        Fit the parameters for thresholder according to the training data column.
        Transform training data column and test data column with fitted parameters.

        Parameters
        ----------
        training_df
            Training data column.
        test_df
            Test data column.
        """
        self.thresholder.fit(training_df)
        return self.thresholder.transform(training_df), self.thresholder.transform(test_df)
