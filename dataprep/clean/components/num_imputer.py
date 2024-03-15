"""
Implement numerical imputer component.
"""

from typing import Any, Tuple, Dict
import dask.dataframe as dd

from .num_imputation import operator_dic


class NumImputer:
    """Numerical imputer for imputing missing values in numerical columns
    Attributes:
        impute_type
            Name of numerical imputer
        imputer
            Imputer
        null_values
            Specified null values which should be recognized
    """

    def __init__(self, num_pipe_info: Dict[str, Any]) -> None:
        """
        This function initiate numerical imputer.

        Parameters
        ----------
        num_pipe_info
            Information of pipeline managing numerical columns,
            including the arrangement of components, name of operators
            and other information should be provided, such as filling value for imputation.
        """

        self.null_values = num_pipe_info["num_null_value"]
        if isinstance(num_pipe_info["num_imputation"], str):
            impute_type = num_pipe_info["num_imputation"]
            self.impute_type = impute_type
            self.imputer: Any = operator_dic[self.impute_type](self.null_values)
        # elif isinstance(num_pipe_info['num_imputation'], object):
        else:
            self.imputer = num_pipe_info["num_imputation"]()

    def fit(self, col_df: dd.Series) -> Any:
        """
        Fit the parameters for imputer according to the provided column.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        self.imputer.fit(col_df)
        return self

    def transform(self, col_df: dd.Series) -> dd.Series:
        """
        Transform the provided data column with the fitted parameters.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        return self.imputer.transform(col_df)

    def fit_transform(
        self, training_df: dd.Series, test_df: dd.Series
    ) -> Tuple[dd.Series, dd.Series]:
        """
        Fit the parameters for imputer according to the training data column.
        Transform training data column and test data column with fitted parameters.

        Parameters
        ----------
        training_df
            Training data column.
        test_df
            Test data column.
        """
        self.imputer.fit(training_df)
        return self.imputer.transform(training_df), self.imputer.transform(test_df)
