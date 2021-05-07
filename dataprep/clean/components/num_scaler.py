"""
Implement numerical scaler component.
"""

from typing import Any, Tuple, Dict
import dask.dataframe as dd

from .num_scaling import operator_dic


class NumScaler:
    """Numerical scaler for scaling numerical values in numerical columns
    Attributes:
        scale_type
            Name of numerical scaler
        scaler
            Scaler
    """

    def __init__(self, num_pipe_info: Dict[str, Any]) -> None:
        """
        This function initiate numerical scaler.

        Parameters
        ----------
        num_pipe_info
            Information of pipeline managing numerical columns,
            including the arrangement of components, name of operators
            and other information should be provided, such as filling value for imputation.
        """

        if isinstance(num_pipe_info["num_scaling"], str):
            scale_type = num_pipe_info["num_scaling"]
            self.scale_type = scale_type
            self.scaler: Any = operator_dic[self.scale_type]()
        # elif isinstance(num_pipe_info['num_scaling'], object):
        else:
            self.scaler = num_pipe_info["num_scaling"]()

    def fit(self, col_df: dd.Series) -> Any:
        """
        Fit the parameters for scaler according to the provided column.

        Parameters
        ----------
        col_df
            Provided data column.
        """
        self.scaler.fit(col_df)
        return self

    def transform(self, col_df: dd.Series) -> dd.Series:
        """
        Transform the provided data column with the fitted parameters.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        return self.scaler.transform(col_df)

    def fit_transform(
        self, training_df: dd.Series, test_df: dd.Series
    ) -> Tuple[dd.Series, dd.Series]:
        """
        Fit the parameters for scaler according to the training data column.
        Transform training data column and test data column with fitted parameters.

        Parameters
        ----------
        training_df
            Training data column.
        test_df
            Test data column.
        """
        self.scaler.fit(training_df)
        return self.scaler.transform(training_df), self.scaler.transform(test_df)
