"""
Implement categorical imputer component.
"""

from typing import Any, Tuple, Dict
import dask.dataframe as dd

from .cat_imputation import operator_dic


class CatImputer:
    """Categorical imputer for imputing missing values in categorical columns
    Attributes:
        impute_type
            Name of categorical imputer
        imputer
            Imputer
        null_values
            Specified null values which should be recognized
        fill_value
            Value used for imputing missing values
    """

    def __init__(self, cat_pipe_info: Dict[str, Any]) -> None:
        """
        This function initiate categorical imputer.

        Parameters
        ----------
        cat_pipe_info
            Information of pipeline managing categorical columns,
            including the arrangement of components, name of operators
            and other information should be provided, such as filling value for imputation.
        """

        if isinstance(cat_pipe_info["cat_imputation"], str):
            impute_type = cat_pipe_info["cat_imputation"]
            fill_value = cat_pipe_info["fill_val"]
            self.null_values = cat_pipe_info["cat_null_value"]
            if len(impute_type) == 0:
                self.impute_type = "constant"
                self.fill_value = "Missing"
            else:
                self.impute_type = impute_type
                self.fill_value = fill_value
            self.imputer: Any = operator_dic[self.impute_type](self.null_values, self.fill_value)
        # elif isinstance(cat_pipe_info['cat_imputation'], object):
        else:
            self.imputer = cat_pipe_info["cat_imputation"]()

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
