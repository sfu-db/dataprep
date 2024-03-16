"""
Implement categorical encoder component.
"""

from typing import Any, Tuple, Dict
import dask.dataframe as dd

from .cat_encoding import operator_dic


class CatEncoder:
    """Categorical encoder for encoding categorical columns
    Attributes:
        encode_type
            Name of categorical encoder
        encoder
            Encoder object
    """

    def __init__(self, cat_pipe_info: Dict[str, Any]) -> None:
        """
        This function initiate categorical encoder.

        Parameters
        ----------
        cat_pipe_info
            Information of pipeline managing categorical columns,
            including the arrangement of components, name of operators
            and other information should be provided, such as filling value for imputation.
        """

        if isinstance(cat_pipe_info["cat_encoding"], str):
            encode_type = cat_pipe_info["cat_encoding"]
            self.encode_type = encode_type
            self.encoder = operator_dic[self.encode_type]()
        # elif isinstance(cat_pipe_info['cat_encoding'], object):
        else:
            self.encoder = cat_pipe_info["cat_encoding"]()

    def fit(self, col_df: dd.Series) -> Any:
        """
        Fit the parameters for encoder according to the provided column.

        Parameters
        ----------
        col_df
            Provided data column.
        """

        self.encoder.fit(col_df)
        return self

    def transform(self, col_df: dd.Series) -> dd.Series:
        """
        Transform the provided data column with the fitted parameters.

        Parameters
        ----------
        col_df
            Provided data column.
        """
        return self.encoder.transform(col_df)

    def fit_transform(
        self, training_df: dd.Series, test_df: dd.Series
    ) -> Tuple[dd.Series, dd.Series]:
        """
        Fit the parameters for encoder according to the training data column.
        Transform training data column and test data column with fitted parameters.

        Parameters
        ----------
        training_df
            Training data column.
        test_df
            Test data column.
        """
        self.encoder.fit(training_df)
        return self.encoder.transform(training_df), self.encoder.transform(test_df)
