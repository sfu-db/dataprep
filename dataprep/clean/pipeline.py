"""
Implement pipeline class
"""

from typing import Any, Tuple, Dict

import dask.dataframe as dd

from .components import component_dic
from .utils import NULL_VALUES


class Pipeline:
    """Pipeline for managing categorical column and numerical column.
    Attributes:
        cat_pipeline
            List of pipeline components for categorical column
        num_pipeline
            List of pipeline components for numerical column
        cat_pipe_info
            Other information of components in cat_pipeline, such as filling value for imputation.
        num_pipe_info
            Other information of components in num_pipeline, such as filling value for imputation.
        cat_pipe_with_ops
            Generated pipeline with specific operators. Using the components in cat_pipeline.
        num_pipe_with_ops
            Generated pipeline with specific operators. Using the components in num_pipeline.
    """

    def __init__(self, cat_pipe_info: Dict[str, Any], num_pipe_info: Dict[str, Any]) -> None:
        """
        This function initiate categorical pipeline and numerical pipeline.

        Parameters
        ----------
        cat_pipe_info
            Information of pipeline managing categorical columns,
            including the arrangement of components, name of operators
            and other information should be provided, such as filling value for imputation.
        num_pipe_info
            Information of pipeline managing numerical columns,
            including the arrangement of components, name of operators
            and other information should be provided, such as filling value for imputation.
        """

        self.cat_pipeline = cat_pipe_info["cat_pipeline"]
        self.num_pipeline = num_pipe_info["num_pipeline"]

        self.cat_pipe_info = cat_pipe_info
        self.num_pipe_info = num_pipe_info

        self.cat_pipe_with_ops = self.generate_cat_pipe(cat_pipe_info)
        self.num_pipe_with_ops = self.generate_num_pipe(num_pipe_info)

        self.is_num_type = True

    def fit(
        self,
        training_df: dd.Series,
        test_df: dd.Series,
    ) -> Any:
        """
        Fit the parameters for cleaning according to the training data column.

        Parameters
        ----------
        training_df
            Training data column.
        test_df
            Test data column.
        """
        self.fit_transform(training_df, test_df)
        return self

    def transform(self, col_df: dd.Series) -> dd.Series:
        """
        Transform data column with fitted parameters.

        Parameters
        ----------
        col_df
            Data column.
        """
        result = col_df
        if not self.is_num_type:
            for i in range(len(self.cat_pipe_with_ops)):
                result = self.cat_pipe_with_ops[i].transform(result)
        else:
            for i in range(len(self.num_pipe_with_ops)):
                result = self.num_pipe_with_ops[i].transform(result)
        return result

    def fit_transform(
        self, training_df: dd.Series, test_df: dd.Series
    ) -> Tuple[dd.Series, dd.Series]:
        """
        Fit the parameters for cleaning according to the training data column.
        Transform training data column and test data column with fitted parameters.

        Parameters
        ----------
        training_df
            Training data column.
        test_df
            Test data column.
        """

        temp_training_df = training_df
        temp_test_df = test_df
        self.is_num_type = True
        for _, value in temp_training_df.iteritems():
            if isinstance(value, str):
                if not value.isnumeric() and not value in NULL_VALUES:
                    self.is_num_type = False
                    break
        if not self.is_num_type:
            for i in range(len(self.cat_pipe_with_ops)):
                temp_training_df, temp_test_df = self.cat_pipe_with_ops[i].fit_transform(
                    temp_training_df, temp_test_df
                )
        else:
            for i in range(len(self.num_pipe_with_ops)):
                temp_training_df, temp_test_df = self.num_pipe_with_ops[i].fit_transform(
                    temp_training_df, temp_test_df
                )
        return temp_training_df, temp_test_df

    def generate_cat_pipe(self, cat_pipe_info: Dict[str, Any]) -> Any:
        """
        This function is used to generate categorical pipeline with specific operators.

        Parameters
        ----------
        cat_pipe_info
            Information of pipeline managing categorical columns,
            including the arrangement of components, name of operators
            and other information should be provided, such as filling value for imputation.
        """
        result = []
        for component in self.cat_pipeline:
            if cat_pipe_info[component] is None:
                continue
            result.append(component_dic[component](cat_pipe_info))
        return result

    def generate_num_pipe(self, num_pipe_info: Dict[str, Any]) -> Any:
        """
        This function is used to generate numerical pipeline with specific operators.

        Parameters
        ----------
        num_pipe_info
            Information of pipeline managing numerical columns,
            including the arrangement of components, name of operators
            and other information should be provided, such as filling value for imputation.
        """
        result = []
        for component in self.num_pipeline:
            if num_pipe_info[component] is None:
                continue
            result.append(component_dic[component](num_pipe_info))
        return result
