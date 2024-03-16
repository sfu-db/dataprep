"""
Conduct a set of operations that would be useful for
cleaning and standardizing a full Pandas DataFrame.
"""

# pylint: disable-msg=relative-beyond-top-level
# pylint: disable-msg=cyclic-import
# type: ignore

from typing import Any

import pandas as pd

from IPython.display import IFrame, display

from dataprep.clean.gui.clean_gui import launch


def clean_df_gui(
    df: pd.DataFrame,
) -> Any:
    """
    This function shows the GUI of clean module.

    Parameters
    ----------
    df
        A Pandas DataFrame containing the data to be cleaned.
    """
    # pylint: disable=too-many-arguments
    # pylint: disable-msg=too-many-locals
    # pylint:disable=too-many-branches
    # type: ignore

    return UserInterface(df).display()


class UserInterface:
    """
    A user interface used by clean module.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def display(self) -> None:
        """Display the GUI."""
        launch(self.df)

        path_to_local_server = "http://localhost:7680"
        display(IFrame(path_to_local_server, width=900, height=500))
