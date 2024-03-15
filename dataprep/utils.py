"""Utility functions used by the whole library."""

from typing import Any
import webbrowser
from tempfile import NamedTemporaryFile
from IPython.core.display import display, HTML
import pandas as pd


def is_notebook() -> Any:
    """
    :return: whether it is running in jupyter notebook
    """
    try:
        # pytype: disable=import-error
        from IPython import get_ipython  # pylint: disable=import-outside-toplevel

        # pytype: enable=import-error

        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        return False
    except (NameError, ImportError):
        return False


def display_html(html_content: str) -> None:
    """Writes HTML content to a file and displays in browser."""
    if is_notebook():
        display(HTML(html_content))
    else:
        with NamedTemporaryFile(suffix=".html", mode="w", delete=False) as tmpf:
            tmpf.write(html_content)
            tmpf.flush()
            webbrowser.open_new_tab("file://" + tmpf.name)


def display_dataframe(df: pd.DataFrame) -> None:
    """Styles and displays dataframe in browser."""
    display_html(get_styled_schema(df))


def get_styled_schema(df: pd.DataFrame) -> Any:
    """Adds CSS styling to dataframe."""
    styled_df = df.style.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background", "white"),
                    ("font-weight", "bold"),
                    ("text-align", "right"),
                    ("font-family", "arial"),
                    ("font-size", "13"),
                ],
            },
            {"selector": "td", "props": [("font-family", "arial")]},
            {
                "selector": "tr:nth-of-type(odd)",
                "props": [("background", "#f5f5f5"), ("font-size", "13"), ("text-align", "right")],
            },
            {
                "selector": "tr:nth-of-type(even)",
                "props": [("background", "#white"), ("font-size", "13"), ("text-align", "right")],
            },
            {"selector": "tr:hover", "props": [("background-color", "e0f1ff")]},
        ]
    )

    return styled_df.render()
