"""Utility functions used by the whole library."""
from typing import Any
import webbrowser
from jinja2 import PackageLoader
from tensorboard.data.experimental.experiment_from_dev import pandas

loader = PackageLoader("dataprep", "assets/tempfile")


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
    
def display_html(html_content : str) -> None:
    """Writes HTML content to a file and displays in browser."""
    if is_notebook():
        from IPython.core.display import display, HTML
        display(HTML(html_content))
    else:
        file_path = (loader.provider.module_path + '/' + loader.package_path + '/tmpf.html') 

        with open(file_path, 'w') as tmpf:
            tmpf.write(html_content)
            tmpf.flush()
            webbrowser.open_new_tab("file://" + tmpf.name)
            
def display_dataframe(df: pandas.DataFrame) -> None:
    """Styles and displays dataframe in browser."""
    display_html(get_styled_schema(df))

def get_styled_schema(df: pandas.DataFrame) -> str:
    """Adds CSS styling to dataframe."""
    styled_df = df.style.set_table_styles(
    [{'selector': 'th',
      'props': [('background', '#7CAE00'), 
                ('color', 'white'),
                ('font-family', 'verdana')]},

     {'selector': 'td',
      'props': [('font-family', 'verdana')]},

     {'selector': 'tr:nth-of-type(odd)',
      'props': [('background', '#DCDCDC')]}, 

     {'selector': 'tr:nth-of-type(even)',
      'props': [('background', 'white')]},
     
     {'selector': 'tr:hover',
      'props': [('background-color', 'yellow')]}
    ]
    ).hide_index()
    
    return styled_df.render()
