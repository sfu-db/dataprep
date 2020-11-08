"""This module handles displaying information on how to connect and query."""
from jinja2 import Environment, PackageLoader
from ..utils import display_html
from typing import Any, Dict

loader = PackageLoader("dataprep", "connector/assets")
ENV_LOADER = Environment(loader = loader)

def info_UI(dbname: str, tbs: Dict[str, Any]) -> None:
    """Fills out info.txt template file. Renders the template to an html file.
    
    Parameters
    ----------
    dbname
        Name of the website
    tbs
        Table containing info to be displayed.
    """
    template = ENV_LOADER.get_template('info.txt')
    
    jinja_vars = {
        'dbname': dbname,
        'tbs' : tbs
    } 
     
    html_content = template.render(jinja_vars)
    
    display_html(html_content)