# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from pathlib import Path
from typing import cast

import toml

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "DataPrep"
author = "SFU Database Systems Lab"
copyright = f"©2020 {author}."

# The full version, including alpha/beta/rc tags
def get_version() -> str:
    """
    Get the library version from pyproject.toml
    """
    path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = toml.loads(open(str(path)).read())
    return cast(str, pyproject["tool"]["poetry"]["version"])


release = get_version()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "docs.source.bokeh.theme",
]

# autodoc_typehints = "description"
# # Napoleon settings
# napoleon_google_docstring = False
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = False
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True
# napoleon_use_keyword = True
# napoleon_custom_sections = None

# autodoc_default_options = {
#     "members": True,
#     "member-order": "bysource",
#     "special-members": "__init__",
# }

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_context = {
    "AUTHOR": author,
    "DESCRIPTION": "DataPrep, documentation site.",
    "SITEMAP_BASE_URL": "https://sfu-db.github.io/dataprep/",  # Trailing slash is needed
    "VERSION": release,
}

html_theme = "bokeh"

html_theme_path = ["."]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

nbsphinx_execute = "auto"
nbsphinx_allow_errors = False  # exception cells should be run apriori

nbsphinx_execute_arguments = []

if "DATAPREP_DOC_KERNEL" in os.environ:
    nbsphinx_kernel_name = os.environ["DATAPREP_DOC_KERNEL"]
