.. _installation:

============
Installation
============

.. toctree::
    :maxdepth: 2
    :hidden:

To install DataPrep via pip from `PyPi <https://pypi.org/project/dataprep/>`_, run

::

    pip install dataprep


To install DataPrep with Anaconda or Miniconda from the 
`Anaconda <https://docs.continuum.io/anaconda/>`_ distribution, first run

::

    conda install -c conda-forge mamba 


to install the mamba installer. Then, use mamba to install DataPrep:

::

    mamba install -c conda-forge -c sfu-db dataprep

`Mamba <https://github.com/TheSnakePit/mamba>`_ is a conda compatible installer that features fast package resolving.
Since `conda-forge` is a big repository, using `mamba` to install DataPrep is more efficient.

Supported Platforms
===================

We currently support the following platforms:

* Jupyter Notebook
* Google Colab
* Kaggle Notebooks
* VSCode Python Notebook
* Standalone Python