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
`Anaconda <https://docs.continuum.io/anaconda/>`_ distribution, run

::

    conda install -c conda-forge dataprep 

Supported Platforms
===================

We currently support the following platforms:

* Jupyter Notebook
* Google Colab
* Kaggle Notebooks
* VSCode Python Notebook
* Standalone Python

Deploy Dataprep.EDA on Hadoop Yarn
===================================

We could deploy dataprep.eda on Yarn Cluster via `Dask-Yarn <https://yarn.dask.org/en/latest/>`_


Install dask-yarn via pip, run

::

    pip install dask-yarn

Package conda environment via `conda-pack <https://conda.github.io/conda-pack/>`_

::

    conda-pack

Upload the archive to HDFS

::

    hadoop fs -put my_env.tar.gz /

Start a Yarn cluster and do some dataprep.eda work

::

    from dask_yarn import YarnCluster
    from dask.distributed import Client
    from dataprep.eda import create_report
    import pandas as pd

    # Create a cluster where each worker has two cores and 10GiB of memory
    cluster = YarnCluster(environment='hdfs://nameservice/my_env.tar.gz', n_workers=4, worker_cores=2, worker_memory='10GiB')

    # Connect to the cluster
    client = Client(cluster)

    # Do some dataprep.eda work here
    df = pd.read_csv('data.csv')
    create_report(df)

    # Close the resources
    client.close()
    cluster.shutdown()
