"""This module implements load dataset related functions"""

import os
from os.path import dirname
from typing import List

import pandas as pd
import dask.dataframe as dd
from sqlalchemy.engine.base import Engine
from sqlalchemy import create_engine


def get_dataset_names() -> List[str]:
    """
    Get all available dataset names. It is all csv file names in 'data' folder.

    Returns
    -------
    datasets: list
        A list of all available dataset names.

    """
    module_path = dirname(__file__)
    files = os.listdir(f"{module_path}/data")
    csv_files = list(filter(lambda x: x.endswith(".csv"), files))

    # remove suffix csv and get dataset names
    datasets = list(map(lambda f: os.path.splitext(f)[0], csv_files))

    return datasets


def get_db_names() -> List[str]:
    """
    Get all available database names. It is all csv file names in 'database' folder.

    Returns
    -------
    datasets: list
        A list of all available dataset names.

    """
    module_path = dirname(__file__)
    files = os.listdir(f"{module_path}/database")
    db_files = list(filter(lambda x: x.endswith(".db"), files))

    # remove suffix csv and get dataset names
    db_names = list(map(lambda f: os.path.splitext(f)[0], db_files))

    return db_names


def _get_dataset_path(name: str) -> str:
    """
    Given a dataset name, output the file path.
    """
    # Remove suffix 'csv' and transform to lower case
    lower_name = name.lower()
    if lower_name.endswith(".csv"):
        lower_name = os.path.splitext(lower_name)[0]

    if lower_name not in get_dataset_names():
        raise ValueError(
            f"Dataset {name} is not found. You may want to try get_dataset_names()"
            + " to get all available dataset names"
        )

    module_path = dirname(__file__)
    path = f"{module_path}/data/{lower_name}.csv"
    return path


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load dataset of the given name.

    Parameters
    ----------
    name: str
        Dataset name. The dataset will be loaded from 'data/{name}.csv'.

    Returns
    -------
    df : dataframe
        A dataframe of corresponding dataset.

    Examples
    --------
    Load titanic dataset:
    >>> from dataprep.datasets import load_dataset
    >>> df = load_dataset('titanic')

    Get all available dataset names:
    >>> from dataprep.datasets import get_dataset_names
    >>> get_dataset_names()
    ['iris', 'titanic', 'adult', 'house_prices_train', 'house_prices_test']
    """
    path = _get_dataset_path(name)
    df = pd.read_csv(path)
    return df


def load_db(name: str) -> Engine:
    """
    Load a database file

    Parameters
    ----------
    name: str
        Name of the database file

    Returns
    -------
    db_url : str
        SQLite url
    """
    file_name = name.lower()
    if not file_name.endswith(".db"):
        file_name += ".db"

    db_file_path = str(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "database", file_name)
    )
    db_engine = create_engine(f"sqlite:///{db_file_path}")
    return db_engine


def _load_dataset_as_dask(name: str) -> dd.DataFrame:
    """
    Return a dask dataframe from dd.read_csv. Used for testing.
    """
    path = _get_dataset_path(name)
    ddf = dd.read_csv(path)
    return ddf
