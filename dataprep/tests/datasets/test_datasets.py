"""
module for testing the functions inside datasets
"""

from ...datasets import get_dataset_names, get_db_names, load_dataset, load_db


def test_get_dataset_names() -> None:
    names = get_dataset_names()
    assert len(names) > 0


def test_get_db_names() -> None:
    names = get_db_names()
    assert len(names) > 0


def test_load_dataset() -> None:
    dataset_names = get_dataset_names()
    for name in dataset_names:
        df = load_dataset(name)
        assert len(df) > 0


def test_load_db() -> None:
    dataset_names = get_db_names()
    for name in dataset_names:
        db = load_db(name)
