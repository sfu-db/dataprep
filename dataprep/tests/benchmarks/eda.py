#%%
from functools import partial
from ...datasets import load_dataset
from ...eda import create_report


def report_func(df, **kwargs):
    create_report(df, **kwargs)


def test_create_report(benchmark):
    df = load_dataset("titanic")
    benchmark(partial(report_func), df)
