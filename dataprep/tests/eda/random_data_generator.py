import string
from typing import Any, Dict, List, Optional, Tuple, Mapping, Callable, Union
import pandas as pd
import numpy as np
import pytest


def _resolve_random_state(random_state: Union[int, np.random.RandomState]) -> np.random.RandomState:
    """Return a RandomState based on Input Integer (as seed) or RandomState"""
    if isinstance(random_state, int):
        return np.random.RandomState(random_state)
    elif isinstance(random_state, np.random.RandomState):
        return random_state
    else:
        raise NotImplementedError(
            f"The random_state must be an integer or np.random.RandomState, "
            f"current type: {type(random_state)}"
        )


def _gen_random_int_series(
    size: int, low: int = -100, high: int = 100, random_state: Union[int, np.random.RandomState] = 0
) -> pd.Series:
    """Return a randonly generated int Series, where the value is in [low, high]"""
    rand = _resolve_random_state(random_state)
    arr = rand.randint(low=low, high=high, size=size)
    return pd.Series(arr)


def _gen_random_float_series(
    size: int, random_state: Union[int, np.random.RandomState] = 0
) -> pd.Series:
    """Return a randonly generated float Series, with normal distribution"""
    rand = _resolve_random_state(random_state)
    arr = rand.normal(size=size)
    return pd.Series(arr)


def _gen_random_bool_series(
    size: int, random_state: Union[int, np.random.RandomState] = 0
) -> pd.Series:
    """Return a randonly generated boolean Series"""
    rand = _resolve_random_state(random_state)
    arr = rand.choice([True, False], size=size)
    return pd.Series(arr)


def _gen_random_datatime_series(
    size: int,
    start: str = "1/1/2018",
    end: str = "1/1/2019",
    random_state: Union[int, np.random.RandomState] = 0,
) -> pd.Series:
    """Return a randonly generated datetime Series, where time in [start, end]"""
    rand = _resolve_random_state(random_state)
    population = pd.date_range(start, end)
    arr = rand.choice(population, size=size)
    return pd.Series(arr)


def _gen_random_string_series(
    size: int,
    min_len: int = 1,
    max_len: int = 100,
    random_state: Union[int, np.random.RandomState] = 0,
) -> pd.Series:
    """Return a randonly generated string Series, where string length is in [min_len, max_len]"""
    rand = _resolve_random_state(random_state)
    population = list(string.printable)
    lst = []
    for _ in range(size):
        curr_len = rand.randint(min_len, max_len)
        randstr = "".join(rand.choice(population, size=curr_len))
        lst.append(randstr)
    return pd.Series(lst)


def gen_constant_series(size: int, value: Any) -> pd.Series:
    """Return a constant pd.Series with given size and fill in given value"""
    return pd.Series(value, index=range(size))


def gen_random_series(
    size: int,
    dtype: str = "object",
    na_ratio: float = 0.0,
    str_max_len: int = 100,
    random_state: Union[int, np.random.RandomState] = 0,
) -> pd.Series:
    """
    Return a randomly generated Pandas Series.

    Parameters
    ----------
    size: int
        The size of the generated series
    dtype: string
        The type of the generated series.
        Chosen from 'int', 'float', 'boolean', 'datetime', 'string' and 'object'.
    na_ratio: float
        The ratio of NA values in the series. Should be in [0.0, 1.0]
    str_max_len: int
        The max len of random string
    seed: int
        generator seed
    """

    gen_func: Mapping[str, Callable[..., pd.Series]] = {
        "int": _gen_random_int_series,
        "float": _gen_random_float_series,
        "boolean": _gen_random_bool_series,
        "datetime": _gen_random_datatime_series,
        "string": _gen_random_string_series,
    }
    if (dtype not in gen_func) and dtype != "object":
        raise NotImplementedError(f"dtype {dtype} generator is not implemented.")

    rand = _resolve_random_state(random_state)

    # Generate non-NA series then replace some with NA.
    # This can keep the type as the original type rather than object.
    population_list = []
    for curr_type in gen_func:
        if dtype in [curr_type, "object"]:
            if curr_type != "string":
                rand_series = gen_func[curr_type](size, random_state=rand)
            else:
                rand_series = gen_func[curr_type](size, max_len=str_max_len, random_state=rand)
            population_list.append(rand_series)
    object_population = pd.concat(population_list, ignore_index=True)
    object_series = pd.Series(rand.choice(object_population, size=size))

    # Replace some values with NA.
    na_pos = object_series.sample(frac=na_ratio, random_state=rand).index
    if not na_pos.empty:
        object_series[na_pos] = np.nan
    return object_series


def gen_random_dataframe(
    nrows: int = 30,
    ncols: int = 30,
    na_ratio: float = 0.0,
    str_col_name_max_len: int = 100,
    random_state: Union[int, np.random.RandomState] = 0,
) -> pd.DataFrame:
    """
    Return a randomly generated dataframe.
    The column name, data types are both randomly generated.
    Note that if na_ratio is not 0.0, then the column type may not contain all types,
    since there is a type transform when add NA to some series, e.g., boolean.

    Parameters
    ----------
    nrows: int
        Number of rows of the generated dataframe.
    na_ratio:
        Ratio of NA values.
    str_col_name_max_len:
        max length of string column name
    ncols: int
        Number of columns of the generated dataframe.
    seed: int
        Random Seed
    """

    rand = _resolve_random_state(random_state)
    dtypes = ["int", "float", "boolean", "datetime", "string", "object"]

    # Generate random columns
    col_types = rand.choice(dtypes, size=ncols)
    series_list = {}
    for i in range(ncols):
        series = gen_random_series(nrows, dtype=col_types[i], na_ratio=na_ratio, random_state=rand)
        series_list[i] = series
    df = pd.DataFrame(series_list)

    # Generate random column names and index.
    col_names = gen_random_series(
        size=ncols,
        dtype="object",
        na_ratio=0.1,
        str_max_len=str_col_name_max_len,
        random_state=rand,
    )
    df.columns = col_names
    df.index = gen_random_series(
        df.index.shape[0], na_ratio=0.1, str_max_len=str_col_name_max_len, random_state=rand
    )
    return df


def gen_test_df() -> pd.DataFrame:
    rand = np.random.RandomState(0)
    nrows = 30
    data = {}
    data[0] = gen_random_dataframe(nrows=nrows, ncols=10, random_state=rand).reset_index(drop=True)
    data[1] = gen_random_dataframe(
        nrows=nrows, ncols=10, na_ratio=0.1, random_state=rand
    ).reset_index(drop=True)
    data[2] = pd.Series([np.nan] * nrows, name="const_na")
    data[3] = pd.Series(["s"] * nrows, name="const_str")
    data[4] = pd.Series([0] * nrows, name="const_zero")
    data[5] = pd.Series([-1] * nrows, name="const_neg")
    data[6] = pd.Series([1] * nrows, name="const_pos")
    data[7] = pd.Series([0, 1, np.nan] * (nrows // 3), name="small_distinct_miss")
    data[8] = gen_random_series(size=nrows, dtype="string", random_state=rand).rename("str_no_miss")
    data[9] = gen_random_series(size=nrows, dtype="string", na_ratio=0.1, random_state=rand).rename(
        "str_miss"
    )
    data[10] = gen_random_series(size=nrows, dtype="float", random_state=rand).rename("num_no_miss")
    data[11] = gen_random_series(size=nrows, dtype="float", na_ratio=0.1, random_state=rand).rename(
        "num_miss"
    )
    data[12] = pd.Series(["a", "b"] * (nrows // 2), name="category_no_miss", dtype="category")
    data[13] = pd.Series(["a", np.nan] * (nrows // 2), name="category_miss", dtype="category")
    df = pd.concat(data.values(), axis=1)
    df.index = gen_random_series(df.index.shape[0], na_ratio=0.1, str_max_len=100, random_state=2)
    return df


@pytest.fixture(scope="module")  # type: ignore
def random_df() -> pd.DataFrame:
    return gen_test_df()
