from datetime import datetime as DateTime
from datetime import timedelta as TimeDelta

import pandas as pd

from ...eda.dtypes import is_nominal, is_continuous


def test_dtypes() -> None:
    df = pd.DataFrame(data=[["a", "c", False]], columns=["S", "C", "B"])
    df["C"] = df["C"].astype("category")

    for col in df.columns:
        assert is_nominal(df[col].dtype)

    df = pd.DataFrame(
        data=[[complex(3, 1), 1, 1.1, TimeDelta(1), DateTime.now(),]],
        columns=["IM", "I", "F", "TD", "DT"],
    )

    for col in df.columns:
        assert is_continuous(df[col].dtype)
