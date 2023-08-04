from collections import defaultdict
from typing import Any, Tuple, List, Callable, Union, DefaultDict
import pandas as pd
import dask.dataframe as dd

Step = Union[Tuple[str, Callable[..., Any]], Callable[..., Any]]


class Pipeline:
    params: DefaultDict[str, Any]

    def __init__(self, steps: List[Step]) -> None:
        if len(steps) == 0:
            raise ValueError("empty steps")

        self.steps = steps
        self.params = defaultdict(dict)

    def set_params(self, step_name: str, **params: Any):
        self.params[step_name].update(params)

    def run(self) -> Any:
        if isinstance(self.steps[0], Callable):
            df = self.steps[0](**self.params.get(self.steps[0].__name__, {}))
        elif isinstance(self.steps[0], tuple):
            name, step = self.steps[0]
            df = step(**self.params.get(name, {}))
        elif isinstance(self.steps[0], (pd.DataFrame, dd.DataFrame)):
            df = self.steps[0]
        else:
            raise ValueError("Unknow step 0")

        for step in self.steps[1:]:
            if isinstance(step, Callable):
                df = step(df, **self.params.get(step.__name__, {}))
            elif isinstance(step, tuple):
                name, step = step
                df = step(df, **self.params.get(name, {}))
            else:
                raise ValueError("Unknow step 0")

        return df
