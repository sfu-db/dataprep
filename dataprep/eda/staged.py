"""Decorator to make it cope with two staged computation easily."""

from typing import Any, Callable, Generator, Tuple, Union, cast

import dask

from .intermediate import Intermediate

Decoratee = Callable[..., Generator[Any, Any, Intermediate]]

Completion = Callable[[Any], Intermediate]


def staged(
    func: Decoratee,
) -> Callable[..., Union[Tuple[Any, Completion], Intermediate]]:
    """Transform a two stage computation into a result and a completion function."""

    def staged_imp(
        *args: Any, _staged: bool = False, **kwargs: Any
    ) -> Union[Tuple[Any, Completion], Intermediate]:
        gen = func(*args, **kwargs)

        def completion(computed: Any) -> Intermediate:
            try:
                gen.send(computed)
                raise RuntimeError("Computation didn't stop.")
            except StopIteration as stop:
                return cast(Intermediate, stop.value)

        if _staged:
            return next(gen), completion
        else:
            (computed,) = dask.compute(next(gen))
            return completion(computed)

    return staged_imp
