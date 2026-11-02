"""Per-object caches that never hold the object they belong to."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from functools import wraps
from typing import TypeVar

_Result = TypeVar("_Result")


def memoized(method: Callable[..., _Result]) -> Callable[..., _Result]:
    """Caches ``method``'s result per object and positional arguments, in the object's own
    ``__dict__``. A ``functools.cache`` of a bound method stored on the object would hold the object
    in a reference cycle, so a dropped object would stay in memory until the cycle collector ran."""
    slot = f"_memoized_{method.__name__}"

    @wraps(method)
    def cached(owner: object, *args: Hashable) -> _Result:
        results = owner.__dict__.setdefault(slot, {})
        if args not in results:
            results[args] = method(owner, *args)
        return results[args]

    return cached


def memoized_recent(limit: int) -> Callable[[Callable[..., _Result]], Callable[..., _Result]]:
    """``memoized`` that keeps only the ``limit`` results used last. For a result that can be built
    again from what the object keeps, so what an object holds does not grow with its scope."""

    def decorate(method: Callable[..., _Result]) -> Callable[..., _Result]:
        slot = f"_memoized_{method.__name__}"

        @wraps(method)
        def cached(owner: object, *args: Hashable) -> _Result:
            results = owner.__dict__.setdefault(slot, OrderedDict())
            try:
                results.move_to_end(args)
                return results[args]
            except KeyError:
                value = method(owner, *args)
            results[args] = value
            while len(results) > limit:
                results.popitem(last=False)
            return value

        return cached

    return decorate
