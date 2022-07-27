#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import (
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    Sequence,
    ValuesView,
)
from itertools import chain
from typing import Generic, Optional, TypeVar, overload, TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsKeysAndGetItem

T = TypeVar("T", covariant=True)
S = TypeVar("S")


class Results(Generic[T]):
    _tokens: Sequence[T | Results[T]]
    _names: Mapping[str, Sequence[T | Results[T]]]

    __match_args__ = ("_tokens", "_names")

    def __init__(
        self,
        tokens: Iterable[T | Results[T]] = (),
        names: SupportsKeysAndGetItem[str, Sequence[T | Results[T]]]
        | Iterable[tuple[str, Sequence[T | Results[T]]]] = (),
    ):
        self._tokens = tuple(tokens)
        self._names = dict(names)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._tokens!r}, {dict.__repr__(self._names)})"

    def __add__(self, other: S | Results[S] | Sequence[S | Results[S]] | Mapping[str, Sequence[S | Results[S]]]) -> Results[S | T]:
        match other:
            case Results(tokens, names):
                return Results(
                    chain(self._tokens, tokens),
                    ((k, [*self._names.get(k, ()), *names.get(k, ())])
                        for k in self._names.keys() | names.keys())
                )
            case Mapping():
                ret = Results(
                    self._tokens,
                    ((k, [*self._names.get(k, ()), *other.get(k, ())])
                        for k in self._names.keys() | other.keys())
                )
            case Sequence() if not isinstance(other, str):
                ret = Results(chain(self._tokens, other), self._names)
            case _:
                return NotImplemented
        return ret

    # def __iadd__(self, other: Results[S]) -> Results[S | T]:
    #     if isinstance(other, Results):
    #         self.extend(other._tokens)
    #         self.update(other._names)
    #     elif isinstance(other, list):
    #         self.extend(other)
    #     elif isinstance(other, dict):
    #         self.update(other)
    #     else:
    #         return NotImplemented
    #     return self

    def __len__(self) -> int:
        return len(self._tokens)

    def __contains__(self, key: str) -> bool:
        return key in self._names

    def __iter__(self) -> Iterator[T | Results[T]]:
        # Note: we can use list(res) instead of having to provide res.as_list()
        # if this method iterates over what we should see in a list view of the
        # results.
        return iter(self._tokens)

    @overload
    def __getitem__(self, key: int) -> T | Results[T]:
        ...

    @overload
    def __getitem__(self, key: str | slice) -> Sequence[T | Results[T]]:
        ...

    def __getitem__(
        self, key: int | str | slice
    ) -> T | Results[T] | Sequence[T | Results[T]]:
        # Note: we can use dict(res) instead of having to provide res.as_dict()
        # if this method (along with keys()) returns what we should see in a
        # dictionary view of the results.
        match key:
            case int() | slice():
                return self._tokens[key]
            case str():
                if key not in self._names:
                    raise KeyError(key)
                return self._names[key]
            case _:
                raise TypeError("Results index must be int, slice, or str")

    # @overload
    # def __setitem__(self, key: int, value: T | Results[T]) -> None: ...

    # @overload
    # def __setitem__(self, key: slice, value: Iterable[T | Results[T]]) -> None: ...

    # @overload
    # def __setitem__(self, key: str, value: Iterable[T | Results[T]]) -> None: ...

    # def __setitem__(self, key: int | slice | str, value: T | Results[T] | Iterable[T | Results[T]]) -> None:
    #     if isinstance(key, int):
    #         self._tokens[key] = cast(T | Results[T], value)
    #     elif isinstance(key, slice):
    #         self._tokens[key] = cast(Iterable[T | Results[T]], value)
    #     elif isinstance(key, str):
    #         self._names[key] = list(cast(Iterable[T | Results[T]], value))
    #     else:
    #         raise TypeError("Results index must be int, slice, or str")

    def keys(self) -> KeysView[str]:
        return self._names.keys()

    def items(self) -> ItemsView[str, Sequence[T | Results[T]]]:
        return self._names.items()

    def values(self) -> ValuesView[Sequence[T | Results[T]]]:
        return self._names.values()

    def flattened(self) -> Iterator[T]:
        for t in self._tokens:
            match t:
                case Results():
                    yield from t.flattened()
                case _:
                    yield t

    # def append(self, value: T | Results[T], name: Optional[str] = None) -> None:
    #     self._tokens.append(value)
    #     if name is not None:
    #         self._names[name].append(value)

    # def extend(self, tokens: Iterable[T | Results[T]]) -> None:
    #     self._tokens += tokens

    # def update(self, names: Mapping[str, Sequence[T | Results[T]]]) -> None:
    #     for k, v in names.items():
    #         self._names[k].extend(v)

    # There is a meaningful distinction between "empty list" and "not present".
    # An empty list means the element was reached and matched, but contained no
    # tokens.  A simple example:
    #
    #     >>> p = Opt(Suppress("x")("foo"))
    #     >>> res = p.parse(text)
    #     >>> "foo" in p.parse("x")
    #     True
    #     >>> "foo" in p.parse("")
    #     False
    #

    def get(self, key: str, default: Optional[S] = None) -> Sequence[T | Results[T]] | Optional[S]:
        return self._names.get(key, default)

    @overload
    def first(self, key: str, default: None = ...) -> T | Results[T] | None: ...

    @overload
    def first(self, key: str, default: S = ...) -> T | Results[T] | S: ...

    def first(self, key: str, default: Optional[S] = None) -> T | Results[T] | Optional[S]:
        tokens = self._names.get(key)
        if not tokens:
            return default
        return tokens[0]

    @overload
    def last(self, key: str, default: None = ...) -> T | Results[T] | None: ...

    @overload
    def last(self, key: str, default: S = ...) -> T | Results[T] | S: ...

    def last(self, key: str, default: Optional[S] = None) -> T | Results[T] | Optional[S]:
        tokens = self._names.get(key)
        if not tokens:
            return default
        return tokens[-1]
