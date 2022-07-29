#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Callable, Iterator, Iterable, Sequence, Mapping
from itertools import chain
import re
from types import EllipsisType
from typing import Any, ClassVar, Generic, Optional, TypeVar, Union, cast, overload, TypeGuard


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
S = TypeVar("S")

SPACE = re.compile(r"\s*")


class ParseException(Exception):
    def __init__(self, expected: str):
        super().__init__(f"expected {expected}")
        self.expected = expected


class Element(Generic[T_co]):
    def __add__(self, other: Element[S]) -> Concat[T_co | S]:
        return Concat([self, other])

    def __or__(self, other: Element[S]) -> AnyOf[T_co | S]:
        return AnyOf([self, other])

    def __getitem__(self, reps: EllipsisType | int | slice | tuple[int] | tuple[int | EllipsisType, int | EllipsisType]) -> Repeat[T_co]:
        if reps is Ellipsis:
            return Repeat(self)
        elif isinstance(reps, int):
            return Repeat(self, reps, reps)
        elif isinstance(reps, slice) and reps.step is None:
            return Repeat(self, lbound=reps.start, ubound=reps.start, stop_on=reps.stop)
        elif isinstance(reps, tuple):
            if len(reps) == 1:
                lbound, = cast(tuple[int], reps)
                return Repeat(self, lbound=lbound)
            elif len(reps) == 2:
                lbound, ubound = cast(tuple[int, int | slice], reps)
                if isinstance(ubound, slice):
                    return Repeat(self, lbound=lbound, ubound=ubound.start, stop_on=ubound.stop)
                else:
                    return Repeat(self, lbound=lbound, ubound=ubound)
        raise TypeError("malformed Element[...] syntax")

    def __mul__(self, other: int | tuple[int] | tuple[int, int]) -> Repeat[T_co]:
        return self[other]

    def __call__(self, name: str, as_list: bool = False) -> Named[T_co]:
        if name is None:
            return self
        return Named(self, name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    @staticmethod
    def skip_space(s: str, loc: int) -> int:
        m = SPACE.match(s, loc)
        assert m
        return m.end()

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T_co]]]:
        yield from ()

    def parse_all(
        self, s: str, loc: int = 0, partial: bool = False, skip_space: bool = True
    ) -> Iterator[Sequence[T_co]]:
        if skip_space:
            loc = Element.skip_space(s, loc)
        end = len(s)
        for res_loc, result in self.parse_at(s, loc):
            if skip_space:
                res_loc = Element.skip_space(s, res_loc)
            if partial or res_loc == end:
                yield list(result)

    def parse(
        self, s: str, loc: int = 0, partial: bool = False, unambiguous: bool = False
    ) -> Optional[Sequence[T_co]]:
        it = self.parse_all(s, loc, partial=partial)
        try:
            result = next(it)
        except StopIteration:
            return None
        if unambiguous:
            # Make sure there isn't a second way to parse
            try:
                next(it)
            except StopIteration:
                pass
            else:
                raise ValueError("ambiguous parse")
        return result

    def matches(self, s: str, loc: int = 0) -> bool:
        return self.parse(s, loc) is not None

    def search(self, s: str, loc: int = 0) -> Iterator[tuple[int, Sequence[T_co]]]:
        for start in range(loc, len(s) + 1):
            for result in self.parse_all(s, start, partial=True):
                yield start, result


# def NoMatch():
#     return AnyOf([])

class NoMatch(Element[Any]):
    pass


class Literal(Element[str]):
    __match_args__ = ("text",)

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.text!r})"

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[str]]]:
        if s.startswith(self.text, loc):
            yield loc + len(self.text), (self.text,)


class RegexBase(Element[T]):
    __match_args__ = ("pattern_text",)

    def __init__(self, pattern: Union[re.Pattern[str], str]):
        super().__init__()
        if isinstance(pattern, str):
            self.pattern_text = pattern
            self.pattern = re.compile(pattern)
        else:
            self.pattern_text = pattern.pattern
            self.pattern = pattern

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.pattern_text!r})"


class Regex(RegexBase[str]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[str]]]:
        m = self.pattern.match(s, pos=loc)
        if m:
            yield m.end(), (m[0],)


class RegexGroupList(RegexBase[str]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[str]]]:
        m = self.pattern.match(s, pos=loc)
        if m:
            yield m.end(), m.groups()


class RegexGroupDict(RegexBase[Mapping[str, str]]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[Mapping[str, str]]]]:
        m = self.pattern.match(s, pos=loc)
        if m:
            yield m.end(), (m.groupdict(),)


class AnyChar(Element[str]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[str]]]:
        if loc < len(s):
            yield loc + 1, (s[loc],)


class AssociativeOp(Element[T]):
    OPERATOR: ClassVar[str] = ""

    def __init__(self, exprs: Iterable[Element[T]]):
        super().__init__()
        self.exprs = list(self.flatten_exprs(exprs))

    def __repr__(self) -> str:
        if len(self.exprs) < 2:
            return f"{type(self).__name__}({self.exprs!r})"
        op = f" {type(self).OPERATOR} "
        return "(" + op.join(map(repr, self.exprs)) + ")"

    @classmethod
    def flatten_exprs(cls, exprs: Iterable[Element[T]]) -> Iterator[Element[T]]:
        for expr in exprs:
            if isinstance(expr, cls):
                yield from cls.flatten_exprs(expr.exprs)
            else:
                yield expr


class Concat(AssociativeOp[T]):
    OPERATOR = "+"

    def __init__(self, exprs: Iterable[Element[T]]):
        super().__init__(exprs)

    def parse_at_rec(self, s: str, loc: int, idx: int) -> Iterator[tuple[int, Iterable[T]]]:
        if idx == len(self.exprs):
            yield loc, []
            return
        expr: Element[T] | Concat[T] = self.exprs[idx]
        for first_loc, first_res in expr.parse_at(s, loc):
            for rest_loc, rest_res in self.parse_at_rec(s, first_loc, idx + 1):
                yield rest_loc, chain(first_res, rest_res)

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        yield from self.parse_at_rec(s, loc, 0)


class AnyOf(AssociativeOp[T]):
    OPERATOR = "|"

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        for expr in self.exprs:
            yield from expr.parse_at(s, loc)


class StringStart(Element[T]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        if loc == 0:
            yield loc, ()


class StringEnd(Element[T]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        if loc == len(s):
            yield loc, ()


class LineStart(Element[T]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        if loc == 0 or s[loc - 1] == "\n":
            yield loc, ()


class LineEnd(Element[T]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        if loc == len(s) or s[loc] == "\n":
            yield loc, ()


class Char(Element[str]):
    def __init__(self, chars: Iterable[str]):
        super().__init__()
        self.chars = frozenset(chars)

    def __repr__(self) -> str:
        chars_str = "".join(sorted(self.chars))
        return f"{type(self).__name__}({chars_str!r})"

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[str]]]:
        try:
            c = s[loc]
        except IndexError:
            return
        if c in self.chars:
            yield loc + 1, (c,)


class NotChar(Element[str]):
    def __init__(self, chars: Iterable[str]):
        super().__init__()
        self.chars = frozenset(chars)

    def __repr__(self) -> str:
        chars_str = "".join(sorted(self.chars))
        return f"{type(self).__name__}({chars_str!r})"

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[str]]]:
        try:
            c = s[loc]
        except IndexError:
            return
        if c not in self.chars:
            yield loc + 1, (c,)


def needs_wrap(obj: ElementContainer[T, Any], expr: Any) -> TypeGuard[ElementContainer[T, str]]:
    return isinstance(expr, str)

def wraps_str(arg: ElementContainer[T, Any]) -> TypeGuard[ElementContainer[T, str]]:
    return True


# T: type of Results produced by the container
# S: type of Results consumed from its subexpression
class ElementContainer(Element[T], Generic[T, S]):
    expr: Element[S]

    @overload
    def __init__(self: ElementContainer[Any, str], expr: str): ...

    @overload
    def __init__(self, expr: Element[S] | str): ...

    def __init__(self, expr: Element[S] | str):
        super().__init__()
        if isinstance(expr, str):
            self_ = cast(ElementContainer[T, str], self)
            self_.expr = Literal(expr)
        else:
            self.expr = expr

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.expr!r})"


class Named(ElementContainer[Mapping[str, Sequence[T]], T]):
    @overload
    def __init__(self: Named[str], expr: str, name: str): ...

    @overload
    def __init__(self, expr: Element[T], name: str): ...

    def __init__(self, expr: Element[T] | str, name: str):
        super().__init__(expr)
        self.name = name

    def __repr__(self) -> str:
        return f"{self.expr!r}({self.name!r})"

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[Mapping[str, Sequence[T]]]]]:
        for loc, res in self.expr.parse_at(s, loc):
            yield loc, ({self.name: list(res)},)


class Group(ElementContainer[Sequence[T], T]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[Sequence[T]]]]:
        for loc, res in self.expr.parse_at(s, loc):
            yield loc, (list(res),)


class Suppress(ElementContainer[T, Any]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        for loc, res in self.expr.parse_at(s, loc):
            yield loc, ()


class First(ElementContainer[T, T]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        for loc, res in self.expr.parse_at(s, loc):
            yield loc, res
            return


class Filter(ElementContainer[T, S]):
    def __init__(self, expr: Element[S], combine_fn: Callable[[Iterable[S]], T]):
        super().__init__(expr)
        self.fn = combine_fn

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        for loc, res in self.expr.parse_at(s, loc):
            yield loc, (self.fn(res),)


class SkipToAny(ElementContainer[T, Any]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        for start in range(loc, len(s) + 1):
            if self.expr.parse_at(s, start):
                yield start, ()


class AsKeyword(ElementContainer[T, T]):
    DEFAULT_KEYWORD_CHARS = frozenset(
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
    )

    def __init__(self, expr: Element[T], keyword_chars: Optional[Iterable[str]] = None):
        super().__init__(expr)
        if keyword_chars is None:
            self.keyword_chars = type(self).DEFAULT_KEYWORD_CHARS
        else:
            self.keyword_chars = frozenset(keyword_chars)

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        if loc > 0 and s[loc - 1] in self.keyword_chars:
            return
        for end, res in self.expr.parse_at(s, loc):
            if end < len(s) and s[end] in self.keyword_chars:
                continue
            yield end, res


class Longest(ElementContainer[T, T]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        end, res = max(self.expr.parse_at(s, loc), key=lambda t: t[0], default=(loc, None))
        if res is not None:
            yield end, res


class Repeat(ElementContainer[T, T]):
    def __init__(
        self,
        expr: Element[T],
        lbound: int | EllipsisType = 0,
        ubound: int | EllipsisType = 0,
        stop_on: Optional[Element[Any]] = None,
        skip_spaces: bool = True,
        greedy: bool = True,
    ):
        super().__init__(expr)
        self.lbound = 0 if isinstance(lbound, EllipsisType) else lbound
        self.ubound = 0 if isinstance(ubound, EllipsisType) else ubound
        self.skip_spaces = skip_spaces
        self.greedy = greedy
        self.stop_on = stop_on

    def __repr__(self) -> str:
        if self.lbound == self.ubound:
            if self.lbound == 0:
                return f"{self.expr!r}[...]"
            return f"{self.expr!r}[{self.ubound}]"
        if self.lbound == 0 and self.ubound == 1:
            return f"Opt({self.expr!r})"
        if self.ubound == 0:
            return f"{self.expr!r}[{self.lbound}, ...]"
        if self.lbound == 0:
            return f"{self.expr!r}[..., {self.ubound}]"
        return f"{self.expr!r}[{self.lbound}, {self.ubound}]"

    def parse_at_rec(
        self, s: str, loc: int, reps: int
    ) -> Iterator[tuple[int, Iterable[T]]]:
        if reps > 0 and self.skip_spaces:
            loc = Element.skip_space(s, loc)
        if not self.greedy and reps >= self.lbound:
            # Reluctant: yield the shorter result first
            yield loc, []
        if self.stop_on is None or not self.stop_on.matches(s, loc):
            if not self.ubound or reps < self.ubound:
                for first_loc, first_res in self.expr.parse_at(s, loc):
                    for rest_loc, rest_res in self.parse_at_rec(s, first_loc, reps + 1):
                        yield rest_loc, chain(first_res, rest_res)
        if self.greedy and reps >= self.lbound:
            # Greedy: yield the shorter result last
            yield loc, []

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        yield from self.parse_at_rec(s, loc, 0)


class FollowedBy(ElementContainer[T, Any]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        for res in self.expr.parse_at(s, loc):
            yield loc, ()
            return


class NotFollowedBy(ElementContainer[T, Any]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        for res in self.expr.parse_at(s, loc):
            return
        yield loc, ()


class PrecededBy(ElementContainer[T, Any]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        for start in range(loc, -1, -1):
            if self.expr.matches(s[:loc], start):
                yield loc, ()
                return


class NotPrecededBy(ElementContainer[T, Any]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Iterable[T]]]:
        for start in range(loc, -1, -1):
            if self.expr.matches(s[:loc], start):
                return
        yield loc, ()


# class Opt(Element):
#     def __init__(self, expr: ElementArg):
#         super().__init__()
#         self.expr = Literal(expr)

#     def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
#         yield from self.expr.parse_at(s, loc)
#         yield loc, []


def Opt(expr: Element[T]) -> Element[T]:
    return Repeat(expr, ubound=1)

def ZeroOrMore(expr: Element[T]) -> Element[T]:
    return Repeat(expr)

def OneOrMore(expr: Element[T]) -> Element[T]:
    return Repeat(expr, lbound=1)


# class Keyword(Element):
#     DEFAULT_KEYWORD_CHARS = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$")

#     def __init__(self, text: str, keyword_chars: Iterable[str] = DEFAULT_KEYWORD_CHARS):
#         super().__init__()
#         self.text = text
#         self.keyword_chars = frozenset(keyword_chars)

#     def __repr__(self) -> str:
#         return f"{type(self).__name__}({self.text!r})"

#     def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
#         if loc > 0 and s[loc - 1] in self.keyword_chars:
#             return
#         if not s.startswith(self.text, loc):
#             return
#         end = loc + len(self.text)
#         if end < len(s) and s[end] in self.keyword_chars:
#             return
#         yield end, Results([self.text])


def Empty() -> Element[Any]:
    return Concat([])

def Keyword(text: str, keyword_chars: Optional[Iterable[str]] = None) -> Element[str]:
    return AsKeyword(Literal(text), keyword_chars)

def SkipTo(expr: Element[Any]) -> Element[Any]:
    return First(SkipToAny(expr))

def Combine(expr: Element[str]) -> Element[str]:
    return Filter(expr, "".join)

def Word(chars: Iterable[str]) -> Element[str]:
    return Regex("[{}]+".format(re.escape("".join(chars))))

def CharsNotIn(chars: Iterable[str]) -> Element[str]:
    return Regex("[^{}]+".format(re.escape("".join(chars))))

def Whitespace() -> Element[str]:
    #return Combine(Repeat(Char(" \n\t\r")))
    return Regex(r"\s+")

def LongestOf(exprs: Iterable[Element[T]]) -> Element[T]:
    return Longest(AnyOf(exprs))

def MatchFirst(exprs: Iterable[Element[T]]) -> Element[T]:
    return First(AnyOf(exprs))
