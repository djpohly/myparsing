#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Callable, Iterator, Iterable, Sequence
import re
from results import Results
from types import EllipsisType
from typing import ClassVar, Generic, Optional, TypeVar, Union, cast


E = TypeVar("E", bound="Element[object]")
T = TypeVar("T", covariant=True)
S = TypeVar("S")

SPACE = re.compile(r"\s*")


class ParseException(Exception):
    def __init__(self, expected: str):
        super().__init__(f"expected {expected}")
        self.expected = expected


# class Results:
#     def __init__(
#         self,
#         toklist: Optional[Iterable] = None,
#         tokdict: Optional[
#             Union[
#                 SupportsKeysAndGetItem[str, list],
#                 Iterable[tuple[str, list]],
#             ]
#         ] = None,
#     ):
#         if toklist is None:
#             toklist = []
#         if tokdict is None:
#             tokdict = []
#         self.toklist = list(toklist)
#         self.tokdict = defaultdict(list, tokdict)

#     def __repr__(self) -> str:
#         return f"{type(self).__name__}({self.toklist!r}, {dict.__repr__(self.tokdict)})"

#     def __add__(self, other: Results) -> Results:
#         if not isinstance(other, Results):
#             return NotImplemented

#         newlist = self.toklist + other.toklist
#         newdict = self.tokdict.copy()
#         for k, v in other.tokdict.items():
#             newdict[k] += v
#         return Results(newlist, newdict)

#     def __iadd__(self, other: Results) -> None:
#         if not isinstance(other, Results):
#             return NotImplemented

#         self.toklist += other.toklist
#         for k, v in other.tokdict.items():
#             self.tokdict[k] += v

#     def __iter__(self):
#         return iter(self.toklist)

#     @overload
#     def __getitem__(self, key: int) -> Any:
#         ...

#     @overload
#     def __getitem__(self, key: str) -> list:
#         ...

#     def __getitem__(self, key: Union[int, str]):
#         if isinstance(key, int):
#             return self.toklist[key]
#         if key not in self.tokdict:
#             raise KeyError(key)
#         return self.tokdict.get(key)

#     def copy(self) -> Results:
#         return Results(self.toklist, self.tokdict)


class Element(Generic[T]):
    def __add__(self, other: Element[S]) -> And[T | S]:
        return And([self, other])

    def __radd__(self, other: Element[S]) -> And[T | S]:
        return And([other, self])

    def __or__(self, other: Element[S]) -> Or[T | S]:
        return Or([self, other])

    def __ror__(self, other: Element[S]) -> Or[T | S]:
        return Or([other, self])

    def __getitem__(self, reps: EllipsisType | int | slice | tuple[int] | tuple[int | EllipsisType, int | EllipsisType]) -> Repeat[T]:
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

    def __mul__(self, other: int | tuple[int] | tuple[int, int]) -> Repeat[T]:
        return self[other]

    def __call__(self, name: str) -> Named[T]:
        if name is None:
            return self
        return Named(self, name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    # @staticmethod
    # @overload
    # def wrap_literal(element: str) -> Literal: ...

    # @staticmethod
    # @overload
    # def wrap_literal(element: E) -> E: ...

    # @staticmethod
    # @overload
    # def wrap_literal(element: None) -> None: ...

    # @staticmethod
    # def wrap_literal(element: E | str | None) -> E | Literal | None:
    #     if element is None:
    #         return None
    #     if isinstance(element, str):
    #         return Literal(element)
    #     return element

    @staticmethod
    def skip_space(s: str, loc: int) -> int:
        m = SPACE.match(s, loc)
        assert m
        return m.end()

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[T]]]:
        yield from ()

    def parse_all(
        self, s: str, loc: int = 0, partial: bool = False, skip_space: bool = True
    ) -> Iterator[Results[T]]:
        if skip_space:
            loc = Element.skip_space(s, loc)
        end = len(s)
        for res_loc, result in self.parse_at(s, loc):
            if skip_space:
                res_loc = Element.skip_space(s, res_loc)
            if partial or res_loc == end:
                yield result

    def parse(
        self, s: str, loc: int = 0, partial: bool = False, unambiguous: bool = False
    ) -> Optional[Results[T]]:
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

    def search(self, s: str, loc: int = 0) -> Iterator[tuple[int, Results[T]]]:
        for start in range(loc, len(s) + 1):
            for result in self.parse_all(s, start, partial=True):
                yield start, result


class NoMatch(Element[object]):
    pass


class Literal(Element[str]):
    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.text!r})"

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[str]]]:
        if s.startswith(self.text, loc):
            yield loc + len(self.text), Results([self.text])


def Empty() -> Element[str]:
    return Literal("")


# def Empty():
#     return And([])


# T: type of Results generated by this container
# S: type of Results generated by its subexpression
class ElementContainer(Element[T], Generic[T, S]):
    def __init__(self, expr: Element[S]):
        super().__init__()
        self.expr = expr

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.expr!r})"


class Suppress(ElementContainer[object, object]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[object]]]:
        for loc, res in self.expr.parse_at(s, loc):
            yield loc, Results()


class First(ElementContainer[T, T]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[T]]]:
        for loc, res in self.expr.parse_at(s, loc):
            yield loc, res
            return


class Named(ElementContainer[T, T]):
    def __init__(self, expr: Element[T], name: str):
        super().__init__(expr)
        self.name = name

    def __repr__(self) -> str:
        return f"{self.expr!r}({self.name!r})"

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[T]]]:
        for loc, res in self.expr.parse_at(s, loc):
            yield loc, res + {self.name: list(res)}


class Group(ElementContainer[T, T]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[T]]]:
        for loc, res in self.expr.parse_at(s, loc):
            yield loc, Results([res], cast(dict[str, Sequence[T | Results[T]]], dict(res)))


class Filter(ElementContainer[T, S]):
    def __init__(self, expr: Element[S], combine_fn: Callable[[Results[S]], Results[T]]):
        super().__init__(expr)
        self.fn = combine_fn

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[T]]]:
        for loc, res in self.expr.parse_at(s, loc):
            yield loc, self.fn(res)


def _combine_filter(res: Results[str]) -> Results[str]:
    return Results(["".join(res.flattened())])


def Combine(expr: Element[str]) -> Element[str]:
    return Filter(expr, _combine_filter)


class SkipToAny(ElementContainer[object, object]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[object]]]:
        for start in range(loc, len(s) + 1):
            if self.expr.parse_at(s, start):
                yield start, Results()


def SkipTo(expr: Element[object]) -> Element[object]:
    return First(SkipToAny(expr))


class Regex(Element[str]):
    def __init__(self, pattern: Union[re.Pattern[str], str]):
        super().__init__()
        if isinstance(pattern, str):
            self.pattern = re.compile(pattern)
        else:
            self.pattern = pattern

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.pattern.pattern!r})"

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[str]]]:
        m = self.pattern.match(s, pos=loc)
        if m:
            yield m.end(), Results([m[0]])


class RegexGroups(Regex):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[str]]]:
        m = self.pattern.match(s, pos=loc)
        if m:
            yield m.end(), Results(m.groups(), m.groupdict())


class AnyChar(Element[str]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[str]]]:
        if loc < len(s):
            yield loc + 1, Results([s[loc]])


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

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[T]]]:
        if loc > 0 and s[loc - 1] in self.keyword_chars:
            return
        for end, res in self.expr.parse_at(s, loc):
            if end < len(s) and s[end] in self.keyword_chars:
                continue
            yield end, res


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


def Keyword(text: str, keyword_chars: Optional[Iterable[str]] = None) -> Element[str]:
    return AsKeyword(Literal(text), keyword_chars)


class AssociativeOp(Element[T]):
    OPERATOR: ClassVar[str] = ""

    def __init__(self, exprs: Iterable[Element[T]]):
        super().__init__()
        # self.exprs: list[Element] = []
        # for expr in map(Element.wrap_literal, exprs):
        #     if isinstance(expr, type(self)) and self.can_inline(expr):
        #         self.exprs += expr.exprs
        #     else:
        #         self.exprs.append(expr)
        # self.exprs = list(map(Element.wrap_literal, exprs))
        self.exprs = list(exprs)

    def __repr__(self) -> str:
        if len(self.exprs) < 2:
            return f"{type(self).__name__}({self.exprs!r})"
        op = f" {type(self).OPERATOR} "
        return "(" + op.join(map(repr, self.exprs)) + ")"


class And(AssociativeOp[T]):
    OPERATOR = "+"

    def __init__(self, exprs: Iterable[Element[T]], skip_spaces: bool = True):
        self.skip_spaces = skip_spaces
        super().__init__(exprs)

    def can_inline(self, other: And[T]) -> bool:
        return other.skip_spaces == self.skip_spaces

    def parse_at_rec(self, s: str, loc: int, idx: int) -> Iterator[tuple[int, Results[T]]]:
        if idx == len(self.exprs):
            yield loc, Results()
            return
        if idx > 0 and self.skip_spaces:
            loc = Element.skip_space(s, loc)
        for first_loc, first_res in self.exprs[idx].parse_at(s, loc):
            for rest_loc, rest_res in self.parse_at_rec(s, first_loc, idx + 1):
                yield rest_loc, first_res + rest_res

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[T]]]:
        yield from self.parse_at_rec(s, loc, 0)


class Or(AssociativeOp[T]):
    OPERATOR = "|"

    def can_inline(self, other: Or[T]) -> bool:
        return True

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[T]]]:
        for expr in self.exprs:
            yield from expr.parse_at(s, loc)


# def NoMatch():
#     return Or([])


def MatchFirst(exprs: Iterable[Element[T]]) -> Element[T]:
    return First(Or(exprs))


class Repeat(ElementContainer[T, T]):
    def __init__(
        self,
        expr: Element[T],
        lbound: int | EllipsisType = 0,
        ubound: int | EllipsisType = 0,
        stop_on: Optional[Element[object]] = None,
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
    ) -> Iterator[tuple[int, Results[T]]]:
        if reps > 0 and self.skip_spaces:
            loc = Element.skip_space(s, loc)
        if not self.greedy and reps >= self.lbound:
            # Reluctant: yield the shorter result first
            yield loc, Results()
        if self.stop_on is None or not self.stop_on.matches(s, loc):
            if not self.ubound or reps < self.ubound:
                for first_loc, first_res in self.expr.parse_at(s, loc):
                    for rest_loc, rest_res in self.parse_at_rec(s, first_loc, reps + 1):
                        yield rest_loc, first_res + rest_res
        if self.greedy and reps >= self.lbound:
            # Greedy: yield the shorter result last
            yield loc, Results()

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[T]]]:
        yield from self.parse_at_rec(s, loc, 0)


# class Opt(Element):
#     def __init__(self, expr: ElementArg):
#         super().__init__()
#         self.expr = Element.wrap_literal(expr)

#     def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
#         yield from self.expr.parse_at(s, loc)
#         yield loc, []


def Opt(expr: Element[T]) -> Element[T]:
    return Repeat(expr, ubound=1)


def ZeroOrMore(expr: Element[T]) -> Element[T]:
    return Repeat(expr)


def OneOrMore(expr: Element[T]) -> Element[T]:
    return Repeat(expr, lbound=1)


class FollowedBy(ElementContainer[object, object]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[object]]]:
        for res in self.expr.parse_at(s, loc):
            yield loc, Results()
            return


class NotFollowedBy(ElementContainer[object, object]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[object]]]:
        for res in self.expr.parse_at(s, loc):
            return
        yield loc, Results()


class StringStart(Element[object]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[object]]]:
        if loc == 0:
            yield loc, Results()


class StringEnd(Element[object]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[object]]]:
        if loc == len(s):
            yield loc, Results()


class LineStart(Element[object]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[object]]]:
        if loc == 0 or s[loc - 1] == "\n":
            yield loc, Results()


class LineEnd(Element[object]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[object]]]:
        if loc == len(s) or s[loc] == "\n":
            yield loc, Results()


class Char(Element[str]):
    def __init__(self, chars: Iterable[str]):
        super().__init__()
        self.chars = frozenset(chars)

    def __repr__(self) -> str:
        chars_str = "".join(sorted(self.chars))
        return f"{type(self).__name__}({chars_str!r})"

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[str]]]:
        try:
            c = s[loc]
        except IndexError:
            return
        if c in self.chars:
            yield loc + 1, Results([c])


class NotChar(Element[str]):
    def __init__(self, chars: Iterable[str]):
        super().__init__()
        self.chars = frozenset(chars)

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[str]]]:
        try:
            c = s[loc]
        except IndexError:
            return
        if c not in self.chars:
            yield loc + 1, Results([c])


class PrecededBy(ElementContainer[object, object]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[object]]]:
        for start in range(loc, -1, -1):
            if self.expr.matches(s[:loc], start):
                yield loc, Results()
                return


class NotPrecededBy(ElementContainer[object, object]):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results[object]]]:
        for start in range(loc, -1, -1):
            if self.expr.matches(s[:loc], start):
                return
        yield loc, Results()


def Word(chars: Iterable[str]) -> Element[str]:
    return Combine(Repeat(Char(chars), lbound=1, skip_spaces=False))


def CharsNotIn(chars: Iterable[str]) -> Element[str]:
    return Combine(Repeat(NotChar(chars), lbound=1, skip_spaces=False))
