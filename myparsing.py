#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Iterable, Mapping
from dataclasses import dataclass, field
import re
from typing import Any, ClassVar, Optional, TypeVar, Union, overload


ElementArg = Union["Element", str]
E = TypeVar("E", bound="Element")
AO = TypeVar("AO", bound="AssociativeOp")

SPACE = re.compile(r"\s*")


class ParseException(Exception):
    def __init__(self, expected: str):
        super().__init__(f"expected {expected}")
        self.expected = expected


class Results:
    def __init__(self, toklist: Optional[list] = None, tokdict: Optional[Union[Mapping[str, Union[list, Results]], Iterable[tuple[str, Union[list, Results]]]]] = None):
        if toklist is None:
            toklist = []
        if tokdict is None:
            tokdict = []
        self.toklist = toklist
        self.tokdict = defaultdict(list, tokdict)

# Union[Mapping[str, list], Iterable[tuple[str, list]]]
# Overload(
#   _typeshed.SupportsKeysAndGetItem[_KT`1, _VT`2]
#   typing.Iterable[Tuple[_KT`1, _VT`2]]
# )

    def __add__(self, other: Results) -> Results:
        if not isinstance(other, Results):
            return NotImplemented

        newlist = self.toklist + other.toklist
        newdict = self.tokdict.copy()
        for k, v in other.tokdict.items():
            newdict[k] += v
        return Results(newlist, newdict)

    def __iadd__(self, other: Results) -> None:
        if not isinstance(other, Results):
            return NotImplemented

        self.toklist += other.toklist
        for k, v in other.tokdict.items():
            self.tokdict[k] += v

    def __iter__(self):
        return iter(self.toklist)

    @overload
    def __getitem__(self, key: int) -> Any: ...

    @overload
    def __getitem__(self, key: str) -> list: ...

    def __getitem__(self, key: Union[int, str]):
        if isinstance(key, int):
            return self.toklist[key]
        if key not in self.tokdict:
            raise KeyError(key)
        return self.tokdict.get(key)


class Element:
    def __add__(self, other: ElementArg) -> And:
        return And([self, other])

    def __radd__(self, other: ElementArg) -> And:
        return And([other, self])

    def __or__(self, other: ElementArg) -> Or:
        return Or([self, other])

    def __ror__(self, other: ElementArg) -> Or:
        return Or([other, self])

    def __getitem__(self, reps) -> Repeat:
        if reps is Ellipsis:
            return Repeat(self)
        if isinstance(reps, int):
            return Repeat(self, reps, reps)
        if isinstance(reps, slice):
            if reps.step is None:
                count, stop_on = reps.start, reps.stop
                return Repeat(self, lbound=count, ubound=count, stop_on=stop_on)
        if isinstance(reps, tuple):
            if len(reps) == 1:
                return Repeat(self, lbound=reps[0])
            if len(reps) == 2:
                lbound = 0 if reps[0] is Ellipsis else reps[0]
                ubound = 0 if reps[1] is Ellipsis else reps[1]
                return Repeat(self, lbound=lbound, ubound=ubound)
        raise TypeError("malformed Element[...] syntax")

    def __mul__(self, other: Union[int, tuple]) -> Repeat:
        return self[other]

    @overload
    def __call__(self: E) -> E: ...

    @overload
    def __call__(self, name: str) -> Named: ...

    def __call__(self, name=None):
        if name is None:
            return self
        return Named(self, name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    @staticmethod
    def wrap_literal(element: ElementArg) -> Element:
        if element is None:
            return None
        if isinstance(element, str):
            return Literal(element)
        return element

    @staticmethod
    def skip_space(s: str, loc: int) -> int:
        m = SPACE.match(s, loc)
        assert m
        return m.end()

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        yield from ()

    def parse_all(self, s: str, loc: int = 0, partial: bool = False, skip_space: bool = True) -> Iterator[Results]:
        if skip_space:
            loc = Element.skip_space(s, loc)
        end = len(s)
        for res_loc, result in self.parse_at(s, loc):
            if skip_space:
                res_loc = Element.skip_space(s, res_loc)
            if partial or res_loc == end:
                yield result

    def parse(self, s: str, loc: int = 0, partial: bool = False, unambiguous: bool = False) -> Optional[Results]:
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

    def search(self, s: str, loc: int = 0) -> Iterator[tuple[int, Results]]:
        for start in range(loc, len(s) + 1):
            for result in self.parse_all(s, start, partial=True):
                yield start, result


class NoMatch(Element):
    pass


class ElementContainer(Element):
    def __init__(self, expr: ElementArg):
        super().__init__()
        self.expr = Element.wrap_literal(expr)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.expr!r})"


class Suppress(ElementContainer):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        for loc, res in self.expr.parse_at(s, loc):
            yield loc, Results()


class First(ElementContainer):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        for loc, res in self.expr.parse_at(s, loc):
            yield loc, res
            return


class Named(ElementContainer):
    def __init__(self, expr: ElementArg, name: str):
        super().__init__(expr)
        self.name = name

    def __repr__(self) -> str:
        return f"{self.expr!r}({self.name!r})"

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        for loc, res in self.expr.parse_at(s, loc):
            res.tokdict[self.name] += res.toklist
            yield loc, res


class Group(ElementContainer):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        for loc, res in self.expr.parse_at(s, loc):
            yield loc, Results([res.toklist], res.tokdict)


class NamedGroup(ElementContainer):
    def __init__(self, expr: ElementArg, name: str):
        super().__init__(expr)
        self.name = name

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        for loc, res in self.expr.parse_at(s, loc):
            yield loc, Results([res.toklist], {self.name: res})



class Filter(ElementContainer):
    def __init__(self, expr: ElementArg, combine_fn: Callable[[Results], Results]):
            super().__init__(expr)
            self.fn = combine_fn

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        for loc, res in self.expr.parse_at(s, loc):
            yield loc, self.fn(res)


def _combine_filter(res: Results):
    res.toklist = ["".join(res.toklist)]
    return res

def Combine(expr: ElementArg) -> Element:
    return Filter(expr, _combine_filter)


class SkipToAny(ElementContainer):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        for start in range(loc, len(s) + 1):
            if self.expr.parse_at(s, start):
                yield start, Results()


def SkipTo(expr: ElementArg) -> Element:
    return First(SkipToAny(expr))


class Regex(Element):
    def __init__(self, pattern: Union[re.Pattern[str], str]):
        super().__init__()
        if isinstance(pattern, str):
            self.pattern = re.compile(pattern)
        else:
            self.pattern = pattern

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.pattern.pattern!r})"

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        m = self.pattern.match(s, pos=loc)
        if m:
            yield m.end(), Results([m[0]])


class Literal(Element):
    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.text!r})"

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        if s.startswith(self.text, loc):
            yield loc + len(self.text), Results([self.text])


class AsKeyword(ElementContainer):
    DEFAULT_KEYWORD_CHARS = frozenset("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$")

    def __init__(self, expr: ElementArg, keyword_chars: Optional[Iterable[str]] = None):
        super().__init__(expr)
        if keyword_chars is None:
            self.keyword_chars = type(self).DEFAULT_KEYWORD_CHARS
        else:
            self.keyword_chars = frozenset(keyword_chars)

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
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


def Keyword(text: str, keyword_chars: Optional[Iterable[str]] = None):
    return AsKeyword(Literal(text), keyword_chars)

def Empty():
    return Literal("")

# def Empty():
#     return And([])


class AssociativeOp(Element):
    OPERATOR: ClassVar[str] = ""

    def __init__(self, exprs: Iterable[ElementArg]):
        super().__init__()
        # self.exprs: list[Element] = []
        # for expr in map(Element.wrap_literal, exprs):
        #     if isinstance(expr, type(self)) and self.can_inline(expr):
        #         self.exprs += expr.exprs
        #     else:
        #         self.exprs.append(expr)
        self.exprs = list(map(Element.wrap_literal, exprs))

    def __repr__(self) -> str:
        if len(self.exprs) < 2:
            return f"{type(self).__name__}({self.exprs!r})"
        op = f" {type(self).OPERATOR} "
        return "(" + op.join(map(repr, self.exprs)) + ")"

    def can_inline(self: AO, other: AO) -> bool:
        return False


class And(AssociativeOp):
    OPERATOR = "+"

    def __init__(self, exprs: Iterable[ElementArg], skip_spaces=True):
        self.skip_spaces = skip_spaces
        super().__init__(exprs)

    def can_inline(self, other: And) -> bool:
        return other.skip_spaces == self.skip_spaces

    def parse_at_rec(self, s: str, loc: int, idx: int) -> Iterator[tuple[int, Results]]:
        if idx == len(self.exprs):
            yield loc, Results()
            return
        if idx > 0 and self.skip_spaces:
            loc = Element.skip_space(s, loc)
        for first_loc, first_res in self.exprs[idx].parse_at(s, loc):
            for rest_loc, rest_res in self.parse_at_rec(s, first_loc, idx + 1):
                yield rest_loc, first_res + rest_res

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        yield from self.parse_at_rec(s, loc, 0)


class Or(AssociativeOp):
    OPERATOR = "|"

    def can_inline(self, other: Or) -> bool:
        return True

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        for expr in self.exprs:
            yield from expr.parse_at(s, loc)


# def NoMatch():
#     return Or([])


def MatchFirst(exprs: Iterable[ElementArg]):
    return First(Or(exprs))


class Repeat(ElementContainer):
    def __init__(self, expr: ElementArg, lbound: int = 0, ubound: int = 0, stop_on: Optional[ElementArg] = None, skip_spaces: bool = True, greedy: bool = True):
        super().__init__(expr)
        self.lbound = lbound
        self.ubound = ubound
        self.skip_spaces = skip_spaces
        self.greedy = greedy
        self.stop_on = None if stop_on is None else Element.wrap_literal(stop_on)

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

    def parse_at_rec(self, s: str, loc: int, reps: int) -> Iterator[tuple[int, Results]]:
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

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        yield from self.parse_at_rec(s, loc, 0)


# class Opt(Element):
#     def __init__(self, expr: ElementArg):
#         super().__init__()
#         self.expr = Element.wrap_literal(expr)

#     def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
#         yield from self.expr.parse_at(s, loc)
#         yield loc, []


def Opt(expr: ElementArg):
    return Repeat(expr, ubound=1)

def ZeroOrMore(expr: ElementArg):
    return Repeat(expr)

def OneOrMore(expr: ElementArg):
    return Repeat(expr, lbound=1)


class FollowedBy(ElementContainer):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        for res in self.expr.parse_at(s, loc):
            yield loc, Results()
            return


class NotFollowedBy(ElementContainer):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        for res in self.expr.parse_at(s, loc):
            return
        yield loc, Results()


class StringStart(Element):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        if loc == 0:
            yield loc, Results()


class StringEnd(Element):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        if loc == len(s):
            yield loc, Results()


class LineStart(Element):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        if loc == 0 or s[loc - 1] == "\n":
            yield loc, Results()


class LineEnd(Element):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        if loc == len(s) or s[loc] == "\n":
            yield loc, Results()


class Char(Element):
    def __init__(self, chars: Iterable[str]):
        super().__init__()
        self.chars = frozenset(chars)

    def __repr__(self) -> str:
        chars_str = "".join(sorted(self.chars))
        return f"{type(self).__name__}({chars_str!r})"

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        try:
            c = s[loc]
        except IndexError:
            return
        if c in self.chars:
            yield loc + 1, Results([c])


class NotChar(Element):
    def __init__(self, chars: Iterable[str]):
        super().__init__()
        self.chars = frozenset(chars)

    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        try:
            c = s[loc]
        except IndexError:
            return
        if c not in self.chars:
            yield loc + 1, Results([c])


class PrecededBy(ElementContainer):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        for start in range(loc, -1, -1):
            if self.expr.matches(s[:loc], start):
                yield loc, Results()
                return


class NotPrecededBy(ElementContainer):
    def parse_at(self, s: str, loc: int) -> Iterator[tuple[int, Results]]:
        for start in range(loc, -1, -1):
            if self.expr.matches(s[:loc], start):
                return
        yield loc, Results()


def Word(chars: Iterable[str]) -> Element:
    return Combine(Repeat(Char(chars), lbound=1, skip_spaces=False))

def CharsNotIn(chars: Iterable[str]) -> Element:
    return Combine(Repeat(NotChar(chars), lbound=1, skip_spaces=False))
