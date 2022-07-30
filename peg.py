#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Callable, Iterator, Iterable, Sequence, Mapping
from itertools import chain
import re
from types import EllipsisType
from typing import Any, ClassVar, Generic, Optional, TypeVar, Union, cast, overload, TYPE_CHECKING


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
S = TypeVar("S")
E = TypeVar("E", bound="Element[Any]")
Results = tuple[int, Optional[Iterable[T]]]

SPACE = re.compile(r"\s*")


class ParseException(Exception):
    def __init__(self, expected: str):
        super().__init__(f"expected {expected}")
        self.expected = expected


class Element(Generic[T_co]):
    @overload
    def __add__(self: Element[None], other: str) -> ConcatSkipSpaces[str]: ...
    @overload
    def __add__(self: Element[None], other: Element[S]) -> ConcatSkipSpaces[S]: ...
    @overload
    def __add__(self: Element[T_co], other: Element[None]) -> ConcatSkipSpaces[T_co]: ...
    @overload
    def __add__(self: Element[T_co], other: str) -> ConcatSkipSpaces[T_co | str]: ...
    @overload
    def __add__(self: Element[T_co], other: Element[S]) -> ConcatSkipSpaces[T_co | S]: ...

    def __add__(self, other: Element[S] | str) -> ConcatSkipSpaces[T_co | S | str]:
        other = cast(Element[S], Element.wrap_literal(other))
        return ConcatSkipSpaces([self, other])

    @overload
    def __or__(self: Element[None], other: str) -> MatchFirst[str]: ...
    @overload
    def __or__(self: Element[None], other: Element[S]) -> MatchFirst[S]: ...
    @overload
    def __or__(self: Element[T_co], other: Element[None]) -> MatchFirst[T_co]: ...
    @overload
    def __or__(self: Element[T_co], other: str) -> MatchFirst[T_co | str]: ...
    @overload
    def __or__(self: Element[T_co], other: Element[S]) -> MatchFirst[T_co | S]: ...

    def __or__(self, other: Element[S] | str) -> MatchFirst[T_co | S | str]:
        other = cast(Element[S], Element.wrap_literal(other))
        return MatchFirst([self, other])

    def __invert__(self) -> NotFollowedBy:
        return NotFollowedBy(self)

    def __getitem__(self, reps: EllipsisType | int | slice | tuple[int] | tuple[int | EllipsisType, int | EllipsisType]) -> RepeatSkipSpaces[T_co]:
        if reps is Ellipsis:
            return RepeatSkipSpaces(self)
        elif isinstance(reps, int):
            return RepeatSkipSpaces(self, reps, reps)
        elif isinstance(reps, slice) and reps.step is None:
            return RepeatSkipSpaces(self, lbound=reps.start, ubound=reps.start, stop_on=reps.stop)
        elif isinstance(reps, tuple):
            if len(reps) == 1:
                lbound, = cast(tuple[int], reps)
                return RepeatSkipSpaces(self, lbound=lbound)
            elif len(reps) == 2:
                lbound, ubound = cast(tuple[int, int | slice], reps)
                if isinstance(ubound, slice):
                    return RepeatSkipSpaces(self, lbound=lbound, ubound=ubound.start, stop_on=ubound.stop)
                else:
                    return RepeatSkipSpaces(self, lbound=lbound, ubound=ubound)
        raise TypeError("malformed Element[...] syntax")

    def __mul__(self, other: int | tuple[int] | tuple[int, int]) -> RepeatSkipSpaces[T_co]:
        return self[other]

    @overload
    def __call__(self: E) -> E: ...
    @overload
    def __call__(self, name: str, as_list: bool = ...) -> Named[T_co]: ...

    def __call__(self: E, name: Optional[str] = None, as_list: bool = False) -> E | Named[T_co]:
        if name is None:
            return self
        return Named(self, name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    @overload
    @staticmethod
    def wrap_literal(expr: E) -> E: ...
    @overload
    @staticmethod
    def wrap_literal(expr: str) -> Literal: ...

    @staticmethod
    def wrap_literal(expr: E | str) -> E | Literal:
        if isinstance(expr, str):
            return Literal(expr)
        return expr

    def parse_at(self, s: str, loc: int) -> Results[T_co]:
        return loc, None

    def parse(self, s: str, loc: int = 0, partial: bool = False) -> Optional[Sequence[T_co]]:
        end, res = self.parse_at(s, loc)
        if res is None or (not partial and end != len(s)):
            return None
        return list(res)

    def matches(self, s: str, loc: int = 0, partial: bool = False) -> bool:
        return self.parse(s, loc, partial) is not None

    def search(self, s: str, loc: int = 0) -> tuple[int, Optional[Sequence[T_co]]]:
        for start in range(loc, len(s) + 1):
            end, res = self.parse_at(s, start)
            if res is not None:
                return start, list(res)
        return loc, None


class NoMatch(Element[None]):
    pass


class Literal(Element[str]):
    __match_args__ = ("text",)

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.text!r})"

    def parse_at(self, s: str, loc: int) -> Results[str]:
        if not s.startswith(self.text, loc):
            return loc, None
        return loc + len(self.text), (self.text,)


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
    def parse_at(self, s: str, loc: int) -> Results[str]:
        m = self.pattern.match(s, pos=loc)
        if not m:
            return loc, None
        return m.end(), (m[0],)


class RegexGroupList(RegexBase[str]):
    def parse_at(self, s: str, loc: int) -> Results[str]:
        m = self.pattern.match(s, pos=loc)
        if not m:
            return loc, None
        return m.end(), m.groups()


class RegexGroupDict(RegexBase[Mapping[str, str]]):
    def parse_at(self, s: str, loc: int) -> Results[Mapping[str, str]]:
        m = self.pattern.match(s, pos=loc)
        if not m:
            return loc, None
        return m.end(), (m.groupdict(),)


class AnyChar(Element[str]):
    def parse_at(self, s: str, loc: int) -> Results[str]:
        if loc >= len(s):
            return loc, None
        return loc + 1, (s[loc],)


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
            # Exclude subclasses from flattening, since they may have different
            # behavior (e.g. skipping spaces)
            if type(expr) is cls:
                yield from cls.flatten_exprs(expr.exprs)
            else:
                yield expr


class Concat(AssociativeOp[T_co]):
    OPERATOR = "+"

    @overload
    def __init__(self: Concat[None], exprs: Iterable[Element[None]]): ...
    @overload
    def __init__(self: Concat[str], exprs: Iterable[str | Element[None]]): ...
    @overload
    def __init__(self: Concat[T_co], exprs: Iterable[Element[T_co] | Element[None]]): ...
    @overload
    def __init__(self: Concat[S | str], exprs: Iterable[Element[S] | Element[None] | str]): ...

    def __init__(self, exprs: Iterable[Element[Any] | str]):
        super().__init__(cast(Element[T_co], Element.wrap_literal(e)) for e in exprs)

    def parse_one(self, i: int, s: str, start: int) -> tuple[int, Optional[Iterable[T_co]]]:
        end, res = self.exprs[i].parse_at(s, start)
        if res is None:
            return start, None
        return end, res

    def parse_at(self, s: str, loc: int) -> Results[T_co]:
        end = loc
        parts = []
        for i in range(len(self.exprs)):
            end, res = self.parse_one(i, s, end)
            if res is None:
                return loc, None
            parts.append(res)
        return end, list(chain.from_iterable(parts))


class ConcatSkipSpaces(Concat[T_co]):
    if TYPE_CHECKING:
        @overload
        def __init__(self: ConcatSkipSpaces[None], exprs: Iterable[Element[None]]): ...
        @overload
        def __init__(self: ConcatSkipSpaces[str], exprs: Iterable[str | Element[None]]): ...
        @overload
        def __init__(self: ConcatSkipSpaces[T_co], exprs: Iterable[Element[T_co] | Element[None]]): ...
        @overload
        def __init__(self: ConcatSkipSpaces[S | str], exprs: Iterable[Element[S] | Element[None] | str]): ...
        def __init__(self, exprs: Iterable[Element[Any] | str]): ...

    def parse_one(self, i: int, s: str, start: int) -> tuple[int, Optional[Iterable[T_co]]]:
        start = cast(re.Match[str], SPACE.match(s, start)).end()
        end, res = self.exprs[i].parse_at(s, start)
        if res is None:
            return start, None
        return end, res


class MatchFirst(AssociativeOp[T_co]):
    OPERATOR = "|"

    @overload
    def __init__(self: MatchFirst[None], exprs: Iterable[Element[None]]): ...
    @overload
    def __init__(self: MatchFirst[str], exprs: Iterable[str | Element[None]]): ...
    @overload
    def __init__(self: MatchFirst[T_co], exprs: Iterable[Element[T_co] | Element[None]]): ...
    @overload
    def __init__(self: MatchFirst[S | str], exprs: Iterable[Element[S] | Element[None] | str]): ...

    def __init__(self, exprs: Iterable[Element[Any] | str]):
        super().__init__(cast(Element[T_co], Element.wrap_literal(e)) for e in exprs)

    def parse_at(self, s: str, loc: int) -> Results[T_co]:
        for expr in self.exprs:
            end, res = expr.parse_at(s, loc)
            if res is not None:
                return end, res
        return loc, None


class MatchLongest(AssociativeOp[T_co]):
    OPERATOR = "^"

    @overload
    def __init__(self: MatchLongest[None], exprs: Iterable[Element[None]]): ...
    @overload
    def __init__(self: MatchLongest[str], exprs: Iterable[str | Element[None]]): ...
    @overload
    def __init__(self: MatchLongest[T_co], exprs: Iterable[Element[T_co] | Element[None]]): ...
    @overload
    def __init__(self: MatchLongest[S | str], exprs: Iterable[Element[S] | Element[None] | str]): ...

    def __init__(self, exprs: Iterable[Element[Any] | str]):
        super().__init__(cast(Element[T_co], Element.wrap_literal(e)) for e in exprs)

    def parse_at(self, s: str, loc: int) -> Results[T_co]:
        return max(
            (expr.parse_at(s, loc) for expr in self.exprs),
            key=lambda t: t[0],
            default=(loc, None)
        )


class StringStart(Element[None]):
    def parse_at(self, s: str, loc: int) -> Results[None]:
        if loc == 0:
            return loc, ()
        return loc, None


class StringEnd(Element[None]):
    def parse_at(self, s: str, loc: int) -> Results[None]:
        if loc == len(s):
            return loc, ()
        return loc, None


class LineStart(Element[None]):
    def parse_at(self, s: str, loc: int) -> Results[None]:
        if loc == 0 or s[loc - 1] == "\n":
            return loc, ()
        return loc, None


class LineEnd(Element[None]):
    def parse_at(self, s: str, loc: int) -> Results[None]:
        if loc == len(s) or s[loc] == "\n":
            return loc, ()
        return loc, None


class Char(Element[str]):
    def __init__(self, chars: Iterable[str]):
        super().__init__()
        self.chars = frozenset(chars)

    def __repr__(self) -> str:
        chars_str = "".join(sorted(self.chars))
        return f"{type(self).__name__}({chars_str!r})"

    def parse_at(self, s: str, loc: int) -> Results[str]:
        try:
            c = s[loc]
        except IndexError:
            return loc, None
        if c in self.chars:
            return loc + 1, (c,)
        return loc, None


class NotChar(Element[str]):
    def __init__(self, chars: Iterable[str]):
        super().__init__()
        self.chars = frozenset(chars)

    def __repr__(self) -> str:
        chars_str = "".join(sorted(self.chars))
        return f"{type(self).__name__}({chars_str!r})"

    def parse_at(self, s: str, loc: int) -> Results[str]:
        try:
            c = s[loc]
        except IndexError:
            return loc, None
        if c not in self.chars:
            return loc + 1, (c,)
        return loc, None


# T: type of Results produced by the container
# S: type of Results consumed from its subexpression
class ElementContainer(Element[T], Generic[T, S]):
    expr: Element[S]

    @overload
    def __init__(self, expr: Element[S]): ...
    @overload
    def __init__(self: ElementContainer[T, str], expr: str): ...
    # XXX extra overload to keep mypy happy with calls to
    # super().__init__(expr) in subclasses.
    @overload
    def __init__(self, expr: Element[S] | str): ...

    def __init__(self, expr: Element[S] | str):
        super().__init__()
        self.expr = cast(Element[S], Element.wrap_literal(expr))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.expr!r})"


class Named(ElementContainer[Mapping[str, Sequence[T]], T]):
    @overload
    def __init__(self, expr: Element[T], name: str): ...
    @overload
    def __init__(self: Named[str], expr: str, name: str): ...

    def __init__(self, expr: Element[T] | str, name: str):
        super().__init__(expr)
        self.name = name

    def __repr__(self) -> str:
        return f"{self.expr!r}({self.name!r})"

    def parse_at(self, s: str, loc: int) -> Results[Mapping[str, Sequence[T]]]:
        end, res = self.expr.parse_at(s, loc)
        if res is None:
            return loc, None
        return end, ({self.name: list(res)},)


class Group(ElementContainer[Sequence[T], T]):
    if TYPE_CHECKING:
        @overload
        def __init__(self, expr: Element[T]): ...
        @overload
        def __init__(self: Group[str], expr: str): ...
        def __init__(self, expr: Element[T] | str): ...

    def parse_at(self, s: str, loc: int) -> Results[Sequence[T]]:
        end, res = self.expr.parse_at(s, loc)
        if res is None:
            return loc, None
        return end, (list(res),)


class Suppress(ElementContainer[None, Any]):
    def parse_at(self, s: str, loc: int) -> Results[None]:
        end, res = self.expr.parse_at(s, loc)
        if res is None:
            return loc, None
        return end, ()


class SkipTo(ElementContainer[None, Any]):
    def parse_at(self, s: str, loc: int) -> Results[None]:
        for start in range(loc, len(s) + 1):
            end, res = self.expr.parse_at(s, start)
            if res is not None:
                return start, ()
        return loc, None


class AsKeyword(ElementContainer[T, T]):
    DEFAULT_KEYWORD_CHARS = frozenset(
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
    )

    @overload
    def __init__(self, expr: Element[T], keyword_chars: Optional[Iterable[str]] = ...): ...
    @overload
    def __init__(self: AsKeyword[str], expr: str, keyword_chars: Optional[Iterable[str]] = ...): ...

    def __init__(self, expr: Element[T] | str, keyword_chars: Optional[Iterable[str]] = None):
        super().__init__(expr)
        if keyword_chars is None:
            self.keyword_chars = type(self).DEFAULT_KEYWORD_CHARS
        else:
            self.keyword_chars = frozenset(keyword_chars)

    def parse_at(self, s: str, loc: int) -> Results[T]:
        if loc > 0 and s[loc - 1] in self.keyword_chars:
            return loc, None
        end, res = self.expr.parse_at(s, loc)
        if res is None:
            return loc, None
        if end < len(s) and s[end] in self.keyword_chars:
            return loc, None
        return end, res


class Repeat(ElementContainer[T, T]):
    @overload
    def __init__(
        self,
        expr: Element[T],
        lbound: int | EllipsisType = 0,
        ubound: int | EllipsisType = 0,
        stop_on: Optional[Element[Any]] = None,
        greedy: bool = True,
    ): ...

    @overload
    def __init__(
        self: Repeat[str],
        expr: str,
        lbound: int | EllipsisType = 0,
        ubound: int | EllipsisType = 0,
        stop_on: Optional[Element[Any]] = None,
        greedy: bool = True,
    ): ...

    def __init__(
        self,
        expr: Element[T] | str,
        lbound: int | EllipsisType = 0,
        ubound: int | EllipsisType = 0,
        stop_on: Optional[Element[Any]] = None,
        greedy: bool = True,
    ):
        super().__init__(expr)
        self.lbound = 0 if isinstance(lbound, EllipsisType) else lbound
        self.ubound = 0 if isinstance(ubound, EllipsisType) else ubound
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

    def parse_one(self, i: int, s: str, start: int) -> tuple[int, Optional[Iterable[T]]]:
        end, res = self.expr.parse_at(s, start)
        if res is None:
            return start, None
        return end, res

    def parse_at(self, s: str, loc: int) -> Results[T]:
        # TODO: implement stop_on, see how pyparsing does it (check every
        # iteration or every character?)
        start = loc
        parts: list[Iterable[T]] = []
        count = 0
        while not self.ubound or count < self.ubound:
            end, res = self.parse_one(count, s, start)

            # If the match fails
            if res is None:
                if count < self.lbound:
                    # Not enough repetitions
                    return loc, None
                # Enough repetitions - we're done
                break

            # If the match is empty
            if end == start:
                # Fill up to lbound with repeats and claim victory
                parts += [list(res)] * (self.lbound - count)
                break

            # Keep going
            parts.append(res)
            start = end
            count += 1
        return start, chain.from_iterable(parts)


class RepeatSkipSpaces(Repeat[T]):
    if TYPE_CHECKING:
        @overload
        def __init__(self, expr: Element[T], lbound: int | EllipsisType = 0, ubound: int | EllipsisType = 0, stop_on: Optional[Element[Any]] = None, greedy: bool = True): ...
        @overload
        def __init__(self: RepeatSkipSpaces[str], expr: str, lbound: int | EllipsisType = 0, ubound: int | EllipsisType = 0, stop_on: Optional[Element[Any]] = None, greedy: bool = True): ...
        def __init__(self, expr: Element[T] | str, lbound: int | EllipsisType = 0, ubound: int | EllipsisType = 0, stop_on: Optional[Element[Any]] = None, greedy: bool = True): ...

    def parse_one(self, i: int, s: str, start: int) -> tuple[int, Optional[Iterable[T]]]:
        start = cast(re.Match[str], SPACE.match(s, start)).end()
        end, res = self.expr.parse_at(s, start)
        if res is None:
            return start, None
        return end, res


class FollowedBy(ElementContainer[None, Any]):
    def parse_at(self, s: str, loc: int) -> Results[None]:
        end, res = self.expr.parse_at(s, loc)
        if res is None:
            return loc, None
        return loc, ()


class NotFollowedBy(ElementContainer[None, Any]):
    def __repr__(self) -> str:
        return f"~{self.expr!r}"

    def parse_at(self, s: str, loc: int) -> Results[None]:
        end, res = self.expr.parse_at(s, loc)
        if res is not None:
            return loc, None
        return loc, ()


class PrecededBy(ElementContainer[None, Any]):
    def parse_at(self, s: str, loc: int) -> Results[None]:
        for start in range(loc, -1, -1):
            if self.expr.matches(s[:loc], start, partial=False):
                return loc, ()
        return loc, None


class NotPrecededBy(ElementContainer[None, Any]):
    def parse_at(self, s: str, loc: int) -> Results[None]:
        for start in range(loc, -1, -1):
            if self.expr.matches(s[:loc], start, partial=False):
                return loc, None
        return loc, ()


# The type of combine_fn here still needs some thought...
class Filter(ElementContainer[T, S]):
    @overload
    def __init__(self, expr: Element[Any], combine_fn: Callable[[Iterable[S]], T]): ...
    @overload
    def __init__(self: Filter[T, str], expr: str, combine_fn: Callable[[Iterable[str]], T]): ...

    def __init__(self, expr: Element[S] | str, combine_fn: Callable[[Iterable[S]], T] | Callable[[Iterable[str]], T]):
        super().__init__(expr)
        self.fn = cast(Callable[[Iterable[S]], Iterable[T]], combine_fn)

    def parse_at(self, s: str, loc: int) -> Results[T]:
        end, res = self.expr.parse_at(s, loc)
        if res is None:
            return loc, None
        return loc, self.fn(res)



# XXX Still not sure if there's a better way to represent this than
# Element[None].  Create a separate class called Nothing to use as the type
# variable?  "Any" and "object" infect everything.
def Empty() -> Element[None]:
    return Concat([])

# class Nothing:
#     pass

# def Empty() -> Element[Nothing]:
#     return Concat([])

def Keyword(text: str, keyword_chars: Optional[Iterable[str]] = None) -> Element[str]:
    return AsKeyword(Literal(text), keyword_chars)

def Word(chars: Iterable[str]) -> Element[str]:
    return Regex("[{}]+".format(re.escape("".join(chars))))

def CharsNotIn(chars: Iterable[str]) -> Element[str]:
    return Regex("[^{}]+".format(re.escape("".join(chars))))

def Whitespace() -> Element[str]:
    return Regex(r"\s+")

def Opt(expr: Element[T]) -> Element[T]:
    return RepeatSkipSpaces(expr, ubound=1)

def ZeroOrMore(expr: Element[T]) -> Element[T]:
    return RepeatSkipSpaces(expr)

def OneOrMore(expr: Element[T]) -> Element[T]:
    return RepeatSkipSpaces(expr, lbound=1)

def Combine(expr: Element[str]) -> Element[str]:
    return Filter(expr, "".join)
