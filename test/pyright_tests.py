from myparsing import *
from typing import Any, Sequence


# String auto-wrapping
reveal_type(Named("hello", "foo"), expected_type=Named[str])
reveal_type(Group("hello"), expected_type=Group[str])
reveal_type(Suppress("hello"), expected_type=Suppress)
reveal_type(First("hello"), expected_type=First[str])
reveal_type(SkipToAny("hello"), expected_type=SkipToAny)
reveal_type(AsKeyword("hello"), expected_type=AsKeyword[str])
reveal_type(Longest("hello"), expected_type=Longest[str])
reveal_type(Repeat("hello"), expected_type=Repeat[str])
reveal_type(RepeatSkipSpaces("hello"), expected_type=RepeatSkipSpaces[str])
def list_len_sum(xs: Iterable[str]):
    return [sum(map(len, xs))]
reveal_type(Filter("hello", list_len_sum), expected_type=Filter[list[int], str])

# Concat

reveal_type(Concat([]), expected_type=Concat[None])  # 1
reveal_type(Concat([Empty(), Empty()]), expected_type=Concat[None])  # 1
reveal_type(Concat(["hello", "world"]), expected_type=Concat[str])  # 2
reveal_type(Concat([Empty(), "hello", Empty(), "world"]), expected_type=Concat[str])  # 2
reveal_type(Concat([Group("hello"), Group("world")]), expected_type=Concat[Sequence[str]])  # 3
reveal_type(Concat([Empty(), Group("hello"), Empty(), Group("world")]), expected_type=Concat[Sequence[str]])  # 3
reveal_type(Concat([Group("hello"), "world"]), expected_type=Concat[str | Sequence[str]])  # 4

reveal_type(Empty() + Empty(), expected_type=ConcatSkipSpaces[None])
reveal_type(Empty() + "hello", expected_type=ConcatSkipSpaces[str])
reveal_type(Empty() + Group("hello"), expected_type=ConcatSkipSpaces[Sequence[str]])

reveal_type(Empty() + "hello" + Empty(), expected_type=ConcatSkipSpaces[str])
reveal_type(Empty() + "hello" + "world", expected_type=ConcatSkipSpaces[str])
reveal_type(Empty() + "hello" + Group("world"), expected_type=ConcatSkipSpaces[str | Sequence[str]])

reveal_type(Empty() + Group("hello") + Empty(), expected_type=ConcatSkipSpaces[Sequence[str]])
reveal_type(Empty() + Group("hello") + "world", expected_type=ConcatSkipSpaces[Sequence[str] | str])
reveal_type(Empty() + Group("hello") + Group("world"), expected_type=ConcatSkipSpaces[Sequence[str]])


# AnyOf

reveal_type(AnyOf([]), expected_type=AnyOf[None])  # 1
reveal_type(AnyOf([Empty(), Empty()]), expected_type=AnyOf[None])  # 1
reveal_type(AnyOf(["hello", "world"]), expected_type=AnyOf[str])  # 2
reveal_type(AnyOf([Empty(), "hello", Empty(), "world"]), expected_type=AnyOf[str])  # 2
reveal_type(AnyOf([Group("hello"), Group("world")]), expected_type=AnyOf[Sequence[str]])  # 3
reveal_type(AnyOf([Empty(), Group("hello"), Empty(), Group("world")]), expected_type=AnyOf[Sequence[str]])  # 3
reveal_type(AnyOf([Group("hello"), "world"]), expected_type=AnyOf[str | Sequence[str]])  # 4

reveal_type(Empty() | Empty(), expected_type=AnyOf[None])
reveal_type(Empty() | "hello", expected_type=AnyOf[str])
reveal_type(Empty() | Group("hello"), expected_type=AnyOf[Sequence[str]])

reveal_type(Empty() | "hello" | Empty(), expected_type=AnyOf[str])
reveal_type(Empty() | "hello" | "world", expected_type=AnyOf[str])
reveal_type(Empty() | "hello" | Group("world"), expected_type=AnyOf[str | Sequence[str]])

reveal_type(Empty() | Group("hello") | Empty(), expected_type=AnyOf[Sequence[str]])
reveal_type(Empty() | Group("hello") | "world", expected_type=AnyOf[Sequence[str] | str])
reveal_type(Empty() | Group("hello") | Group("world"), expected_type=AnyOf[Sequence[str]])


# This is ... okay?  The content of the Group will of course be [], which can
# be a sequence of anything.

reveal_type(Group(Empty()) + "hello", expected_type=ConcatSkipSpaces[Sequence[None] | str])

# Ummm... how will this interact with things later?

reveal_type(Named(Empty(), "foo") + "hello", expected_type=ConcatSkipSpaces[Mapping[str, Sequence[None]] | str])

reveal_type(Suppress(Empty()) + "hello", expected_type=ConcatSkipSpaces[str])
reveal_type(First(Empty()) + "hello", expected_type=ConcatSkipSpaces[str])
reveal_type(AsKeyword(Empty()) + "hello", expected_type=ConcatSkipSpaces[str])
reveal_type(Longest(Empty()) + "hello", expected_type=ConcatSkipSpaces[str])



# Do not do this.  We try not to let you shoot yourself in the foot, but...
reveal_type(Repeat(Empty()) + "hello", expected_type=ConcatSkipSpaces[str])
reveal_type(Empty()[...] + "hello", expected_type=ConcatSkipSpaces[str])
reveal_type(Empty()[3,...] + "hello", expected_type=ConcatSkipSpaces[str])
reveal_type(Empty()[3,6] + "hello", expected_type=ConcatSkipSpaces[str])

reveal_type(RepeatSkipSpaces(Empty()) + "hello", expected_type=ConcatSkipSpaces[str])
