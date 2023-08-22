from __future__ import annotations
import re
from dataclasses import dataclass, asdict, astuple
from typing import Optional
from collections.abc import Iterator, Sequence


@dataclass(frozen=True, order=True)
class Span:
    start: int
    stop: int

    def __post_init__(self):
        Span.validate(self.start, self.stop)

    @classmethod
    def generate(
            cls,
            start_or_stop: int,
            stop: Optional[int] = None,
            /, *,
            include_empty: bool = False,
            **kwargs,
    ) -> Iterator:
        """Returns an iterator over all spans in the specified range."""
        start, stop = Span.validate(start_or_stop, stop)
        offset = 0 if include_empty else 1
        # by returing a generator instead of yielding, we are able to raise validation errors immediately
        return (
            cls(i, j, **kwargs)
            for i in range(start, stop)
            for j in range(i + offset, stop + 1)
        )

    @staticmethod
    def count(
            start_or_stop: int,
            stop: Optional[int] = None,
            /, *,
            include_empty: bool = False,
    ) -> int:
        """Returns the number of spans in the specified range."""
        start, stop = Span.validate(start_or_stop, stop)
        n = stop - start
        num_spans = n * (n + 1) // 2
        if include_empty:
            num_spans += n
        return num_spans

    @staticmethod
    def validate(start_or_stop: int, stop: Optional[int] = None) -> tuple[int, int]:
        """
        Ensures start and stop are both non-negative, and start is not greater than stop.
        If only one argument is provided, it is used as stop and start defaults to zero.
        Returns the start and stop values.
        """
        if stop is None:
            start, stop = 0, start_or_stop
        else:
            start = start_or_stop
        if start < 0:
            raise ValueError(f'expected non-negative start, got {start=}')
        if stop < 0:
            raise ValueError(f'expected non-negative stop, got {stop=}')
        if start > stop:
            raise ValueError(f'{start=} must be <= {stop=}')
        return start, stop

    @staticmethod
    def any_overlap(spans: Sequence[Span]) -> bool:
        """Returns True if any pair of spans in the sequence overlap with each other."""
        for i in range(len(spans)):
            for j in range(i+1, len(spans)):
                if spans[i].overlaps(spans[j]):
                    return True
        return False

    @classmethod
    def string_find(
            cls,
            text: str,
            sub: str,
            *,
            start: Optional[int] = None,
            stop: Optional[int] = None,
            flexible_spaces: bool = False,
            ignore_case: bool = False,
            **kwargs,
    ) -> list:
        """Returns list of matches of sub in text."""
        return list(cls.string_find_iter(
            text=text,
            sub=sub,
            start=start,
            stop=stop,
            flexible_spaces=flexible_spaces,
            ignore_case=ignore_case,
            **kwargs,
        ))

    @classmethod
    def string_find_iter(
            cls,
            text: str,
            sub: str,
            *,
            start: Optional[int] = None,
            stop: Optional[int] = None,
            flexible_spaces: bool = False,
            ignore_case: bool = False,
            **kwargs,
    ) -> Iterator:
        """Iterate over matches of sub in text."""
        if start is None:
            start = 0
        if stop is None:
            stop = len(text)
        if flexible_spaces:
            sub = r'\s+'.join(map(re.escape, sub.split()))
        else:
            sub = re.escape(sub)
        flags = re.IGNORECASE if ignore_case else re.NOFLAG
        for m in re.finditer(sub, text, flags):
            if start <= m.start() and stop >= m.end():
                yield cls(m.start(), m.end(), **kwargs)

    @property
    def first(self) -> Optional[int]:
        """Returns the first element in the span, or None if the span is empty."""
        if not self.is_empty():
            return self.start

    @property
    def last(self) -> Optional[int]:
        """Returns the last element in the span, or None if the span is empty."""
        if not self.is_empty():
            return self.stop - 1

    @property
    def slice(self) -> slice:
        """Returns a slice object equivalent to the span."""
        return slice(self.start, self.stop)

    @property
    def range(self) -> range:
        """Returns a range object equivalent to the span."""
        return range(self.start, self.stop)

    def is_empty(self) -> bool:
        """Returns True if the span is empty."""
        return self.start == self.stop

    def size(self) -> int:
        """Returns the number of elements in the span."""
        return self.stop - self.start

    def to_dict(self) -> dict[str, int]:
        """Converts the span to a dict."""
        return asdict(self)
    
    def to_tuple(self) -> tuple[int, int]:
        """Converts the span to a tuple."""
        return astuple(self)

    def overlaps(self, other: Span) -> bool:
        """Returns True if the spans overlap with each other."""
        if self.is_empty() or other.is_empty():
            return False
        if self.start == other.start or self.stop == other.stop:
            return True
        if self.start < other.start and self.stop > other.start:
            return True
        if self.start > other.start and self.start < other.stop:
            return True
        return False
