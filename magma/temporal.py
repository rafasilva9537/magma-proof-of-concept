"""Temporal tagger: turns date expressions into absolute time windows.

Two uses. On a question, it finds the expressions, and `query_window` turns
them into the time window [τs, τe] of Stage 1. On a turn, it resolves the
`dates_mentioned` of the slow path against the time the turn was spoken
("yesterday" said on 20 Oct means 19 Oct).

Every expression becomes a window of its own granularity: "yesterday" is a
day, "last week" a calendar week, "in June" a month, "2022" a year.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Literal, Protocol, TypedDict

from dateparser.date import DateDataParser
from dateparser.search import search_dates

from magma.config import DEFAULT, DatePreference, TemporalConfig


class Granularity(StrEnum):
    DAY = "DAY"
    WEEK = "WEEK"
    MONTH = "MONTH"
    YEAR = "YEAR"


@dataclass(frozen=True)
class TimeWindow:
    start: datetime  # inclusive
    end: datetime  # exclusive

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment < self.end

    def overlaps(self, other: TimeWindow) -> bool:
        return self.start < other.end and other.start < self.end

    def covers(self, other: TimeWindow) -> bool:
        return self.start <= other.start and other.end <= self.end


@dataclass(frozen=True)
class TemporalMention:
    text: str  # the expression as written
    window: TimeWindow
    granularity: Granularity


def query_window(mentions: list[TemporalMention]) -> TimeWindow | None:
    """The window [τs, τe] of a question: the smallest window covering its most specific mentions.

    "between August 11 and August 15 2023" yields three mentions, the two days
    and the year 2023. The year only qualifies the days, so a mention whose
    window contains another one is left out, and the window covers both days.
    """
    specific = [mention for mention in mentions if not _contains_another(mention, mentions)]
    if not specific:
        return None
    start = min(mention.window.start for mention in specific)
    end = max(mention.window.end for mention in specific)
    return TimeWindow(start=start, end=end)


def _contains_another(mention: TemporalMention, mentions: list[TemporalMention]) -> bool:
    for other in mentions:
        if other.window != mention.window and mention.window.covers(other.window):
            return True
    return False


class TemporalTagger(Protocol):
    def resolve(self, expression: str, reference: datetime, preference: DatePreference) -> TemporalMention | None:
        """One expression on its own, such as an item of `dates_mentioned`; None if it is not a date."""
        ...

    def find(self, text: str, reference: datetime, preference: DatePreference) -> list[TemporalMention]:
        """Every date expression inside a free text, such as a question, in reading order."""
        ...


# A match inside free text is kept only if it has one of these words or a
# four-digit year. Without the check, the search also returns "the 18th"
# (a birthday), "an hour" or "a morning jog" as dates.
TEMPORAL_CUE_WORDS = frozenset(
    """
    january february march april may june july august september october november december
    jan feb mar apr jun jul aug sep sept oct nov dec
    monday tuesday wednesday thursday friday saturday sunday
    yesterday today tomorrow tonight ago last next week weeks weekend month months year years
    """.split()
)
_YEAR = re.compile(r"\b\d{4}\b")
_WORD = re.compile(r"[a-z]+")

_GRANULARITY_OF_PERIOD = {
    "time": Granularity.DAY,
    "part_of_day": Granularity.DAY,
    "day": Granularity.DAY,
    "week": Granularity.WEEK,
    "month": Granularity.MONTH,
    "year": Granularity.YEAR,
}



class _DateparserSettings(TypedDict):
    RELATIVE_BASE: datetime
    PREFER_DATES_FROM: Literal["past", "current_period"]
    RETURN_TIME_AS_PERIOD: bool


class DateparserTagger:
    """TemporalTagger on the `dateparser` library, the only module that imports it."""

    def __init__(self, config: TemporalConfig = DEFAULT.temporal) -> None:
        self._languages = list(config.languages)

    def resolve(self, expression: str, reference: datetime, preference: DatePreference) -> TemporalMention | None:
        parser = DateDataParser(languages=self._languages, settings=self._settings(reference, preference))
        parsed = parser.get_date_data(expression)
        moment: datetime | None = parsed.date_obj
        period: str | None = parsed.period
        if moment is None or period not in _GRANULARITY_OF_PERIOD:
            return None
        granularity = _GRANULARITY_OF_PERIOD[period]
        return TemporalMention(text=expression, window=window_around(moment, granularity), granularity=granularity)

    def find(self, text: str, reference: datetime, preference: DatePreference) -> list[TemporalMention]:
        found: list[tuple[str, datetime]] = (
            search_dates(text, languages=self._languages, settings=self._settings(reference, preference)) or []
        )
        mentions: list[TemporalMention] = []
        for expression, _ in found:
            if not has_temporal_cue(expression):
                continue
            mention = self.resolve(expression, reference, preference)
            if mention is not None:
                mentions.append(mention)
        return mentions

    @staticmethod
    def _settings(reference: datetime, preference: DatePreference) -> _DateparserSettings:
        match preference:
            case DatePreference.PAST:
                prefer: Literal["past", "current_period"] = "past"
            case DatePreference.CURRENT_PERIOD:
                prefer = "current_period"
        return _DateparserSettings(RELATIVE_BASE=reference, PREFER_DATES_FROM=prefer, RETURN_TIME_AS_PERIOD=True)


def has_temporal_cue(expression: str) -> bool:
    lowered = expression.lower()
    if _YEAR.search(lowered):
        return True
    return any(word in TEMPORAL_CUE_WORDS for word in _WORD.findall(lowered))


def window_around(moment: datetime, granularity: Granularity) -> TimeWindow:
    """The calendar day, week (Monday to Sunday), month or year that contains `moment`."""
    day = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    match granularity:
        case Granularity.DAY:
            return TimeWindow(start=day, end=day + timedelta(days=1))
        case Granularity.WEEK:
            monday = day - timedelta(days=day.weekday())
            return TimeWindow(start=monday, end=monday + timedelta(weeks=1))
        case Granularity.MONTH:
            first = day.replace(day=1)
            if first.month == 12:
                return TimeWindow(start=first, end=first.replace(year=first.year + 1, month=1))
            return TimeWindow(start=first, end=first.replace(month=first.month + 1))
        case Granularity.YEAR:
            first = day.replace(month=1, day=1)
            return TimeWindow(start=first, end=first.replace(year=first.year + 1))
