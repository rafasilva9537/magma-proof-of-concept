from datetime import datetime

import pytest

from magma.config import DatePreference
from magma.temporal import (
    DateparserTagger,
    Granularity,
    TemporalMention,
    TimeWindow,
    has_temporal_cue,
    query_window,
    window_around,
)

# Friday, 20 October 2023, the session of the case study in the paper's Appendix E.
SESSION = datetime(2023, 10, 20, 13, 56)
PAST = DatePreference.PAST


@pytest.fixture(scope="module")
def tagger() -> DateparserTagger:
    return DateparserTagger()


def window(start: tuple[int, int, int], end: tuple[int, int, int]) -> TimeWindow:
    return TimeWindow(start=datetime(*start), end=datetime(*end))


def test_windows_follow_the_calendar() -> None:
    moment = datetime(2023, 12, 13, 15, 30)  # a Wednesday
    assert window_around(moment, Granularity.DAY) == window((2023, 12, 13), (2023, 12, 14))
    assert window_around(moment, Granularity.WEEK) == window((2023, 12, 11), (2023, 12, 18))  # Monday to Monday
    assert window_around(moment, Granularity.MONTH) == window((2023, 12, 1), (2024, 1, 1))  # across the year
    assert window_around(moment, Granularity.YEAR) == window((2023, 1, 1), (2024, 1, 1))


def test_window_end_is_exclusive() -> None:
    day = window((2023, 10, 19), (2023, 10, 20))
    assert day.contains(datetime(2023, 10, 19, 23, 59))
    assert not day.contains(datetime(2023, 10, 20))
    assert not day.overlaps(window((2023, 10, 20), (2023, 10, 21)))
    assert window((2023, 1, 1), (2024, 1, 1)).covers(day)
    assert not day.covers(window((2023, 1, 1), (2024, 1, 1)))


def test_yesterday_is_resolved_against_the_session(tagger: DateparserTagger) -> None:
    mention = tagger.resolve("yesterday", SESSION, DatePreference.CURRENT_PERIOD)
    assert mention == TemporalMention(
        text="yesterday", window=window((2023, 10, 19), (2023, 10, 20)), granularity=Granularity.DAY
    )


@pytest.mark.parametrize(
    ("expression", "expected", "granularity"),
    [
        ("last week", window((2023, 10, 9), (2023, 10, 16)), Granularity.WEEK),
        ("in June", window((2023, 6, 1), (2023, 7, 1)), Granularity.MONTH),
        ("2022", window((2022, 1, 1), (2023, 1, 1)), Granularity.YEAR),
        ("last year", window((2022, 1, 1), (2023, 1, 1)), Granularity.YEAR),
        ("2024-01-01", window((2024, 1, 1), (2024, 1, 2)), Granularity.DAY),
        ("1:56 pm on 8 May, 2023", window((2023, 5, 8), (2023, 5, 9)), Granularity.DAY),
    ],
)
def test_each_expression_gets_its_own_granularity(
    tagger: DateparserTagger, expression: str, expected: TimeWindow, granularity: Granularity
) -> None:
    mention = tagger.resolve(expression, SESSION, PAST)
    assert mention is not None
    assert mention.window == expected
    assert mention.granularity is granularity


def test_non_dates_are_not_resolved(tagger: DateparserTagger) -> None:
    assert tagger.resolve("the violin", SESSION, PAST) is None
    assert tagger.resolve("a few days ago", SESSION, PAST) is None  # dateparser does not read vague amounts


def test_preference_decides_ambiguous_dates(tagger: DateparserTagger) -> None:
    past = tagger.resolve("December", SESSION, DatePreference.PAST)
    current = tagger.resolve("December", SESSION, DatePreference.CURRENT_PERIOD)
    assert past is not None and current is not None
    assert past.window.start == datetime(2022, 12, 1)  # the last December that already happened
    assert current.window.start == datetime(2023, 12, 1)  # the December of this year


def test_find_keeps_only_real_dates(tagger: DateparserTagger) -> None:
    question = "How long ago was Caroline's 18th birthday, after a morning jog of an hour?"
    assert tagger.find(question, SESSION, PAST) == []
    found = tagger.find("When did she hike yesterday?", SESSION, PAST)
    assert [mention.text for mention in found] == ["yesterday"]


def test_temporal_cues() -> None:
    assert has_temporal_cue("in May 2023")
    assert has_temporal_cue("last Friday")
    assert has_temporal_cue("2022")
    assert not has_temporal_cue("18th")
    assert not has_temporal_cue("an hour")


def test_query_window_covers_the_most_specific_mentions(tagger: DateparserTagger) -> None:
    mentions = tagger.find("Where was John between August 11 and August 15 2023?", SESSION, PAST)
    assert [mention.text for mention in mentions] == ["August 11", "August 15", "2023"]
    # The year contains both days, so it only qualifies them and is left out of the window.
    assert query_window(mentions) == window((2023, 8, 11), (2023, 8, 16))


def test_query_window_without_mentions() -> None:
    assert query_window([]) is None


def test_query_window_of_a_single_mention(tagger: DateparserTagger) -> None:
    mentions = tagger.find("What did Melanie do in May 2023?", SESSION, PAST)
    assert query_window(mentions) == window((2023, 5, 1), (2023, 6, 1))
