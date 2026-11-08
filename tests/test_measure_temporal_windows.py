from datetime import datetime

from bench.locomo import LocomoQuestion, LocomoSample
from bench.measure_temporal_windows import measure_question, reference_time, summarize
from magma.temporal import DateparserTagger

TAGGER = DateparserTagger()
REFERENCE = datetime(2023, 10, 20)


def question(text: str, category: int = 4) -> LocomoQuestion:
    return LocomoQuestion(question=text, evidence=[], category=category, answer="gold")


def test_reference_is_the_day_of_the_last_session() -> None:
    sample = LocomoSample(
        sample_id="conv-x",
        qa=[],
        conversation={
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [],
            "session_2_date_time": "2:00 pm on 20 October, 2023",
            "session_2": [],
        },
    )
    assert reference_time(sample, TAGGER) == datetime(2023, 10, 20)


def test_question_with_a_date_gets_its_window() -> None:
    result = measure_question(question("What movie did Joanna watch on 1 May, 2022?"), REFERENCE, TAGGER)
    assert result.window is not None
    assert (result.window.start, result.window.end) == (datetime(2022, 5, 1), datetime(2022, 5, 2))
    assert not result.is_narrowed_by_specificity


def test_rejected_candidates_and_narrowed_windows_are_reported() -> None:
    birthday = measure_question(question("How long ago was Caroline's 18th birthday?"), REFERENCE, TAGGER)
    assert birthday.window is None
    assert birthday.rejected_candidates == ["18th"]
    between = measure_question(question("Where was John between August 11 and August 15 2023?"), REFERENCE, TAGGER)
    assert between.is_narrowed_by_specificity  # the year 2023 would otherwise widen the window


def test_summary_counts_by_category() -> None:
    results = [
        measure_question(question("What did Melanie do in May 2023?", category=4), REFERENCE, TAGGER),
        measure_question(question("When did Melanie paint a sunrise?", category=2), REFERENCE, TAGGER),
    ]
    report = summarize(results, sample_size=5, seed=7)
    assert (report.questions, report.with_window) == (2, 1)
    assert [(c.category, c.questions, c.with_window) for c in report.by_category] == [(2, 1, 0), (4, 1, 1)]
    assert [(g.name, g.count) for g in report.mention_granularity] == [("MONTH", 1)]
    assert [example.gold_answer for example in report.examples] == ["gold"]
