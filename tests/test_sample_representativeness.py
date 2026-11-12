"""The sample spread report: shares per category, conversation and position quartile."""

from datetime import datetime

from bench.benchmark import Conversation, Question
from bench.sample_representativeness import compare, quartile, representativeness

END = datetime(2024, 1, 1)


def question(conversation_id: str, number: int, category: str) -> Question:
    return Question(f"{conversation_id}-q{number}", conversation_id, category, "?", "!", ())


def conversation(conversation_id: str, categories: list[str]) -> Conversation:
    questions = tuple(question(conversation_id, number, category) for number, category in enumerate(categories))
    return Conversation(conversation_id, (), END, questions)


def test_quartile_by_relative_position() -> None:
    assert [quartile(index, 5) for index in range(5)] == [0, 1, 2, 3, 3]  # positions 0, .25, .5, .75, 1
    assert quartile(0, 1) == 0


def test_compare_gives_shares_and_the_largest_gap() -> None:
    axis = compare("category", ["A", "A", "B", "B"], ["A", "A", "A", "B"])
    assert [(share.value, share.dataset_share, share.sample_share) for share in axis.shares] == [("A", 50.0, 75.0), ("B", 50.0, 25.0)]
    assert axis.largest_gap_points == 25.0


def test_report_counts_every_axis_and_flags_unknown_ids() -> None:
    conversations = [conversation("c1", ["X", "Y", "X", "Y"]), conversation("c2", ["X", "Y"])]
    report = representativeness(conversations, {"c1-q0", "c1-q3", "c2-q1", "missing"})
    assert report.dataset_questions == 6 and report.sample_questions == 3
    assert report.unknown_ids == ["missing"]
    by_name = {axis.name: axis for axis in report.axes}
    assert [(share.value, share.sample) for share in by_name["conversation"].shares] == [("c1", 2), ("c2", 1)]
    # c1-q0 is Q1; c1-q3 is the last of four, Q4; c2-q1 is the last of two, Q4
    assert [(share.value, share.sample) for share in by_name["position_quartile"].shares] == [("Q1", 1), ("Q2", 0), ("Q3", 0), ("Q4", 2)]
