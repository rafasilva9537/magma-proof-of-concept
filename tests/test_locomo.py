import json
from datetime import datetime
from pathlib import Path

import pytest

from bench.locomo import UNANSWERABLE, LocomoCategory, find_sample, load_samples, session_time, sessions, to_conversation, turn_text

SAMPLE = {
    "sample_id": "conv-x",
    "qa": [{"question": "When did Melanie get a dog?", "answer": "9 May 2023", "evidence": ["D1:2"], "category": 2}],
    "conversation": {
        "speaker_a": "Caroline",
        "speaker_b": "Melanie",
        "session_10_date_time": "2:00 pm on 20 October, 2023",
        "session_10": [{"speaker": "Caroline", "dia_id": "D10:1", "text": "Long time!"}],
        "session_2_date_time": "1:00 pm on 9 May, 2023",
        "session_2": [{"speaker": "Melanie", "dia_id": "D2:1", "text": "Look.", "blip_caption": "a photo of a dog"}],
    },
    "observation": {"ignored": "fields the reader does not use are dropped"},
}


def write(tmp_path: Path) -> Path:
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([SAMPLE]), encoding="utf-8")
    return path


def test_sessions_come_in_numeric_order(tmp_path: Path) -> None:
    sample = find_sample(load_samples(write(tmp_path)), "conv-x")
    found = sessions(sample)
    assert [session.number for session in found] == [2, 10]  # 10 after 2, not before as in text order
    assert found[1].date_time == "2:00 pm on 20 October, 2023"
    assert found[0].turns[0]["dia_id"] == "D2:1"


def test_turn_text_includes_speaker_and_caption(tmp_path: Path) -> None:
    turn = sessions(find_sample(load_samples(write(tmp_path)), "conv-x"))[0].turns[0]
    assert turn_text(turn) == "Melanie: [Image: a photo of a dog] Look."
    assert turn_text(turn, with_speaker=False) == "[Image: a photo of a dog] Look."


def test_questions_are_read(tmp_path: Path) -> None:
    question = load_samples(write(tmp_path))[0]["qa"][0]
    assert question["category"] == 2
    assert question.get("answer") == "9 May 2023"


def test_unknown_sample_raises(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        find_sample(load_samples(write(tmp_path)), "conv-missing")


def test_session_time_reads_the_dataset_format(tmp_path: Path) -> None:
    found = sessions(find_sample(load_samples(write(tmp_path)), "conv-x"))
    assert [session_time(session) for session in found] == [datetime(2023, 5, 9, 13, 0), datetime(2023, 10, 20, 14, 0)]


def test_adapter_dates_turns_by_session_and_asks_at_the_last_session(tmp_path: Path) -> None:
    conversation = to_conversation(find_sample(load_samples(write(tmp_path)), "conv-x"))
    assert conversation.id == "conv-x"
    assert [(turn.speaker, turn.text, turn.timestamp) for turn in conversation.turns] == [
        ("Melanie", "[Image: a photo of a dog] Look.", datetime(2023, 5, 9, 13, 0)),
        ("Caroline", "Long time!", datetime(2023, 10, 20, 14, 0)),
    ]
    assert conversation.reference_time == datetime(2023, 10, 20, 14, 0)
    question = conversation.questions[0]
    assert (question.id, question.category, question.gold, question.evidence) == ("conv-x-q1", LocomoCategory.TEMPORAL, "9 May 2023", ("D1:2",))


def test_adversarial_questions_get_the_unanswerable_gold(tmp_path: Path) -> None:
    sample = dict(SAMPLE)
    sample["qa"] = [
        {"question": "What did Caroline adopt?", "adversarial_answer": "a cat", "evidence": [], "category": 5},
        {"question": "How old is Melanie?", "answer": 30, "evidence": ["D2:1"], "category": 4},
    ]
    path = tmp_path / "adversarial.json"
    path.write_text(json.dumps([sample]), encoding="utf-8")
    questions = to_conversation(load_samples(path)[0]).questions
    assert [(q.category, q.gold) for q in questions] == [(LocomoCategory.ADVERSARIAL, UNANSWERABLE), (LocomoCategory.SINGLE_HOP, "30")]
