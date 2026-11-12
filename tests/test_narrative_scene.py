from datetime import datetime

from magma.graph import VISIBLE_TO_EVERYONE
from magma.ingest import Turn
from narrative.config import DEFAULT_NARRATIVE
from narrative.scene import TurnParts, describe_audience, split_turn, visible_to

START = datetime(2024, 3, 1, 20, 0)


def turn(speaker: str, text: str, present: frozenset[str] = VISIBLE_TO_EVERYONE) -> Turn:
    return Turn(speaker=speaker, text=text, timestamp=START, present=present)


def test_marked_turn_splits_into_action_and_speech() -> None:
    parts = split_turn(turn("Brom", '*picks up the wand* "Nobody attacked me, I dropped it."'), DEFAULT_NARRATIVE)
    assert parts == TurnParts(action_text="picks up the wand", speech_text="Nobody attacked me, I dropped it.")


def test_curly_quotes_and_several_marks_are_joined_in_order() -> None:
    parts = split_turn(turn("Lia", "*smiles* “Hello.” *waves* “Bye.”"), DEFAULT_NARRATIVE)
    assert parts == TurnParts(action_text="smiles waves", speech_text="Hello. Bye.")


def test_text_outside_the_marks_counts_as_speech_never_as_action() -> None:
    # Only the marked action is evidence; unmarked words could be part of a lie.
    parts = split_turn(turn("Brom", "*shrugs* I was in the lab."), DEFAULT_NARRATIVE)
    assert parts == TurnParts(action_text="shrugs", speech_text="I was in the lab.")


def test_turn_without_marks_is_all_speech_as_written() -> None:
    # LoCoMo and FANToM turns; a quoted title stays quoted.
    parts = split_turn(turn("Caroline", 'I read "Becoming" last week.'), DEFAULT_NARRATIVE)
    assert parts == TurnParts(action_text="", speech_text='I read "Becoming" last week.')


def test_a_turn_that_is_one_quotation_loses_the_quotes() -> None:
    parts = split_turn(turn("Lia", '"The key is in the drawer."'), DEFAULT_NARRATIVE)
    assert parts == TurnParts(action_text="", speech_text="The key is in the drawer.")


def test_quotes_inside_marked_speech_keep_the_word_order() -> None:
    parts = split_turn(turn("Lia", '*nods* I read "Becoming" today.'), DEFAULT_NARRATIVE)
    assert parts == TurnParts(action_text="nods", speech_text="I read Becoming today.")


def test_narration_is_all_action() -> None:
    parts = split_turn(turn("narrator", 'Lia casts a disarming spell on Brom. "Stop!" echoes in the hall.'), DEFAULT_NARRATIVE)
    assert parts.speech_text == ""
    assert parts.action_text.startswith("Lia casts")


def test_speaker_always_sees_their_own_turn() -> None:
    assert visible_to(turn("Brom", "hi", frozenset({"Dara"})), DEFAULT_NARRATIVE) == frozenset({"Brom", "Dara"})


def test_narrator_is_not_an_observer() -> None:
    present = frozenset({"Lia", "Brom", "Ceres"})
    assert visible_to(turn("narrator", "A spell flies.", present), DEFAULT_NARRATIVE) == present


def test_turn_in_a_scene_open_to_everyone_stays_open() -> None:
    assert visible_to(turn("Caroline", "hi"), DEFAULT_NARRATIVE) == VISIBLE_TO_EVERYONE


def test_audience_is_who_else_was_there() -> None:
    assert describe_audience(frozenset({"Brom", "Dara", "Ceres"}), "Brom") == "{Ceres, Dara}"
    assert describe_audience(VISIBLE_TO_EVERYONE, "Brom") == "everyone"
    assert describe_audience(frozenset({"Brom"}), "Brom") == "no one"


def test_audience_names_the_viewer_as_you() -> None:
    assert describe_audience(frozenset({"Brom", "Dara", "Ceres"}), "Brom", viewer="Dara") == "{Ceres, you}"
