"""Reader for the LoCoMo dataset (Maharana et al., 2024), in the format of `locomo10.json`.

Each sample is one long conversation between two speakers, split into dated
sessions, plus its questions. A conversation is a JSON object whose keys are
numbered (`session_1`, `session_1_date_time`, `session_2`, …), so the sessions
are read by number, in order.

`to_conversation` is the adapter to the runner's format: every turn of a
session gets the session's start time (LoCoMo dates sessions, not turns), a
shared image enters as its caption, as in the official loader, and the
questions are asked at the start of the last session.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import NotRequired, TypedDict

from pydantic import TypeAdapter

from bench.benchmark import Conversation, Question
from magma.ingest import Turn
from magma.json_types import JSONValue


class LocomoTurn(TypedDict):
    speaker: str
    dia_id: str
    text: str
    blip_caption: NotRequired[str]  # caption of a shared image


class LocomoQuestion(TypedDict):
    question: str
    evidence: list[str]  # dia_id of the turns that hold the answer
    category: int
    answer: NotRequired[str | int]
    adversarial_answer: NotRequired[str]  # questions about something that was never said


class LocomoSample(TypedDict):
    sample_id: str
    conversation: dict[str, JSONValue]
    qa: list[LocomoQuestion]


@dataclass(frozen=True)
class Session:
    number: int
    date_time: str  # as written in the dataset, e.g. "1:56 pm on 8 May, 2023"
    turns: list[LocomoTurn]


class LocomoCategory(StrEnum):
    """The five question types of LoCoMo (Tab. 7 of the paper), numbered 1–5 in the dataset."""

    MULTI_HOP = "MULTI_HOP"
    TEMPORAL = "TEMPORAL"
    OPEN_DOMAIN = "OPEN_DOMAIN"
    SINGLE_HOP = "SINGLE_HOP"
    ADVERSARIAL = "ADVERSARIAL"


_CATEGORY_BY_NUMBER = {
    1: LocomoCategory.MULTI_HOP,
    2: LocomoCategory.TEMPORAL,
    3: LocomoCategory.OPEN_DOMAIN,
    4: LocomoCategory.SINGLE_HOP,
    5: LocomoCategory.ADVERSARIAL,
}

# Adversarial questions ask about something never said and have no answer in the dataset.
# The judge prompt (C.3) expects this gold reference for them.
UNANSWERABLE = "Unanswerable"

_SAMPLES = TypeAdapter(list[LocomoSample])
_TURNS = TypeAdapter(list[LocomoTurn])
_DATE_TIME = TypeAdapter(str)
_SESSION_KEY = re.compile(r"session_(\d+)")
_SESSION_TIME_FORMAT = "%I:%M %p on %d %B, %Y"  # "1:56 pm on 8 May, 2023"


def load_samples(path: Path) -> list[LocomoSample]:
    return _SAMPLES.validate_json(path.read_text(encoding="utf-8"))


def find_sample(samples: list[LocomoSample], sample_id: str) -> LocomoSample:
    for sample in samples:
        if sample["sample_id"] == sample_id:
            return sample
    raise KeyError(f"no LoCoMo sample {sample_id!r}")


def sessions(sample: LocomoSample) -> list[Session]:
    conversation = sample["conversation"]
    numbers = sorted(int(match.group(1)) for key in conversation if (match := _SESSION_KEY.fullmatch(key)))
    return [
        Session(
            number=number,
            date_time=_DATE_TIME.validate_python(conversation[f"session_{number}_date_time"]),
            turns=_TURNS.validate_python(conversation[f"session_{number}"]),
        )
        for number in numbers
    ]


def turn_text(turn: LocomoTurn, with_speaker: bool = True) -> str:
    """The turn as the fast path stores it. A shared image enters as its caption, as in the official loader."""
    text = turn["text"]
    if "blip_caption" in turn:
        text = f"[Image: {turn['blip_caption']}] {text}"
    return f"{turn['speaker']}: {text}" if with_speaker else text


def session_time(session: Session) -> datetime:
    """The moment the session starts. LoCoMo dates sessions, not turns, so every turn of a session gets it."""
    return datetime.strptime(session.date_time, _SESSION_TIME_FORMAT)


def category_of(question: LocomoQuestion) -> LocomoCategory:
    return _CATEGORY_BY_NUMBER[question["category"]]


def gold_answer(question: LocomoQuestion) -> str:
    """The reference answer; adversarial questions get `UNANSWERABLE` (their `adversarial_answer` is the trap)."""
    if category_of(question) is LocomoCategory.ADVERSARIAL:
        return UNANSWERABLE
    return str(question.get("answer", ""))


def to_conversation(sample: LocomoSample) -> Conversation:
    found = sessions(sample)
    turns: list[Turn] = []
    for session in found:
        moment = session_time(session)
        for turn in session.turns:
            turns.append(Turn(speaker=turn["speaker"], text=turn_text(turn, with_speaker=False), timestamp=moment))
    sample_id = sample["sample_id"]
    questions = tuple(
        Question(
            id=f"{sample_id}-q{number}",
            conversation_id=sample_id,
            category=category_of(question).value,
            question=question["question"],
            gold=gold_answer(question),
            evidence=tuple(question["evidence"]),
        )
        for number, question in enumerate(sample["qa"], start=1)
    )
    return Conversation(id=sample_id, turns=tuple(turns), reference_time=session_time(found[-1]), questions=questions)
