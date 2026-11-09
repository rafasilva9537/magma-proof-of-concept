"""Reader for the LoCoMo dataset (Maharana et al., 2024), in the format of `locomo10.json`.

Each sample is one long conversation between two speakers, split into dated
sessions, plus its questions. A conversation is a JSON object whose keys are
numbered (`session_1`, `session_1_date_time`, `session_2`, …), so the sessions
are read by number, in order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NotRequired, TypedDict

from pydantic import TypeAdapter

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
