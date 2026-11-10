"""The one format every benchmark is converted to before the runner sees it.

A conversation is a list of dated turns plus the questions asked about it at
its end. Adapters (LoCoMo now, others later) build these from each dataset, so
the runner, the sampling and the metrics are written once.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from magma.ingest import Turn


@dataclass(frozen=True)
class Question:
    id: str
    conversation_id: str
    category: str  # the dataset's question type, used for stratified sampling and per-category results only
    question: str
    gold: str
    evidence: tuple[str, ...]  # the dataset's ids of the turns that hold the answer


@dataclass(frozen=True)
class Conversation:
    id: str
    turns: tuple[Turn, ...]
    reference_time: datetime  # "now" for the questions: the end of the conversation
    questions: tuple[Question, ...]
