"""Decision models that label a question with one of the query intents, behind the `IntentLabeler` interface.

A decision model is an encoder that reads the question together with one short
description per label and scores every label in a single forward pass. It does
not generate text, so the same question always gets the same label. The GLiNER2
library is imported only here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from magma.config import Intent, IntentDescriptions

if TYPE_CHECKING:
    from gliner2 import AutoExtractor

_TASK_NAME = "intent"


@dataclass(frozen=True)
class LabeledIntent:
    intent: Intent
    confidence: float


class IntentLabeler(Protocol):
    """Labels each question with the intent whose description fits it best."""

    @property
    def name(self) -> str: ...

    def label(self, questions: Sequence[str], descriptions: IntentDescriptions) -> list[LabeledIntent]: ...


class GlinerLabeler:
    """A GLiNER2 decision model (e.g. `fastino/GLiNER2.5-Decide`) run locally."""

    def __init__(self, model_name: str, batch_size: int = 8) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: AutoExtractor | None = None

    @property
    def name(self) -> str:
        return self._model_name

    def label(self, questions: Sequence[str], descriptions: IntentDescriptions) -> list[LabeledIntent]:
        if not questions:
            return []
        if self._model is None:
            from gliner2 import AutoExtractor  # heavy import: only when a model is actually used

            self._model = AutoExtractor.from_pretrained(self._model_name)
        task = {_TASK_NAME: {"labels": {intent.value: descriptions.of(intent) for intent in Intent}}}
        raw_results = self._model.batch_classify_text(
            list(questions), task, batch_size=self._batch_size, include_confidence=True
        )
        return [_parse_result(result) for result in raw_results]


def _parse_result(result: dict[str, dict[str, str | float]]) -> LabeledIntent:
    # GLiNER2 returns {"intent": {"label": "WHY", "confidence": 0.69}} for a single-label task.
    answer = result[_TASK_NAME]
    return LabeledIntent(intent=Intent(str(answer["label"])), confidence=float(answer["confidence"]))
