"""Unit tests for the decision-model classifiers (`magma/intent_model.py` and the model classifiers in `magma/router.py`)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np

from magma.config import DEFAULT, Intent, IntentDescriptions
from magma.embed import Matrix, Vector
from magma.intent_model import LabeledIntent, _parse_result
from magma.router import (
    DatesRemovedClassifier,
    KeywordModelClassifier,
    ModelClassifier,
    Router,
    remove_date_expressions,
)
from magma.temporal import DateparserTagger, TimeWindow

NO_VECTOR = np.zeros(3, dtype=np.float32)


class ScriptedLabeler:
    """Answers from a fixed table and records which questions reached the model."""

    def __init__(self, answers: dict[str, LabeledIntent]) -> None:
        self._answers = answers
        self.asked: list[str] = []
        self.descriptions_seen: list[IntentDescriptions] = []

    @property
    def name(self) -> str:
        return "scripted"

    def label(self, questions: Sequence[str], descriptions: IntentDescriptions) -> list[LabeledIntent]:
        self.asked.extend(questions)
        self.descriptions_seen.append(descriptions)
        return [self._answers[question] for question in questions]


def test_model_classifier_returns_the_model_label_with_its_confidence() -> None:
    labeler = ScriptedLabeler({"What made Jon choose the decor?": LabeledIntent(Intent.WHY, 0.69)})
    match = ModelClassifier(labeler).classify("What made Jon choose the decor?", NO_VECTOR)

    assert match.intent == Intent.WHY
    assert match.reason == "model scripted (confidence 0.690)"
    assert labeler.descriptions_seen == [DEFAULT.router.intent_descriptions]


def test_keyword_model_classifier_asks_the_model_only_without_a_keyword() -> None:
    labeler = ScriptedLabeler({"What color did Joanna choose for her hair?": LabeledIntent(Intent.ENTITY, 0.81)})
    classifier = KeywordModelClassifier(labeler)

    explicit = classifier.classify("When did Melanie paint a sunrise?", NO_VECTOR)
    implicit = classifier.classify("What color did Joanna choose for her hair?", NO_VECTOR)

    assert explicit.intent == Intent.WHEN and explicit.reason == 'keyword "when"'
    assert implicit.intent == Intent.ENTITY and implicit.reason.startswith("model ")
    assert labeler.asked == ["What color did Joanna choose for her hair?"]


def test_gliner_result_is_parsed_into_intent_and_confidence() -> None:
    parsed = _parse_result({"intent": {"label": "WHEN", "confidence": 0.83}})
    assert parsed == LabeledIntent(Intent.WHEN, 0.83)


def test_dates_removed_classifier_hides_context_dates_from_the_inner_classifier() -> None:
    question = "Which city did John recommend to Tim in January 2024?"
    labeler = ScriptedLabeler({"Which city did John recommend to Tim?": LabeledIntent(Intent.ENTITY, 0.9)})
    classifier = DatesRemovedClassifier(ModelClassifier(labeler), DateparserTagger())

    match = classifier.classify(question, NO_VECTOR)

    assert labeler.asked == ["Which city did John recommend to Tim?"]
    assert match.intent == Intent.ENTITY
    assert match.reason.endswith(", dates removed")


def test_remove_date_expressions_keeps_questions_without_dates_unchanged() -> None:
    tagger = DateparserTagger()
    assert remove_date_expressions("When did Melanie paint a sunrise?", tagger) == "When did Melanie paint a sunrise?"
    assert remove_date_expressions("What did John organize on May 8, 2022?", tagger) == "What did John organize?"


def test_router_with_dates_removed_keeps_the_time_window_from_the_full_question() -> None:
    question = "Which city did John recommend to Tim in January 2024?"
    labeler = ScriptedLabeler({"Which city did John recommend to Tim?": LabeledIntent(Intent.ENTITY, 0.9)})
    tagger = DateparserTagger()
    router = Router(OneAxisEmbedder(), tagger, DatesRemovedClassifier(ModelClassifier(labeler), tagger))

    routed = router.route(question, reference_time=datetime(2024, 3, 1))

    assert routed.intent == Intent.ENTITY
    assert routed.time_window == TimeWindow(start=datetime(2024, 1, 1), end=datetime(2024, 2, 1))
    assert "january" in routed.keywords


class OneAxisEmbedder:
    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))

    def embed_query(self, text: str) -> Vector:
        vector: Vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return vector
