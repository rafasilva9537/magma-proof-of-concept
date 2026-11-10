"""Unit tests for the router measurement script (`bench/measure_router.py`)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from bench.locomo import LocomoQuestion
from bench.measure_router import PrelabeledLabeler, build_classifiers, build_report, route_questions
from magma.config import DEFAULT, Intent, IntentDescriptions
from magma.embed import Matrix, Vector, normalize
from magma.intent_model import LabeledIntent


class ThreeAxisEmbedder:
    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        rows = [self._vector(text) for text in texts]
        return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 3), dtype=np.float32)

    def embed_query(self, text: str) -> Vector:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> Vector:
        lowered = text.lower()
        why_axis = float(any(word in lowered for word in ("cause", "caused", "reason", "motivat", "effect", "broke out")))
        when_axis = float(any(word in lowered for word in ("when", "date", "month", "year", "long", "moment", "era")))
        entity_axis = float(any(word in lowered for word in ("who", "role", "hobby", "where", "item", "instrument")))
        raw = np.array([[why_axis, when_axis, entity_axis]], dtype=np.float32)
        if not raw.any():
            raw = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        unit: Vector = normalize(raw)[0]
        return unit


def test_measure_router_counts_intents_and_finds_disagreements() -> None:
    embedder = ThreeAxisEmbedder()
    questions: list[LocomoQuestion] = [
        {"question": "Why did Caroline go to the group?", "evidence": ["D1:1"], "category": 1, "answer": "support"},
        {"question": "When did Melanie start painting?", "evidence": ["D1:2"], "category": 2, "answer": "2022"},
        {
            "question": "Who did Caroline talk to after the meeting?",
            "evidence": ["D1:3"],
            "category": 4,
            "answer": "Melanie",
        },
        {
            "question": "What broke out between the two groups?",
            "evidence": ["D1:4"],
            "category": 1,
            "answer": "a dispute",
        },
    ]
    vectors = embedder.embed_documents([item["question"] for item in questions])
    classifiers = build_classifiers(embedder)
    routings = route_questions(questions, vectors, classifiers)
    report = build_report(
        routings=routings,
        classifier_names=list(classifiers.keys()),
        embed_seconds=0.1,
        route_seconds=0.01,
        sample_size=10,
        seed=7,
    )

    by_name = {summary.name: summary for summary in report.classifiers}

    # "What broke out..." is default ENTITY in keywords_only, but WHY by prototype in keywords_plus_prototypes.
    assert by_name["keywords_only"].overall.why == 1
    assert by_name["keywords_only"].overall.when == 1
    assert by_name["keywords_only"].overall.entity == 2
    assert by_name["keywords_only"].default_matches == 1

    assert by_name["keywords_plus_prototypes"].overall.why == 2
    assert by_name["keywords_plus_prototypes"].prototype_matches == 1
    assert len(report.prototype_additions) == 1
    assert report.prototype_additions[0].question == "What broke out between the two groups?"

    # "Who did Caroline talk to after the meeting?" has "after" (WHEN in official_keywords, before "who").
    assert len(report.official_traps) == 1
    assert report.official_traps[0].first_intent == "WHEN"
    assert report.official_traps[0].second_intent == "ENTITY"


class CountingLabeler:
    """Says WHY for questions containing "broke out", ENTITY otherwise; counts the questions it labels."""

    def __init__(self) -> None:
        self.labeled = 0

    @property
    def name(self) -> str:
        return "counting"

    def label(self, questions: Sequence[str], descriptions: IntentDescriptions) -> list[LabeledIntent]:
        del descriptions
        self.labeled += len(questions)
        return [LabeledIntent(Intent.WHY if "broke out" in q else Intent.ENTITY, 0.9) for q in questions]


def test_model_classifiers_share_one_labeling_pass() -> None:
    embedder = ThreeAxisEmbedder()
    questions: list[LocomoQuestion] = [
        {"question": "When did Melanie start painting?", "evidence": ["D1:1"], "category": 2, "answer": "2022"},
        {"question": "What broke out between the two groups?", "evidence": ["D1:2"], "category": 1, "answer": "a fight"},
        {"question": "What broke out between the two groups?", "evidence": ["D1:3"], "category": 1, "answer": "a fight"},
    ]
    texts = [item["question"] for item in questions]
    counting = CountingLabeler()
    labeler = PrelabeledLabeler(counting, texts, DEFAULT.router.intent_descriptions)
    classifiers = build_classifiers(embedder, labeler)
    routings = route_questions(questions, embedder.embed_documents(texts), classifiers)
    report = build_report(routings, list(classifiers), embed_seconds=0.1, route_seconds=0.01, intent_model="counting")

    by_name = {summary.name: summary for summary in report.classifiers}
    assert counting.labeled == 2  # each distinct question once, in one batch
    assert by_name["keywords_plus_model"].keyword_matches == 1
    assert by_name["keywords_plus_model"].model_matches == 2
    assert by_name["keywords_plus_model"].overall.why == 2
    assert by_name["model_only"].overall.entity == 1  # "When did..." goes to the model too, which says ENTITY here
    assert len(report.model_additions) == 2
