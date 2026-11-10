"""Unit tests for Stage 1 query routing (`magma/router.py`)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pytest

from magma.config import DEFAULT, Intent
from magma.embed import Matrix, Vector, normalize
from magma.router import (
    IntentMatch,
    KeywordClassifier,
    KeywordPrototypeClassifier,
    PrototypeClassifier,
    Router,
)
from magma.temporal import DateparserTagger, TimeWindow


class AxisEmbedder:
    """Maps each text to a 3D unit vector along the axis of its dominant cue word."""

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


def test_keyword_classifier_matches_explicit_interrogatives() -> None:
    classifier = KeywordClassifier()
    zero = np.zeros(3, dtype=np.float32)

    assert classifier.classify("Why did Caroline go to the group?", zero) == IntentMatch(
        intent=Intent.WHY,
        reason='keyword "why"',
    )
    assert classifier.classify("What caused the delay in the project?", zero) == IntentMatch(
        intent=Intent.WHY,
        reason='keyword "what caused"',
    )
    assert classifier.classify("When did Melanie start painting?", zero) == IntentMatch(
        intent=Intent.WHEN,
        reason='keyword "when"',
    )
    assert classifier.classify("What year did they first meet?", zero) == IntentMatch(
        intent=Intent.WHEN,
        reason='keyword "what year"',
    )
    assert classifier.classify("Who helped Caroline with the research?", zero) == IntentMatch(
        intent=Intent.ENTITY,
        reason='keyword "who"',
    )


def test_keyword_classifier_requires_whole_words_and_avoids_mid_sentence_traps() -> None:
    classifier = KeywordClassifier()
    zero = np.zeros(3, dtype=np.float32)

    # "who" is inside "whole", and "after" in the middle of a "Who..." question must not turn it into WHEN.
    assert classifier.classify("What was the whole story?", zero) == IntentMatch(
        intent=Intent.ENTITY,
        reason="default ENTITY",
    )
    assert classifier.classify("Who did Caroline talk to after the meeting?", zero) == IntentMatch(
        intent=Intent.ENTITY,
        reason='keyword "who"',
    )
    assert classifier.classify("Did they have a good time at the lake?", zero) == IntentMatch(
        intent=Intent.ENTITY,
        reason="default ENTITY",
    )


def test_prototype_classifier_picks_the_closest_prototype_in_query_space() -> None:
    embedder = AxisEmbedder()
    classifier = PrototypeClassifier(embedder)

    why_question = "What broke out between the factions?"
    when_question = "In which era did the dynasty fall?"
    entity_question = "What instrument does Melanie play?"

    why_match = classifier.classify(why_question, embedder.embed_query(why_question))
    when_match = classifier.classify(when_question, embedder.embed_query(when_question))
    entity_match = classifier.classify(entity_question, embedder.embed_query(entity_question))

    assert why_match.intent is Intent.WHY
    assert why_match.reason.startswith('prototype "Why did this happen')
    assert when_match.intent is Intent.WHEN
    assert when_match.reason.startswith('prototype "When did this event')
    assert entity_match.intent is Intent.ENTITY
    assert entity_match.reason.startswith('prototype "Who is this person')


def test_hybrid_classifier_uses_keywords_first_and_prototypes_as_fallback() -> None:
    embedder = AxisEmbedder()
    classifier = KeywordPrototypeClassifier(embedder)

    explicit = "Why did Melanie choose the blue instrument?"
    paraphrased = "What broke out between the two groups?"

    explicit_match = classifier.classify(explicit, embedder.embed_query(explicit))
    paraphrased_match = classifier.classify(paraphrased, embedder.embed_query(paraphrased))

    assert explicit_match == IntentMatch(intent=Intent.WHY, reason='keyword "why"')
    assert paraphrased_match.intent is Intent.WHY
    assert paraphrased_match.reason.startswith("prototype ")


def test_router_builds_complete_stage1_output() -> None:
    embedder = AxisEmbedder()
    tagger = DateparserTagger()
    router = Router(embedder=embedder, tagger=tagger, classifier=KeywordPrototypeClassifier(embedder))
    reference = datetime(2023, 5, 20, 14, 0)

    routed = router.route("What instrument did Melanie play on 9 July, 2022?", reference_time=reference)

    assert routed.question == "What instrument did Melanie play on 9 July, 2022?"
    assert routed.intent is Intent.ENTITY
    assert routed.time_window == TimeWindow(start=datetime(2022, 7, 9), end=datetime(2022, 7, 10))
    assert len(routed.temporal_mentions) == 1
    assert routed.keywords == ("instrument", "melanie", "play", "9", "july", "2022")
    assert routed.query_vector.shape == (3,)
    assert pytest.approx(float(np.linalg.norm(routed.query_vector))) == 1.0


def test_router_without_reference_time_leaves_time_window_empty() -> None:
    embedder = AxisEmbedder()
    tagger = DateparserTagger()
    router = Router(embedder=embedder, tagger=tagger, classifier=KeywordPrototypeClassifier(embedder))

    routed = router.route("When did Caroline go to the support group yesterday?")

    assert routed.intent is Intent.WHEN
    assert routed.time_window is None
    assert routed.temporal_mentions == ()
