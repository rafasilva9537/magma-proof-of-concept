"""Query analysis (Stage 1 of Alg. 1, §3.3): intent T_q, time window [τs, τe], query vector v_q and keywords q_key.

The intent T_q ∈ {WHY, WHEN, ENTITY} chooses the edge-weight vector w_Tq for the
beam search. Intent classification is separated behind `IntentClassifier`, so
keyword rules, embedding prototypes, a hybrid of both, or a dedicated model can
be swapped without touching the rest of Stage 1. By default a decision model
reads the question with its date expressions taken out: the dates already set
the time window, and the intent says what the question asks for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from magma.config import DEFAULT, Intent, IntentPhrases, RouterConfig, TemporalConfig
from magma.embed import Embedder, Vector, cosine
from magma.index import tokenize
from magma.intent_model import GlinerLabeler, IntentLabeler
from magma.temporal import TemporalMention, TemporalTagger, TimeWindow, query_window

_INTENT_ORDER: tuple[Intent, ...] = (Intent.WHY, Intent.WHEN, Intent.ENTITY)


@dataclass(frozen=True)
class IntentMatch:
    """The chosen intent and how the classifier arrived at it (shown by `/why`)."""

    intent: Intent
    reason: str


@dataclass(frozen=True)
class RouterOutput:
    """Everything Stage 1 produces for Stage 2 (RRF), Stage 3 (beam) and Stage 4 (linearization)."""

    question: str
    intent: Intent
    intent_reason: str
    time_window: TimeWindow | None
    temporal_mentions: tuple[TemporalMention, ...]
    query_vector: Vector
    keywords: tuple[str, ...]


class IntentClassifier(Protocol):
    """Maps a question (and its already-computed vector v_q) to T_q ∈ {WHY, WHEN, ENTITY}."""

    def classify(self, question: str, query_vector: Vector) -> IntentMatch: ...


class QueryRouter(Protocol):
    """Stage 1 of Alg. 1: analyses a query into intent, time window, dense vector and sparse keywords."""

    def route(self, question: str, reference_time: datetime | None = None) -> RouterOutput: ...


class KeywordClassifier:
    """Matches whole-word interrogative phrases in order (WHY, WHEN, ENTITY), with a fixed fallback intent."""

    def __init__(self, config: RouterConfig = DEFAULT.router) -> None:
        self._patterns = _compile_phrases(config.keywords)
        self._default_intent = config.default_intent

    def match(self, question: str) -> IntentMatch | None:
        """The first matching keyword phrase, or None when no phrase appears as whole words."""
        lowered = question.lower()
        for intent, phrase, pattern in self._patterns:
            if pattern.search(lowered):
                return IntentMatch(intent=intent, reason=f'keyword "{phrase}"')
        return None

    def classify(self, question: str, query_vector: Vector) -> IntentMatch:
        del query_vector
        found = self.match(question)
        if found is not None:
            return found
        return IntentMatch(intent=self._default_intent, reason=f"default {self._default_intent}")


class PrototypeClassifier:
    """Picks the intent whose prototype question is closest in cosine similarity to `query_vector`."""

    def __init__(self, embedder: Embedder, config: RouterConfig = DEFAULT.router) -> None:
        # Prototypes are questions too, so they are embedded with `embed_query` in the same query space as v_q.
        self._prototypes: list[tuple[Intent, str, Vector]] = []
        for intent in _INTENT_ORDER:
            for text in config.prototypes.of(intent):
                self._prototypes.append((intent, text, embedder.embed_query(text)))

    def classify(self, question: str, query_vector: Vector) -> IntentMatch:
        del question
        best_intent, best_text, best_vector = self._prototypes[0]
        best_score = cosine(query_vector, best_vector)
        for intent, text, vector in self._prototypes[1:]:
            score = cosine(query_vector, vector)
            if score > best_score:
                best_intent = intent
                best_text = text
                best_score = score
        return IntentMatch(intent=best_intent, reason=f'prototype "{best_text}" (cosine {best_score:.3f})')


class KeywordPrototypeClassifier:
    """Explicit interrogative keywords first; when none matches, nearest prototype in the embedding space."""

    def __init__(self, embedder: Embedder, config: RouterConfig = DEFAULT.router) -> None:
        self._keywords = KeywordClassifier(config)
        self._prototypes = PrototypeClassifier(embedder, config)

    def classify(self, question: str, query_vector: Vector) -> IntentMatch:
        found = self._keywords.match(question)
        if found is not None:
            return found
        return self._prototypes.classify(question, query_vector)


class ModelClassifier:
    """Asks a decision model which intent description fits the question."""

    def __init__(self, labeler: IntentLabeler, config: RouterConfig = DEFAULT.router) -> None:
        self._labeler = labeler
        self._descriptions = config.intent_descriptions

    def classify(self, question: str, query_vector: Vector) -> IntentMatch:
        del query_vector
        labeled = self._labeler.label([question], self._descriptions)[0]
        return IntentMatch(
            intent=labeled.intent,
            reason=f"model {self._labeler.name} (confidence {labeled.confidence:.3f})",
        )


class KeywordModelClassifier:
    """Explicit interrogative keywords first; when none matches, the decision model."""

    def __init__(self, labeler: IntentLabeler, config: RouterConfig = DEFAULT.router) -> None:
        self._keywords = KeywordClassifier(config)
        self._model = ModelClassifier(labeler, config)

    def classify(self, question: str, query_vector: Vector) -> IntentMatch:
        found = self._keywords.match(question)
        if found is not None:
            return found
        return self._model.classify(question, query_vector)


class DatesRemovedClassifier:
    """Classifies the question with its date expressions taken out.

    A date given as context ("Which city did John recommend in January 2024?")
    narrows the time window, which the temporal tagger already handles; it does
    not make the question ask for a time.
    """

    def __init__(
        self,
        inner: IntentClassifier,
        tagger: TemporalTagger,
        temporal: TemporalConfig = DEFAULT.temporal,
    ) -> None:
        self._inner = inner
        self._tagger = tagger
        self._temporal = temporal

    def classify(self, question: str, query_vector: Vector) -> IntentMatch:
        stripped = remove_date_expressions(question, self._tagger, self._temporal)
        match = self._inner.classify(stripped, query_vector)
        if stripped == question:
            return match
        return IntentMatch(intent=match.intent, reason=f"{match.reason}, dates removed")


# Only the text of each date expression is used, never its resolved date, so any fixed reference works;
# a fixed one keeps the result independent of the day the question is asked.
_DATE_DETECTION_REFERENCE = datetime(2000, 1, 1)


def remove_date_expressions(question: str, tagger: TemporalTagger, temporal: TemporalConfig = DEFAULT.temporal) -> str:
    """The question without the date expressions the tagger finds in it ("on May 8, 2022"), spaces collapsed."""
    stripped = question
    for mention in tagger.find(question, _DATE_DETECTION_REFERENCE, temporal.question_preference):
        stripped = stripped.replace(mention.text, " ")
    stripped = re.sub(r"\s+([?.,!])", r"\1", stripped)
    return re.sub(r"\s{2,}", " ", stripped).strip()


def build_default_classifier(
    tagger: TemporalTagger,
    config: RouterConfig = DEFAULT.router,
    temporal: TemporalConfig = DEFAULT.temporal,
) -> IntentClassifier:
    """The intent classifier used by default: the decision model, on the question without its dates."""
    labeler = GlinerLabeler(config.intent_model)
    return DatesRemovedClassifier(ModelClassifier(labeler, config), tagger, temporal)


class Router:
    """Stage 1 query analyzer."""

    def __init__(
        self,
        embedder: Embedder,
        tagger: TemporalTagger,
        classifier: IntentClassifier,
        temporal: TemporalConfig = DEFAULT.temporal,
    ) -> None:
        self._embedder = embedder
        self._tagger = tagger
        self._classifier = classifier
        self._temporal = temporal

    def route(self, question: str, reference_time: datetime | None = None) -> RouterOutput:
        query_vector = self._embedder.embed_query(question)
        match = self._classifier.classify(question, query_vector)
        mentions = (
            self._tagger.find(question, reference_time, self._temporal.question_preference)
            if reference_time is not None
            else []
        )
        return RouterOutput(
            question=question,
            intent=match.intent,
            intent_reason=match.reason,
            time_window=query_window(mentions),
            temporal_mentions=tuple(mentions),
            query_vector=query_vector,
            keywords=tuple(tokenize(question)),
        )


def _compile_phrases(phrases: IntentPhrases) -> list[tuple[Intent, str, re.Pattern[str]]]:
    compiled: list[tuple[Intent, str, re.Pattern[str]]] = []
    for intent in _INTENT_ORDER:
        for phrase in phrases.of(intent):
            words = r"\s+".join(re.escape(part) for part in phrase.lower().split())
            compiled.append((intent, phrase, re.compile(rf"\b{words}\b")))
    return compiled
