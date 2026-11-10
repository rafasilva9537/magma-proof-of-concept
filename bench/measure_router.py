"""Compare Stage 1 intent classifiers on the 1,986 questions of the LoCoMo dataset.

Eight classifiers are compared on the same questions and query vectors:
- `keywords_only`: explicit interrogative phrases (WHY, WHEN, ENTITY) + default ENTITY;
- `official_keywords`: the phrase lists of `detect_query_intent` in the reference repository;
- `keywords_plus_prototypes`: explicit interrogative phrases first, then nearest prototype in Qwen3-Embedding;
- `prototypes_only`: nearest prototype in Qwen3-Embedding for every question;
- `keywords_plus_model`: explicit interrogative phrases first, then a decision model (`--intent-model`);
- `model_only`: the decision model for every question;
- `keywords_plus_model_no_dates` and `model_only_no_dates`: the same two, with the date
  expressions found by the temporal tagger taken out of the question before classifying.

The report gives the intent counts overall and by LoCoMo category (category 2 is
temporal; category 4 is single-hop factual), how often the hybrid classifier
uses keywords vs prototypes vs the model, and random samples of disagreements
for inspection.

    python -m bench.measure_router --locomo MAGMA-original-repo/data/locomo10.json
    python -m bench.measure_router --intent-model fastino/GLiNER2.5-multi-Decide --out runs/measurements/router_intents_multi.json

The full report is written as JSON to runs/measurements/router_intents.json
(`--no-model` leaves the two model classifiers out).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from bench.locomo import LocomoQuestion, load_samples
from magma.config import DEFAULT, Intent, IntentDescriptions, IntentPhrases, RouterConfig
from magma.embed import Embedder, InstructedEmbedder, Matrix, Vector
from magma.intent_model import GlinerLabeler, IntentLabeler, LabeledIntent
from magma.router import (
    DatesRemovedClassifier,
    IntentClassifier,
    IntentMatch,
    KeywordClassifier,
    KeywordModelClassifier,
    KeywordPrototypeClassifier,
    ModelClassifier,
    PrototypeClassifier,
    remove_date_expressions,
)
from magma.temporal import DateparserTagger, TemporalTagger

# The literal phrase lists of `QueryEngine.detect_query_intent` (`query_engine.py:232-242`).
OFFICIAL_KEYWORDS = IntentPhrases(
    why=("why", "because", "cause", "reason", "lead to", "result"),
    when=("when", "time", "date", "before", "after", "during", "while"),
    entity=("who", "whom", "whose", "which person", "which entity"),
)


@dataclass(frozen=True)
class IntentBreakdown:
    why: int
    when: int
    entity: int

    @property
    def total(self) -> int:
        return self.why + self.when + self.entity


@dataclass(frozen=True)
class CategoryIntentRow:
    category: int
    questions: int
    intents: IntentBreakdown


@dataclass(frozen=True)
class ClassifierSummary:
    name: str
    overall: IntentBreakdown
    by_category: list[CategoryIntentRow]
    keyword_matches: int
    prototype_matches: int
    model_matches: int
    default_matches: int


@dataclass(frozen=True)
class DisagreementExample:
    category: int
    question: str
    gold_answer: str
    first_intent: str
    first_reason: str
    second_intent: str
    second_reason: str


@dataclass(frozen=True)
class QuestionRouting:
    category: int
    question: str
    gold_answer: str
    matches: dict[str, IntentMatch]


@dataclass(frozen=True)
class RouterReport:
    questions: int
    intent_model: str | None
    embed_seconds: float
    label_seconds: float
    route_seconds: float
    classifiers: list[ClassifierSummary]
    prototype_additions: list[DisagreementExample]
    official_traps: list[DisagreementExample]
    model_additions: list[DisagreementExample]
    model_vs_prototypes: list[DisagreementExample]
    dates_removed_changes: list[DisagreementExample]


class PrelabeledLabeler:
    """Labels every question once in batches, then answers from memory (the model classifiers share it)."""

    def __init__(self, labeler: IntentLabeler, questions: Sequence[str], descriptions: IntentDescriptions) -> None:
        self._name = labeler.name
        unique = list(dict.fromkeys(questions))
        self._labels = dict(zip(unique, labeler.label(unique, descriptions), strict=True))

    @property
    def name(self) -> str:
        return self._name

    def label(self, questions: Sequence[str], descriptions: IntentDescriptions) -> list[LabeledIntent]:
        del descriptions  # fixed when the questions were labeled
        return [self._labels[question] for question in questions]


def build_classifiers(
    embedder: Embedder,
    labeler: IntentLabeler | None = None,
    tagger: TemporalTagger | None = None,
    config: RouterConfig = DEFAULT.router,
) -> dict[str, IntentClassifier]:
    official_config = RouterConfig(
        keywords=OFFICIAL_KEYWORDS,
        prototypes=config.prototypes,
        default_intent=config.default_intent,
    )
    classifiers: dict[str, IntentClassifier] = {
        "keywords_only": KeywordClassifier(config),
        "official_keywords": KeywordClassifier(official_config),
        "keywords_plus_prototypes": KeywordPrototypeClassifier(embedder, config),
        "prototypes_only": PrototypeClassifier(embedder, config),
    }
    if labeler is not None:
        classifiers["keywords_plus_model"] = KeywordModelClassifier(labeler, config)
        classifiers["model_only"] = ModelClassifier(labeler, config)
        if tagger is not None:
            classifiers["keywords_plus_model_no_dates"] = DatesRemovedClassifier(
                KeywordModelClassifier(labeler, config), tagger
            )
            classifiers["model_only_no_dates"] = DatesRemovedClassifier(ModelClassifier(labeler, config), tagger)
    return classifiers


def route_questions(
    questions: list[LocomoQuestion],
    vectors: Matrix,
    classifiers: dict[str, IntentClassifier],
) -> list[QuestionRouting]:
    routings: list[QuestionRouting] = []
    for index, item in enumerate(questions):
        text = item["question"]
        vector: Vector = vectors[index]
        gold = str(item.get("answer", item.get("adversarial_answer", "")))
        matches = {name: classifier.classify(text, vector) for name, classifier in classifiers.items()}
        routings.append(
            QuestionRouting(
                category=item["category"],
                question=text,
                gold_answer=gold,
                matches=matches,
            )
        )
    return routings


def summarize_classifier(name: str, routings: list[QuestionRouting]) -> ClassifierSummary:
    overall_counts: Counter[Intent] = Counter()
    by_cat: dict[int, Counter[Intent]] = {}
    keyword_matches = 0
    prototype_matches = 0
    model_matches = 0
    default_matches = 0

    for item in routings:
        match = item.matches[name]
        overall_counts[match.intent] += 1
        cat_counter = by_cat.setdefault(item.category, Counter())
        cat_counter[match.intent] += 1
        if match.reason.startswith("keyword "):
            keyword_matches += 1
        elif match.reason.startswith("prototype "):
            prototype_matches += 1
        elif match.reason.startswith("model "):
            model_matches += 1
        else:
            default_matches += 1

    categories = [
        CategoryIntentRow(
            category=cat,
            questions=sum(by_cat[cat].values()),
            intents=_breakdown(by_cat[cat]),
        )
        for cat in sorted(by_cat)
    ]
    return ClassifierSummary(
        name=name,
        overall=_breakdown(overall_counts),
        by_category=categories,
        keyword_matches=keyword_matches,
        prototype_matches=prototype_matches,
        model_matches=model_matches,
        default_matches=default_matches,
    )


def _breakdown(counts: Counter[Intent]) -> IntentBreakdown:
    return IntentBreakdown(
        why=counts[Intent.WHY],
        when=counts[Intent.WHEN],
        entity=counts[Intent.ENTITY],
    )


def sample_disagreements(
    routings: list[QuestionRouting],
    first: str,
    second: str,
    sample_size: int,
    seed: int,
) -> list[DisagreementExample]:
    diffs = [
        DisagreementExample(
            category=item.category,
            question=item.question,
            gold_answer=item.gold_answer,
            first_intent=item.matches[first].intent.value,
            first_reason=item.matches[first].reason,
            second_intent=item.matches[second].intent.value,
            second_reason=item.matches[second].reason,
        )
        for item in routings
        if item.matches[first].intent != item.matches[second].intent
    ]
    return random.Random(seed).sample(diffs, min(sample_size, len(diffs)))


def build_report(
    routings: list[QuestionRouting],
    classifier_names: list[str],
    embed_seconds: float,
    route_seconds: float,
    intent_model: str | None = None,
    label_seconds: float = 0.0,
    sample_size: int = 20,
    seed: int = 7,
) -> RouterReport:
    summaries = [summarize_classifier(name, routings) for name in classifier_names]
    prototype_additions = sample_disagreements(
        routings,
        first="keywords_only",
        second="keywords_plus_prototypes",
        sample_size=sample_size,
        seed=seed,
    )
    official_traps = sample_disagreements(
        routings,
        first="official_keywords",
        second="keywords_only",
        sample_size=sample_size,
        seed=seed,
    )
    has_model = "keywords_plus_model" in classifier_names
    model_additions = (
        sample_disagreements(routings, "keywords_only", "keywords_plus_model", sample_size, seed) if has_model else []
    )
    model_vs_prototypes = (
        sample_disagreements(routings, "keywords_plus_prototypes", "keywords_plus_model", sample_size, seed)
        if has_model
        else []
    )
    has_dates_removed = "keywords_plus_model_no_dates" in classifier_names
    dates_removed_changes = (
        sample_disagreements(routings, "keywords_plus_model", "keywords_plus_model_no_dates", sample_size, seed)
        if has_dates_removed
        else []
    )
    return RouterReport(
        questions=len(routings),
        intent_model=intent_model,
        embed_seconds=round(embed_seconds, 2),
        label_seconds=round(label_seconds, 2),
        route_seconds=round(route_seconds, 3),
        classifiers=summaries,
        prototype_additions=prototype_additions,
        official_traps=official_traps,
        model_additions=model_additions,
        model_vs_prototypes=model_vs_prototypes,
        dates_removed_changes=dates_removed_changes,
    )


def print_summary(report: RouterReport) -> None:
    print(
        f"Questions: {report.questions} | model {report.intent_model} | "
        f"embed {report.embed_seconds:.2f}s | "
        f"label {report.label_seconds:.2f}s | "
        f"route {report.route_seconds:.3f}s"
    )
    for summary in report.classifiers:
        overall = summary.overall
        print(
            f"\n=== {summary.name} ===\n"
            f"  Overall: WHY={overall.why} ({100 * overall.why / overall.total:.1f}%), "
            f"WHEN={overall.when} ({100 * overall.when / overall.total:.1f}%), "
            f"ENTITY={overall.entity} ({100 * overall.entity / overall.total:.1f}%)\n"
            f"  Sources: keyword={summary.keyword_matches}, "
            f"prototype={summary.prototype_matches}, "
            f"model={summary.model_matches}, "
            f"default={summary.default_matches}"
        )
        for row in summary.by_category:
            b = row.intents
            print(
                f"  cat {row.category} (n={row.questions:3d}): "
                f"WHY={b.why:3d} ({100 * b.why / row.questions:4.1f}%) | "
                f"WHEN={b.when:3d} ({100 * b.when / row.questions:4.1f}%) | "
                f"ENTITY={b.entity:3d} ({100 * b.entity / row.questions:4.1f}%)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Stage 1 intent classifiers on LoCoMo questions.")
    parser.add_argument("--locomo", type=Path, default=Path("MAGMA-original-repo/data/locomo10.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/measurements/router_intents.json"))
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--intent-model", default=DEFAULT.router.intent_model)
    parser.add_argument("--no-model", action="store_true", help="leave the decision-model classifiers out")
    args = parser.parse_args()

    samples = load_samples(args.locomo)
    questions = [question for sample in samples for question in sample["qa"]]
    texts = [question["question"] for question in questions]

    embedder = InstructedEmbedder.from_config(DEFAULT.embedding)
    start_embed = time.perf_counter()
    vectors = embedder.embed_queries(texts)
    embed_seconds = time.perf_counter() - start_embed

    tagger = DateparserTagger()
    labeler: IntentLabeler | None = None
    label_seconds = 0.0
    if not args.no_model:
        model = GlinerLabeler(args.intent_model)
        start_label = time.perf_counter()
        without_dates = [remove_date_expressions(text, tagger) for text in texts]
        labeler = PrelabeledLabeler(model, texts + without_dates, DEFAULT.router.intent_descriptions)
        label_seconds = time.perf_counter() - start_label

    start_route = time.perf_counter()
    classifiers = build_classifiers(embedder, labeler, tagger)
    routings = route_questions(questions, vectors, classifiers)
    route_seconds = time.perf_counter() - start_route

    report = build_report(
        routings=routings,
        classifier_names=list(classifiers.keys()),
        embed_seconds=embed_seconds,
        route_seconds=route_seconds,
        intent_model=None if args.no_model else args.intent_model,
        label_seconds=label_seconds,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(report)


if __name__ == "__main__":
    main()
