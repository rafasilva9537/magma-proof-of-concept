"""Measure Stages 2-4 of the query on a conversation already built by the runner.

Every question of the conversation goes through the router (Stage 1) and the
retrieval (Stages 2-4) over the saved graph; no LLM is called. For each
question the report says where its evidence turns ended up: in the time
channel (with τ only and with τ or the mentioned dates), among the anchors,
among the visited nodes and in the written context. It also counts how often
the drop threshold of the beam search fires, for events and for entities, the
cosine between the questions and the stored nodes, and the size of the
context.

    python -m bench.measure_retrieval --locomo MAGMA-original-repo/data/locomo10.json \
        --memory results/magma-sample-build-seed7/memories/conv-30 --sample conv-30

The memory folder holds the graph.json and vectors.npz written by the runner.
Raw data goes to runs/measurements/retrieval/<sample>/: report.json and one line per question in questions.jsonl.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from bench.benchmark import Question
from bench.locomo import LocomoSample, find_sample, load_samples, sessions, to_conversation
from magma.bm25_index import BM25Index
from magma.config import DEFAULT, Config
from magma.embed import InstructedEmbedder
from magma.faiss_index import FaissVectorIndex
from magma.graph import GraphStore, load_graph
from magma.index import load_vectors
from magma.networkx_graph import NetworkXGraphStore
from magma.retrieve import NodeVectors, Retriever, events_in_window
from magma.router import Router, RouterOutput, build_default_classifier
from magma.schema import MAGMA_PROFILE, MagmaNode
from magma.temporal import DateparserTagger

PERCENTILES = (1, 5, 25, 50, 75, 95)


@dataclass(frozen=True)
class LoadedMemory:
    graph: GraphStore
    vectors: FaissVectorIndex
    keywords: BM25Index


@dataclass(frozen=True)
class QuestionMeasure:
    question_id: str
    category: str
    intent: str
    has_window: bool
    evidence: tuple[str, ...]  # event ids of the evidence turns found in the graph
    missing_evidence: tuple[str, ...]  # dia_ids of the dataset with no matching turn
    window_events_tau: int  # events of the time channel with τ only
    window_events_dates: int  # events of the time channel with τ or a mentioned date
    evidence_in_window_tau: int
    evidence_in_window_dates: int
    evidence_in_anchors: int
    evidence_visited: int
    evidence_in_context: int
    visited_events: int
    visited_entities: int
    deepest_level: int
    transitions_checked: int
    dropped_events: int
    dropped_entities: int
    context_tokens: int
    omitted_events: int
    event_cosines: tuple[float, ...]  # cos(v_q, v) for every event of the graph
    entity_cosines: tuple[float, ...]  # cos(v_q, v) for every entity, with v the encoding of its name


@dataclass(frozen=True)
class Group:
    name: str
    questions: int
    with_evidence: int
    # Shares of the questions with evidence that have at least one evidence turn at each step.
    anchor_recall: float
    visit_recall: float
    context_recall: float
    found_by_traversal_only: float  # evidence visited, none of it among the anchors
    lost_to_budget: float  # evidence visited, none of it written in the context
    mean_visited_events: float
    mean_visited_entities: float
    mean_context_tokens: float
    share_with_omissions: float
    mean_deepest_level: float


@dataclass(frozen=True)
class TimeChannelCheck:
    """Questions with a time window and evidence: is an evidence turn in the time channel?"""

    questions: int
    with_tau_only: int
    with_tau_or_dates: int
    mean_window_events_tau: float
    mean_window_events_dates: float


@dataclass(frozen=True)
class DropCheck:
    transitions_checked: int
    dropped_events: int
    dropped_entities: int
    event_cosine_percentiles: dict[str, float]  # over every (question, event) pair
    entity_cosine_percentiles: dict[str, float]  # over every (question, entity) pair
    event_share_below: float  # share of (question, event) pairs under the drop threshold
    entity_share_below: float


@dataclass(frozen=True)
class RetrievalReport:
    sample: str
    events: int
    entities: int
    drop_threshold: float
    token_budget: int
    intents: dict[str, int]
    overall: Group
    by_category: list[Group]
    by_intent: list[Group]
    time_channel: TimeChannelCheck
    drops: DropCheck


def event_ids_by_dia_id(sample: LocomoSample) -> dict[str, str]:
    """The event id the fast path gave each turn (e1, e2, ... in conversation order), by LoCoMo dia_id."""
    ids: dict[str, str] = {}
    number = 0
    for session in sessions(sample):
        for turn in session.turns:
            number += 1
            ids[turn["dia_id"]] = f"e{number}"
    return ids


def load_memory(directory: Path, dimension: int) -> LoadedMemory:
    """The graph and vectors saved by the runner; BM25 is rebuilt from the event texts, as the fast path built it."""
    graph = NetworkXGraphStore(MAGMA_PROFILE)
    load_graph(directory / "graph.json", graph)
    vectors = FaissVectorIndex(dimension)
    load_vectors(directory / "vectors.npz", vectors)
    keywords = BM25Index()
    for node in graph.nodes(MagmaNode.EVENT):
        keywords.add(node.id, node.text)
    return LoadedMemory(graph=graph, vectors=vectors, keywords=keywords)


def measure_question(
    question: Question,
    ids: dict[str, str],
    route: RouterOutput,
    retriever: Retriever,
    graph: GraphStore,
    node_vectors: NodeVectors,
) -> QuestionMeasure:
    evidence = tuple(dict.fromkeys(ids[dia_id] for dia_id in question.evidence if dia_id in ids))
    missing = tuple(dia_id for dia_id in question.evidence if dia_id not in ids)
    window_tau = {node.id for node in events_in_window(graph, route.time_window, uses_mentioned_dates=False)}
    window_dates = {node.id for node in events_in_window(graph, route.time_window, uses_mentioned_dates=True)}

    found = retriever.retrieve(route)
    anchors = {anchor.node_id for anchor in found.anchors.anchors}
    visited = {visit.node_id for visit in found.traversal.visits}
    entities = graph.nodes(MagmaNode.ENTITY)
    events = graph.nodes(MagmaNode.EVENT)
    dropped = found.traversal.dropped_by_node_type
    return QuestionMeasure(
        question_id=question.id,
        category=question.category,
        intent=route.intent.value,
        has_window=route.time_window is not None,
        evidence=evidence,
        missing_evidence=missing,
        window_events_tau=len(window_tau),
        window_events_dates=len(window_dates),
        evidence_in_window_tau=_count_in(evidence, window_tau),
        evidence_in_window_dates=_count_in(evidence, window_dates),
        evidence_in_anchors=_count_in(evidence, anchors),
        evidence_visited=_count_in(evidence, visited),
        evidence_in_context=_count_in(evidence, set(found.context.included)),
        visited_events=sum(1 for node_id in visited if graph.node(node_id).type == MagmaNode.EVENT),
        visited_entities=sum(1 for node_id in visited if graph.node(node_id).type == MagmaNode.ENTITY),
        deepest_level=max(visit.depth for visit in found.traversal.visits) if found.traversal.visits else 0,
        transitions_checked=found.traversal.transitions_checked,
        dropped_events=dropped.get(MagmaNode.EVENT.value, 0),
        dropped_entities=dropped.get(MagmaNode.ENTITY.value, 0),
        context_tokens=found.context.tokens,
        omitted_events=len(found.context.omitted),
        event_cosines=tuple(node_vectors.similarities(events, route.query_vector).values()),
        entity_cosines=tuple(node_vectors.similarities(entities, route.query_vector).values()),
    )


def summarize(
    sample: str,
    measures: list[QuestionMeasure],
    graph: GraphStore,
    config: Config = DEFAULT,
) -> RetrievalReport:
    drop_threshold = config.traversal.drop_threshold
    categories = sorted({measure.category for measure in measures})
    intents = sorted({measure.intent for measure in measures})
    timed = [measure for measure in measures if measure.has_window and measure.evidence]
    event_cosines = [value for measure in measures for value in measure.event_cosines]
    entity_cosines = [value for measure in measures for value in measure.entity_cosines]
    return RetrievalReport(
        sample=sample,
        events=len(graph.nodes(MagmaNode.EVENT)),
        entities=len(graph.nodes(MagmaNode.ENTITY)),
        drop_threshold=drop_threshold,
        token_budget=config.context.token_budget,
        intents=dict(Counter(measure.intent for measure in measures)),
        overall=_group("OVERALL", measures),
        by_category=[_group(name, [m for m in measures if m.category == name]) for name in categories],
        by_intent=[_group(name, [m for m in measures if m.intent == name]) for name in intents],
        time_channel=TimeChannelCheck(
            questions=len(timed),
            with_tau_only=sum(1 for measure in timed if measure.evidence_in_window_tau > 0),
            with_tau_or_dates=sum(1 for measure in timed if measure.evidence_in_window_dates > 0),
            mean_window_events_tau=_mean([measure.window_events_tau for measure in timed]),
            mean_window_events_dates=_mean([measure.window_events_dates for measure in timed]),
        ),
        drops=DropCheck(
            transitions_checked=sum(measure.transitions_checked for measure in measures),
            dropped_events=sum(measure.dropped_events for measure in measures),
            dropped_entities=sum(measure.dropped_entities for measure in measures),
            event_cosine_percentiles=_percentiles(event_cosines),
            entity_cosine_percentiles=_percentiles(entity_cosines),
            event_share_below=_share_below(event_cosines, drop_threshold),
            entity_share_below=_share_below(entity_cosines, drop_threshold),
        ),
    )


def _group(name: str, measures: list[QuestionMeasure]) -> Group:
    with_evidence = [measure for measure in measures if measure.evidence]
    return Group(
        name=name,
        questions=len(measures),
        with_evidence=len(with_evidence),
        anchor_recall=_share([measure.evidence_in_anchors > 0 for measure in with_evidence]),
        visit_recall=_share([measure.evidence_visited > 0 for measure in with_evidence]),
        context_recall=_share([measure.evidence_in_context > 0 for measure in with_evidence]),
        found_by_traversal_only=_share(
            [measure.evidence_visited > 0 and measure.evidence_in_anchors == 0 for measure in with_evidence]
        ),
        lost_to_budget=_share(
            [measure.evidence_visited > 0 and measure.evidence_in_context == 0 for measure in with_evidence]
        ),
        mean_visited_events=_mean([measure.visited_events for measure in measures]),
        mean_visited_entities=_mean([measure.visited_entities for measure in measures]),
        mean_context_tokens=_mean([measure.context_tokens for measure in measures]),
        share_with_omissions=_share([measure.omitted_events > 0 for measure in measures]),
        mean_deepest_level=_mean([measure.deepest_level for measure in measures]),
    )


def _count_in(evidence: tuple[str, ...], found: set[str]) -> int:
    return sum(1 for node_id in evidence if node_id in found)


def _mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _share(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def _share_below(values: list[float], threshold: float) -> float:
    return sum(1 for value in values if value < threshold) / len(values) if values else 0.0


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    computed = np.percentile(np.asarray(values, dtype=np.float64), PERCENTILES)
    return {f"p{p}": round(float(value), 4) for p, value in zip(PERCENTILES, computed, strict=True)}


def print_summary(report: RetrievalReport) -> None:
    print(f"{report.sample}: {report.events} events, {report.entities} entities; intents {report.intents}")
    print(
        "group        questions  evidence  anchors  visited  context  by-beam  lost  events  entities  depth"
        "  tokens  omissions"
    )
    for group in [report.overall, *report.by_category, *report.by_intent]:
        print(
            f"{group.name:<12} {group.questions:>9} {group.with_evidence:>9} {group.anchor_recall:>8.2f}"
            f" {group.visit_recall:>8.2f} {group.context_recall:>8.2f} {group.found_by_traversal_only:>8.2f}"
            f" {group.lost_to_budget:>5.2f} {group.mean_visited_events:>7.1f} {group.mean_visited_entities:>9.1f}"
            f" {group.mean_deepest_level:>6.1f} {group.mean_context_tokens:>7.0f} {group.share_with_omissions:>10.2f}"
        )
    time_channel = report.time_channel
    print(
        f"time channel: {time_channel.questions} questions with a window and evidence; evidence in the channel with"
        f" tau only {time_channel.with_tau_only}, with tau or dates {time_channel.with_tau_or_dates};"
        f" mean events {time_channel.mean_window_events_tau:.1f} / {time_channel.mean_window_events_dates:.1f}"
    )
    drops = report.drops
    print(
        f"drop threshold {report.drop_threshold}: {drops.dropped_events} events and {drops.dropped_entities} entities"
        f" dropped of {drops.transitions_checked} steps; pairs below: events {drops.event_share_below:.4f},"
        f" entities {drops.entity_share_below:.4f}"
    )
    print(f"event cosines {drops.event_cosine_percentiles}")
    print(f"entity cosines {drops.entity_cosine_percentiles}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--locomo", type=Path, required=True, help="path to locomo10.json")
    parser.add_argument("--memory", type=Path, required=True, help="folder with the graph.json and vectors.npz")
    parser.add_argument("--sample", default="conv-30", help="LoCoMo conversation id of the saved memory")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/measurements/retrieval"))
    args = parser.parse_args()

    sample = find_sample(load_samples(args.locomo), args.sample)
    conversation = to_conversation(sample)
    ids = event_ids_by_dia_id(sample)
    embedder = InstructedEmbedder.from_config()
    memory = load_memory(args.memory, embedder.dimension)
    event_count = len(memory.graph.nodes(MagmaNode.EVENT))
    if event_count != len(ids):
        raise SystemExit(f"{args.memory} has {event_count} events, {args.sample} has {len(ids)} turns")

    tagger = DateparserTagger()
    router = Router(embedder, tagger, build_default_classifier(tagger))
    node_vectors = NodeVectors(memory.graph, memory.vectors, embedder)
    retriever = Retriever.create(memory.graph, memory.vectors, memory.keywords, embedder, embedder)
    reference_time: datetime = conversation.reference_time
    measures = [
        measure_question(
            question, ids, router.route(question.question, reference_time), retriever, memory.graph, node_vectors
        )
        for question in conversation.questions
    ]
    report = summarize(args.sample, measures, memory.graph)

    output = args.output_dir / args.sample
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "questions.jsonl").write_text(
        "".join(json.dumps(asdict(measure), ensure_ascii=False) + "\n" for measure in measures), encoding="utf-8"
    )
    print_summary(report)
    print(f"raw data in {output}")


if __name__ == "__main__":
    main()
