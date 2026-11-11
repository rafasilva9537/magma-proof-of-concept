"""Measure the slow path (Alg. 3) on one LoCoMo conversation, with the real LLM and embedding model.

Every turn goes through the fast path, then the queue is drained by the slow
path, as the benchmark does before asking questions. The report counts what
consolidation adds (entities, resolved dates, causal edges), how large the
2-hop neighborhood shown to the causal prompt gets, where in that list the
chosen causes sit, the tokens of each prompt and the failures. A random sample
of causal edges is kept for checking by hand.

    python -m bench.measure_consolidation --locomo MAGMA-original-repo/data/locomo10.json --sample conv-26

The causal prompt shows the candidates the config allows; --all-candidates
shows the whole neighborhood, and --candidate-limit another number. A run
replaces the folder of the same name, so runs to be compared take a --label:

    python -m bench.measure_consolidation --locomo ... --sample conv-30 --all-candidates --label all-candidates

The LLM is read from the LLM_* variables (see .env.example); its calls are
cached, so a second run costs nothing. Raw data goes to
runs/measurements/consolidation/<sample>/: the report, the graph and the log of
every LLM call.
"""

from __future__ import annotations

import argparse
import json
import queue
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel

from bench.locomo import LocomoSample, find_sample, load_samples, session_time, sessions, turn_text
from magma.bm25_index import BM25Index
from magma.config import DEFAULT, IngestConfig
from magma.embed import Embedder, InstructedEmbedder
from magma.faiss_index import FaissVectorIndex
from magma.graph import GraphStore, save_graph
from magma.ingest import Consolidation, FastPathIngestion, SlowPathConsolidation, Turn, entity_key
from magma.llm import LLMClient
from magma.networkx_graph import NetworkXGraphStore
from magma.schema import MAGMA_PROFILE, MagmaNode
from magma.temporal import DateparserTagger

CONFIDENCE_BINS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)  # lower edges; δ = 0.5 keeps only what is above the first
RANK_CUTS = (5, 10, 20, 50)


@dataclass(frozen=True)
class Distribution:
    count: int
    mean: float
    median: float
    p90: float
    maximum: float


@dataclass(frozen=True)
class NamedCount:
    name: str
    count: int


@dataclass(frozen=True)
class ShareWithin:
    rank: int
    share: float  # of the causal edges whose cause was at this rank or better


@dataclass(frozen=True)
class MergedEntity:
    key: str
    forms: list[str]


@dataclass(frozen=True)
class TagUsage:
    tag: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency_s: float
    prompt_tokens_per_call: Distribution


@dataclass(frozen=True)
class CauseExample:
    cause_id: str
    cause_text: str
    effect_id: str
    effect_text: str
    confidence: float
    reason: str
    rank: int  # position of the cause in the candidate list, from 1


@dataclass(frozen=True)
class ConsolidationReport:
    sample_id: str
    model: str
    turns: int
    candidate_limit: int | None
    causal_min_confidence: float
    seconds: float
    entities: int
    mentions_edges: int
    entities_per_event: Distribution
    top_entities: list[NamedCount]  # by the number of events that mention them
    merged_entities: int  # keys reached by more than one written form
    merge_examples: list[MergedEntity]
    date_expressions: int
    resolved_dates: int
    date_granularity: list[NamedCount]
    unresolved_dates: list[NamedCount]
    neighborhood_events: Distribution
    neighborhood_by_quarter: list[float]  # mean size in each quarter of the conversation
    causal_edges: int
    events_with_cause: int
    causes_per_event: Distribution
    confidence: list[NamedCount]
    cause_rank: Distribution
    cause_rank_within: list[ShareWithin]
    cause_turn_gap: Distribution  # how many events back the cause is
    errors: list[NamedCount]
    error_examples: list[str]
    usage: list[TagUsage]
    examples: list[CauseExample]


def ingest_conversation(
    sample: LocomoSample,
    fast_path: FastPathIngestion,
    slow_path: SlowPathConsolidation,
    max_turns: int | None = None,
) -> list[Consolidation]:
    """Fast path for every turn (or the first `max_turns`, for a cheap pilot), then drain the slow path."""
    ingested = 0
    for session in sessions(sample):
        moment = session_time(session)
        for turn in session.turns:
            if max_turns is not None and ingested == max_turns:
                return slow_path.drain()
            fast_path.ingest(Turn(speaker=turn["speaker"], text=turn_text(turn, with_speaker=False), timestamp=moment))
            ingested += 1
    return slow_path.drain()


def distribution(values: list[float]) -> Distribution:
    if not values:
        return Distribution(count=0, mean=0.0, median=0.0, p90=0.0, maximum=0.0)
    array = np.array(values, dtype=np.float64)
    return Distribution(
        count=len(values),
        mean=round(float(array.mean()), 2),
        median=round(float(np.median(array)), 2),
        p90=round(float(np.percentile(array, 90)), 2),
        maximum=round(float(array.max()), 2),
    )


class LoggedCall(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency_s: float


class LoggedRecord(BaseModel):
    """The fields of a line of the LLM log that the report uses."""

    tag: str
    response: LoggedCall


def read_calls(log_path: Path) -> list[LoggedRecord]:
    """One entry per LLM call, from the log the client writes."""
    return [LoggedRecord.model_validate_json(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def summarize(
    sample_id: str,
    model: str,
    config: IngestConfig,
    graph: GraphStore,
    results: list[Consolidation],
    calls: list[LoggedRecord],
    seconds: float,
    sample_size: int,
    seed: int,
) -> ConsolidationReport:
    events = graph.nodes(MagmaNode.EVENT)
    position = {event.id: index for index, event in enumerate(events)}
    entity_nodes = graph.nodes(MagmaNode.ENTITY)
    mentions_of = {entity.id: len(graph.neighbors(entity.id)) for entity in entity_nodes}

    forms_of_key: dict[str, set[str]] = {}
    for event in events:
        names = event.attributes.get("entities", [])
        for name in names if isinstance(names, list) else []:
            if isinstance(name, str) and entity_key(name):
                forms_of_key.setdefault(entity_key(name), set()).add(name.strip())
    merged = sorted((key, sorted(forms)) for key, forms in forms_of_key.items() if len(forms) > 1)

    granularity: Counter[str] = Counter()
    unresolved: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    error_examples: list[str] = []
    ranks: list[float] = []
    gaps: list[float] = []
    confidences: Counter[str] = Counter()
    chosen: list[CauseExample] = []
    for result in results:
        for mention in result.resolved_dates:
            granularity[mention.granularity.value] += 1
        for expression in result.unresolved_dates:
            unresolved[expression.lower()] += 1
        for error in result.errors:
            errors[error.split(":", 1)[0]] += 1
            if len(error_examples) < sample_size:
                error_examples.append(error[:300])
        for edge in result.causes:
            rank = result.candidates.index(edge.source) + 1
            ranks.append(rank)
            gaps.append(position[edge.target] - position[edge.source])
            confidences[_confidence_bin(edge.weight)] += 1
            reason = edge.attributes.get("reason", "")
            chosen.append(
                CauseExample(
                    cause_id=edge.source,
                    cause_text=graph.node(edge.source).text,
                    effect_id=edge.target,
                    effect_text=graph.node(edge.target).text,
                    confidence=edge.weight,
                    reason=reason if isinstance(reason, str) else "",
                    rank=rank,
                )
            )

    sizes = [float(result.neighborhood_events) for result in results]
    quarters = [chunk for chunk in np.array_split(np.array(sizes, dtype=np.float64), 4) if chunk.size]
    rng = random.Random(seed)
    return ConsolidationReport(
        sample_id=sample_id,
        model=model,
        turns=len(events),
        candidate_limit=config.causal_candidate_limit,
        causal_min_confidence=config.causal_min_confidence,
        seconds=round(seconds, 1),
        entities=len(entity_nodes),
        mentions_edges=sum(mentions_of.values()),
        entities_per_event=distribution([float(len(result.entities)) for result in results]),
        top_entities=[NamedCount(graph.node(node_id).text, count) for node_id, count in Counter(mentions_of).most_common(10)],
        merged_entities=len(merged),
        merge_examples=[MergedEntity(key, forms) for key, forms in merged[:sample_size]],
        date_expressions=sum(granularity.values()) + sum(unresolved.values()),
        resolved_dates=sum(granularity.values()),
        date_granularity=[NamedCount(name, count) for name, count in granularity.most_common()],
        unresolved_dates=[NamedCount(name, count) for name, count in unresolved.most_common(30)],
        neighborhood_events=distribution(sizes),
        neighborhood_by_quarter=[round(float(chunk.mean()), 1) for chunk in quarters],
        causal_edges=len(ranks),
        events_with_cause=sum(1 for result in results if result.causes),
        causes_per_event=distribution([float(len(result.causes)) for result in results]),
        confidence=[NamedCount(name, count) for name, count in sorted(confidences.items())],
        cause_rank=distribution(ranks),
        cause_rank_within=[ShareWithin(cut, round(sum(1 for r in ranks if r <= cut) / len(ranks), 3)) for cut in RANK_CUTS] if ranks else [],
        cause_turn_gap=distribution(gaps),
        errors=[NamedCount(name, count) for name, count in errors.most_common()],
        error_examples=error_examples,
        usage=_usage_by_tag(calls),
        examples=rng.sample(chosen, min(sample_size, len(chosen))),
    )


def _confidence_bin(confidence: float) -> str:
    lower = max(edge for edge in CONFIDENCE_BINS if edge <= max(confidence, CONFIDENCE_BINS[0]))
    return f">= {lower:.1f}"


def _usage_by_tag(calls: list[LoggedRecord]) -> list[TagUsage]:
    usage: list[TagUsage] = []
    for tag in sorted({call.tag for call in calls}):
        mine = [call for call in calls if call.tag == tag]
        usage.append(
            TagUsage(
                tag=tag,
                calls=len(mine),
                prompt_tokens=sum(call.response.prompt_tokens for call in mine),
                completion_tokens=sum(call.response.completion_tokens for call in mine),
                reasoning_tokens=sum(call.response.reasoning_tokens for call in mine),
                latency_s=round(sum(call.response.latency_s for call in mine), 1),
                prompt_tokens_per_call=distribution([float(call.response.prompt_tokens) for call in mine]),
            )
        )
    return usage


def print_summary(report: ConsolidationReport) -> None:
    print(f"{report.sample_id}: {report.turns} turns consolidated in {report.seconds} s with {report.model}")
    print(f"entities: {report.entities} nodes, {report.mentions_edges} MENTIONS, {report.merged_entities} merged keys")
    print(f"dates: {report.resolved_dates} of {report.date_expressions} expressions resolved")
    n = report.neighborhood_events
    print(f"2-hop neighborhood (earlier events): median {n.median}, p90 {n.p90}, max {n.maximum}; by quarter {report.neighborhood_by_quarter}")
    print(f"causal edges: {report.causal_edges} on {report.events_with_cause} events; cause rank median {report.cause_rank.median}")
    print("cause within rank: " + ", ".join(f"{s.rank}: {s.share:.0%}" for s in report.cause_rank_within))
    print("errors: " + (", ".join(f"{e.name} {e.count}" for e in report.errors) or "none"))
    for usage in report.usage:
        print(f"{usage.tag}: {usage.calls} calls, {usage.prompt_tokens} prompt + {usage.completion_tokens} completion tokens")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--locomo", type=Path, required=True, help="path to locomo10.json")
    parser.add_argument("--sample", default="conv-26", help="LoCoMo conversation id")
    candidates = parser.add_mutually_exclusive_group()
    candidates.add_argument(
        "--candidate-limit",
        type=int,
        default=DEFAULT.ingest.causal_candidate_limit,
        help="causal candidates shown to the prompt; default: the one in the config",
    )
    candidates.add_argument("--all-candidates", action="store_true", help="show the whole 2-hop neighborhood, as Alg. 3 is written")
    parser.add_argument("--max-turns", type=int, default=None, help="only the first turns, for a cheap pilot")
    parser.add_argument("--sample-size", type=int, default=15, help="causal edges kept for checking by hand")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/measurements/consolidation"))
    parser.add_argument("--label", default=None, help="suffix of the run folder, e.g. the LLM settings being compared")
    return parser


def candidate_limit(arguments: argparse.Namespace) -> int | None:
    """The limit asked on the command line; None is the whole neighborhood."""
    if arguments.all_candidates:
        return None
    limit: int | None = arguments.candidate_limit
    return limit


def main() -> None:
    arguments = build_parser().parse_args()

    load_dotenv()
    run_name = arguments.sample if arguments.max_turns is None else f"{arguments.sample}-first-{arguments.max_turns}"
    if arguments.label:
        run_name = f"{run_name}-{arguments.label}"
    output = arguments.output_dir / run_name
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "llm_calls.jsonl"
    log_path.unlink(missing_ok=True)  # the log of this run only

    config = replace(DEFAULT.ingest, causal_candidate_limit=candidate_limit(arguments))
    embedder: Embedder = InstructedEmbedder.from_config()
    llm = LLMClient.from_env("LLM", log_path=log_path)
    graph = NetworkXGraphStore(MAGMA_PROFILE)
    consolidation_queue: queue.Queue[str] = queue.Queue()
    fast_path = FastPathIngestion(graph, FaissVectorIndex(embedder.dimension), BM25Index(), embedder, consolidation_queue, config)
    slow_path = SlowPathConsolidation(graph, llm, DateparserTagger(), consolidation_queue, config)

    sample = find_sample(load_samples(arguments.locomo), arguments.sample)
    start = time.perf_counter()
    results = ingest_conversation(sample, fast_path, slow_path, arguments.max_turns)
    seconds = time.perf_counter() - start

    report = summarize(
        arguments.sample, llm.settings.model, config, graph, results, read_calls(log_path), seconds, arguments.sample_size, arguments.seed
    )
    print_summary(report)
    save_graph(graph, output / "graph.json")
    (output / "report.json").write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report, graph and LLM log written to {output}")


if __name__ == "__main__":
    main()
