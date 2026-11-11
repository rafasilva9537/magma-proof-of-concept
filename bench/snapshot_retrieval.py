"""Record what the query stages of the `magma` profile produce on a saved memory: the regression snapshot.

For every question of the conversation, the router (Stage 1), the retrieval
(Stages 2-4) and the C.2 prompt run over the saved graph, without calling the
LLM. The trace of each question is written as JSON: intent, time window,
anchors, the steps of the beam search, the linearized context and the final
prompt (as a sha256, since it repeats the context). The snapshot also keeps a fingerprint of the ingestion (the slow-path
prompts and the settings of embedding, dates and ingestion), so that a change
there shows up without rebuilding the graph.

    python -m bench.snapshot_retrieval --locomo MAGMA-original-repo/data/locomo10.json \
        --memory tests/regression/data/conv-30 --sample conv-30 --output tests/regression/conv-30.snapshot.json

tests/regression/ compares the current code with the snapshot recorded at the v-original tag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from bench.locomo import find_sample, load_samples, to_conversation
from bench.measure_retrieval import load_memory
from magma.config import DEFAULT, Config
from magma.embed import InstructedEmbedder
from magma.ingest import CAUSAL_SYSTEM_PROMPT, EXTRACTION_INPUT, EXTRACTION_SYSTEM_PROMPT, NO_CONTEXT
from magma.json_types import JSONObject
from magma.memory import build_trace
from magma.qa import SynthesizedAnswer, build_qa_messages, dynamic_instruction
from magma.retrieve import Retriever
from magma.router import QueryRouter, Router, build_default_classifier
from magma.temporal import DateparserTagger

INGESTION_PROMPTS = {
    "extraction_system": EXTRACTION_SYSTEM_PROMPT,
    "extraction_input": EXTRACTION_INPUT,
    "no_context": NO_CONTEXT,
    "causal_system": CAUSAL_SYSTEM_PROMPT,
}


class IngestionFingerprint(TypedDict):
    prompts: dict[str, str]  # prompt name -> sha256 of its text
    settings: JSONObject  # the embedding, temporal and ingestion settings, as JSON


class QuestionTrace(TypedDict):
    question_id: str
    question: str
    trace: JSONObject


class Snapshot(TypedDict):
    sample: str
    reference_time: str
    ingestion: IngestionFingerprint
    questions: list[QuestionTrace]


class AskedQuestion(TypedDict):
    question_id: str
    question: str


def ingestion_fingerprint(config: Config = DEFAULT) -> IngestionFingerprint:
    """What decides the graph a conversation turns into, apart from the LLM answers kept in the cache."""
    prompts = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in INGESTION_PROMPTS.items()}
    # Through JSON, so that tuples and enums compare equal to what a snapshot file gives back.
    settings: JSONObject = json.loads(
        json.dumps({"embedding": asdict(config.embedding), "temporal": asdict(config.temporal), "ingest": asdict(config.ingest)})
    )
    return IngestionFingerprint(prompts=prompts, settings=settings)


def snapshot_question(question: AskedQuestion, reference_time: datetime, router: QueryRouter, retriever: Retriever) -> QuestionTrace:
    """Stages 1-4 and the C.2 prompt for one question; the answer is left empty because no LLM is called."""
    route = router.route(question["question"], reference_time)
    retrieval = retriever.retrieve(route)
    prompt = SynthesizedAnswer(
        answer="",
        dynamic_instruction=dynamic_instruction(route.intent),
        messages=build_qa_messages(route.question, retrieval.context.text, route.intent),
    )
    trace = build_trace(route, retrieval, prompt)
    # The prompt repeats the context, which is kept in full; its hash is enough to catch any other change.
    trace["messages"] = hashlib.sha256(json.dumps(trace["messages"], ensure_ascii=False).encode("utf-8")).hexdigest()
    return QuestionTrace(question_id=question["question_id"], question=question["question"], trace=trace)


def take_snapshot(memory_dir: Path, sample: str, questions: list[AskedQuestion], reference_time: datetime) -> Snapshot:
    """The snapshot of a memory saved by the runner, with the models and settings of the `magma` profile."""
    embedder = InstructedEmbedder.from_config()
    memory = load_memory(memory_dir, embedder.dimension)
    tagger = DateparserTagger()
    router = Router(embedder, tagger, build_default_classifier(tagger))
    retriever = Retriever.create(memory.graph, memory.vectors, memory.keywords, embedder, embedder)
    traces = [snapshot_question(question, reference_time, router, retriever) for question in questions]
    return Snapshot(
        sample=sample, reference_time=reference_time.isoformat(), ingestion=ingestion_fingerprint(), questions=traces
    )


def differences(recorded: Snapshot, current: Snapshot) -> list[str]:
    """What changed between two snapshots: "ingestion" and the ids of the questions whose trace differs."""
    changed: list[str] = []
    if recorded["ingestion"] != current["ingestion"]:
        changed.append("ingestion")
    current_traces = {question["question_id"]: question["trace"] for question in current["questions"]}
    for question in recorded["questions"]:
        if current_traces.get(question["question_id"]) != question["trace"]:
            changed.append(question["question_id"])
    return changed


def load_snapshot(path: Path) -> Snapshot:
    snapshot: Snapshot = json.loads(path.read_text(encoding="utf-8"))
    return snapshot


def asked_questions(snapshot: Snapshot) -> list[AskedQuestion]:
    return [AskedQuestion(question_id=question["question_id"], question=question["question"]) for question in snapshot["questions"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--locomo", type=Path, required=True, help="path to locomo10.json")
    parser.add_argument("--memory", type=Path, required=True, help="folder with the graph.json and vectors.npz")
    parser.add_argument("--sample", default="conv-30", help="LoCoMo conversation id of the saved memory")
    parser.add_argument("--output", type=Path, required=True, help="snapshot file to write")
    arguments = parser.parse_args()

    conversation = to_conversation(find_sample(load_samples(arguments.locomo), arguments.sample))
    questions = [AskedQuestion(question_id=question.id, question=question.question) for question in conversation.questions]
    snapshot = take_snapshot(arguments.memory, arguments.sample, questions, conversation.reference_time)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(snapshot['questions'])} questions of {arguments.sample} written to {arguments.output}")


if __name__ == "__main__":
    main()
