import json
from dataclasses import replace

from bench.snapshot_retrieval import IngestionFingerprint, QuestionTrace, Snapshot, differences, ingestion_fingerprint
from magma.config import DEFAULT


def snapshot(traces: dict[str, str], ingestion: IngestionFingerprint | None = None) -> Snapshot:
    questions = [QuestionTrace(question_id=qid, question="?", trace={"context": context}) for qid, context in traces.items()]
    return Snapshot(
        sample="conv-x", reference_time="2023-01-01T00:00:00", ingestion=ingestion or ingestion_fingerprint(), questions=questions
    )


def test_fingerprint_survives_a_round_trip_through_json() -> None:
    fingerprint = ingestion_fingerprint()
    assert json.loads(json.dumps(fingerprint)) == fingerprint


def test_fingerprint_changes_with_an_ingestion_setting() -> None:
    changed = replace(DEFAULT, ingest=replace(DEFAULT.ingest, causal_candidate_limit=None))
    assert ingestion_fingerprint(changed) != ingestion_fingerprint()
    assert ingestion_fingerprint(changed)["prompts"] == ingestion_fingerprint()["prompts"]


def test_fingerprint_ignores_query_settings() -> None:
    changed = replace(DEFAULT, retrieval=replace(DEFAULT.retrieval, top_k=5))
    assert ingestion_fingerprint(changed) == ingestion_fingerprint()


def test_differences_name_the_questions_that_changed() -> None:
    recorded = snapshot({"q1": "a", "q2": "b", "q3": "c"})
    assert differences(recorded, snapshot({"q1": "a", "q2": "B", "q3": "c"})) == ["q2"]
    assert differences(recorded, snapshot({"q1": "a", "q2": "b"})) == ["q3"]  # a question that is gone counts
    assert differences(recorded, recorded) == []


def test_differences_report_a_changed_ingestion() -> None:
    other = ingestion_fingerprint(replace(DEFAULT, ingest=replace(DEFAULT.ingest, causal_min_confidence=0.6)))
    assert differences(snapshot({"q1": "a"}), snapshot({"q1": "a"}, other)) == ["ingestion"]
