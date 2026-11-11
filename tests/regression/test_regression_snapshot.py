"""The `magma` profile is the control group of the comparison: its ingestion and its query stages stay as recorded.

The snapshot was written at the v-original tag by `python -m bench.snapshot_retrieval` over the memory in
data/conv-30 (the smallest LoCoMo conversation, built with the 50 causal candidates). A change in magma/ that
alters it must be deliberate: record the reason, write the snapshot again and measure Architecture A again.
"""

import os
from datetime import datetime
from pathlib import Path

import pytest

from bench.snapshot_retrieval import asked_questions, differences, ingestion_fingerprint, load_snapshot, take_snapshot

HERE = Path(__file__).parent
SNAPSHOT = HERE / "conv-30.snapshot.json"
MEMORY = HERE / "data" / "conv-30"


def test_ingestion_is_frozen() -> None:
    # The prompts of the slow path and the settings that shape the graph; the LLM answers live in the cache.
    assert load_snapshot(SNAPSHOT)["ingestion"] == ingestion_fingerprint()


def test_snapshot_covers_every_question_of_the_conversation() -> None:
    snapshot = load_snapshot(SNAPSHOT)
    assert snapshot["sample"] == "conv-30"
    assert len(snapshot["questions"]) == 105


@pytest.mark.skipif(os.getenv("RUN_MODEL_TESTS") != "1", reason="loads the embedding and intent models; set RUN_MODEL_TESTS=1")
def test_query_stages_give_the_recorded_output() -> None:
    recorded = load_snapshot(SNAPSHOT)
    current = take_snapshot(
        MEMORY, recorded["sample"], asked_questions(recorded), datetime.fromisoformat(recorded["reference_time"])
    )
    assert differences(recorded, current) == []
