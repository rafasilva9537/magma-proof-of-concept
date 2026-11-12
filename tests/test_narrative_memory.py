"""The narrative facade end to end, with a scripted LLM: each viewer's answer is built only from what they saw."""

import json
from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np

from magma.embed import Matrix, Vector, normalize
from magma.ingest import Turn
from magma.llm import ChatRequest, ClientSettings, Completion, LLMClient
from magma.memory import MagmaMemory
from magma.router import KeywordClassifier
from magma.temporal import DateparserTagger
from narrative.memory import NarrativeMemory

START = datetime(2024, 3, 1, 20, 0)
HALL = frozenset({"Lia", "Brom", "Ceres"})
CORRIDOR = frozenset({"Dara"})
SPELL = "Lia casts a disarming spell on Brom in the hall."
BROM = '*picks up the wand* "Nobody attacked me, I dropped the wand."'


class ScriptedBackend:
    def complete(self, request: ChatRequest) -> Completion:
        if "response_format" in request:
            user = request["messages"][1]["content"]
            reply: object = {"causes": []}
            if "- Speech: Nobody attacked me" in user:
                reply = {"entities": ["Brom"], "claims": [{"statement": "Brom dropped the wand", "about": ["Brom"]}]}
            return Completion(text=json.dumps(reply), prompt_tokens=50, completion_tokens=5, reasoning_tokens=0)
        return Completion(text="scripted answer", prompt_tokens=120, completion_tokens=4, reasoning_tokens=0)


class SameVectorEmbedder:
    @property
    def dimension(self) -> int:
        return 2

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return normalize(np.array([(1.0, 1.0) for _ in texts], dtype=np.float32))

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self.embed_documents([text])[0]
        return vector

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def build(memory_class: type[MagmaMemory]) -> MagmaMemory:
    llm = LLMClient(ClientSettings(model="test-model"), backend=ScriptedBackend(), cache_dir=None)
    embedder = SameVectorEmbedder()
    memory = memory_class.create(llm, embedder, DateparserTagger(), embedder, classifier=KeywordClassifier())
    memory.add_turn(Turn("narrator", SPELL, START, HALL))
    memory.add_turn(Turn("Brom", BROM, START + timedelta(minutes=1), CORRIDOR))
    memory.consolidate()
    return memory


def test_each_viewer_is_answered_from_what_they_saw() -> None:
    memory = build(NarrativeMemory)
    try:
        question = "What happened to the wand?"
        later = START + timedelta(hours=1)
        dara = memory.query(question, later, viewer="Dara").trace
        lia = memory.query(question, later, viewer="Lia").trace
        everyone = memory.query(question, later).trace

        assert dara["viewer"] == "Dara"
        assert dara["included"] == ["e2"]
        assert "disarming" not in str(dara["context"])
        assert '"Nobody attacked me, I dropped the wand."' in str(dara["context"])

        assert lia["included"] == ["e1"]
        assert "wand" not in str(lia["context"]).replace("disarming spell", "")  # Lia never heard Brom

        included = everyone["included"]
        assert isinstance(included, list) and sorted(str(node_id) for node_id in included) == ["e1", "e2"]
    finally:
        memory.close()


def test_original_memory_has_no_observers() -> None:
    memory = build(MagmaMemory)
    try:
        question = "What happened to the wand?"
        later = START + timedelta(hours=1)
        dara = memory.query(question, later, viewer="Dara").trace
        omniscient = memory.query(question, later).trace
        assert dara["included"] == omniscient["included"]
        assert "disarming" in str(dara["context"])  # the control group leaks by design
    finally:
        memory.close()
