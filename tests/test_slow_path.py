import json
import queue
import re
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from magma.config import DEFAULT
from magma.graph import Edge, Node
from magma.ingest import CAUSAL_SYSTEM_PROMPT, EXTRACTION_SYSTEM_PROMPT, NO_CONTEXT, SlowPathConsolidation, entity_key
from magma.llm import ChatRequest, ClientSettings, Completion, LLMClient, LLMUnavailable
from magma.networkx_graph import NetworkXGraphStore
from magma.schema import MAGMA_PROFILE, MagmaEdge, MagmaNode
from magma.temporal import DateparserTagger

START = datetime(2023, 5, 8, 13, 0)
_EVENT_ID = re.compile(r"\[(e\d+)\]")
_TEXT = re.compile(r"- Text: (.*)")


class ScriptedBackend:
    """Answers the extraction prompt by the turn text and the causal prompt by the id of the new event."""

    def __init__(self, extractions: dict[str, object], causes: dict[str, object] | None = None) -> None:
        self.extractions = extractions
        self.causes = causes or {}
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        system, user = request["messages"][0]["content"], request["messages"][1]["content"]
        if system == CAUSAL_SYSTEM_PROMPT:
            match = _EVENT_ID.search(user)
            assert match is not None
            reply = self.causes.get(match.group(1), {"causes": []})
        else:
            text = _TEXT.search(user)
            assert text is not None
            reply = self.extractions.get(text.group(1), {})
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return Completion(text=content, prompt_tokens=100, completion_tokens=20, reasoning_tokens=0)

    def prompts(self, system: str) -> list[str]:
        return [r["messages"][1]["content"] for r in self.requests if r["messages"][0]["content"] == system]


class Conversation:
    """Events added by hand, linked by NEXT, as the fast path leaves them."""

    def __init__(self, backend: ScriptedBackend, limit: int | None = None) -> None:
        self.graph = NetworkXGraphStore(MAGMA_PROFILE)
        self.queue: queue.Queue[str] = queue.Queue()
        self.backend = backend
        llm = LLMClient(ClientSettings(model="test-model"), backend=backend, cache_dir=None)
        config = replace(DEFAULT.ingest, causal_candidate_limit=limit)
        self.slow_path = SlowPathConsolidation(self.graph, llm, DateparserTagger(), self.queue, config)

    def say(self, speaker: str, text: str, day: int = 0) -> str:
        events = self.graph.nodes(MagmaNode.EVENT)
        event = Node(
            id=f"e{len(events) + 1}",
            type=MagmaNode.EVENT,
            text=f"{speaker}: {text}",
            timestamp=START + timedelta(days=day, minutes=len(events)),
            attributes={"speaker": speaker},
        )
        self.graph.add_node(event)
        if events:
            self.graph.add_edge(Edge(source=events[-1].id, target=event.id, type=MagmaEdge.NEXT))
        self.queue.put(event.id)
        return event.id


def test_entity_key_merges_case_spacing_and_possessive() -> None:
    assert entity_key("Caroline's") == entity_key("  caroline ") == "caroline"
    assert entity_key("New   York") == "new york"
    assert entity_key("Mel") != entity_key("Melanie")


def test_extraction_creates_shared_entities_and_stores_attributes() -> None:
    backend = ScriptedBackend(
        {
            "I went to the support group.": {
                "entities": ["Caroline", "LGBTQ support group"],
                "topic": "support group",
                "semantic_facts": ["Caroline went to an LGBTQ support group"],
                "summary": "Caroline went to the support group.",
            },
            "That's great, Caroline!": {"entities": ["caroline's"], "summary": "Melanie congratulates Caroline."},
        }
    )
    talk = Conversation(backend)
    first = talk.say("Caroline", "I went to the support group.")
    second = talk.say("Melanie", "That's great, Caroline!")
    done = talk.slow_path.drain()

    assert [result.new_entities for result in done] == [["entity:caroline", "entity:lgbtq support group"], []]
    assert done[1].entities == ["entity:caroline"]  # "caroline's" is the same node
    assert talk.graph.node("entity:caroline").text == "Caroline"  # first form seen
    assert talk.graph.has_edge(first, "entity:caroline", MagmaEdge.MENTIONS)
    assert talk.graph.has_edge(second, "entity:caroline", MagmaEdge.MENTIONS)
    stored = talk.graph.node(first).attributes
    assert stored["speaker"] == "Caroline"
    assert stored["topic"] == "support group"
    assert stored["semantic_facts"] == ["Caroline went to an LGBTQ support group"]


def test_context_of_c1_is_the_previous_summary() -> None:
    backend = ScriptedBackend({"Hi!": {"summary": "Caroline greets Melanie."}})
    talk = Conversation(backend)
    talk.say("Caroline", "Hi!")
    talk.say("Melanie", "Hello!")
    talk.slow_path.drain()
    extraction_prompts = backend.prompts(EXTRACTION_SYSTEM_PROMPT)
    assert f"- Context: {NO_CONTEXT}" in extraction_prompts[0]
    assert "- Text: Hello!" in extraction_prompts[1]
    assert "- Context: Caroline greets Melanie." in extraction_prompts[1]


def test_dates_are_resolved_from_the_event_timestamp() -> None:
    backend = ScriptedBackend({"I ran a charity race yesterday.": {"dates_mentioned": ["yesterday", "soonish"]}})
    talk = Conversation(backend)
    event = talk.say("Melanie", "I ran a charity race yesterday.")
    [result] = talk.slow_path.drain()
    assert [(m.window.start, m.window.end) for m in result.resolved_dates] == [(datetime(2023, 5, 7), datetime(2023, 5, 8))]
    assert result.unresolved_dates == ["soonish"]
    stored = talk.graph.node(event).attributes
    assert stored["dates_mentioned"] == [
        {"text": "yesterday", "start": "2023-05-07T00:00:00", "end": "2023-05-08T00:00:00", "granularity": "DAY"}
    ]
    assert stored["unresolved_dates"] == ["soonish"]


def test_causes_come_from_the_earlier_neighborhood_above_delta() -> None:
    # e1 adopts a dog; e2 is unrelated; e3 is a later turn about the dog, reached from e1 through the entity (2 hops).
    backend = ScriptedBackend(
        {
            "I adopted a dog, Oscar.": {"entities": ["Oscar"]},
            "Took Oscar to the vet today.": {"entities": ["Oscar"]},
        },
        causes={
            "e3": {
                "causes": [
                    {"event_id": "e1", "confidence": 0.9, "reason": "the vet visit is for the adopted dog"},
                    {"event_id": "e2", "confidence": 0.5, "reason": "at δ, not above it"},
                    {"event_id": "e9", "confidence": 0.95, "reason": "not a candidate"},
                ]
            }
        },
    )
    talk = Conversation(backend)
    talk.say("Melanie", "I adopted a dog, Oscar.")
    talk.say("Caroline", "Nice weather.")
    talk.say("Melanie", "Took Oscar to the vet today.", day=30)
    done = talk.slow_path.drain()

    assert done[0].candidates == []  # the first event has nothing earlier: no causal call
    assert done[2].candidates == ["e2", "e1"]  # e2 at 1 hop (NEXT), e1 at 2 hops (NEXT and through Oscar)
    assert [(edge.source, edge.target, edge.weight) for edge in done[2].causes] == [("e1", "e3", 0.9)]
    assert talk.graph.has_edge("e1", "e3", MagmaEdge.CAUSES)
    assert not talk.graph.has_edge("e2", "e3", MagmaEdge.CAUSES)
    assert len(backend.prompts(CAUSAL_SYSTEM_PROMPT)) == 2  # for e2 and e3, not for e1


def test_later_events_are_neither_candidates_nor_paths() -> None:
    talk = Conversation(ScriptedBackend({}))
    talk.say("Caroline", "One.")
    talk.say("Melanie", "Two.")
    talk.say("Caroline", "Three.")
    # Before e2 is consolidated, e3 already exists (the fast path ran ahead) and is linked to e1.
    talk.graph.add_edge(Edge(source="e3", target="e1", type=MagmaEdge.SIMILAR_TO, weight=0.8))
    done = talk.slow_path.drain()
    assert done[1].candidates == ["e1"]
    assert done[0].neighborhood_events == 0


def test_candidate_limit_keeps_the_closest_hops_first() -> None:
    talk = Conversation(ScriptedBackend({}), limit=1)
    for number in range(4):
        talk.say("Caroline", f"Turn {number}.")
    talk.graph.add_edge(Edge(source="e4", target="e1", type=MagmaEdge.SIMILAR_TO, weight=0.8))
    done = talk.slow_path.drain()
    assert done[3].neighborhood_events == 3  # e3 and e1 at 1 hop, e2 at 2 hops
    assert done[3].candidates == ["e3"]  # 1 hop, the most recent first


def test_invalid_json_is_recorded_and_consolidation_goes_on() -> None:
    backend = ScriptedBackend({"Hi!": "not json"}, causes={"e2": "{\"causes\": [{\"event_id\": \"e1\"}]}"})
    talk = Conversation(backend)
    first = talk.say("Caroline", "Hi!")
    second = talk.say("Melanie", "Hello!")
    done = talk.slow_path.drain()
    assert done[0].errors[0].startswith("extraction:")
    assert done[1].errors[0].startswith("causal:")  # confidence missing
    assert talk.graph.node(first).attributes["consolidation_errors"] == done[0].errors
    assert talk.graph.node(second).attributes["consolidation_errors"] == done[1].errors
    assert talk.queue.empty()


class FailingOnce:
    """The causal call of e2 fails the first time, as when the LLM stays unavailable."""

    def __init__(self, inner: ScriptedBackend) -> None:
        self.inner = inner
        self.failed = False

    def complete(self, request: ChatRequest) -> Completion:
        is_causal = request["messages"][0]["content"] == CAUSAL_SYSTEM_PROMPT
        if is_causal and not self.failed:
            self.failed = True
            raise LLMUnavailable("down")
        return self.inner.complete(request)


def test_event_interrupted_by_an_unavailable_llm_is_consolidated_first_next_time() -> None:
    inner = ScriptedBackend({"Hi!": {"entities": ["Caroline"]}, "Hello!": {"entities": ["Caroline"]}})
    talk = Conversation(inner)
    talk.slow_path = SlowPathConsolidation(
        talk.graph,
        LLMClient(ClientSettings(model="test-model"), backend=FailingOnce(inner), cache_dir=None),
        DateparserTagger(),
        talk.queue,
    )
    talk.say("Caroline", "Hi!")
    talk.say("Melanie", "Hello!")
    talk.say("Caroline", "Bye!")
    with pytest.raises(LLMUnavailable):
        talk.slow_path.drain()
    done = talk.slow_path.drain()
    assert [result.event_id for result in done] == ["e2", "e3"]
    assert done[0].entities == ["entity:caroline"]  # extracted again, without a duplicate edge
