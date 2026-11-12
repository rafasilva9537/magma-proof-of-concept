"""Viewer views, claim status and the two-section linearization, on a scene built by hand.

The scene (the magic example of the design's ontology decision):
- e1, narration seen by Lia, Brom and Ceres: Lia disarms Brom in the hall.
- e2, Brom to Dara: *picks up the wand* "Nobody attacked me, I dropped it."
  Two claims: e2.c1 (denial) "Someone attacked Brom", e2.c2 "Brom dropped the wand".
  e1 refutes both: the denial (someone did attack Brom) and the assertion (INVALIDATES).
- e3, Ceres to everyone: "I saw nothing." (no claim extracted)
Dara never saw e1, so for her the claim is unverified; Brom saw it.
"""

from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from magma.bm25_index import BM25Index
from magma.config import DEFAULT, Intent
from magma.faiss_index import FaissVectorIndex
from magma.graph import VISIBLE_TO_EVERYONE, Edge, GraphStore, Node, UnknownNode
from magma.networkx_graph import NetworkXGraphStore
from magma.retrieve import Traversal, Visit
from magma.router import RouterOutput
from narrative.linearize import ViewerLinearizer
from narrative.schema import NARRATIVE_PROFILE, ClaimStatus
from narrative.status import ClaimView, status_for
from narrative.view import MaskedLexicalIndex, MaskedVectorIndex, ReadOnlyView, ViewerGraph, visible_ids

START = datetime(2024, 3, 1, 20, 0)
HALL = frozenset({"Lia", "Brom", "Ceres"})
CORRIDOR = frozenset({"Brom", "Dara"})


class WordCounter:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


def event(node_id: str, minute: int, speaker: str, action: str, speech: str, visible: frozenset[str]) -> Node:
    text = f"{speaker}: " + " ".join(part for part in (f"*{action}*" if action else "", speech) if part)
    return Node(
        id=node_id,
        type="EVENT",
        text=text,
        timestamp=START + timedelta(minutes=minute),
        attributes={"speaker": speaker, "action_text": action, "speech_text": speech},
        visible_to=visible,
    )


def claim(node_id: str, statement: str, claim_type: str, visible: frozenset[str]) -> Node:
    return Node(
        id=node_id,
        type="CLAIM",
        text=statement,
        attributes={"speaker": "Brom", "claim_type": claim_type},
        visible_to=visible,
    )


def scene() -> GraphStore:
    graph = NetworkXGraphStore(NARRATIVE_PROFILE)
    graph.add_node(event("e1", 0, "narrator", "Lia casts a disarming spell on Brom in the hall.", "", HALL))
    graph.add_node(event("e2", 1, "Brom", "picks up the wand", "Nobody attacked me, I dropped it.", CORRIDOR))
    graph.add_node(claim("e2.c1", "Someone attacked Brom", "DENIAL", CORRIDOR))
    graph.add_node(claim("e2.c2", "Brom dropped the wand", "ASSERTION", CORRIDOR))
    graph.add_node(event("e3", 2, "Ceres", "", "I saw nothing.", VISIBLE_TO_EVERYONE))
    graph.add_node(Node(id="entity:brom", type="ENTITY", text="Brom"))
    graph.add_edge(Edge(source="e1", target="e2", type="NEXT"))
    graph.add_edge(Edge(source="e2", target="e3", type="NEXT"))
    graph.add_edge(Edge(source="e2", target="e2.c1", type="STATED"))
    graph.add_edge(Edge(source="e2", target="e2.c2", type="STATED"))
    graph.add_edge(Edge(source="e2", target="entity:brom", type="MENTIONS", attributes={"role": "ACTOR"}))
    graph.add_edge(Edge(source="e1", target="e2.c1", type="INVALIDATES"))
    graph.add_edge(Edge(source="e1", target="e2.c2", type="INVALIDATES"))
    return graph


def route(intent: Intent = Intent.WHEN) -> RouterOutput:
    return RouterOutput(
        question="What happened to Brom's wand?",
        intent=intent,
        intent_reason="test",
        time_window=None,
        temporal_mentions=(),
        query_vector=np.array([1.0, 0.0], dtype=np.float32),
        keywords=("wand",),
    )


def anchor(node_id: str, score: float) -> Visit:
    return Visit(node_id=node_id, depth=0, score=score, transition_score=None)


def reached(node_id: str, parent: str, transition: float) -> Visit:
    return Visit(node_id=node_id, depth=1, score=transition, transition_score=transition, parent_id=parent, edge_type="NEXT")


def traversal(*visits: Visit) -> Traversal:
    return Traversal(visits=visits, transitions_checked=0, dropped_by_node_type={})


# --- the viewer's view --------------------------------------------------------------------------


def test_view_hides_what_the_viewer_did_not_see() -> None:
    dara = ViewerGraph(scene(), "Dara")
    assert [node.id for node in dara.nodes()] == ["e2", "e2.c1", "e2.c2", "e3", "entity:brom"]
    assert not dara.has_node("e1")
    with pytest.raises(UnknownNode):
        dara.node("e1")
    assert "e1" not in {neighbor.node.id for neighbor in dara.neighbors("e2")}
    assert all("e1" not in (edge.source, edge.target) for edge in dara.edges())
    assert not dara.has_edge("e1", "e2.c2", "INVALIDATES")


def test_view_keeps_everything_the_viewer_saw() -> None:
    brom = ViewerGraph(scene(), "Brom")
    assert {node.id for node in brom.nodes()} == {"e1", "e2", "e2.c1", "e2.c2", "e3", "entity:brom"}
    assert brom.has_edge("e1", "e2.c2", "INVALIDATES")


def test_view_is_read_only() -> None:
    with pytest.raises(ReadOnlyView):
        ViewerGraph(scene(), "Dara").add_node(Node(id="x", type="ENTITY", text="x"))


def test_masked_indexes_never_return_an_invisible_hit() -> None:
    graph = scene()
    vectors = FaissVectorIndex(2)
    keywords = BM25Index(DEFAULT.lexical_index)
    for node_id, vector in (("e1", (1.0, 0.0)), ("e2", (0.8, 0.6)), ("e3", (0.0, 1.0))):
        vectors.add(node_id, np.array(vector, dtype=np.float32))
        keywords.add(node_id, graph.node(node_id).text)
    visible = visible_ids(ViewerGraph(graph, "Dara"))

    hits = MaskedVectorIndex(vectors, visible).search(np.array([1.0, 0.0], dtype=np.float32), top_k=3)
    assert [hit.node_id for hit in hits] == ["e2", "e3"]
    assert [hit.node_id for hit in MaskedLexicalIndex(keywords, visible).search("spell hall Brom", top_k=3)] == ["e2"]
    assert [hit.node_id for hit in keywords.search("spell hall", top_k=3)] == ["e1"]  # the full index still has it


# --- status by viewer ---------------------------------------------------------------------------


def test_claim_is_refuted_only_for_who_saw_the_proof() -> None:
    graph = scene()
    assert status_for(ViewerGraph(graph, "Brom"), "e2.c2") == ClaimView(ClaimStatus.REFUTED, ("e1",))
    assert status_for(ViewerGraph(graph, "Dara"), "e2.c2") == ClaimView(ClaimStatus.UNVERIFIED, ())
    assert status_for(graph, "e2.c2").status is ClaimStatus.REFUTED  # the omniscient reader


def precedence_scene() -> GraphStore:
    """Claims c1..c4 of one speech; f1 and f2 are facts."""
    graph = NetworkXGraphStore(NARRATIVE_PROFILE)
    graph.add_node(event("f1", 0, "narrator", "fact one", "", VISIBLE_TO_EVERYONE))
    graph.add_node(event("f2", 1, "narrator", "fact two", "", VISIBLE_TO_EVERYONE))
    graph.add_node(event("s", 2, "Brom", "", "four things", VISIBLE_TO_EVERYONE))
    for number in range(1, 5):
        graph.add_node(claim(f"c{number}", f"claim {number}", "ASSERTION", VISIBLE_TO_EVERYONE))
        graph.add_edge(Edge(source="s", target=f"c{number}", type="STATED"))
    return graph


def test_a_fact_outweighs_what_anyone_said() -> None:
    graph = precedence_scene()
    graph.add_edge(Edge(source="f1", target="c1", type="SUPPORTS"))
    graph.add_edge(Edge(source="c1", target="c2", type="CONTRADICTS"))
    graph.add_edge(Edge(source="f2", target="c1", type="INVALIDATES"))
    assert status_for(graph, "c1") == ClaimView(ClaimStatus.REFUTED, ("f2",))  # refuting beats confirming
    assert status_for(graph, "c2") == ClaimView(ClaimStatus.CONTESTED, ("c1",))  # contradiction counts both ways


def test_supersedes_retracts_the_older_version_only() -> None:
    graph = precedence_scene()
    graph.add_edge(Edge(source="c4", target="c3", type="SUPERSEDES"))
    assert status_for(graph, "c3") == ClaimView(ClaimStatus.RETRACTED, ("c4",))
    assert status_for(graph, "c4").status is ClaimStatus.UNVERIFIED


# --- two-section linearization ------------------------------------------------------------------


def test_witness_of_the_proof_reads_own_words_and_the_refuted_claims() -> None:
    graph = scene()
    linearizer = ViewerLinearizer(ViewerGraph(graph, "Brom"), WordCounter(), viewer="Brom")
    context = linearizer.linearize(route(), traversal(anchor("e2", 0.05), reached("e1", "e2", 2.5)))
    assert context.text == "\n".join(
        [
            "What you witnessed:",
            "<t:2024-03-01 20:00> Lia casts a disarming spell on Brom in the hall. <ref:e1>",
            "<t:2024-03-01 20:01> You: *picks up the wand* <ref:e2>",
            "",
            "What was said:",
            '<t:2024-03-01 20:01> You said to {Dara}: "Nobody attacked me, I dropped it." <ref:e2>',
            '  - denied: "Someone attacked Brom" [REFUTED by <ref:e1>] <ref:e2.c1>',
            '  - "Brom dropped the wand" [REFUTED by <ref:e1>] <ref:e2.c2>',
        ]
    )
    assert context.included == ("e1", "e2")


def test_listener_who_missed_the_proof_reads_only_what_was_said() -> None:
    graph = scene()
    linearizer = ViewerLinearizer(ViewerGraph(graph, "Dara"), WordCounter(), viewer="Dara")
    context = linearizer.linearize(route(), traversal(anchor("e2", 0.05)))
    assert context.text == "\n".join(
        [
            "What you witnessed:",
            "<t:2024-03-01 20:01> Brom: *picks up the wand* <ref:e2>",
            "",
            "What was said:",
            '<t:2024-03-01 20:01> Brom said to {you}: "Nobody attacked me, I dropped it." <ref:e2>',
        ]
    )  # both claims are UNVERIFIED for Dara, so no claim line


def test_a_claim_reached_by_the_beam_brings_its_event() -> None:
    graph = scene()
    linearizer = ViewerLinearizer(ViewerGraph(graph, "Dara"), WordCounter(), viewer="Dara")
    context = linearizer.linearize(route(), traversal(anchor("e3", 0.05), reached("e2.c2", "e3", 2.0)))
    assert context.included == ("e2", "e3")
    assert '"Nobody attacked me, I dropped it." <ref:e2>' in context.text


def test_turn_that_is_only_speech_has_no_witnessed_line() -> None:
    graph = scene()
    linearizer = ViewerLinearizer(graph, WordCounter())
    context = linearizer.linearize(route(), traversal(anchor("e3", 0.05)))
    assert context.text == "\n".join(
        [
            "What was said:",
            '<t:2024-03-01 20:02> Ceres said to everyone: "I saw nothing." <ref:e3>',
        ]
    )  # as in a LoCoMo turn: the context is the turn as said


def test_budget_folds_an_event_together_with_its_claims() -> None:
    graph = scene()
    small = replace(DEFAULT, context=replace(DEFAULT.context, token_budget=30))
    linearizer = ViewerLinearizer(ViewerGraph(graph, "Brom"), WordCounter(), small, viewer="Brom")
    context = linearizer.linearize(route(), traversal(anchor("e1", 0.05), reached("e2", "e1", 2.0)))
    assert context.omitted == ("e2",)
    assert "<ref:e2" not in context.text
    assert "...1 intermediate event..." in context.text
