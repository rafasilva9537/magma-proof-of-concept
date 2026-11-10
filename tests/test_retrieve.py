"""Unit tests for Stages 2-4 of the query process (`magma/retrieve.py`), on small graphs built by hand."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from magma.bm25_index import BM25Index
from magma.config import DEFAULT, Ablation, Config, Intent
from magma.embed import Matrix, Vector, normalize
from magma.faiss_index import FaissVectorIndex
from magma.graph import Edge, Node
from magma.json_types import JSONValue
from magma.networkx_graph import NetworkXGraphStore
from magma.retrieve import (
    Anchor,
    AnchorSet,
    BeamSearch,
    ChannelRanks,
    GraphLinearizer,
    NodeVectors,
    Retriever,
    RRFAnchorSelector,
    Traversal,
    Visit,
    anchor_of,
    brevity_code,
    format_event,
    salience_order,
    transition_score,
)
from magma.router import RouterOutput
from magma.schema import MAGMA_PROFILE, EdgeType, MagmaNode, NodeRole, NodeType, Producer, Profile
from magma.temporal import TimeWindow

QUERY = np.array([1.0, 0.0, 0.0], dtype=np.float32)
MAY_8 = datetime(2023, 5, 8, 13, 56)


def at_cosine(value: float) -> Vector:
    """A unit vector whose cosine with QUERY is `value`."""
    unit: Vector = normalize(np.array([[value, math.sqrt(1 - value * value), 0.0]], dtype=np.float32))[0]
    return unit


class TableEmbedder:
    """Embeds each text by looking it up; counts the texts it embedded."""

    def __init__(self, table: dict[str, Vector]) -> None:
        self._table = table
        self.embedded: list[str] = []

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        self.embedded.extend(texts)
        return np.asarray([self._table[text] for text in texts], dtype=np.float32).reshape(len(texts), 3)

    def embed_query(self, text: str) -> Vector:
        return self._table[text]


class WordCounter:
    def count_tokens(self, text: str) -> int:
        return len(text.split())


class ToyMemory:
    def __init__(self, profile: Profile = MAGMA_PROFILE) -> None:
        self.graph = NetworkXGraphStore(profile)
        self.vectors = FaissVectorIndex(3)
        self.keywords = BM25Index()

    def event(
        self,
        node_id: str,
        text: str,
        cosine_to_query: float,
        when: datetime = MAY_8,
        dates: Sequence[tuple[datetime, datetime]] = (),
    ) -> None:
        mentioned: list[JSONValue] = [
            {"text": "a date", "start": start.isoformat(), "end": end.isoformat(), "granularity": "YEAR"}
            for start, end in dates
        ]
        self.graph.add_node(
            Node(id=node_id, type=MagmaNode.EVENT, text=text, timestamp=when, attributes={"dates_mentioned": mentioned})
        )
        self.vectors.add(node_id, at_cosine(cosine_to_query))
        self.keywords.add(node_id, text)

    def entity(self, node_id: str, name: str) -> None:
        self.graph.add_node(Node(id=node_id, type=MagmaNode.ENTITY, text=name))

    def link(self, source: str, target: str, edge_type: str) -> None:
        self.graph.add_edge(Edge(source=source, target=target, type=edge_type))


def route(
    intent: Intent = Intent.ENTITY,
    keywords: tuple[str, ...] = (),
    window: TimeWindow | None = None,
) -> RouterOutput:
    return RouterOutput(
        question="q",
        intent=intent,
        intent_reason="test",
        time_window=window,
        temporal_mentions=(),
        query_vector=QUERY,
        keywords=keywords,
    )


def anchors(*scored: tuple[str, float]) -> AnchorSet:
    return AnchorSet(
        anchors=tuple(Anchor(node_id=node_id, score=score, ranks=ChannelRanks()) for node_id, score in scored),
        time_channel_size=0,
    )


def with_traversal(
    beam_width: int = DEFAULT.traversal.beam_width,
    max_nodes: int = DEFAULT.traversal.max_nodes,
) -> Config:
    return replace(DEFAULT, traversal=replace(DEFAULT.traversal, beam_width=beam_width, max_nodes=max_nodes))


# --- Stage 2 ---------------------------------------------------------------


def five_events() -> ToyMemory:
    memory = ToyMemory()
    memory.event("e1", "Caroline: I love pottery", 0.9)
    memory.event("e2", "Melanie: the pottery class was really fun today", 0.5)
    memory.event("e3", "Caroline: thanks", 0.3)
    memory.event("e4", "Melanie: see you soon", 0.2)
    memory.event("e5", "Caroline: bye", 0.1)
    return memory


def test_rrf_sums_weighted_reciprocal_ranks() -> None:
    memory = five_events()
    selector = RRFAnchorSelector(memory.graph, memory.vectors, memory.keywords)

    found = selector.select(route(keywords=("pottery",)))

    by_id = {anchor.node_id: anchor for anchor in found.anchors}
    assert by_id["e1"].ranks == ChannelRanks(vector=1, keyword=1, time=None)
    assert by_id["e2"].ranks == ChannelRanks(vector=2, keyword=2, time=None)
    assert by_id["e1"].score == pytest.approx(1.0 / 61 + 2.0 / 61)  # w_vec = 1, w_keyword = 2, k = 60
    assert by_id["e3"].score == pytest.approx(1.0 / 63)  # vector only
    assert [anchor.node_id for anchor in found.anchors] == ["e1", "e2", "e3", "e4", "e5"]
    assert found.time_channel_size == 0


def test_top_k_keeps_the_best_anchors() -> None:
    memory = five_events()
    config = replace(DEFAULT, retrieval=replace(DEFAULT.retrieval, top_k=2))
    selector = RRFAnchorSelector(memory.graph, memory.vectors, memory.keywords, config)

    found = selector.select(route(keywords=("pottery",)))

    assert [anchor.node_id for anchor in found.anchors] == ["e1", "e2"]


def test_every_event_in_the_window_shares_the_mean_rank() -> None:
    memory = ToyMemory()
    memory.event("e1", "Caroline: hi", 0.9, when=datetime(2023, 4, 30))
    memory.event("e2", "Caroline: I went to a support group", 0.2, when=datetime(2023, 5, 7))
    memory.event("e3", "Melanie: nice", 0.1, when=datetime(2023, 5, 8))
    memory.event("e4", "Melanie: bye", 0.1, when=datetime(2023, 5, 25))
    selector = RRFAnchorSelector(memory.graph, memory.vectors, memory.keywords)
    may = TimeWindow(start=datetime(2023, 5, 1), end=datetime(2023, 6, 1))

    found = selector.select(route(window=may))

    by_id = {anchor.node_id: anchor for anchor in found.anchors}
    assert found.time_channel_size == 3
    assert by_id["e2"].ranks.time == 2.0  # (N + 1) / 2 with N = 3
    assert by_id["e2"].score == pytest.approx(1.0 / 62 + 1.0 / 62)  # 2nd in the vector channel, and the window
    assert by_id["e1"].ranks.time is None  # outside the window: only the vector channel


def test_a_mentioned_date_puts_the_event_in_the_window() -> None:
    # "We went camping last year", said in August 2023, for a question about 2022.
    memory = ToyMemory()
    year_2022 = (datetime(2022, 1, 1), datetime(2023, 1, 1))
    memory.event("e1", "Melanie: we went camping last year", 0.5, when=datetime(2023, 8, 14), dates=[year_2022])
    memory.event("e2", "Caroline: cool", 0.4, when=datetime(2023, 8, 14))
    window = TimeWindow(start=datetime(2022, 1, 1), end=datetime(2023, 1, 1))
    only_tau = replace(DEFAULT, retrieval=replace(DEFAULT.retrieval, time_channel_uses_mentioned_dates=False))

    with_dates = RRFAnchorSelector(memory.graph, memory.vectors, memory.keywords).select(route(window=window))
    tau_only = RRFAnchorSelector(memory.graph, memory.vectors, memory.keywords, only_tau).select(route(window=window))

    assert with_dates.time_channel_size == 1
    assert {anchor.node_id: anchor.ranks.time for anchor in with_dates.anchors}["e1"] == 1.0
    assert tau_only.time_channel_size == 0


def test_a_hit_on_a_facet_counts_for_its_anchor_with_the_best_rank() -> None:
    claim = NodeType(name="CLAIM", description="what was said", role=NodeRole.FACET, is_embedded=True)
    stated = EdgeType(
        name="STATED",
        description="the event states the claim",
        sources=frozenset({MagmaNode.EVENT.value}),
        targets=frozenset({"CLAIM"}),
        is_directed=True,
        producer=Producer.LLM_EXTRACT,
        phrase="{source} states {target}",
        is_anchoring=True,
    )
    memory = ToyMemory(MAGMA_PROFILE.extend("with-claims", node_types=(claim,), edge_types=(stated,)))
    memory.event("e1", "Caroline: hello", 0.2)
    memory.event("e2", "Melanie: I was at the lab", 0.3)
    memory.graph.add_node(Node(id="c1", type="CLAIM", text="Melanie was at the lab", timestamp=None))
    memory.vectors.add("c1", at_cosine(0.95))
    memory.link("e2", "c1", "STATED")
    selector = RRFAnchorSelector(memory.graph, memory.vectors, memory.keywords)

    found = selector.select(route())

    assert anchor_of(memory.graph, "c1") == "e2"
    assert [anchor.node_id for anchor in found.anchors] == ["e2", "e1"]  # the claim does not take a slot
    assert found.anchors[0].ranks.vector == 1  # the claim's rank, better than e2's own (2)


# --- Stage 3 ---------------------------------------------------------------


def causal_chain() -> ToyMemory:
    """e1 -NEXT-> e2, e1 -CAUSES-> e3, e3 -NEXT-> e4; e5 is unrelated."""
    memory = ToyMemory()
    memory.event("e1", "a", 0.9)
    memory.event("e2", "b", 0.6)
    memory.event("e3", "c", 0.6)
    memory.event("e4", "d", 0.6)
    memory.event("e5", "e", 0.6)
    memory.link("e1", "e2", "NEXT")
    memory.link("e1", "e3", "CAUSES")
    memory.link("e3", "e4", "NEXT")
    return memory


def beam(memory: ToyMemory, config: Config = DEFAULT, embedder: TableEmbedder | None = None) -> BeamSearch:
    node_vectors = NodeVectors(memory.graph, memory.vectors, embedder or TableEmbedder({}))
    return BeamSearch(memory.graph, node_vectors, config)


def test_transition_score_is_eq_5() -> None:
    assert transition_score(phi=5.0, similarity=0.6, lambda1=1.0, lambda2=0.5) == pytest.approx(math.exp(5.3))


def test_beam_keeps_the_best_neighbors_and_decays_the_path_score() -> None:
    memory = causal_chain()
    search = beam(memory, with_traversal(beam_width=1))

    found = search.traverse(route(Intent.WHY), anchors(("e1", 0.05)))

    visits = {visit.node_id: visit for visit in found.visits}
    assert list(visits) == ["e1", "e3", "e4"]  # depth 1: CAUSES (w 5.0) beats NEXT (w 1.5); width 1 drops e2
    s_e3 = math.exp(5.0 + 0.5 * 0.6)
    assert visits["e3"].score == pytest.approx(0.05 * 0.9 + s_e3)
    assert visits["e3"].transition_score == pytest.approx(s_e3)
    assert visits["e4"].score == pytest.approx(visits["e3"].score * 0.9 + math.exp(1.5 + 0.5 * 0.6))
    assert (visits["e4"].depth, visits["e4"].parent_id, visits["e4"].edge_type) == (2, "e3", "NEXT")


def test_a_neighbor_below_the_drop_threshold_is_skipped() -> None:
    memory = ToyMemory()
    memory.event("e1", "a", 0.9)
    memory.event("e2", "b", 0.1)
    memory.link("e1", "e2", "NEXT")

    found = beam(memory).traverse(route(), anchors(("e1", 0.05)))

    assert [visit.node_id for visit in found.visits] == ["e1"]
    assert found.dropped_by_node_type == {"EVENT": 1}
    assert found.transitions_checked == 1


def entity_bridge() -> tuple[ToyMemory, TableEmbedder]:
    """e1 and e5 both mention Caroline; nothing else links them."""
    memory = ToyMemory()
    memory.event("e1", "Caroline: I joined a group", 0.9)
    memory.event("e5", "Caroline: the group helped me", 0.7)
    memory.entity("ent:caroline", "Caroline")
    memory.link("e1", "ent:caroline", "MENTIONS")
    memory.link("e5", "ent:caroline", "MENTIONS")
    return memory, TableEmbedder({"Caroline": at_cosine(0.4)})


def test_entities_are_scored_by_the_encoding_of_their_name() -> None:
    memory, embedder = entity_bridge()

    found = beam(memory, embedder=embedder).traverse(route(Intent.ENTITY), anchors(("e1", 0.05)))

    visits = {visit.node_id: visit for visit in found.visits}
    assert list(visits) == ["e1", "ent:caroline", "e5"]
    assert visits["ent:caroline"].similarity == pytest.approx(0.4)
    assert visits["ent:caroline"].transition_score == pytest.approx(math.exp(6.0 + 0.5 * 0.4))
    assert embedder.embedded == ["Caroline"]  # once, then kept


def test_no_entity_ablation_never_traverses_entity_edges() -> None:
    memory, embedder = entity_bridge()
    search = beam(memory, DEFAULT.with_ablation(Ablation.NO_ENTITY), embedder)

    found = search.traverse(route(Intent.ENTITY), anchors(("e1", 0.05)))

    assert [visit.node_id for visit in found.visits] == ["e1"]


def test_only_causal_ablation_follows_causal_edges_alone() -> None:
    memory = causal_chain()

    search = beam(memory, DEFAULT.with_ablation(Ablation.ONLY_CAUSAL))

    found = search.traverse(route(Intent.WHEN), anchors(("e1", 0.05)))

    assert [visit.node_id for visit in found.visits] == ["e1", "e3"]  # e2 and e4 hang on NEXT edges


def test_without_adaptive_policy_every_intent_uses_the_mean_weights() -> None:
    memory = causal_chain()
    search = beam(memory, DEFAULT.with_ablation(Ablation.NO_ADAPTIVE_POLICY))

    found = search.traverse(route(Intent.ENTITY), anchors(("e1", 0.05)))

    e3 = {visit.node_id: visit for visit in found.visits}["e3"]
    mean_causal = (5.0 + 3.0 + 3.0) / 3
    assert e3.transition_score == pytest.approx(math.exp(mean_causal + 0.5 * 0.6))


def test_the_search_stops_at_the_node_budget() -> None:
    memory = causal_chain()

    search = beam(memory, with_traversal(max_nodes=2, beam_width=1))

    found = search.traverse(route(Intent.WHY), anchors(("e1", 0.05)))

    assert [visit.node_id for visit in found.visits] == ["e1", "e3"]


# --- Stage 4 ---------------------------------------------------------------


def reached(node_id: str, transition: float) -> Visit:
    return Visit(
        node_id=node_id, depth=1, score=transition, transition_score=transition, parent_id="e1", edge_type="NEXT"
    )


def anchor_visit(node_id: str, rrf: float) -> Visit:
    return Visit(node_id=node_id, depth=0, score=rrf, transition_score=None)


def timeline() -> ToyMemory:
    memory = ToyMemory()
    memory.event("e1", "Caroline: one", 0.5, when=datetime(2023, 5, 1, 10, 0))
    memory.event("e2", "Melanie: two", 0.5, when=datetime(2023, 5, 2, 10, 0))
    memory.event("e3", "Caroline: three", 0.5, when=datetime(2023, 5, 3, 10, 0))
    memory.event("e4", "Melanie: four", 0.5, when=datetime(2023, 5, 4, 10, 0))
    return memory


def test_each_event_is_written_as_eq_7() -> None:
    node = Node(id="e7", type=MagmaNode.EVENT, text="Caroline: hi", timestamp=MAY_8)

    assert format_event(node) == "<t:2023-05-08 13:56> Caroline: hi <ref:e7>"
    assert brevity_code(3) == "...3 intermediate events..."
    assert brevity_code(1) == "...1 intermediate event..."


def test_salience_puts_anchors_first_then_reached_nodes_by_transition_score() -> None:
    visits = [anchor_visit("e1", 0.02), reached("e2", 15.0), anchor_visit("e3", 0.05), reached("e4", 400.0)]

    assert [visit.node_id for visit in salience_order(visits)] == ["e3", "e1", "e4", "e2"]


def test_when_orders_by_time_and_entity_by_salience() -> None:
    memory = timeline()
    linearizer = GraphLinearizer(memory.graph, WordCounter())
    traversal = Traversal(
        visits=(anchor_visit("e3", 0.05), reached("e1", 400.0), reached("e4", 15.0), anchor_visit("e2", 0.02)),
        transitions_checked=0,
        dropped_by_node_type={},
    )

    when = linearizer.linearize(route(Intent.WHEN), traversal)
    entity = linearizer.linearize(route(Intent.ENTITY), traversal)

    assert when.included == ("e1", "e2", "e3", "e4")
    assert entity.included == ("e3", "e2", "e1", "e4")
    assert when.text.splitlines()[0] == "<t:2023-05-01 10:00> Caroline: one <ref:e1>"


def test_why_puts_causes_before_effects_even_at_the_same_time() -> None:
    memory = ToyMemory()
    memory.event("e1", "Caroline: I quit", 0.5)
    memory.event("e2", "Melanie: why?", 0.5)
    memory.event("e3", "Caroline: my boss yelled", 0.5)
    memory.link("e3", "e1", "CAUSES")
    traversal = Traversal(
        visits=(anchor_visit("e1", 0.05), anchor_visit("e2", 0.04), anchor_visit("e3", 0.03)),
        transitions_checked=0,
        dropped_by_node_type={},
    )

    why = GraphLinearizer(memory.graph, WordCounter()).linearize(route(Intent.WHY), traversal)
    no_causal = GraphLinearizer(memory.graph, WordCounter(), DEFAULT.with_ablation(Ablation.NO_CAUSAL))

    assert why.included == ("e2", "e3", "e1")  # e2 and e3 are ready first (graph order breaks the tie in time)
    assert no_causal.linearize(route(Intent.WHY), traversal).included == ("e1", "e2", "e3")


def test_over_the_budget_the_least_salient_events_fold_into_one_code() -> None:
    memory = timeline()
    # Each line has 5 words; a brevity code has 3. Two full lines and one code fit in 13.
    config = replace(DEFAULT, context=replace(DEFAULT.context, token_budget=13))
    traversal = Traversal(
        visits=(anchor_visit("e1", 0.05), anchor_visit("e4", 0.04), reached("e2", 15.0), reached("e3", 20.0)),
        transitions_checked=0,
        dropped_by_node_type={},
    )

    context = GraphLinearizer(memory.graph, WordCounter(), config).linearize(route(Intent.WHEN), traversal)

    assert context.included == ("e1", "e4")
    assert context.omitted == ("e2", "e3")  # least salient first
    assert context.text.splitlines()[1] == "...2 intermediate events..."
    assert context.tokens == 13


def test_entity_nodes_guide_the_search_but_are_not_written() -> None:
    memory, embedder = entity_bridge()
    retriever = Retriever.create(memory.graph, memory.vectors, memory.keywords, embedder, WordCounter())

    found = retriever.retrieve(route(Intent.ENTITY))

    assert "ent:caroline" in {visit.node_id for visit in found.traversal.visits}
    assert set(found.context.included) == {"e1", "e5"}
    assert "<ref:ent:caroline>" not in found.context.text
