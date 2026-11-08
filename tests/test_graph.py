import json
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from magma.config import Layer
from magma.graph import (
    EVERYONE,
    Edge,
    GraphStore,
    InvalidGraphChange,
    Node,
    UnknownNode,
    load_graph,
    save_graph,
)
from magma.networkx_graph import NetworkXGraphStore
from magma.schema import MAGMA_PROFILE, MagmaEdge, MagmaNode, Profile, UnknownType

STORE_FACTORIES: list[Callable[[Profile], GraphStore]] = [NetworkXGraphStore]
START = datetime(2023, 5, 8, 13, 56)


@pytest.fixture(params=STORE_FACTORIES)
def make_store(request: pytest.FixtureRequest) -> Callable[[Profile], GraphStore]:
    factory: Callable[[Profile], GraphStore] = request.param
    return factory


@pytest.fixture
def store(make_store: Callable[[Profile], GraphStore]) -> GraphStore:
    return make_store(MAGMA_PROFILE)


def event(node_id: str, minute: int = 0, text: str = "") -> Node:
    return Node(id=node_id, type=MagmaNode.EVENT, text=text or node_id, timestamp=START + timedelta(minutes=minute))


def entity(name: str) -> Node:
    return Node(id=name, type=MagmaNode.ENTITY, text=name)


def edge(source: str, target: str, edge_type: MagmaEdge, weight: float = 1.0) -> Edge:
    return Edge(source=source, target=target, type=edge_type, weight=weight)


def small_conversation(store: GraphStore) -> None:
    """e1 -> e2 -> e3 in time; e1 causes e3; e1 and e2 are similar; e1 and e3 mention Melanie."""
    for minute, node_id in enumerate(["e1", "e2", "e3"]):
        store.add_node(event(node_id, minute))
    store.add_node(entity("Melanie"))
    store.add_edge(edge("e1", "e2", MagmaEdge.NEXT))
    store.add_edge(edge("e2", "e3", MagmaEdge.NEXT))
    store.add_edge(edge("e1", "e3", MagmaEdge.CAUSES, weight=0.8))
    store.add_edge(edge("e1", "e2", MagmaEdge.SIMILAR_TO, weight=0.71))
    store.add_edge(edge("e1", "Melanie", MagmaEdge.MENTIONS))
    store.add_edge(edge("e3", "Melanie", MagmaEdge.MENTIONS))


def test_nodes_come_back_in_insertion_order_and_by_type(store: GraphStore) -> None:
    small_conversation(store)
    assert [node.id for node in store.nodes()] == ["e1", "e2", "e3", "Melanie"]
    assert [node.id for node in store.nodes(MagmaNode.EVENT)] == ["e1", "e2", "e3"]
    assert store.node("e2").timestamp == START + timedelta(minutes=1)
    assert store.has_node("Melanie") and not store.has_node("Caroline")


def test_unknown_node_raises(store: GraphStore) -> None:
    with pytest.raises(UnknownNode):
        store.node("missing")
    with pytest.raises(UnknownNode):
        store.neighbors("missing")
    store.add_node(event("e1"))
    with pytest.raises(UnknownNode):
        store.add_edge(edge("e1", "missing", MagmaEdge.NEXT))


def test_node_must_match_the_profile(store: GraphStore) -> None:
    with pytest.raises(UnknownType):
        store.add_node(Node(id="x", type="EPISODE", text="x"))
    with pytest.raises(InvalidGraphChange):
        store.add_node(Node(id="e1", type=MagmaNode.EVENT, text="no timestamp"))
    with pytest.raises(InvalidGraphChange):
        store.add_node(Node(id="Melanie", type=MagmaNode.ENTITY, text="Melanie", timestamp=START))
    with pytest.raises(InvalidGraphChange):
        store.add_node(Node(id="e1", type=MagmaNode.EVENT, text="hidden", timestamp=START, visible_to=frozenset()))


def test_duplicate_node_is_rejected(store: GraphStore) -> None:
    store.add_node(event("e1"))
    with pytest.raises(InvalidGraphChange):
        store.add_node(event("e1", text="other"))


def test_edge_must_match_the_profile(store: GraphStore) -> None:
    store.add_node(event("e1"))
    store.add_node(entity("Melanie"))
    with pytest.raises(InvalidGraphChange):
        store.add_edge(edge("Melanie", "e1", MagmaEdge.MENTIONS))  # MENTIONS goes from event to entity
    with pytest.raises(InvalidGraphChange):
        store.add_edge(edge("e1", "Melanie", MagmaEdge.NEXT))
    with pytest.raises(InvalidGraphChange):
        store.add_edge(edge("e1", "e1", MagmaEdge.SIMILAR_TO))
    with pytest.raises(UnknownType):
        store.add_edge(Edge(source="e1", target="Melanie", type="REFERS_TO"))


def test_same_pair_holds_one_edge_per_type(store: GraphStore) -> None:
    small_conversation(store)
    between_e1_and_e2 = [e.type for e in store.edges() if {e.source, e.target} == {"e1", "e2"}]
    assert sorted(between_e1_and_e2) == [MagmaEdge.NEXT, MagmaEdge.SIMILAR_TO]
    with pytest.raises(InvalidGraphChange):
        store.add_edge(edge("e1", "e2", MagmaEdge.NEXT))


def test_undirected_edge_exists_in_both_directions(store: GraphStore) -> None:
    small_conversation(store)
    assert store.has_edge("e2", "e1", MagmaEdge.SIMILAR_TO)
    with pytest.raises(InvalidGraphChange):
        store.add_edge(edge("e2", "e1", MagmaEdge.SIMILAR_TO))
    assert not store.has_edge("e3", "e1", MagmaEdge.CAUSES)
    store.add_edge(edge("e3", "e1", MagmaEdge.CAUSES))  # a directed type may hold the reverse edge


def test_neighbors_follow_edges_in_both_directions(store: GraphStore) -> None:
    small_conversation(store)
    around_e2 = [(n.node.id, n.edge.type, n.is_outgoing) for n in store.neighbors("e2")]
    assert around_e2 == [
        ("e3", MagmaEdge.NEXT, True),
        ("e1", MagmaEdge.NEXT, False),
        ("e1", MagmaEdge.SIMILAR_TO, False),
    ]


def test_neighbors_by_layer(store: GraphStore) -> None:
    small_conversation(store)
    assert [n.node.id for n in store.neighbors("e1", Layer.CAUSAL)] == ["e3"]
    assert [n.node.id for n in store.neighbors("e1", Layer.SEMANTIC)] == ["e2"]
    assert [n.node.id for n in store.neighbors("Melanie", Layer.ENTITY)] == ["e1", "e3"]
    assert store.neighbors("e3", Layer.SEMANTIC) == []


def test_replace_node_keeps_edges(store: GraphStore) -> None:
    small_conversation(store)
    enriched = Node(
        id="e1",
        type=MagmaNode.EVENT,
        text="e1",
        timestamp=START,
        attributes={"topic": "music", "entities": ["Melanie"]},
    )
    store.replace_node(enriched)
    assert store.node("e1").attributes["topic"] == "music"
    assert len(store.neighbors("e1")) == 4
    with pytest.raises(InvalidGraphChange):
        store.replace_node(Node(id="e1", type=MagmaNode.ENTITY, text="e1"))
    with pytest.raises(UnknownNode):
        store.replace_node(event("missing"))


def test_backend_missing_storage_methods_cannot_be_created() -> None:
    class Incomplete(GraphStore):
        def edges(self) -> list[Edge]:
            return []

    with pytest.raises(TypeError):
        Incomplete(MAGMA_PROFILE)  # type: ignore[abstract]  # the point of the test


def test_visibility_defaults_to_everyone() -> None:
    public = event("e1")
    private = Node(id="e2", type=MagmaNode.EVENT, text="whisper", timestamp=START, visible_to=frozenset({"alice"}))
    assert public.visible_to == {EVERYONE}
    assert public.is_visible_to("bob")
    assert private.is_visible_to("alice")
    assert not private.is_visible_to("bob")
    assert private.is_visible_to(None)


def test_save_and_load_round_trip(store: GraphStore, make_store: Callable[[Profile], GraphStore], tmp_path: Path) -> None:
    small_conversation(store)
    store.replace_node(
        Node(
            id="e2",
            type=MagmaNode.EVENT,
            text="Melanie: I played the violin — à noite.",
            timestamp=START + timedelta(minutes=1),
            attributes={"dates_mentioned": ["2023-05-07"], "facts": [{"subject": "Melanie", "plays": "violin"}]},
            visible_to=frozenset({"alice", "bob"}),
        )
    )
    path = tmp_path / "graphs" / "conversation.json"
    save_graph(store, path)

    loaded = make_store(MAGMA_PROFILE)
    load_graph(path, loaded)
    assert loaded.nodes() == store.nodes()
    assert loaded.edges() == store.edges()

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["profile"] == "magma"
    assert written["nodes"][1]["visible_to"] == ["alice", "bob"]
    assert written["nodes"][1]["timestamp"] == "2023-05-08T13:57:00"


def test_load_checks_the_target_store(
    store: GraphStore, make_store: Callable[[Profile], GraphStore], tmp_path: Path
) -> None:
    small_conversation(store)
    path = tmp_path / "graph.json"
    save_graph(store, path)

    with pytest.raises(InvalidGraphChange):
        load_graph(path, store)  # not empty

    other_profile = MAGMA_PROFILE.extend("other")
    with pytest.raises(InvalidGraphChange):
        load_graph(path, make_store(other_profile))

    broken = tmp_path / "broken.json"
    broken.write_text('{"profile": "magma", "nodes": [{"id": "e1"}], "edges": []}', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_graph(broken, make_store(MAGMA_PROFILE))
