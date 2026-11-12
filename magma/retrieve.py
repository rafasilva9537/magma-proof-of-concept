"""Stages 2-4 of the query process (§3.3, Alg. 1): anchors, traversal and linearization.

- Stage 2 (Eq. 4): a weighted reciprocal rank fusion of the vector, keyword and
  time channels picks the anchors, the entry points into the graph.
- Stage 3 (Alg. 1, Eqs. 5-6): a heuristic beam search expands the anchors along
  the graph layers, steered by the intent weights w_Tq and by how similar each
  neighbor is to the query.
- Stage 4 (Eq. 7): the retrieved events are ordered for the intent, written as
  `<t:τ> content <ref:id>`, and, over the token budget, the least salient ones
  become "...N intermediate events...".

Each stage sits behind a small Protocol, so one can be replaced without
touching the others. Ablations (Tabs. 4 and 5) act here, at query time, on the
same graph: the edges of a removed layer are never traversed.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypedDict

from pydantic import TypeAdapter

from magma.config import DEFAULT, Config, Intent, Layer
from magma.embed import Embedder, Vector, cosine
from magma.graph import GraphStore, Node
from magma.index import LexicalIndex, VectorIndex
from magma.router import RouterOutput
from magma.schema import NodeRole
from magma.temporal import TimeWindow

# ---------------------------------------------------------------------------
# Stage 2: anchors by reciprocal rank fusion (Eq. 4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelRanks:
    """Where an anchor stood in each channel; None when the channel did not return it."""

    vector: int | None = None  # 1-based
    keyword: int | None = None  # 1-based
    time: float | None = None  # every event in the window shares the mean rank (N + 1) / 2


@dataclass(frozen=True)
class Anchor:
    node_id: str
    score: float  # weighted RRF of Eq. 4
    ranks: ChannelRanks


@dataclass(frozen=True)
class AnchorSet:
    anchors: tuple[Anchor, ...]  # best first
    time_channel_size: int  # N: events in the time window (0 without a window)


class AnchorSelector(Protocol):
    """Stage 2: the entry points of the traversal."""

    def select(self, route: RouterOutput) -> AnchorSet: ...


class MentionedDate(TypedDict):
    """One item of `dates_mentioned` in A_i, as the slow path writes it."""

    text: str
    start: str
    end: str
    granularity: str


_MENTIONED_DATES = TypeAdapter(list[MentionedDate])


def mentioned_windows(node: Node) -> list[TimeWindow]:
    """The resolved dates a node's content mentions (the time of the fact, not of the turn)."""
    records = _MENTIONED_DATES.validate_python(node.attributes.get("dates_mentioned", []))
    return [
        TimeWindow(start=datetime.fromisoformat(record["start"]), end=datetime.fromisoformat(record["end"]))
        for record in records
    ]


def is_in_time_window(node: Node, window: TimeWindow, uses_mentioned_dates: bool) -> bool:
    """τ inside [τs, τe], or, when allowed, a mentioned date overlapping it."""
    if node.timestamp is not None and window.contains(node.timestamp):
        return True
    if not uses_mentioned_dates:
        return False
    return any(window.overlaps(mentioned) for mentioned in mentioned_windows(node))


def events_in_window(graph: GraphStore, window: TimeWindow | None, uses_mentioned_dates: bool) -> list[Node]:
    """The anchors of the time channel, in graph order; none without a window."""
    if window is None:
        return []
    return [
        node
        for node in graph.nodes()
        if _role(graph, node) is NodeRole.ANCHOR and is_in_time_window(node, window, uses_mentioned_dates)
    ]


def anchor_of(graph: GraphStore, node_id: str) -> str | None:
    """The anchor a search hit counts for: itself, the anchor of a facet, or None for an abstract node."""
    node = graph.node(node_id)
    match _role(graph, node):
        case NodeRole.ANCHOR:
            return node.id
        case NodeRole.FACET:
            for neighbor in graph.neighbors(node.id):
                is_anchoring = graph.profile.edge_type(neighbor.edge.type).is_anchoring
                if is_anchoring and not neighbor.is_outgoing:
                    return neighbor.node.id
            raise ValueError(f"facet {node.id!r} has no anchor")
        case NodeRole.ABSTRACT:
            return None


class RRFAnchorSelector:
    """Weighted RRF over three channels: score(n) = Σ_c w_c / (k + r_c(n)).

    A hit on a facet counts for its anchor, with the best rank among them, so
    the top-K holds distinct anchors.
    """

    def __init__(
        self,
        graph: GraphStore,
        vectors: VectorIndex,
        keywords: LexicalIndex,
        config: Config = DEFAULT,
    ) -> None:
        self._graph = graph
        self._vectors = vectors
        self._keywords = keywords
        self._retrieval = config.retrieval

    def select(self, route: RouterOutput) -> AnchorSet:
        top_k = self._retrieval.top_k
        vector_ranks = self._anchor_ranks([hit.node_id for hit in self._vectors.search(route.query_vector, top_k)])
        keyword_ranks = self._anchor_ranks(
            [hit.node_id for hit in self._keywords.search(" ".join(route.keywords), top_k)]
        )
        in_window = events_in_window(self._graph, route.time_window, self._retrieval.time_channel_uses_mentioned_dates)
        time_rank = (len(in_window) + 1) / 2
        time_members = {node.id for node in in_window}

        position = {node.id: index for index, node in enumerate(self._graph.nodes())}
        candidates = set(vector_ranks) | set(keyword_ranks) | time_members
        anchors = [
            self._fuse(node_id, vector_ranks, keyword_ranks, time_rank if node_id in time_members else None)
            for node_id in candidates
        ]
        anchors.sort(key=lambda anchor: (-anchor.score, position[anchor.node_id]))
        return AnchorSet(anchors=tuple(anchors[:top_k]), time_channel_size=len(in_window))

    def _anchor_ranks(self, hit_ids: list[str]) -> dict[str, int]:
        ranks: dict[str, int] = {}
        for rank, hit_id in enumerate(hit_ids, start=1):
            anchor_id = anchor_of(self._graph, hit_id)
            if anchor_id is not None and anchor_id not in ranks:
                ranks[anchor_id] = rank
        return ranks

    def _fuse(
        self,
        node_id: str,
        vector_ranks: dict[str, int],
        keyword_ranks: dict[str, int],
        time_rank: float | None,
    ) -> Anchor:
        ranks = ChannelRanks(vector=vector_ranks.get(node_id), keyword=keyword_ranks.get(node_id), time=time_rank)
        weights = self._retrieval.channel_weights
        k = self._retrieval.rrf_k
        score = 0.0
        if ranks.vector is not None:
            score += weights.vector / (k + ranks.vector)
        if ranks.keyword is not None:
            score += weights.keyword / (k + ranks.keyword)
        if ranks.time is not None:
            score += weights.time / (k + ranks.time)
        return Anchor(node_id=node_id, score=score, ranks=ranks)


# ---------------------------------------------------------------------------
# Stage 3: heuristic beam search (Alg. 1, Eqs. 5-6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Visit:
    """A node the traversal kept, and the step that brought it in."""

    node_id: str
    depth: int  # 0 for anchors
    score: float  # cumulative score of Alg. 1; for an anchor, its RRF score
    transition_score: float | None  # s_uv of Eq. 5 for the step that kept it; None for an anchor
    parent_id: str | None = None
    edge_type: str | None = None
    similarity: float | None = None  # sim(n_j, q) of Eq. 5

    @property
    def is_anchor(self) -> bool:
        return self.transition_score is None


@dataclass(frozen=True)
class Traversal:
    visits: tuple[Visit, ...]  # anchors first, then depth by depth, best first within a depth
    transitions_checked: int  # neighbor steps scored or dropped
    dropped_by_node_type: dict[str, int]  # steps to a neighbor less similar to the query than the drop threshold


class GraphTraversal(Protocol):
    """Stage 3: expands the anchors into the retrieved subgraph."""

    def traverse(self, route: RouterOutput, anchors: AnchorSet) -> Traversal: ...


def transition_score(phi: float, similarity: float, lambda1: float, lambda2: float) -> float:
    """Eq. 5: S(n_j | n_i, q) = exp(λ1·φ(type(e_ij), T_q) + λ2·sim(n_j, q))."""
    return math.exp(lambda1 * phi + lambda2 * similarity)


class NodeVectors:
    """The vector v of every node the beam search scores.

    An embedded node (an event) has its vector in the index. A node that is not
    embedded (an entity) gets the encoding of its content, its name, the same
    rule Alg. 2 applies to events; it is computed the first time the node is
    reached and kept, so ingestion does not change.
    """

    def __init__(self, graph: GraphStore, vectors: VectorIndex, embedder: Embedder) -> None:
        self._graph = graph
        self._vectors = vectors
        self._embedder = embedder
        self._computed: dict[str, Vector] = {}

    def similarities(self, nodes: list[Node], query_vector: Vector) -> dict[str, float]:
        """cos(v, v_q) for each node, by id."""
        self._compute_missing(nodes)
        return {node.id: cosine(self._vector(node), query_vector) for node in nodes}

    def _vector(self, node: Node) -> Vector:
        if self._is_embedded(node):
            return self._vectors.vector(node.id)
        return self._computed[node.id]

    def _compute_missing(self, nodes: list[Node]) -> None:
        missing: dict[str, Node] = {}
        for node in nodes:
            if not self._is_embedded(node) and node.id not in self._computed:
                missing[node.id] = node
        if not missing:
            return
        encoded = self._embedder.embed_documents([node.text for node in missing.values()])
        for node_id, vector in zip(missing, encoded, strict=True):
            self._computed[node_id] = vector

    def _is_embedded(self, node: Node) -> bool:
        return self._graph.profile.node_type(node.type).is_embedded


class BeamSearch:
    """Alg. 1: per depth, score every unvisited neighbor of the frontier and keep the BeamWidth best.

    s_uv = exp(λ1·w_Tq[layer] + λ2·sim(v, q)) and score_v = score_u·γ + s_uv. A
    neighbor less similar to the query than the drop threshold is skipped. The
    search stops at MaxDepth, when nothing new is reachable, or when the visited
    set reaches the node budget.
    """

    def __init__(self, graph: GraphStore, node_vectors: NodeVectors, config: Config = DEFAULT) -> None:
        self._graph = graph
        self._node_vectors = node_vectors
        self._config = config

    def traverse(self, route: RouterOutput, anchors: AnchorSet) -> Traversal:
        traversal = self._config.traversal
        visited: dict[str, Visit] = {}
        for anchor in anchors.anchors:
            visited[anchor.node_id] = Visit(node_id=anchor.node_id, depth=0, score=anchor.score, transition_score=None)
        frontier = [anchor.node_id for anchor in anchors.anchors]
        checked = 0
        dropped: dict[str, int] = {}

        for depth in range(1, traversal.max_depth + 1):
            steps = self._steps_from(frontier, visited)
            similarities = self._node_vectors.similarities(
                list({step.neighbor.id: step.neighbor for step in steps}.values()), route.query_vector
            )
            candidates: dict[str, Visit] = {}
            for step in steps:
                checked += 1
                similarity = similarities[step.neighbor.id]
                if similarity < traversal.drop_threshold:
                    dropped[step.neighbor.type] = dropped.get(step.neighbor.type, 0) + 1
                    continue
                visit = self._score(route.intent, step, similarity, depth)
                best = candidates.get(visit.node_id)
                if best is None or visit.score > best.score:
                    candidates[visit.node_id] = visit
            if not candidates:
                break
            # sorted() is stable: equal scores keep the order in which the neighbors were reached.
            kept = sorted(candidates.values(), key=lambda visit: -visit.score)[: traversal.beam_width]
            for visit in kept:
                visited[visit.node_id] = visit
            frontier = [visit.node_id for visit in kept]
            if len(visited) >= traversal.max_nodes:
                break
        return Traversal(visits=tuple(visited.values()), transitions_checked=checked, dropped_by_node_type=dropped)

    def _steps_from(self, frontier: list[str], visited: dict[str, Visit]) -> list[_Step]:
        """Every edge from the frontier to an unvisited node, over the layers the ablation keeps."""
        active_layers = self._config.active_layers
        steps: list[_Step] = []
        for node_id in frontier:
            for neighbor in self._graph.neighbors(node_id):
                if neighbor.node.id in visited:
                    continue
                edge_type = self._graph.profile.edge_type(neighbor.edge.type)
                # An anchoring edge has no layer: a facet is read together with its anchor, not reached by a hop.
                if edge_type.layer is None or edge_type.layer not in active_layers:
                    continue
                steps.append(
                    _Step(
                        parent=visited[node_id],
                        neighbor=neighbor.node,
                        edge_type=edge_type.name,
                        layer=edge_type.layer,
                        weight_multiplier=edge_type.weight_multiplier,
                    )
                )
        return steps

    def _score(self, intent: Intent, step: _Step, similarity: float, depth: int) -> Visit:
        traversal = self._config.traversal
        phi = self._config.layer_weights(intent).of(step.layer) * step.weight_multiplier  # Eq. 6
        s_uv = transition_score(phi, similarity, traversal.lambda1, traversal.lambda2)
        return Visit(
            node_id=step.neighbor.id,
            depth=depth,
            score=step.parent.score * traversal.decay + s_uv,
            transition_score=s_uv,
            parent_id=step.parent.node_id,
            edge_type=step.edge_type,
            similarity=similarity,
        )


@dataclass(frozen=True)
class _Step:
    parent: Visit
    neighbor: Node
    edge_type: str
    layer: Layer
    weight_multiplier: float


# ---------------------------------------------------------------------------
# Stage 4: linearization (Eq. 7) with salience-based token budgeting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearContext:
    text: str  # C_prompt
    included: tuple[str, ...]  # node ids written in full, in prompt order
    omitted: tuple[str, ...]  # node ids folded into brevity codes, least salient first
    tokens: int


class ContextLinearizer(Protocol):
    """Stage 4: turns the retrieved subgraph into the context of the answer prompt."""

    def linearize(self, route: RouterOutput, traversal: Traversal) -> LinearContext: ...


class TokenCounter(Protocol):
    def count_tokens(self, text: str) -> int: ...


def salience_order(visits: list[Visit]) -> list[Visit]:
    """Most salient first: the anchors by RRF score, then the other nodes by the s_uv that brought them in.

    The anchors come first because the three channels chose them; their RRF
    scores (~0.01-0.07) are on another scale than any s_uv (from e^0.5 up), so
    the two are never compared.
    """
    anchors = [visit for visit in visits if visit.is_anchor]
    reached = [visit for visit in visits if not visit.is_anchor]
    anchors.sort(key=lambda visit: -visit.score)
    reached.sort(key=lambda visit: -(visit.transition_score or 0.0))
    return anchors + reached


def format_event(node: Node) -> str:
    """One block of Eq. 7: <t:τ_i> content <ref:id>."""
    assert node.timestamp is not None  # only anchors are written, and every anchor has a timestamp
    return f"<t:{node.timestamp:%Y-%m-%d %H:%M}> {node.text} <ref:{node.id}>"


def brevity_code(count: int) -> str:
    noun = "event" if count == 1 else "events"
    return f"...{count} intermediate {noun}..."


class GraphLinearizer:
    """Orders the retrieved events for the intent and fits them in the token budget.

    WHEN: by timestamp. WHY: causes before effects (topological sort of the
    causal edges among them; ties and unrelated events by timestamp). ENTITY:
    by salience. Entity nodes guide the traversal but are not written: the
    events that mention them carry the content. Over the budget, the least
    salient event is folded into a brevity code, one at a time; consecutive
    folded events share one code.
    """

    def __init__(self, graph: GraphStore, tokens: TokenCounter, config: Config = DEFAULT) -> None:
        self._graph = graph
        self._tokens = tokens
        self._config = config

    def linearize(self, route: RouterOutput, traversal: Traversal) -> LinearContext:
        events = self._events(traversal)
        ordered = self._order(route.intent, events)
        least_salient_first = [visit.node_id for visit in reversed(salience_order(events))]
        omitted: list[str] = []
        text = self._render(ordered, set(omitted))
        tokens = self._tokens.count_tokens(text)
        while tokens > self._config.context.token_budget and len(omitted) < len(ordered):
            omitted.append(least_salient_first[len(omitted)])
            text = self._render(ordered, set(omitted))
            tokens = self._tokens.count_tokens(text)
        folded = set(omitted)
        return LinearContext(
            text=text,
            included=tuple(node_id for node_id in ordered if node_id not in folded),
            omitted=tuple(omitted),
            tokens=tokens,
        )

    def _events(self, traversal: Traversal) -> list[Visit]:
        """The visits written as blocks: the anchors the traversal kept."""
        return [visit for visit in traversal.visits if self._is_event(visit.node_id)]

    def _order(self, intent: Intent, events: list[Visit]) -> list[str]:
        match intent:
            case Intent.WHEN:
                return self._by_time([visit.node_id for visit in events])
            case Intent.WHY:
                return self._causes_first([visit.node_id for visit in events])
            case Intent.ENTITY:
                return [visit.node_id for visit in salience_order(events)]

    def _by_time(self, node_ids: list[str]) -> list[str]:
        position = self._positions()
        return sorted(node_ids, key=lambda node_id: (self._time_of(node_id), position[node_id]))

    def _causes_first(self, node_ids: list[str]) -> list[str]:
        """Kahn's algorithm over the causal edges among the events, the earliest ready event first."""
        selected = set(node_ids)
        position = self._positions()
        effects: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        pending_causes: dict[str, int] = {node_id: 0 for node_id in node_ids}
        if Layer.CAUSAL in self._config.active_layers:
            for node_id in node_ids:
                for neighbor in self._graph.neighbors(node_id, Layer.CAUSAL):
                    if neighbor.is_outgoing and neighbor.node.id in selected:
                        effects[node_id].append(neighbor.node.id)
                        pending_causes[neighbor.node.id] += 1
        ready: list[tuple[datetime, int, str]] = []
        for node_id in node_ids:
            if pending_causes[node_id] == 0:
                heapq.heappush(ready, (self._time_of(node_id), position[node_id], node_id))
        ordered: list[str] = []
        while ready:
            _, _, node_id = heapq.heappop(ready)
            ordered.append(node_id)
            for effect in effects[node_id]:
                pending_causes[effect] -= 1
                if pending_causes[effect] == 0:
                    heapq.heappush(ready, (self._time_of(effect), position[effect], effect))
        # Causes always precede their effects in time, so the causal graph has no cycle; if one ever appeared,
        # the events left in it would still be written, in time order.
        written = set(ordered)
        left = [node_id for node_id in node_ids if node_id not in written]
        return ordered + self._by_time(left)

    def _render(self, ordered: list[str], omitted: set[str]) -> str:
        lines: list[str] = []
        folded_run = 0
        for node_id in ordered:
            if node_id in omitted:
                folded_run += 1
                continue
            if folded_run:
                lines.append(brevity_code(folded_run))
                folded_run = 0
            lines.append(format_event(self._graph.node(node_id)))
        if folded_run:
            lines.append(brevity_code(folded_run))
        return "\n".join(lines)

    def _is_event(self, node_id: str) -> bool:
        return _role(self._graph, self._graph.node(node_id)) is NodeRole.ANCHOR

    def _time_of(self, node_id: str) -> datetime:
        timestamp = self._graph.node(node_id).timestamp
        assert timestamp is not None  # only anchors are ordered
        return timestamp

    def _positions(self) -> dict[str, int]:
        return {node.id: index for index, node in enumerate(self._graph.nodes())}


# ---------------------------------------------------------------------------
# Stages 2-4 together
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Retrieval:
    anchors: AnchorSet
    traversal: Traversal
    context: LinearContext


class Retriever:
    """Runs Stages 2, 3 and 4 on the output of the router."""

    def __init__(self, anchors: AnchorSelector, traversal: GraphTraversal, linearizer: ContextLinearizer) -> None:
        self._anchors = anchors
        self._traversal = traversal
        self._linearizer = linearizer

    @staticmethod
    def create(
        graph: GraphStore,
        vectors: VectorIndex,
        keywords: LexicalIndex,
        embedder: Embedder,
        tokens: TokenCounter,
        config: Config = DEFAULT,
    ) -> Retriever:
        """The stages of the paper: weighted RRF, beam search and the graph linearizer."""
        return Retriever(
            RRFAnchorSelector(graph, vectors, keywords, config),
            BeamSearch(graph, NodeVectors(graph, vectors, embedder), config),
            GraphLinearizer(graph, tokens, config),
        )

    def retrieve(self, route: RouterOutput) -> Retrieval:
        anchors = self._anchors.select(route)
        traversal = self._traversal.traverse(route, anchors)
        context = self._linearizer.linearize(route, traversal)
        return Retrieval(anchors=anchors, traversal=traversal, context=context)


def _role(graph: GraphStore, node: Node) -> NodeRole:
    return graph.profile.node_type(node.type).role
