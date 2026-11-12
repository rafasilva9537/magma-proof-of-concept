"""The MAGMA Narrativa memory: the MAGMA facade with observers, claims and viewer-scoped queries.

Ingestion is the narrative fast and slow paths over a `narrative` graph. A
query from a viewer runs the paper's Stages 1-4 on what that viewer can see:
the router and the stages are the same objects as in the magma profile, and
only the graph and indexes they read are the viewer's views. Stage 4 writes the
two sections of the narrative linearization. The answer prompt is the same
C.2, so the memory is the only thing that differs from the original.
"""

from __future__ import annotations

import queue
from datetime import datetime

from magma.bm25_index import BM25Index
from magma.config import DEFAULT, Config
from magma.embed import Embedder
from magma.faiss_index import FaissVectorIndex
from magma.graph import GraphStore
from magma.index import LexicalIndex, VectorIndex
from magma.llm import LLMClient
from magma.memory import MagmaMemory, Reply, build_trace
from magma.networkx_graph import NetworkXGraphStore
from magma.qa import AdaptiveQASynthesizer, AnswerSynthesizer
from magma.retrieve import BeamSearch, NodeVectors, Retriever, RRFAnchorSelector, TokenCounter
from magma.router import IntentClassifier, QueryRouter, Router, build_default_classifier
from magma.temporal import TemporalTagger
from narrative.config import DEFAULT_NARRATIVE, NarrativeConfig
from narrative.ingest import NarrativeConsolidation, NarrativeFastPath, SharedWrites
from narrative.linearize import ViewerLinearizer
from narrative.schema import NARRATIVE_PROFILE
from narrative.view import MaskedLexicalIndex, MaskedVectorIndex, ViewerGraph, visible_ids


class NarrativeMemory(MagmaMemory):
    def __init__(
        self,
        graph: GraphStore,
        fast_path: NarrativeFastPath,
        slow_path: NarrativeConsolidation,
        vectors: FaissVectorIndex,
        keywords: LexicalIndex,
        node_vectors: NodeVectors,
        router: QueryRouter,
        tokens: TokenCounter,
        synthesizer: AnswerSynthesizer,
        llm: LLMClient,
        config: Config = DEFAULT,
        narrative: NarrativeConfig = DEFAULT_NARRATIVE,
    ) -> None:
        self._keywords = keywords
        self._node_vectors = node_vectors  # entity vectors computed once, shared by every viewer
        self._tokens = tokens
        self._config = config
        self._narrative = narrative
        omniscient = self._retriever_for(graph, vectors, None)
        super().__init__(graph, fast_path, slow_path, vectors, router, omniscient, synthesizer, llm)

    @staticmethod
    def create(
        llm: LLMClient,
        embedder: Embedder,
        tagger: TemporalTagger,
        tokens: TokenCounter,
        config: Config = DEFAULT,
        classifier: IntentClassifier | None = None,
        narrative: NarrativeConfig = DEFAULT_NARRATIVE,
    ) -> NarrativeMemory:
        """An empty memory with the in-memory backends (NetworkX graph, FAISS, BM25) and the `narrative` profile."""
        graph = NetworkXGraphStore(NARRATIVE_PROFILE)
        vectors = FaissVectorIndex(embedder.dimension)
        keywords = BM25Index(config.lexical_index)
        consolidation_queue: queue.Queue[str] = queue.Queue()
        shared = SharedWrites(graph)
        fast_path = NarrativeFastPath(graph, vectors, keywords, embedder, consolidation_queue, shared, config.ingest, narrative)
        slow_path = NarrativeConsolidation(
            graph, llm, tagger, consolidation_queue, vectors, keywords, embedder, shared, config.ingest, config.temporal
        )
        intent_classifier = classifier or build_default_classifier(tagger, config.router, config.temporal)
        router = Router(embedder, tagger, intent_classifier, config.temporal)
        node_vectors = NodeVectors(graph, vectors, embedder)
        synthesizer = AdaptiveQASynthesizer(llm)
        return NarrativeMemory(
            graph, fast_path, slow_path, vectors, keywords, node_vectors, router, tokens, synthesizer, llm, config, narrative
        )

    def _answer(self, question: str, reference_time: datetime, viewer: str | None) -> Reply:
        route = self._router.route(question, reference_time)
        retrieval = self._retriever_for(self._graph, self._vectors, viewer).retrieve(route)
        synthesized = self._synthesizer.synthesize(route, retrieval.context)
        trace = build_trace(route, retrieval, synthesized)
        trace["viewer"] = viewer
        return Reply(answer=synthesized.answer, trace=trace)

    def _retriever_for(self, graph: GraphStore, vectors: VectorIndex, viewer: str | None) -> Retriever:
        """Stages 2-4 over what `viewer` can see; `None` reads the whole graph."""
        seen_graph: GraphStore = graph
        seen_vectors: VectorIndex = vectors
        seen_keywords: LexicalIndex = self._keywords
        if viewer is not None:
            view = ViewerGraph(graph, viewer)
            visible = visible_ids(view)
            seen_graph = view
            seen_vectors = MaskedVectorIndex(vectors, visible)
            seen_keywords = MaskedLexicalIndex(self._keywords, visible)
        return Retriever(
            RRFAnchorSelector(seen_graph, seen_vectors, seen_keywords, self._config),
            BeamSearch(seen_graph, self._node_vectors, self._config),
            ViewerLinearizer(seen_graph, self._tokens, self._config, viewer, self._narrative),
        )
