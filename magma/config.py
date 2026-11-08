"""All hyperparameters in one place.

Values come from Table 6 of the MAGMA paper; where the paper gives a range or
nothing at all, a single value is chosen here. Ablations are config switches
applied at query time over the same graph.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class Layer(StrEnum):
    """The four edge layers of the memory graph."""

    TEMPORAL = "TEMPORAL"
    CAUSAL = "CAUSAL"
    SEMANTIC = "SEMANTIC"
    ENTITY = "ENTITY"


class Intent(StrEnum):
    """Query intent detected by the router."""

    WHY = "WHY"
    WHEN = "WHEN"
    ENTITY = "ENTITY"


class Ablation(StrEnum):
    """Ablation: run the system with one component removed, to measure its contribution.

    Table 4 of the paper removes one component; Table 5 keeps a single graph layer.
    """

    NO_ADAPTIVE_POLICY = "NO_ADAPTIVE_POLICY"
    NO_CAUSAL = "NO_CAUSAL"
    NO_TEMPORAL = "NO_TEMPORAL"
    NO_ENTITY = "NO_ENTITY"
    ONLY_CAUSAL = "ONLY_CAUSAL"
    ONLY_TEMPORAL = "ONLY_TEMPORAL"
    ONLY_ENTITY = "ONLY_ENTITY"

    @property
    def kept_layers(self) -> tuple[Layer, ...]:
        match self:
            case Ablation.NO_ADAPTIVE_POLICY:
                return tuple(Layer)
            case Ablation.NO_CAUSAL:
                return (Layer.TEMPORAL, Layer.SEMANTIC, Layer.ENTITY)
            case Ablation.NO_TEMPORAL:
                return (Layer.CAUSAL, Layer.SEMANTIC, Layer.ENTITY)
            case Ablation.NO_ENTITY:
                return (Layer.TEMPORAL, Layer.CAUSAL, Layer.SEMANTIC)
            case Ablation.ONLY_CAUSAL:
                return (Layer.CAUSAL,)
            case Ablation.ONLY_TEMPORAL:
                return (Layer.TEMPORAL,)
            case Ablation.ONLY_ENTITY:
                return (Layer.ENTITY,)


@dataclass(frozen=True)
class LayerWeights:
    """One weight per graph layer: a row of w_Tq."""

    temporal: float
    causal: float
    semantic: float
    entity: float

    def of(self, layer: Layer) -> float:
        match layer:
            case Layer.TEMPORAL:
                return self.temporal
            case Layer.CAUSAL:
                return self.causal
            case Layer.SEMANTIC:
                return self.semantic
            case Layer.ENTITY:
                return self.entity

    def keep_only(self, layers: tuple[Layer, ...]) -> LayerWeights:
        """Zero the weight of every layer not listed."""
        return LayerWeights(
            temporal=self.temporal if Layer.TEMPORAL in layers else 0.0,
            causal=self.causal if Layer.CAUSAL in layers else 0.0,
            semantic=self.semantic if Layer.SEMANTIC in layers else 0.0,
            entity=self.entity if Layer.ENTITY in layers else 0.0,
        )


@dataclass(frozen=True)
class IntentWeights:
    """The w_Tq matrix: one row of layer weights per query intent."""

    why: LayerWeights
    when: LayerWeights
    entity: LayerWeights

    def of(self, intent: Intent) -> LayerWeights:
        match intent:
            case Intent.WHY:
                return self.why
            case Intent.WHEN:
                return self.when
            case Intent.ENTITY:
                return self.entity

    def average(self) -> LayerWeights:
        rows = (self.why, self.when, self.entity)
        return LayerWeights(
            temporal=sum(r.temporal for r in rows) / 3,
            causal=sum(r.causal for r in rows) / 3,
            semantic=sum(r.semantic for r in rows) / 3,
            entity=sum(r.entity for r in rows) / 3,
        )


@dataclass(frozen=True)
class ChannelWeights:
    """Weight of each retrieval channel in the reciprocal rank fusion."""

    vector: float = 1.0
    keyword: float = 2.0
    time: float = 1.0


@dataclass(frozen=True)
class LLMConfig:
    temperature: float = 0.0
    seed: int = 42
    max_output_tokens: int = 1024


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    # Qwen3-Embedding is instruction-aware: the task description goes on queries only, never on stored texts.
    query_instruction: str = "Given a question about a conversation, retrieve the memory entries that answer it"
    batch_size: int = 32


class DatePreference(StrEnum):
    """How an ambiguous date ("in December", "on Friday") is read against the reference time."""

    PAST = "PAST"  # the most recent one that already happened
    CURRENT_PERIOD = "CURRENT_PERIOD"  # the one in the current week, month or year


@dataclass(frozen=True)
class TemporalConfig:
    languages: tuple[str, ...] = ("en",)
    # A question asks about what already happened; a turn talks about plans as well as about the past.
    question_preference: DatePreference = DatePreference.PAST
    mention_preference: DatePreference = DatePreference.CURRENT_PERIOD


@dataclass(frozen=True)
class LexicalIndexConfig:
    """BM25 of the lexical channel. The paper does not give values; these are the usual Okapi defaults."""

    k1: float = 1.5  # saturation of term frequency
    b: float = 0.75  # normalization by document length
    epsilon: float = 0.25  # floor, as a fraction of the mean IDF, for terms found in more than half the documents


@dataclass(frozen=True)
class IngestConfig:
    semantic_edge_threshold: float = 0.6  # provisional: set after measuring the embedding's cosine distribution
    semantic_edge_max_neighbors: int = 5
    causal_min_confidence: float = 0.5
    neighborhood_hops: int = 2


@dataclass(frozen=True)
class RetrievalConfig:
    rrf_k: int = 60
    top_k: int = 20
    channel_weights: ChannelWeights = ChannelWeights()


@dataclass(frozen=True)
class TraversalConfig:
    max_depth: int = 5
    max_nodes: int = 200
    drop_threshold: float = 0.15
    beam_width: int = 10
    decay: float = 0.9
    lambda1: float = 1.0
    lambda2: float = 0.5
    intent_weights: IntentWeights = IntentWeights(
        why=LayerWeights(temporal=1.5, causal=5.0, semantic=2.5, entity=2.5),
        when=LayerWeights(temporal=4.0, causal=3.0, semantic=2.5, entity=2.5),
        entity=LayerWeights(temporal=0.5, causal=3.0, semantic=2.5, entity=6.0),
    )


@dataclass(frozen=True)
class ContextConfig:
    token_budget: int = 2500


@dataclass(frozen=True)
class Config:
    llm: LLMConfig = LLMConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    lexical_index: LexicalIndexConfig = LexicalIndexConfig()
    temporal: TemporalConfig = TemporalConfig()
    ingest: IngestConfig = IngestConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    traversal: TraversalConfig = TraversalConfig()
    context: ContextConfig = ContextConfig()
    ablation: Ablation | None = None

    @property
    def active_layers(self) -> tuple[Layer, ...]:
        if self.ablation is None:
            return tuple(Layer)
        return self.ablation.kept_layers

    @property
    def adaptive(self) -> bool:
        return self.ablation is not Ablation.NO_ADAPTIVE_POLICY

    def layer_weights(self, intent: Intent) -> LayerWeights:
        """Weight vector w_Tq for an intent, after the active ablation.

        Without the adaptive policy every query gets the same row: the average of the three intents.
        """
        table = self.traversal.intent_weights
        row = table.of(intent) if self.adaptive else table.average()
        return row.keep_only(self.active_layers)

    def with_ablation(self, ablation: Ablation | None) -> Config:
        return replace(self, ablation=ablation)


DEFAULT = Config()
