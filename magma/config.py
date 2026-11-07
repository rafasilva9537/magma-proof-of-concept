"""All hyperparameters in one place.

Values come from Table 6 of the MAGMA paper; where the paper gives a range or
nothing at all, a single value is chosen here. Ablations are config switches
applied at query time over the same graph.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

Layer = Literal["TEMP", "CAUSAL", "SEM", "ENT"]
Intent = Literal["WHY", "WHEN", "ENTITY"]
Ablation = Literal[
    "no_adaptive_policy",
    "no_causal",
    "no_temporal",
    "no_entity",
    "only_causal",
    "only_temporal",
    "only_entity",
]

LAYERS: tuple[Layer, ...] = ("TEMP", "CAUSAL", "SEM", "ENT")
INTENTS: tuple[Intent, ...] = ("WHY", "WHEN", "ENTITY")

# Table 4 removes one component; Table 5 keeps a single graph layer.
ABLATION_LAYERS: dict[Ablation, tuple[Layer, ...]] = {
    "no_adaptive_policy": LAYERS,
    "no_causal": ("TEMP", "SEM", "ENT"),
    "no_temporal": ("CAUSAL", "SEM", "ENT"),
    "no_entity": ("TEMP", "CAUSAL", "SEM"),
    "only_causal": ("CAUSAL",),
    "only_temporal": ("TEMP",),
    "only_entity": ("ENT",),
}


@dataclass(frozen=True)
class LayerWeights:
    """One weight per graph layer: a row of w_Tq."""

    temp: float
    causal: float
    sem: float
    ent: float

    def of(self, layer: Layer) -> float:
        if layer == "TEMP":
            return self.temp
        if layer == "CAUSAL":
            return self.causal
        if layer == "SEM":
            return self.sem
        return self.ent

    def keep_only(self, layers: tuple[Layer, ...]) -> LayerWeights:
        """Zero the weight of every layer not listed."""
        return LayerWeights(
            temp=self.temp if "TEMP" in layers else 0.0,
            causal=self.causal if "CAUSAL" in layers else 0.0,
            sem=self.sem if "SEM" in layers else 0.0,
            ent=self.ent if "ENT" in layers else 0.0,
        )


@dataclass(frozen=True)
class IntentWeights:
    """The w_Tq matrix: one row of layer weights per query intent."""

    why: LayerWeights
    when: LayerWeights
    entity: LayerWeights

    def of(self, intent: Intent) -> LayerWeights:
        if intent == "WHY":
            return self.why
        if intent == "WHEN":
            return self.when
        return self.entity

    def average(self) -> LayerWeights:
        rows = (self.why, self.when, self.entity)
        return LayerWeights(
            temp=sum(r.temp for r in rows) / 3,
            causal=sum(r.causal for r in rows) / 3,
            sem=sum(r.sem for r in rows) / 3,
            ent=sum(r.ent for r in rows) / 3,
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
class IngestConfig:
    sem_threshold: float = 0.6  # provisional: set after measuring the embedding's cosine distribution
    sem_max_neighbors: int = 5
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
        why=LayerWeights(temp=1.5, causal=5.0, sem=2.5, ent=2.5),
        when=LayerWeights(temp=4.0, causal=3.0, sem=2.5, ent=2.5),
        entity=LayerWeights(temp=0.5, causal=3.0, sem=2.5, ent=6.0),
    )


@dataclass(frozen=True)
class ContextConfig:
    token_budget: int = 2500


@dataclass(frozen=True)
class Config:
    llm: LLMConfig = LLMConfig()
    ingest: IngestConfig = IngestConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    traversal: TraversalConfig = TraversalConfig()
    context: ContextConfig = ContextConfig()
    ablation: Ablation | None = None

    def __post_init__(self) -> None:
        if self.ablation is not None and self.ablation not in ABLATION_LAYERS:
            raise ValueError(f"unknown ablation {self.ablation!r}; expected one of {sorted(ABLATION_LAYERS)}")

    @property
    def active_layers(self) -> tuple[Layer, ...]:
        if self.ablation is None:
            return LAYERS
        return ABLATION_LAYERS[self.ablation]

    @property
    def adaptive(self) -> bool:
        return self.ablation != "no_adaptive_policy"

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
