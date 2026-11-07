"""All hyperparameters in one place.

Values come from Table 6 of the MAGMA paper; where the paper gives a range or
nothing at all, a single value is chosen here. Ablations are config switches
applied at query time over the same graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Mapping

LAYERS = ("TEMP", "CAUSAL", "SEM", "ENT")
INTENTS = ("WHY", "WHEN", "ENTITY")

# Table 4 removes one component; Table 5 keeps a single graph layer.
ABLATIONS = {
    "no_adaptive_policy": None,
    "no_causal": ("TEMP", "SEM", "ENT"),
    "no_temporal": ("CAUSAL", "SEM", "ENT"),
    "no_entity": ("TEMP", "CAUSAL", "SEM"),
    "only_causal": ("CAUSAL",),
    "only_temporal": ("TEMP",),
    "only_entity": ("ENT",),
}


def _frozen(d: Mapping) -> Mapping:
    return MappingProxyType({k: MappingProxyType(dict(v)) if isinstance(v, Mapping) else v for k, v in d.items()})


DEFAULT_INTENT_WEIGHTS = _frozen({
    "WHY":    {"CAUSAL": 5.0, "TEMP": 1.5, "ENT": 2.5, "SEM": 2.5},
    "WHEN":   {"TEMP": 4.0, "CAUSAL": 3.0, "ENT": 2.5, "SEM": 2.5},
    "ENTITY": {"ENT": 6.0, "CAUSAL": 3.0, "SEM": 2.5, "TEMP": 0.5},
})


@dataclass(frozen=True)
class LLMConfig:
    temperature: float = 0.0
    seed: int = 42
    max_output_tokens: int = 1024


@dataclass(frozen=True)
class IngestConfig:
    sem_threshold: float = 0.6
    sem_max_neighbors: int = 5
    causal_min_confidence: float = 0.5
    neighborhood_hops: int = 2


@dataclass(frozen=True)
class RetrievalConfig:
    rrf_k: int = 60
    top_k: int = 20
    vector_min_sim: float = 0.2
    channel_weights: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType({"vector": 1.0, "keyword": 2.0, "time": 1.0})
    )


@dataclass(frozen=True)
class TraversalConfig:
    max_depth: int = 5
    max_nodes: int = 200
    drop_threshold: float = 0.15
    beam_width: int = 10
    decay: float = 0.9
    lambda1: float = 1.0
    lambda2: float = 0.5
    intent_weights: Mapping[str, Mapping[str, float]] = DEFAULT_INTENT_WEIGHTS


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
    ablation: str | None = None

    def __post_init__(self):
        if self.ablation is not None and self.ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation {self.ablation!r}; expected one of {sorted(ABLATIONS)}")
        for intent in INTENTS:
            weights = self.traversal.intent_weights.get(intent)
            if weights is None or set(weights) != set(LAYERS):
                raise ValueError(f"intent {intent} must define a weight for each of {LAYERS}")

    @property
    def active_layers(self) -> tuple[str, ...]:
        layers = ABLATIONS.get(self.ablation) if self.ablation else None
        return layers or LAYERS

    @property
    def adaptive(self) -> bool:
        return self.ablation != "no_adaptive_policy"

    def layer_weights(self, intent: str) -> dict[str, float]:
        """Weight vector w_Tq for an intent, after the active ablation."""
        table = self.traversal.intent_weights
        if self.adaptive:
            weights = dict(table[intent])
        else:
            weights = {layer: sum(table[i][layer] for i in INTENTS) / len(INTENTS) for layer in LAYERS}
        return {layer: (w if layer in self.active_layers else 0.0) for layer, w in weights.items()}

    def with_ablation(self, ablation: str | None) -> "Config":
        return replace(self, ablation=ablation)


DEFAULT = Config()
