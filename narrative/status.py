"""The status of a claim as a viewer sees it, read from the epistemic effect of the edges that reach it.

Given the graph the viewer can see (`ViewerGraph`), only the evidence that
viewer witnessed counts: whoever did not see the proof still finds the claim
unverified. The effect declared by each edge type decides, never its name, so a
new refuting edge type counts without code here.

When the evidence disagrees, the stronger kind wins: a fact that refutes, then a
fact that confirms, then a newer version of the same speaker, then a claim of
another speaker. Facts come before what anyone said, because speech is never
proof.
"""

from __future__ import annotations

from dataclasses import dataclass

from magma.graph import GraphStore
from magma.schema import EpistemicEffect
from narrative.schema import ClaimStatus


@dataclass(frozen=True)
class ClaimView:
    status: ClaimStatus
    evidence: tuple[str, ...]  # ids of the nodes behind the status, in graph order; empty when UNVERIFIED


# Effect of an edge -> status it gives, strongest first. Only CONTRADICTS counts in both directions:
# the other effects act on the claim the edge points to.
_PRECEDENCE: tuple[tuple[EpistemicEffect, ClaimStatus], ...] = (
    (EpistemicEffect.REFUTES, ClaimStatus.REFUTED),
    (EpistemicEffect.SUPPORTS, ClaimStatus.SUPPORTED),
    (EpistemicEffect.RETRACTS, ClaimStatus.RETRACTED),
    (EpistemicEffect.CONTRADICTS, ClaimStatus.CONTESTED),
)


def status_for(graph: GraphStore, claim_id: str) -> ClaimView:
    """`graph` is the viewer's view; pass the full store for the omniscient status."""
    evidence: dict[EpistemicEffect, list[str]] = {}
    for neighbor in graph.neighbors(claim_id):
        effect = graph.profile.edge_type(neighbor.edge.type).effect
        if effect is EpistemicEffect.NONE:
            continue
        points_at_claim = not neighbor.is_outgoing
        if points_at_claim or effect is EpistemicEffect.CONTRADICTS:
            evidence.setdefault(effect, []).append(neighbor.node.id)
    for effect, status in _PRECEDENCE:
        if effect in evidence:
            return ClaimView(status=status, evidence=tuple(evidence[effect]))
    return ClaimView(status=ClaimStatus.UNVERIFIED, evidence=())
