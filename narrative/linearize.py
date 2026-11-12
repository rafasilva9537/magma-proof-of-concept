"""Stage 4 for the narrative profile: the retrieved events in two sections, from one viewer's side.

- "What you witnessed": per event, what could be seen (the action) and the act
  of speaking ("B spoke to {A, C}"), without the content.
- "What was said": the claims of each event, with speaker, audience and, when
  it is not UNVERIFIED, the status the viewer sees with its evidence, e.g.
  `<t:...> B told {A, C}: "B was in the lab" [REFUTED by <ref:e35>] <ref:e12.c1>`.
  A speech with no claim extracted is written whole, so its content is not lost.

Facets are never written alone: a claim the beam search reached brings in its
event, and an event brings all its claims, so the budget folds an event with
its claims. Order and budgeting are the paper's (Eq. 7), from `GraphLinearizer`.
Printing UNVERIFIED on every claim would cost tokens and make the reader doubt
everything in speech-only conversations, so only other statuses are written.
"""

from __future__ import annotations

from dataclasses import replace

from magma.config import DEFAULT, Config
from magma.graph import GraphStore, Node
from magma.retrieve import GraphLinearizer, TokenCounter, Traversal, Visit, anchor_of, brevity_code
from narrative.config import DEFAULT_NARRATIVE, NarrativeConfig
from narrative.scene import describe_audience
from narrative.schema import ClaimStatus, ClaimType
from narrative.status import ClaimView, status_for

WITNESSED_HEADER = "What you witnessed:"
SAID_HEADER = "What was said:"
YOU = "You"


class ViewerLinearizer(GraphLinearizer):
    def __init__(
        self,
        graph: GraphStore,
        tokens: TokenCounter,
        config: Config = DEFAULT,
        viewer: str | None = None,
        narrative: NarrativeConfig = DEFAULT_NARRATIVE,
    ) -> None:
        """`graph` is the viewer's view (or the full store for the omniscient reader, `viewer=None`)."""
        super().__init__(graph, tokens, config)
        self._viewer = viewer
        self._narrative = narrative

    def _events(self, traversal: Traversal) -> list[Visit]:
        """One visit per event: a visited claim counts for its event, with the more salient of the two visits."""
        best: dict[str, Visit] = {}
        for visit in traversal.visits:
            event_id = anchor_of(self._graph, visit.node_id)
            if event_id is None:
                continue  # an entity: it guided the traversal, the events that mention it carry the content
            candidate = visit if event_id == visit.node_id else replace(visit, node_id=event_id)
            current = best.get(event_id)
            if current is None or _is_more_salient(candidate, current):
                best[event_id] = candidate
        return list(best.values())

    def _render(self, ordered: list[str], omitted: set[str]) -> str:
        witnessed: list[str] = []
        said: list[str] = []
        folded_run = 0
        for node_id in ordered:
            if node_id in omitted:
                folded_run += 1
                continue
            if folded_run:
                witnessed.append(brevity_code(folded_run))
                folded_run = 0
            event = self._graph.node(node_id)
            witnessed.append(self._witnessed_line(event))
            said.extend(self._said_lines(event))
        if folded_run:
            witnessed.append(brevity_code(folded_run))
        if not witnessed:
            return ""
        lines = [WITNESSED_HEADER, *witnessed]
        if said:
            lines += ["", SAID_HEADER, *said]
        return "\n".join(lines)

    def _witnessed_line(self, event: Node) -> str:
        speaker = _text(event, "speaker")
        action = _text(event, "action_text")
        speech = _text(event, "speech_text")
        parts: list[str] = []
        if action:
            is_narration = not speaker or speaker == self._narrative.narrator
            parts.append(action if is_narration else f"{self._who(speaker)}: *{action}*")
        if speech:
            parts.append(f"{self._who(speaker)} spoke to {describe_audience(event.visible_to, speaker, self._viewer)}")
        observed = "; ".join(parts) if parts else event.text
        return f"{_time(event)} {observed} <ref:{event.id}>"

    def _said_lines(self, event: Node) -> list[str]:
        speaker = _text(event, "speaker")
        speech = _text(event, "speech_text")
        if not speech:
            return []
        who = self._who(speaker)
        audience = describe_audience(event.visible_to, speaker, self._viewer)
        claims = self._claims_of(event)
        if not claims:
            return [f'{_time(event)} {who} said to {audience}: "{speech}" <ref:{event.id}>']
        lines: list[str] = []
        for claim in claims:
            verb = "denied to" if claim.attributes.get("claim_type") == ClaimType.DENIAL.value else "told"
            status = _status_text(status_for(self._graph, claim.id))
            lines.append(f'{_time(event)} {who} {verb} {audience}: "{claim.text}"{status} <ref:{claim.id}>')
        return lines

    def _claims_of(self, event: Node) -> list[Node]:
        claims: list[Node] = []
        for neighbor in self._graph.neighbors(event.id):
            if neighbor.is_outgoing and self._graph.profile.edge_type(neighbor.edge.type).is_anchoring:
                claims.append(neighbor.node)
        return claims

    def _who(self, speaker: str) -> str:
        return YOU if self._viewer is not None and speaker == self._viewer else speaker


def _is_more_salient(candidate: Visit, current: Visit) -> bool:
    """An anchor chosen by the channels beats a reached node; among reached ones, the higher s_uv."""
    if current.is_anchor:
        return False
    if candidate.is_anchor:
        return True
    return (candidate.transition_score or 0.0) > (current.transition_score or 0.0)


def _status_text(view: ClaimView) -> str:
    if view.status is ClaimStatus.UNVERIFIED:
        return ""
    references = ", ".join(f"<ref:{node_id}>" for node_id in view.evidence)
    return f" [{view.status.value} by {references}]"


def _time(event: Node) -> str:
    assert event.timestamp is not None  # only events are written, and every event is an anchor
    return f"<t:{event.timestamp:%Y-%m-%d %H:%M}>"


def _text(event: Node, name: str) -> str:
    value = event.attributes.get(name)
    return value if isinstance(value, str) else ""
