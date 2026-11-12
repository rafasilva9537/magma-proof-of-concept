"""What the scene says about a turn, with no LLM: who sees it, and which part is action and which is speech.

Visibility comes from the scene metadata (`present`), never from a model: the
orchestrator of a story knows who is in the room. The speaker always sees their
own turn; the narrator is not a character and sees nothing by being the one who
narrates.

A turn in the narrative format `*action* "speech"` is split by its marks. Only
the action part is observable evidence, so anything not marked as action
counts as speech: a turn with no marks at all (as in LoCoMo) is all speech, kept
as written, and a narrator's turn is all action.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from magma.graph import EVERYONE
from magma.ingest import Turn
from narrative.config import NarrativeConfig

_ACTION = re.compile(r"\*([^*]+)\*")
_QUOTE = re.compile(r"[\"“”]")
_WHOLLY_QUOTED = re.compile(r"^[\"“]([^\"“”]*)[\"”]$")


@dataclass(frozen=True)
class TurnParts:
    action_text: str  # what anyone present could see happen
    speech_text: str  # what was said; its content is only a claim


def split_turn(turn: Turn, config: NarrativeConfig) -> TurnParts:
    if turn.speaker == config.narrator:
        return TurnParts(action_text=turn.text.strip(), speech_text="")
    text = turn.text.strip()
    actions = [match.strip() for match in _ACTION.findall(text)]
    if not actions:
        # Unmarked text (as in LoCoMo) is kept as written, quotes of titles included; only a turn that is one
        # quotation loses its marks.
        wholly_quoted = _WHOLLY_QUOTED.match(text)
        return TurnParts(action_text="", speech_text=wholly_quoted.group(1).strip() if wholly_quoted else text)
    without_actions = _QUOTE.sub("", _ACTION.sub(" ", text))
    return TurnParts(action_text=" ".join(actions), speech_text=" ".join(without_actions.split()))


def visible_to(turn: Turn, config: NarrativeConfig) -> frozenset[str]:
    """Who sees the turn: everyone present, and the speaker unless it is the narrator."""
    if EVERYONE in turn.present:
        return frozenset({EVERYONE})
    if turn.speaker == config.narrator:
        return turn.present
    return turn.present | {turn.speaker}


def describe_audience(visible: frozenset[str], speaker: str, viewer: str | None = None) -> str:
    """The listeners of a turn as text: "everyone", "no one" or "{A, C}", with the viewer as "you"."""
    if EVERYONE in visible:
        return "everyone"
    listeners = sorted(visible - {speaker})
    if not listeners:
        return "no one"
    return "{" + ", ".join(YOU_LISTENING if name == viewer else name for name in listeners) + "}"


YOU_LISTENING = "you"
