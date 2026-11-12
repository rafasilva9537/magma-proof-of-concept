"""C1 Full Context: the answer prompt reads the whole transcript instead of a retrieved context.

The baseline of Tab. 1 of the paper. Every turn is kept as written, one line per
turn in the format of Eq. 7 (`<t:τ> speaker: text <ref:eN>`, numbered like the
events of the graph memories, so a cited reference means the same turn in every
system). The question goes through the same router and the same C.2 prompt as
MAGMA: the memory is the only thing that differs.

The visible variant keeps only the turns the viewer witnessed (everyone
present, and the speaker of their own turn): the simplest visibility filter,
with no structure and no parameter. With no viewer, both variants read every
turn.

A transcript longer than the model's context window is not cut: the question
is left unanswered and the trace records the overflow, so the run goes on and
the report counts it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from magma.config import DEFAULT, Config
from magma.embed import Embedder
from magma.ingest import Turn
from magma.json_types import JSONObject, JSONValue
from magma.llm import ContextWindowExceeded, LLMClient, Usage
from magma.memory import MemorySystem, Reply
from magma.qa import AdaptiveQASynthesizer, AnswerSynthesizer
from magma.retrieve import LinearContext, TokenCounter
from magma.router import IntentClassifier, QueryRouter, Router, build_default_classifier
from magma.temporal import TemporalTagger
from narrative.config import DEFAULT_NARRATIVE, NarrativeConfig
from narrative.scene import visible_to


class FullContextMemory(MemorySystem):
    """Keeps the turns; answers from all of them, or from the ones the viewer saw (`visible_only`)."""

    def __init__(
        self,
        router: QueryRouter,
        synthesizer: AnswerSynthesizer,
        tokens: TokenCounter,
        llm: LLMClient,
        visible_only: bool = False,
        narrative: NarrativeConfig = DEFAULT_NARRATIVE,
    ) -> None:
        super().__init__()
        self._router = router
        self._synthesizer = synthesizer
        self._tokens = tokens
        self._llm = llm
        self._visible_only = visible_only
        self._narrative = narrative
        self._turns: list[Turn] = []

    @staticmethod
    def create(
        llm: LLMClient,
        embedder: Embedder,
        tagger: TemporalTagger,
        tokens: TokenCounter,
        config: Config = DEFAULT,
        classifier: IntentClassifier | None = None,
        visible_only: bool = False,
    ) -> FullContextMemory:
        intent_classifier = classifier or build_default_classifier(tagger, config.router, config.temporal)
        router = Router(embedder, tagger, intent_classifier, config.temporal)
        return FullContextMemory(router, AdaptiveQASynthesizer(llm), tokens, llm, visible_only)

    def consolidate(self) -> None:
        """Nothing runs in the background: a turn is kept as it arrives."""

    def usage(self) -> Usage:
        return self._llm.total()

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        lines = [self._line(number, turn) for number, turn in enumerate(self._turns, start=1)]
        (directory / "transcript.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _add_turn(self, turn: Turn) -> None:
        self._turns.append(turn)

    def _answer(self, question: str, reference_time: datetime, viewer: str | None) -> Reply:
        route = self._router.route(question, reference_time)
        context = self._context(viewer)
        trace: JSONObject = {
            "intent": route.intent.value,
            "intent_reason": route.intent_reason,
            "viewer": viewer,
            "visible_only": self._visible_only,
            "included": list[JSONValue](context.included),
            "context_tokens": context.tokens,
            "context_overflow": False,
        }
        try:
            synthesized = self._synthesizer.synthesize(route, context)
        except ContextWindowExceeded as error:
            trace["context_overflow"] = True
            trace["overflow_error"] = str(error)
            return Reply(answer="", trace=trace)
        trace["dynamic_instruction"] = synthesized.dynamic_instruction
        trace["messages"] = json.loads(json.dumps(synthesized.messages))
        return Reply(answer=synthesized.answer, trace=trace)

    def _context(self, viewer: str | None) -> LinearContext:
        """The transcript the prompt reads: every turn, or only those the viewer saw."""
        lines: list[str] = []
        included: list[str] = []
        for number, turn in enumerate(self._turns, start=1):
            if self._is_shown(turn, viewer):
                lines.append(self._line(number, turn))
                included.append(f"e{number}")
        text = "\n".join(lines)
        return LinearContext(text=text, included=tuple(included), omitted=(), tokens=self._tokens.count_tokens(text))

    def _is_shown(self, turn: Turn, viewer: str | None) -> bool:
        if not self._visible_only or viewer is None:
            return True
        visible = visible_to(turn, self._narrative)
        return "*" in visible or viewer in visible

    @staticmethod
    def _line(number: int, turn: Turn) -> str:
        return f"<t:{turn.timestamp:%Y-%m-%d %H:%M}> {turn.speaker}: {turn.text} <ref:e{number}>"
