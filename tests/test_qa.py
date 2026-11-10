"""Unit tests for the adaptive QA prompt (`magma/qa.py`, Appendix C.2)."""

from __future__ import annotations

import numpy as np

from magma.config import Intent
from magma.llm import ChatRequest, ClientSettings, Completion, LLMClient
from magma.qa import (
    FACTUAL_INSTRUCTION,
    MULTI_HOP_INSTRUCTION,
    QA_TAG,
    TEMPORAL_INSTRUCTION,
    AdaptiveQASynthesizer,
    build_qa_messages,
    dynamic_instruction,
)
from magma.retrieve import LinearContext
from magma.router import RouterOutput


class CannedQABackend:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        return Completion(text=self.reply, prompt_tokens=120, completion_tokens=6, reasoning_tokens=0)


def routed(question: str, intent: Intent) -> RouterOutput:
    return RouterOutput(
        question=question,
        intent=intent,
        intent_reason=f"keyword {intent.value.lower()}",
        time_window=None,
        temporal_mentions=(),
        query_vector=np.array([1.0, 0.0], dtype=np.float32),
        keywords=("melanie",),
    )


def test_each_router_intent_maps_to_its_c2_instruction() -> None:
    assert dynamic_instruction(Intent.WHEN) == TEMPORAL_INSTRUCTION
    assert dynamic_instruction(Intent.WHY) == MULTI_HOP_INSTRUCTION
    assert dynamic_instruction(Intent.ENTITY) == FACTUAL_INSTRUCTION


def test_qa_messages_include_the_abstention_rule_the_dynamic_instruction_and_the_context() -> None:
    context = "<t:2023-05-08 13:00> Melanie: I painted a sunrise yesterday. <ref:e1>"
    messages = build_qa_messages("When did Melanie paint a sunrise?", context, Intent.WHEN)

    assert [message["role"] for message in messages] == ["system", "user"]
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert '"Information not found"' in system
    assert f"4. {TEMPORAL_INSTRUCTION}" in system
    assert "category_specific_constraints" not in system
    assert "category_specific_constraints" not in user
    assert context in user
    assert "- Question: When did Melanie paint a sunrise?" in user
    assert user.endswith("Answer:")


def test_synthesizer_calls_the_llm_under_the_qa_tag_and_strips_the_reply() -> None:
    backend = CannedQABackend("  7 May 2023 \n")
    llm = LLMClient(ClientSettings(model="test-model"), backend=backend, cache_dir=None)
    synthesizer = AdaptiveQASynthesizer(llm)
    context = LinearContext(
        text="<t:2023-05-08 13:00> Melanie: I painted a sunrise yesterday. <ref:e1>",
        included=("e1",),
        omitted=(),
        tokens=15,
    )

    synthesized = synthesizer.synthesize(routed("When did Melanie paint a sunrise?", Intent.WHEN), context)

    assert synthesized.answer == "7 May 2023"
    assert synthesized.dynamic_instruction == TEMPORAL_INSTRUCTION
    assert len(backend.requests) == 1
    assert "response_format" not in backend.requests[0]
    assert llm.usage[QA_TAG].calls == 1
