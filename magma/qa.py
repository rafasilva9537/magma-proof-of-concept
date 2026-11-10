"""Context synthesis (Appendix C.2): answers a question from the linearized context and the router's intent.

The system prompt states the grounding rules (use only the context, reply
"Information not found" when the answer is absent, stay concise) and injects a
fourth instruction chosen from the router's intent T_q, without benchmark
category labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from magma.config import Intent
from magma.llm import LLMClient, Message
from magma.retrieve import LinearContext
from magma.router import RouterOutput

QA_TAG = "qa"

# Appendix C.2 of the paper, split into the role + instructions (system) and the context + query (user).
# The orphan `{category_specific_constraints}` placeholder printed in the figure is omitted: the paper only
# defines the four Dynamic Instruction Injection Candidates below, injected at item 4.
QA_SYSTEM_PROMPT = """System Role: You are a precision QA assistant operating on retrieved memory contexts. Your goal is to answer the user's question accurately using only the provided information.

Instructions:
1. Use ONLY information explicitly stated in the context.
2. If the answer is not present, respond exactly with "Information not found".
3. Be concise (typically 1–10 words) unless detailed reasoning is required.
4. {dynamic_instruction}"""

QA_USER_PROMPT = """Context:
{context}

Current Query:
- Question: {question}

Answer:"""

MULTI_HOP_INSTRUCTION = (
    "Connect related facts across different nodes. For comparison queries (e.g., 'both/all'), "
    "identify commonalities between entities rather than listing individual details."
)
TEMPORAL_INSTRUCTION = (
    "Resolve relative dates (e.g., 'yesterday') using the event timestamps. Output dates strictly in "
    "'D Month YYYY' format. Calculate durations if asked."
)
OPEN_DOMAIN_INSTRUCTION = (
    "Make reasonable inferences based on the user's personality traits, interests, and past behaviors. "
    "Support hypothetical ('would/could') reasoning with evidence."
)
FACTUAL_INSTRUCTION = (
    "Extract the specific entity, name, or method requested. Do not add explanations. "
    "Return the exact fact matching the query intent."
)


def dynamic_instruction(intent: Intent) -> str:
    """The C.2 instruction for the router's intent: one source of routing, no oracle category label."""
    match intent:
        case Intent.WHEN:
            return TEMPORAL_INSTRUCTION
        case Intent.WHY:
            return MULTI_HOP_INSTRUCTION
        case Intent.ENTITY:
            return FACTUAL_INSTRUCTION


def build_qa_messages(question: str, context: str, intent: Intent) -> list[Message]:
    instruction = dynamic_instruction(intent)
    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT.format(dynamic_instruction=instruction)},
        {"role": "user", "content": QA_USER_PROMPT.format(context=context, question=question)},
    ]


@dataclass(frozen=True)
class SynthesizedAnswer:
    answer: str
    dynamic_instruction: str
    messages: list[Message]


class AnswerSynthesizer(Protocol):
    """Turns the router output and the linearized context into the final answer."""

    def synthesize(self, route: RouterOutput, context: LinearContext) -> SynthesizedAnswer: ...


class AdaptiveQASynthesizer:
    """Runs the C.2 prompt against the memory LLM."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def synthesize(self, route: RouterOutput, context: LinearContext) -> SynthesizedAnswer:
        instruction = dynamic_instruction(route.intent)
        messages = build_qa_messages(route.question, context.text, route.intent)
        result = self._llm.chat(messages, tag=QA_TAG)
        return SynthesizedAnswer(answer=result.text.strip(), dynamic_instruction=instruction, messages=messages)
