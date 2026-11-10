"""LLM-as-a-Judge with the paper's evaluation prompt (Appendix C.3).

The judge scores a candidate answer against the gold reference on the rubric
1.0 / 0.8 / 0.6 / 0.4 / 0.2 / 0.0 and returns `{"score", "reasoning"}`. Every
question goes to the judge, adversarial ones included: their gold reference is
"Unanswerable", and the rubric itself asks the candidate to state the lack of
information. The judge model is configured apart from the model under test
(JUDGE_* variables), so it can be of another family.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from magma.llm import InvalidStructuredOutput, LLMClient, Message

# Prompt C.3 of the paper, as printed: the rubric and constraints, then the input line and the output schema.
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing the semantic fidelity of a memory retrieval system. Score the Candidate Answer against the Gold Reference on a continuous scale [0.0, 1.0].

Scoring Rubric:
- 1.0 (Exact Alignment): Captures all key entities, temporal markers, and causal relationships. Semantically equivalent.
- 0.8 (Substantially Correct): Main point is accurate but lacks minor nuances or secondary details.
- 0.6 (Partial Match): Contains valid information but misses key constraints (e.g., wrong date but correct event).
- 0.4 (Tangential): Touches on the topic but misses the core information requirement.
- 0.2 (Incoherent): Factually incorrect with only minimal topical overlap.
- 0.0 (Contradiction/Hallucination): Completely unrelated or contradicts the ground truth.

Evaluation Constraints:
1. Temporal Flexibility: Accept relative time references (e.g., "next Tuesday") if they resolve to the same period as the Gold Reference.
2. Semantic Equivalence: Prioritize informational content over lexical matching.
3. Adversarial Handling: If the Gold Reference states "Unanswerable", the Candidate MUST explicitly state lack of information. Any hallucinated fact results in 0.0.

Output: JSON {"score": float, "reasoning": "concise explanation"}"""

JUDGE_INPUT = "Input: Question: {question} | Gold: {gold} | Candidate: {generated}"

JUDGE_TAG = "judge"


class JudgeVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


@dataclass(frozen=True)
class Grade:
    """The judge's score, or the reason it gave none (a reply that is not a valid verdict)."""

    score: float | None
    reasoning: str
    error: str | None = None


class Judge:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        return self._llm

    def grade(self, question: str, gold: str, candidate: str) -> Grade:
        """Score `candidate`; an invalid verdict is kept as an error, not retried (a retry would hit the cache)."""
        messages: list[Message] = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_INPUT.format(question=question, gold=gold, generated=candidate)},
        ]
        try:
            verdict = self._llm.chat_structured(messages, JudgeVerdict, tag=JUDGE_TAG).value
        except InvalidStructuredOutput as error:
            return Grade(score=None, reasoning="", error=str(error))
        return Grade(score=verdict.score, reasoning=verdict.reasoning)
