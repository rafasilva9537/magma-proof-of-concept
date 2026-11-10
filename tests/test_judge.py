import json
from pathlib import Path

from bench.judge import JUDGE_SYSTEM_PROMPT, Judge
from magma.llm import ChatRequest, ClientSettings, Completion, LLMClient


class JudgeBackend:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        return Completion(text=self.reply, prompt_tokens=300, completion_tokens=30, reasoning_tokens=0)


def judge_with(reply: str, tmp_path: Path) -> tuple[Judge, JudgeBackend]:
    backend = JudgeBackend(reply)
    return Judge(LLMClient(ClientSettings(model="judge-model"), backend=backend, cache_dir=tmp_path)), backend


def test_judge_sends_prompt_c3_with_question_gold_and_candidate(tmp_path: Path) -> None:
    judge, backend = judge_with(json.dumps({"score": 0.8, "reasoning": "misses the year"}), tmp_path)
    grade = judge.grade("When did Melanie paint a sunrise?", "2022", "Last year")
    assert (grade.score, grade.reasoning, grade.error) == (0.8, "misses the year", None)

    request = backend.requests[0]
    assert request["messages"][0]["content"] == JUDGE_SYSTEM_PROMPT
    assert request["messages"][1]["content"] == "Input: Question: When did Melanie paint a sunrise? | Gold: 2022 | Candidate: Last year"
    assert request["response_format"] == {"type": "json_object"}


def test_rubric_keeps_the_adversarial_rule() -> None:
    assert 'If the Gold Reference states "Unanswerable", the Candidate MUST explicitly state lack of information' in JUDGE_SYSTEM_PROMPT


def test_invalid_verdict_is_an_error_not_a_score(tmp_path: Path) -> None:
    judge, _ = judge_with(json.dumps({"score": 1.7}), tmp_path)  # outside [0, 1]
    grade = judge.grade("q", "gold", "candidate")
    assert grade.score is None
    assert grade.error is not None
