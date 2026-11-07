import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from magma.llm import (
    ChatRequest,
    ClientSettings,
    Completion,
    ContextWindowExceeded,
    InvalidStructuredOutput,
    LLMClient,
    Message,
)


class FakeBackend:
    """Stands in for the API: returns canned replies and records every request."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.requests: list[ChatRequest] = []

    def complete(self, request: ChatRequest) -> Completion:
        self.requests.append(request)
        return Completion(text=self.replies.pop(0), prompt_tokens=10, completion_tokens=5, reasoning_tokens=3)


def make_client(
    tmp_path: Path,
    replies: list[str],
    context_window: int | None = None,
    extra_body: dict[str, bool] | None = None,
) -> tuple[LLMClient, FakeBackend]:
    backend = FakeBackend(replies)
    settings = ClientSettings(model="test-model", context_window=context_window, extra_body=dict(extra_body or {}))
    client = LLMClient(settings, backend=backend, cache_dir=tmp_path / "cache", log_path=tmp_path / "llm_calls.jsonl")
    return client, backend


MSG: list[Message] = [{"role": "user", "content": "Hello"}]


def test_second_identical_call_is_served_from_cache(tmp_path: Path) -> None:
    llm, backend = make_client(tmp_path, ["hi"])
    first = llm.chat(MSG)
    second = llm.chat(MSG)
    assert len(backend.requests) == 1
    assert (first.text, first.cached) == ("hi", False)
    assert (second.text, second.cached) == ("hi", True)
    assert second.prompt_tokens == 10


def test_cache_survives_a_new_client(tmp_path: Path) -> None:
    llm, _ = make_client(tmp_path, ["hi"])
    llm.chat(MSG)
    again, backend = make_client(tmp_path, [])
    assert again.chat(MSG).cached
    assert backend.requests == []


def test_different_parameters_are_not_mixed_in_cache(tmp_path: Path) -> None:
    llm, backend = make_client(tmp_path, ["a", "b"])
    llm.chat(MSG)
    llm.chat(MSG, max_tokens=7)
    assert len(backend.requests) == 2


def test_request_uses_deterministic_settings(tmp_path: Path) -> None:
    llm, backend = make_client(tmp_path, ["hi"], extra_body={"allow_fallbacks": False})
    llm.chat(MSG)
    request = backend.requests[0]
    assert request["temperature"] == 0.0
    assert request["seed"] == 42
    assert request.get("extra_body") == {"allow_fallbacks": False}
    assert "response_format" not in request


def test_every_call_is_logged_raw(tmp_path: Path) -> None:
    llm, _ = make_client(tmp_path, ["hi"])
    llm.chat(MSG, tag="qa")
    llm.chat(MSG, tag="qa")
    lines = [json.loads(line) for line in (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["request"]["messages"] == MSG
    assert lines[0]["response"]["text"] == "hi"
    assert [line["response"]["cached"] for line in lines] == [False, True]


def test_usage_is_counted_per_tag(tmp_path: Path) -> None:
    llm, _ = make_client(tmp_path, ["a", "b"])
    llm.chat(MSG, tag="extract")
    llm.chat([{"role": "user", "content": "Other"}], tag="qa")
    llm.chat(MSG, tag="extract")
    assert llm.usage["extract"].calls == 2
    assert llm.usage["extract"].cached_calls == 1
    total = llm.total()
    assert (total.calls, total.prompt_tokens, total.completion_tokens, total.reasoning_tokens) == (3, 30, 15, 9)


class Answer(BaseModel):
    answer: str
    confidence: float


def test_structured_output_is_validated(tmp_path: Path) -> None:
    llm, backend = make_client(tmp_path, ['```json\n{"answer": "violin", "confidence": 0.9}\n```'])
    out = llm.chat_structured(MSG, Answer)
    assert out.value == Answer(answer="violin", confidence=0.9)
    assert backend.requests[0].get("response_format") == {"type": "json_object"}


def test_invalid_structured_output_raises(tmp_path: Path) -> None:
    llm, _ = make_client(tmp_path, ['{"answer": "violin"}'])
    with pytest.raises(InvalidStructuredOutput):
        llm.chat_structured(MSG, Answer)


def test_prompt_larger_than_window_is_refused_before_calling(tmp_path: Path) -> None:
    llm, backend = make_client(tmp_path, ["hi"], context_window=50)
    with pytest.raises(ContextWindowExceeded):
        llm.chat([{"role": "user", "content": "word " * 200}])
    assert backend.requests == []


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("JUDGE_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("JUDGE_CONTEXT_WINDOW", "1000")
    monkeypatch.setenv("JUDGE_EXTRA_BODY", '{"provider": {"allow_fallbacks": false}}')
    settings = ClientSettings.from_env("JUDGE")
    assert settings.model == "judge-model"
    assert settings.context_window == 1000
    assert settings.extra_body == {"provider": {"allow_fallbacks": False}}


def test_settings_require_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NOPE_MODEL", raising=False)
    with pytest.raises(RuntimeError):
        ClientSettings.from_env("NOPE")
