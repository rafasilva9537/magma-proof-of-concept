"""Client for any OpenAI-compatible chat endpoint.

Every call is cached on disk and appended, raw, to a JSONL log, so results can
be audited and metrics recomputed without calling the model again. A call that
fails (provider down, timeout, empty or cut reply) is retried and never cached;
if it keeps failing, the client raises and the run stops.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, NotRequired, Protocol, TypedDict, TypeVar

import tiktoken
from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError, omit
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from magma.config import DEFAULT, LLMConfig
from magma.json_types import JSONObject


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class ResponseFormat(TypedDict):
    type: Literal["json_object"]


class ChatRequest(TypedDict):
    model: str
    messages: list[Message]
    temperature: float
    seed: int
    max_tokens: int
    response_format: NotRequired[ResponseFormat]
    extra_body: NotRequired[JSONObject]


@dataclass(frozen=True)
class Completion:
    """What a backend returns for one request."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    finish_reason: str | None = "stop"
    provider: str | None = None  # who served it, when a gateway routes between providers (OpenRouter)
    model: str | None = None  # the model id the server reports, e.g. with its dated variant


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency_s: float
    cached: bool
    finish_reason: str | None = "stop"
    provider: str | None = None
    model: str | None = None
    attempts: int = 1  # 0 when served from the cache


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class Structured(Generic[T]):
    value: T
    result: LLMResult


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency_s: float = 0.0

    def add(self, r: LLMResult) -> None:
        self.calls += 1
        self.cached_calls += int(r.cached)
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.reasoning_tokens += r.reasoning_tokens
        self.latency_s += r.latency_s

    def merge(self, other: Usage) -> None:
        self.calls += other.calls
        self.cached_calls += other.cached_calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.latency_s += other.latency_s


class ContextWindowExceeded(ValueError):
    pass


class InvalidStructuredOutput(ValueError):
    pass


class TransientBackendError(RuntimeError):
    """The call failed for a reason worth retrying: the provider is down, slow, rate limiting, or sent no usable reply."""


class ReplyCutByTokenLimit(RuntimeError):
    """The reply hit max_tokens: a configuration problem that retrying would only repeat."""


class LLMUnavailable(RuntimeError):
    """Every attempt failed. The run has to stop: going on would build results on a missing answer."""


class ChatBackend(Protocol):
    def complete(self, request: ChatRequest) -> Completion: ...


class OpenAIBackend:
    """Sends requests through the official `openai` SDK. Retries are left to LLMClient, so there is one retry policy."""

    def __init__(self, api_key: str | None, base_url: str | None, timeout_s: float):
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s, max_retries=0)

    def complete(self, request: ChatRequest) -> Completion:
        try:
            response = self._client.chat.completions.create(
                model=request["model"],
                messages=[_to_sdk_message(m) for m in request["messages"]],
                temperature=request["temperature"],
                seed=request["seed"],
                max_tokens=request["max_tokens"],
                response_format={"type": "json_object"} if "response_format" in request else omit,
                extra_body=request.get("extra_body"),
            )
        except (APIConnectionError, RateLimitError, InternalServerError) as error:  # timeouts are connection errors
            raise TransientBackendError(str(error)) from error
        if not response.choices:
            raise TransientBackendError("the server returned no choices")
        usage = response.usage
        details = usage.completion_tokens_details if usage else None
        provider = (response.model_extra or {}).get("provider")
        return Completion(
            text=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            reasoning_tokens=(details.reasoning_tokens or 0) if details else 0,
            finish_reason=response.choices[0].finish_reason,
            provider=provider if isinstance(provider, str) else None,
            model=response.model,
        )


def _to_sdk_message(m: Message) -> ChatCompletionMessageParam:
    if m["role"] == "system":
        return {"role": "system", "content": m["content"]}
    if m["role"] == "assistant":
        return {"role": "assistant", "content": m["content"]}
    return {"role": "user", "content": m["content"]}


@dataclass(frozen=True)
class ClientSettings:
    model: str
    api_key: str | None = None
    base_url: str | None = None
    context_window: int | None = None
    extra_body: JSONObject = field(default_factory=dict)

    @staticmethod
    def from_env(prefix: str) -> ClientSettings:
        """Read <PREFIX>_MODEL, _API_KEY, _BASE_URL, _CONTEXT_WINDOW and _EXTRA_BODY."""
        model = _env(prefix, "MODEL")
        if model is None:
            raise RuntimeError(f"{prefix}_MODEL is not set")
        window = _env(prefix, "CONTEXT_WINDOW")
        extra = _env(prefix, "EXTRA_BODY")
        extra_body: JSONObject = json.loads(extra) if extra else {}
        return ClientSettings(
            model=model,
            api_key=_env(prefix, "API_KEY"),
            base_url=_env(prefix, "BASE_URL"),
            context_window=int(window) if window else None,
            extra_body=extra_body,
        )


def _env(prefix: str, name: str) -> str | None:
    value = os.getenv(f"{prefix}_{name}")
    return value or None


_ENCODING = tiktoken.get_encoding("cl100k_base")
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def count_tokens(messages: list[Message]) -> int:
    """Approximate prompt size; the provider's tokenizer may differ slightly."""
    per_message_overhead = 4
    return sum(len(_ENCODING.encode(m["content"])) + per_message_overhead for m in messages)


class LLMClient:
    def __init__(
        self,
        settings: ClientSettings,
        backend: ChatBackend | None = None,
        cache_dir: Path | None = Path(".cache/llm"),
        log_path: Path | None = None,
        config: LLMConfig = DEFAULT.llm,
        sleep: Callable[[float], None] = time.sleep,  # replaced in tests, so retries do not wait
    ):
        self.settings = settings
        self.backend = backend or OpenAIBackend(settings.api_key, settings.base_url, config.request_timeout_s)
        self.cache_dir = cache_dir
        self.log_path = log_path
        self.config = config
        self._sleep = sleep
        self.usage: dict[str, Usage] = {}
        self._lock = threading.Lock()

    @staticmethod
    def from_env(prefix: str = "LLM", log_path: Path | None = None) -> LLMClient:
        return LLMClient(ClientSettings.from_env(prefix), log_path=log_path)

    def chat(self, messages: list[Message], tag: str = "default", max_tokens: int | None = None) -> LLMResult:
        request = self._build_request(messages, max_tokens, json_output=False)
        return self._send(request, tag)

    def chat_structured(
        self,
        messages: list[Message],
        schema: type[T],
        tag: str = "default",
        max_tokens: int | None = None,
    ) -> Structured[T]:
        """Ask for a JSON object and validate it against a pydantic model."""
        request = self._build_request(messages, max_tokens, json_output=True)
        result = self._send(request, tag)
        return Structured(value=_parse(result.text, schema, tag), result=result)

    def total(self) -> Usage:
        out = Usage()
        for u in self.usage.values():
            out.merge(u)
        return out

    def _build_request(self, messages: list[Message], max_tokens: int | None, json_output: bool) -> ChatRequest:
        output_budget = max_tokens or self.config.max_output_tokens
        window = self.settings.context_window
        if window is not None:
            needed = count_tokens(messages) + output_budget
            if needed > window:
                raise ContextWindowExceeded(f"prompt needs ~{needed} tokens; window is {window}")

        request: ChatRequest = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "max_tokens": output_budget,
        }
        if json_output:
            request["response_format"] = {"type": "json_object"}
        if self.settings.extra_body:
            request["extra_body"] = self.settings.extra_body
        return request

    def _send(self, request: ChatRequest, tag: str) -> LLMResult:
        key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        result = self._cache_get(key)
        if result is None:
            result = self._call_with_retries(request, tag)
            self._cache_put(key, result)
        with self._lock:
            self.usage.setdefault(tag, Usage()).add(result)
            self._log(key, tag, request, result)
        return result

    def _call_with_retries(self, request: ChatRequest, tag: str) -> LLMResult:
        """Only a complete reply is returned (and then cached); a failed attempt is retried after a doubling wait."""
        delay = self.config.first_retry_delay_s
        failures: list[str] = []
        for attempt in range(1, self.config.max_attempts + 1):
            start = time.perf_counter()
            try:
                completion = self.backend.complete(request)
                _check_usable(completion)
            except TransientBackendError as error:
                failures.append(f"attempt {attempt}: {error}")
                if attempt < self.config.max_attempts:
                    self._sleep(delay)
                    delay *= 2
                continue
            return LLMResult(
                text=completion.text,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                reasoning_tokens=completion.reasoning_tokens,
                latency_s=time.perf_counter() - start,
                cached=False,
                finish_reason=completion.finish_reason,
                provider=completion.provider,
                model=completion.model,
                attempts=attempt,
            )
        raise LLMUnavailable(f"[{tag}] no usable reply after {self.config.max_attempts} attempts: " + "; ".join(failures))

    def _cache_path(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, key: str) -> LLMResult | None:
        path = self._cache_path(key)
        if path is None or not path.exists():
            return None
        entry = CacheEntry.model_validate_json(path.read_text(encoding="utf-8"))
        return LLMResult(
            text=entry.text,
            prompt_tokens=entry.prompt_tokens,
            completion_tokens=entry.completion_tokens,
            reasoning_tokens=entry.reasoning_tokens,
            latency_s=entry.latency_s,
            cached=True,
            finish_reason=entry.finish_reason,
            provider=entry.provider,
            model=entry.model,
            attempts=0,
        )

    def _cache_put(self, key: str, result: LLMResult) -> None:
        path = self._cache_path(key)
        if path is None:
            return
        entry = CacheEntry(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            reasoning_tokens=result.reasoning_tokens,
            latency_s=result.latency_s,
            finish_reason=result.finish_reason,
            provider=result.provider,
            model=result.model,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(entry.model_dump_json(), encoding="utf-8")
        tmp.replace(path)  # atomic: a crash never leaves a half-written entry

    def _log(self, key: str, tag: str, request: ChatRequest, result: LLMResult) -> None:
        if self.log_path is None:
            return
        record: LogRecord = {
            "ts": time.time(),
            "tag": tag,
            "key": key,
            "request": request,
            "response": {
                "text": result.text,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "reasoning_tokens": result.reasoning_tokens,
                "latency_s": result.latency_s,
                "cached": result.cached,
                "finish_reason": result.finish_reason,
                "provider": result.provider,
                "model": result.model,
                "attempts": result.attempts,
            },
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class CacheEntry(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency_s: float
    finish_reason: str | None = "stop"
    provider: str | None = None
    model: str | None = None


class LoggedResponse(TypedDict):
    text: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency_s: float
    cached: bool
    finish_reason: str | None
    provider: str | None
    model: str | None
    attempts: int


class LogRecord(TypedDict):
    ts: float
    tag: str
    key: str
    request: ChatRequest
    response: LoggedResponse


def _check_usable(completion: Completion) -> None:
    """A reply that did not finish normally is not an answer, and is never cached.

    A reply cut by the token limit would be cut again on every attempt, so it stops the run at
    once; any other unfinished or empty reply is a provider failure, worth retrying.
    """
    if completion.finish_reason == TOKEN_LIMIT_FINISH_REASON:
        raise ReplyCutByTokenLimit(f"the reply hit max_tokens after {completion.completion_tokens} tokens; raise the limit")
    if completion.finish_reason != COMPLETE_FINISH_REASON:
        raise TransientBackendError(f"reply ended with finish_reason={completion.finish_reason!r}")
    if not completion.text.strip():
        raise TransientBackendError("the reply is empty")


COMPLETE_FINISH_REASON = "stop"
TOKEN_LIMIT_FINISH_REASON = "length"


def _parse(text: str, schema: type[T], tag: str) -> T:
    raw = text.strip()
    fenced = _FENCE.match(raw)
    if fenced:
        raw = fenced.group(1)
    try:
        return schema.model_validate_json(raw)
    except ValidationError as err:
        raise InvalidStructuredOutput(f"[{tag}] response does not match {schema.__name__}: {err}") from err
