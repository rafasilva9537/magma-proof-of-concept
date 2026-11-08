"""Client for any OpenAI-compatible chat endpoint.

Every call is cached on disk and appended, raw, to a JSONL log, so results can
be audited and metrics recomputed without calling the model again.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, Literal, NotRequired, Protocol, TypedDict, TypeVar

import tiktoken
from openai import OpenAI, omit
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


@dataclass(frozen=True)
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency_s: float
    cached: bool


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


class ChatBackend(Protocol):
    def complete(self, request: ChatRequest) -> Completion: ...


class OpenAIBackend:
    """Sends requests through the official `openai` SDK."""

    def __init__(self, api_key: str | None, base_url: str | None):
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, request: ChatRequest) -> Completion:
        response = self._client.chat.completions.create(
            model=request["model"],
            messages=[_to_sdk_message(m) for m in request["messages"]],
            temperature=request["temperature"],
            seed=request["seed"],
            max_tokens=request["max_tokens"],
            response_format={"type": "json_object"} if "response_format" in request else omit,
            extra_body=request.get("extra_body"),
        )
        usage = response.usage
        details = usage.completion_tokens_details if usage else None
        return Completion(
            text=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            reasoning_tokens=(details.reasoning_tokens or 0) if details else 0,
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
    ):
        self.settings = settings
        self.backend = backend or OpenAIBackend(settings.api_key, settings.base_url)
        self.cache_dir = cache_dir
        self.log_path = log_path
        self.config = config
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
            start = time.perf_counter()
            completion = self.backend.complete(request)
            result = LLMResult(
                text=completion.text,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
                reasoning_tokens=completion.reasoning_tokens,
                latency_s=time.perf_counter() - start,
                cached=False,
            )
            self._cache_put(key, result)
        with self._lock:
            self.usage.setdefault(tag, Usage()).add(result)
            self._log(key, tag, request, result)
        return result

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


class LoggedResponse(TypedDict):
    text: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    latency_s: float
    cached: bool


class LogRecord(TypedDict):
    ts: float
    tag: str
    key: str
    request: ChatRequest
    response: LoggedResponse


def _parse(text: str, schema: type[T], tag: str) -> T:
    raw = text.strip()
    fenced = _FENCE.match(raw)
    if fenced:
        raw = fenced.group(1)
    try:
        return schema.model_validate_json(raw)
    except ValidationError as err:
        raise InvalidStructuredOutput(f"[{tag}] response does not match {schema.__name__}: {err}") from err
