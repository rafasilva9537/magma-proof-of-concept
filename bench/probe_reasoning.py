"""Check whether a provider actually reasons when asked to: tokens of reasoning per way of asking.

Sends the same prompts with several request bodies (provider and reasoning
settings) and reports, for each, the reasoning tokens, the output tokens, the
latency and the provider that answered. One prompt needs a few steps of
thought; the other is the real extraction task of the slow path (prompt C.1).

    python -m bench.probe_reasoning

The model, key and URL come from the LLM_* variables; the request bodies below
replace LLM_EXTRA_BODY for this probe only. Raw data goes to
runs/measurements/reasoning_probe/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAIError

from magma.ingest import EXTRACTION_INPUT, EXTRACTION_SYSTEM_PROMPT
from magma.json_types import JSONObject
from magma.llm import ChatBackend, ClientSettings, LLMClient, Message


@dataclass(frozen=True)
class Variant:
    name: str
    extra_body: JSONObject


def _pinned(provider: str, reasoning: JSONObject, require_parameters: bool = True) -> JSONObject:
    return {
        "provider": {"order": [provider], "allow_fallbacks": False, "require_parameters": require_parameters},
        "reasoning": reasoning,
    }


VARIANTS = (
    Variant("inference-net, effort medium (current)", _pinned("inference-net", {"effort": "medium"})),
    Variant("inference-net, effort high", _pinned("inference-net", {"effort": "high"})),
    Variant("inference-net, enabled + max_tokens 2000", _pinned("inference-net", {"enabled": True, "max_tokens": 2000})),
    # Z.ai does not accept `seed`, which every request sends, so it cannot require all parameters.
    Variant("z-ai, effort medium", _pinned("z-ai", {"effort": "medium"}, require_parameters=False)),
)

PROMPTS: dict[str, list[Message]] = {
    "puzzle": [
        {
            "role": "user",
            "content": (
                "Ana is older than Bruno. Carla is younger than Bruno but older than Davi. "
                "Eva is older than Ana. Who is the second youngest? Answer with the name only."
            ),
        }
    ],
    "extraction": [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": EXTRACTION_INPUT.format(
                speaker="Caroline",
                text="I went to a LGBTQ support group yesterday and it was so powerful.",
                prev_summary="Melanie greets Caroline and asks how she has been.",
            ),
        },
    ],
}


@dataclass(frozen=True)
class ProbeResult:
    variant: str
    prompt: str
    provider: str | None
    reasoning_tokens: int
    completion_tokens: int
    prompt_tokens: int
    latency_s: float
    answer: str
    error: str | None


def probe(
    settings: ClientSettings,
    variants: tuple[Variant, ...],
    log_path: Path | None,
    cache_dir: Path | None,
    backend: ChatBackend | None = None,
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for variant in variants:
        llm = LLMClient(replace(settings, extra_body=variant.extra_body), backend=backend, cache_dir=cache_dir, log_path=log_path)
        for prompt_name, messages in PROMPTS.items():
            try:
                result = llm.chat(messages, tag=prompt_name)
            except (OpenAIError, RuntimeError) as error:  # an unknown provider is refused by the gateway
                results.append(ProbeResult(variant.name, prompt_name, None, 0, 0, 0, 0.0, "", str(error)[:300]))
                continue
            results.append(
                ProbeResult(
                    variant=variant.name,
                    prompt=prompt_name,
                    provider=result.provider,
                    reasoning_tokens=result.reasoning_tokens,
                    completion_tokens=result.completion_tokens,
                    prompt_tokens=result.prompt_tokens,
                    latency_s=round(result.latency_s, 2),
                    answer=result.text.strip()[:200],
                    error=None,
                )
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/measurements/reasoning_probe"))
    arguments = parser.parse_args()

    load_dotenv()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = arguments.output_dir / "llm_calls.jsonl"
    log_path.unlink(missing_ok=True)
    results = probe(ClientSettings.from_env("LLM"), VARIANTS, log_path, cache_dir=Path(".cache/llm"))
    for result in results:
        outcome = result.error or f"reasoning {result.reasoning_tokens}, output {result.completion_tokens}, {result.latency_s} s, {result.provider}"
        print(f"{result.variant} | {result.prompt}: {outcome}")
    report = arguments.output_dir / "report.json"
    report.write_text(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report written to {report}")


if __name__ == "__main__":
    main()
