from bench.probe_reasoning import PROMPTS, Variant, probe
from magma.llm import ChatRequest, ClientSettings, Completion


class CountingBackend:
    """Reasons only when the request asks for effort "high"; refuses the provider "nobody"."""

    def complete(self, request: ChatRequest) -> Completion:
        body = request.get("extra_body", {})
        if body.get("provider") == {"order": ["nobody"]}:
            raise RuntimeError("no endpoints found")
        thinks = body.get("reasoning") == {"effort": "high"}
        return Completion(text="Carla", prompt_tokens=50, completion_tokens=300 if thinks else 5, reasoning_tokens=295 if thinks else 0, provider="P")


def test_each_variant_is_sent_with_its_own_body() -> None:
    variants = (
        Variant("high", {"reasoning": {"effort": "high"}}),
        Variant("low", {"reasoning": {"effort": "low"}}),
        Variant("missing", {"provider": {"order": ["nobody"]}}),
    )
    results = probe(ClientSettings(model="m"), variants, log_path=None, cache_dir=None, backend=CountingBackend())
    by_variant = {(r.variant, r.prompt): r for r in results}
    assert len(results) == 3 * len(PROMPTS)
    assert by_variant[("high", "puzzle")].reasoning_tokens == 295
    assert by_variant[("low", "extraction")].reasoning_tokens == 0
    assert by_variant[("missing", "puzzle")].error == "no endpoints found"
