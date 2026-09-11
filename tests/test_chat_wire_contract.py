"""The chat wire contract the gateway added on 2026-09-11 — decode/encode
tests, no network.

Covers the reasoning state a tool loop must hand back, the cache key that
keeps a conversation on one provider shard, and the Gemini 3 signature that
now rides text blocks as well as tool_use blocks. Wire contract source of
truth: backend internal/server/convert.go (ChatRequest, ContentBlock) and
internal/server/sse.go (the thought_signature event).
"""

from __future__ import annotations

from quantum_sdk import ChatMessage, ChatRequest, ChatResponse, ContentBlock
from quantum_sdk.client import _parse_sse_event


def _req(**kw: object) -> ChatRequest:
    return ChatRequest(model="gpt-5.6", messages=[ChatMessage(role="user", content="hi")], **kw)


def test_prompt_cache_key_rides_only_when_set() -> None:
    """Omitted, the gateway derives a key from the caller's identity — so an
    absent field is a real default, not an oversight to paper over."""
    assert "prompt_cache_key" not in _req().to_dict()
    assert _req(prompt_cache_key="conv-7f3a").to_dict()["prompt_cache_key"] == "conv-7f3a"


def test_reasoning_block_round_trips_verbatim_and_in_place() -> None:
    """The provider's reasoning item is opaque, and its POSITION among the
    tool calls is the state the provider reads back."""
    reasoning = {"id": "rs_abc", "summary": [], "encrypted_content": "Zm9v"}
    resp = ChatResponse.from_dict(
        {
            "id": "req_1",
            "model": "gpt-5.6",
            "stop_reason": "tool_use",
            "content": [
                {"type": "reasoning", "reasoning": reasoning, "minted_by": "gpt-5.6"},
                {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"q": "x"}},
            ],
        }
    )

    assert len(resp.content) == 2
    block = resp.content[0]
    assert block.type == "reasoning"
    assert block.minted_by == "gpt-5.6"
    # Opaque: the SDK must not have reshaped the provider's item.
    assert block.reasoning == reasoning

    # Echoed back on the next turn's assistant message, unchanged and in the
    # same order.
    echoed = ChatMessage(role="assistant", content_blocks=resp.content).to_dict()
    blocks = echoed["content_blocks"]
    assert [b["type"] for b in blocks] == ["reasoning", "tool_use"]
    assert blocks[0]["reasoning"] == reasoning
    assert blocks[0]["minted_by"] == "gpt-5.6"


def test_a_block_with_no_reasoning_state_sends_neither_field() -> None:
    """Absent is not empty: a null reasoning item is not something a provider
    will accept back."""
    d = ContentBlock(type="text", text="hello").to_dict()
    for key in ("reasoning", "minted_by", "thought_signature"):
        assert key not in d


def test_gemini_signs_a_turn_that_ends_in_text() -> None:
    """Gemini 3 puts the signature on the TEXT block, not only on tool_use."""
    resp = ChatResponse.from_dict(
        {
            "id": "r",
            "model": "gemini-3.5-flash",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hi", "thought_signature": "c2ln"}],
        }
    )
    assert resp.content[0].thought_signature == "c2ln"
    assert resp.content[0].to_dict()["thought_signature"] == "c2ln"


def test_thought_signature_sse_event_is_parsed() -> None:
    """The gateway sends it just before done on a stream that ended in text."""
    ev = _parse_sse_event('{"type":"thought_signature","thought_signature":"c2ln"}')
    assert ev.type == "thought_signature"
    assert ev.thought_signature == "c2ln"

    # And on the atomic tool_use event, which a streaming tool loop needs.
    tool = _parse_sse_event(
        '{"type":"tool_use","id":"call_1","name":"lookup","input":{},"thought_signature":"c2ln"}'
    )
    assert tool.thought_signature == "c2ln"

    # Absent on an event that carried none.
    assert _parse_sse_event('{"type":"content_delta","delta":{"text":"hi"}}').thought_signature is None


def test_provider_options_is_an_open_map() -> None:
    """Nested per-provider objects, the flat region entry, and a key this SDK
    version never heard of all reach the gateway untouched."""
    opts = {
        "openai": {
            "reasoning_summary": "detailed",
            "reasoning_mode": "pro",
            "verbosity": "low",
            "text_format": "json_object",
            "a_key_this_sdk_never_heard_of": 42,
        },
        "xai": {"native_files": True},
        "region": "europe",
    }
    sent = _req(provider_options=opts).to_dict()["provider_options"]
    assert sent == opts


def test_reasoning_effort_carries_every_tier() -> None:
    for tier in ("none", "low", "medium", "high", "xhigh", "max"):
        assert _req(reasoning_effort=tier).to_dict()["reasoning_effort"] == tier
