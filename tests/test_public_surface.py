"""Public surface invariants — one type per concept, and every export resolves.

types_ext.py used to redeclare 19 types that already existed in types.py /
errors.py / realtime.py under a different casing (TtsRequest vs TTSRequest,
RagResult vs RAGResult, ...). Python is case-sensitive, so both survived and
__init__ exported both as unrelated classes. The worst was ApiError: a
dataclass shadowing the real APIError exception, so any handler written
against the idiomatic casing raised TypeError instead of catching.
"""

from __future__ import annotations

import collections
import inspect

import pytest

import quantum_sdk


def test_all_exports_resolve():
    unresolved = [name for name in quantum_sdk.__all__ if not hasattr(quantum_sdk, name)]
    assert unresolved == []


def test_all_has_no_duplicate_entries():
    counts = collections.Counter(quantum_sdk.__all__)
    assert [name for name, n in counts.items() if n > 1] == []


def test_api_error_casing_is_one_catchable_exception():
    assert quantum_sdk.ApiError is quantum_sdk.APIError
    assert issubclass(quantum_sdk.ApiError, BaseException)

    with pytest.raises(quantum_sdk.ApiError):
        raise quantum_sdk.APIError(402, "INSUFFICIENT_BALANCE", "no funds", "req-1")


@pytest.mark.parametrize(
    "name",
    [
        "TtsRequest",
        "TtsResponse",
        "SttRequest",
        "SttResponse",
        "RagResult",
        "RagCorpus",
        "RagSearchRequest",
        "RagSearchResponse",
        "SurrealRagResult",
        "SurrealRagProvider",
        "SurrealRagProvidersResponse",
        "SurrealRagSearchRequest",
        "SurrealRagSearchResponse",
        "ResponseMeta",
        "RealtimeSession",
    ],
)
def test_mixed_case_variants_are_gone(name):
    """The ALL-CAPS acronym spelling is the only one, per the other six SDKs."""
    import quantum_sdk.types_ext as types_ext

    assert not hasattr(types_ext, name), f"{name} is back in types_ext"


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("ContactResponse", "quantum_sdk.types"),
        ("CreditTier", "quantum_sdk.credits"),
        ("RealtimeSessionResponse", "quantum_sdk.realtime"),
        ("TTSRequest", "quantum_sdk.types"),
        ("SurrealRAGProvider", "quantum_sdk.types"),
    ],
)
def test_diverged_types_export_the_kept_definition(name, module):
    obj = getattr(quantum_sdk, name)
    assert inspect.getmodule(obj).__name__ == module


def test_long_form_audio_aliases_point_at_the_canonical_types():
    """The Rust SDK spells these out; they must not be a second class."""
    assert quantum_sdk.TextToSpeechRequest is quantum_sdk.TTSRequest
    assert quantum_sdk.TextToSpeechResponse is quantum_sdk.TTSResponse
    assert quantum_sdk.SpeechToTextRequest is quantum_sdk.STTRequest
    assert quantum_sdk.SpeechToTextResponse is quantum_sdk.STTResponse
    assert quantum_sdk.SurrealRagProviderInfo is quantum_sdk.SurrealRAGProvider


def test_merged_fields_survived_the_dedupe():
    """Fields the deleted copies had, that the Rust reference confirms."""
    # types_ext.ContactResponse was `status: str`; types.py had `success: bool`.
    contact = quantum_sdk.ContactResponse.from_dict({"status": "ok", "message": "sent"})
    assert contact.status == "ok"
    assert contact.success is True

    # SurrealRagProviderInfo keys on `provider`, not `name` (rag.rs).
    providers = quantum_sdk.SurrealRAGProvidersResponse.from_dict(
        {"providers": [{"provider": "xai", "chunk_count": 12}], "request_id": "req-2"}
    )
    assert providers.providers[0].provider == "xai"
    assert providers.providers[0].chunk_count == 12
    assert providers.request_id == "req-2"

    # RealtimeSession's signed_url/provider/ws_url() folded into realtime.py.
    session = quantum_sdk.RealtimeSessionResponse(
        ephemeral_token="",
        url="",
        session_id="s1",
        signed_url="wss://elevenlabs.example/signed",
        provider="elevenlabs",
    )
    assert session.ws_url() == "wss://elevenlabs.example/signed"
