"""The TTS request contract on POST /qai/v1/audio/tts, and the voice catalogue
on GET /qai/v1/voices — encode/decode tests, no network.

Gemini exposes no parameters for tone, accent or pace — the steering is prose
in ``instructions`` plus inline tags inside ``text`` — so these tests pin the
field names the handler actually decodes. Wire contract source of truth:
backend internal/server/routes_media.go (ttsRequest, ttsVoiceSettings,
ttsSpeaker) and internal/server/routes_voice.go (voiceResponse).
"""

from __future__ import annotations

from quantum_sdk import TTSRequest, TTSSpeaker, TTSVoiceSettings, VoiceInfo


def test_text_alone_is_a_complete_request() -> None:
    """The gateway supplies gemini-3.1-flash-tts-preview + Laomedeia when the
    request names neither. Sending an empty model would pin the request to a
    model that does not exist."""
    d = TTSRequest(text="Hello").to_dict()

    assert d["text"] == "Hello"
    for key in ("model", "voice", "speakers", "voice_settings", "instructions"):
        assert key not in d, f"{key} sent on a request that set none"


def test_the_steering_fields_use_the_wire_names() -> None:
    d = TTSRequest(
        model="gemini-3.1-flash-tts-preview",
        text="[excited] Hi! [whispers] can you keep a secret?",
        voice="Laomedeia",
        output_format="wav",
        speed=1.1,
        instructions="Read aloud with a natural British accent",
        language="en-GB",
        sample_rate=24000,
        bit_rate=128000,
    ).to_dict()

    assert d["model"] == "gemini-3.1-flash-tts-preview"
    assert d["voice"] == "Laomedeia"
    # output_format rides as "format" — the handler reads no other key.
    assert d["format"] == "wav"
    assert "output_format" not in d
    assert d["speed"] == 1.1
    assert d["instructions"] == "Read aloud with a natural British accent"
    assert d["language"] == "en-GB"
    assert d["sample_rate"] == 24000
    assert d["bit_rate"] == 128000


def test_a_two_speaker_dialogue_serializes() -> None:
    """Two speakers, labelled to match the lines the text carries."""
    d = TTSRequest(
        text="Lacey: Hi there.\nCustomer: [excited] Hi!",
        instructions="Lacey is calm; the customer is cheerful",
        speakers=[
            TTSSpeaker(name="Lacey", voice="Laomedeia"),
            TTSSpeaker(name="Customer", voice="Puck"),
        ],
    ).to_dict()

    speakers = d["speakers"]
    assert len(speakers) == 2, "gemini takes exactly two speakers"
    assert speakers[0] == {"name": "Lacey", "voice": "Laomedeia"}
    assert speakers[1] == {"name": "Customer", "voice": "Puck"}


def test_voice_settings_omit_what_was_not_set() -> None:
    """0.0 stability is a real setting the provider honours, so a zeroed
    object silently retunes the voice rather than leaving the default alone."""
    d = TTSRequest(
        text="hi",
        voice_settings=TTSVoiceSettings(stability=0.4, use_speaker_boost=True),
    ).to_dict()

    vs = d["voice_settings"]
    assert vs["stability"] == 0.4
    assert vs["use_speaker_boost"] is True
    assert "similarity_boost" not in vs
    assert "style" not in vs


def test_the_voice_listing_decodes_every_documented_field() -> None:
    """The catalogue a picker is built from."""
    gemini = VoiceInfo.from_dict(
        {
            "voice_id": "Laomedeia",
            "name": "Laomedeia",
            "category": "premade",
            "provider": "gemini",
            "model": "gemini-3.1-flash-tts-preview",
            "is_cloned": False,
        }
    )
    assert gemini.provider == "gemini"
    # The model to pass back for this voice, so a caller never hardcodes the
    # provider-to-model mapping.
    assert gemini.model == "gemini-3.1-flash-tts-preview"
    assert gemini.is_cloned is False

    el = VoiceInfo.from_dict(
        {
            "voice_id": "el_7f3",
            "name": "Rachel",
            "category": "cloned",
            "provider": "elevenlabs",
            "model": "eleven_multilingual_v2",
            "is_cloned": True,
            "description": "warm narrator",
            "preview_url": "https://cdn/x.mp3",
        }
    )
    assert el.is_cloned is True
    assert el.description == "warm narrator"
    assert el.preview_url == "https://cdn/x.mp3"
