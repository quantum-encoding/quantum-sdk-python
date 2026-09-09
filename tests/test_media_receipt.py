"""Media receipt fields — decode tests (no network).

Covers duration_seconds on both the video and music responses, and usage on
the video response. Wire contract source of truth: backend routes_media.go
(videoGenResponse, musicGenResponse, mediaTokenUsage, mediaTokenUsageOf).
"""

from __future__ import annotations

from quantum_sdk import MusicResponse, VideoResponse


def test_video_receipt_carries_duration_and_every_usage_bucket() -> None:
    """A token-billed video: on Gemini Omni the tokens are the whole cost
    basis, so every bucket has to survive the wire."""
    resp = VideoResponse.from_dict(
        {
            "videos": [
                {"base64": "AAAA", "format": "mp4", "size_bytes": 184320, "index": 0}
            ],
            "model": "gemini-omni-video",
            "duration_seconds": 8.5,
            "usage": {
                "prompt_tokens": 412,
                "completion_tokens": 49232,
                "reasoning_tokens": 96,
                "cached_tokens": 128,
                "total_tokens": 49868,
            },
            "cost_ticks": 1247000000,
            "balance_after": 73,
            "request_id": "qai_req_2f1c8ab0-91d",
        }
    )

    assert resp.duration_seconds == 8.5
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 412
    assert resp.usage.completion_tokens == 49232
    assert resp.usage.reasoning_tokens == 96
    assert resp.usage.cached_tokens == 128
    assert resp.usage.total_tokens == 49868
    assert resp.balance_after == 73


def test_video_receipt_without_them_reports_none_not_zero() -> None:
    """A per-second model reports no tokens and the gateway sends no usage
    object at all. A zeroed object would read as a token-billed call that
    spent nothing, and a 0 duration would misstate the basis of cost_ticks.
    A gateway predating these fields must still decode."""
    resp = VideoResponse.from_dict(
        {
            "videos": [
                {"base64": "AAAA", "format": "mp4", "size_bytes": 184320, "index": 0}
            ],
            "model": "veo-2",
            "cost_ticks": 3200000000,
            "balance_after": 41,
            "request_id": "qai_req_7d5e0c14-33a",
        }
    )

    assert resp.duration_seconds is None, "a duration was invented"
    assert resp.usage is None, "a usage block was invented"


def test_music_receipt_carries_the_generated_duration() -> None:
    """Music is duration-metered, so the generated length is the basis of the
    charge and has to survive the wire."""
    resp = MusicResponse.from_dict(
        {
            "audio_clips": [
                {"base64": "SUQz", "format": "mp3", "size_bytes": 2941184, "index": 0}
            ],
            "model": "lyria-002",
            "duration_seconds": 184.0,
            "cost_ticks": 1840000000,
            "balance_after": 57,
            "request_id": "qai_req_bb31f907-4c1",
        }
    )

    assert resp.duration_seconds == 184.0
    assert resp.balance_after == 57


def test_music_receipt_without_a_duration_reports_none() -> None:
    """A provider that reports no length leaves it absent, not 0 — a zero
    would claim a measured empty track."""
    resp = MusicResponse.from_dict(
        {
            "audio_clips": [],
            "model": "eleven-music",
            "cost_ticks": 600000000,
            "request_id": "r",
        }
    )

    assert resp.duration_seconds is None, "a duration was invented"
