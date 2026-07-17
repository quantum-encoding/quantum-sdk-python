"""HeyGen v3 gateway routes — mock-gateway tests (no network).

Covers the 9 routes' wire contract: path, method, auth header, request body,
query-string encoding, and response decode, for both Client and AsyncClient.
Wire contract source of truth: backend routes_heygen_v3.go.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import httpx
import pytest

from quantum_sdk import (
    AsyncClient,
    AvatarAudioInput,
    AvatarRealtimeRequest,
    Client,
    InsufficientBalanceError,
    VideoBatchSubmitRequest,
    VideoTemplateDimension,
    VideoTemplateGenerateRequest,
    VideoTemplateSubtitles,
)
from quantum_sdk.errors import APIError

BASE_URL = "https://gw.test"
API_KEY = "qai_test_key"


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> Client:
    transport = httpx.MockTransport(handler)
    return Client(API_KEY, base_url=BASE_URL, http_client=httpx.Client(transport=transport))


def make_async_client(handler: Callable[[httpx.Request], httpx.Response]) -> AsyncClient:
    transport = httpx.MockTransport(handler)
    return AsyncClient(API_KEY, base_url=BASE_URL, http_client=httpx.AsyncClient(transport=transport))


def capture(
    status_code: int,
    body: dict[str, Any],
    seen: list[httpx.Request],
    headers: dict[str, str] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status_code, json=body, headers=headers or {})

    return handler


# ---------------------------------------------------------------------------
# 1. POST /qai/v1/avatar/realtime — create session (prepaid)
# ---------------------------------------------------------------------------

CREATE_BODY = {
    "stream_id": "rt_9f2c1a",
    "status": "pending",
    "prepaid_seconds": 300,
    "cost_ticks": 345000000000,
    "request_id": "req_abc123def456",
}


def _assert_create_request(req: httpx.Request) -> None:
    assert req.method == "POST"
    assert req.url.path == "/qai/v1/avatar/realtime"
    assert req.headers["Authorization"] == f"Bearer {API_KEY}"
    assert req.headers["Content-Type"] == "application/json"
    sent = json.loads(req.content)
    assert sent == {
        "type": "text_stream",
        "avatar_id": "Abigail_expressive_2024112501",
        "voice_id": "73c0b6a2e29d4d38aca41454bf58c955",
        "text": "Hello! Let me think about that...",
        "max_duration_seconds": 300,
    }


def _create_request() -> AvatarRealtimeRequest:
    return AvatarRealtimeRequest(
        session_type="text_stream",
        avatar_id="Abigail_expressive_2024112501",
        voice_id="73c0b6a2e29d4d38aca41454bf58c955",
        text="Hello! Let me think about that...",
        max_duration_seconds=300,
    )


def _assert_create_response(resp) -> None:
    assert resp.stream_id == "rt_9f2c1a"
    assert resp.status == "pending"
    assert resp.prepaid_seconds == 300
    assert resp.cost_ticks == 345000000000
    assert resp.request_id == "req_abc123def456"
    # Receipt pattern: balance_after filled from the X-QAI-Balance-After header.
    assert resp.balance_after == 655000000000


def test_create_avatar_realtime_session() -> None:
    seen: list[httpx.Request] = []
    client = make_client(
        capture(200, CREATE_BODY, seen, headers={
            "X-QAI-Cost-Ticks": "345000000000",
            "X-QAI-Balance-After": "655000000000",
        })
    )
    resp = client.create_avatar_realtime_session(_create_request())
    _assert_create_request(seen[0])
    _assert_create_response(resp)


def test_create_avatar_realtime_session_async() -> None:
    seen: list[httpx.Request] = []
    client = make_async_client(
        capture(200, CREATE_BODY, seen, headers={
            "X-QAI-Cost-Ticks": "345000000000",
            "X-QAI-Balance-After": "655000000000",
        })
    )
    resp = asyncio.run(client.create_avatar_realtime_session(_create_request()))
    _assert_create_request(seen[0])
    _assert_create_response(resp)


def test_create_avatar_realtime_audio_union_omits_empty_optionals() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, CREATE_BODY, seen))
    client.create_avatar_realtime_session(
        AvatarRealtimeRequest(
            session_type="audio",
            avatar_id="av_1",
            max_duration_seconds=120,
            audio=AvatarAudioInput(input_type="base64", media_type="audio/mpeg", data="AQID"),
        )
    )
    sent = json.loads(seen[0].content)
    # voice_id / text must be omitted for audio sessions (upstream rejects them).
    assert "voice_id" not in sent
    assert "text" not in sent
    assert sent["audio"] == {"type": "base64", "media_type": "audio/mpeg", "data": "AQID"}


def test_create_avatar_realtime_insufficient_balance() -> None:
    err_body = {
        "error": {
            "message": "out of credits — top up to continue",
            "type": "insufficient_balance",
            "code": "INSUFFICIENT_BALANCE",
        }
    }
    client = make_client(capture(402, err_body, []))
    with pytest.raises(InsufficientBalanceError) as exc_info:
        client.create_avatar_realtime_session(_create_request())
    assert exc_info.value.status_code == 402
    assert exc_info.value.code == "INSUFFICIENT_BALANCE"


# ---------------------------------------------------------------------------
# 2. GET /qai/v1/avatar/realtime/{id} — session status
# ---------------------------------------------------------------------------

STATUS_BODY = {
    "stream_id": "rt_9f2c1a",
    "status": "streaming",
    "hls_url": "https://cdn.heygen.com/realtime/rt_9f2c1a/index.m3u8",
    "request_id": "req_abc123def457",
}


def test_get_avatar_realtime_session() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, STATUS_BODY, seen))
    resp = client.get_avatar_realtime_session("rt_9f2c1a")
    req = seen[0]
    assert req.method == "GET"
    assert req.url.path == "/qai/v1/avatar/realtime/rt_9f2c1a"
    assert req.headers["Authorization"] == f"Bearer {API_KEY}"
    assert resp.stream_id == "rt_9f2c1a"
    assert resp.status == "streaming"
    assert resp.hls_url == "https://cdn.heygen.com/realtime/rt_9f2c1a/index.m3u8"
    # Omitted-when-empty optionals decode as None.
    assert resp.error_message is None
    assert resp.end_reason is None
    assert resp.request_id == "req_abc123def457"


def test_get_avatar_realtime_session_async() -> None:
    seen: list[httpx.Request] = []
    client = make_async_client(capture(200, STATUS_BODY, seen))
    resp = asyncio.run(client.get_avatar_realtime_session("rt_9f2c1a"))
    assert seen[0].url.path == "/qai/v1/avatar/realtime/rt_9f2c1a"
    assert resp.hls_url == STATUS_BODY["hls_url"]


def test_get_avatar_realtime_session_completed_end_reason() -> None:
    body = {
        "stream_id": "rt_9f2c1a",
        "status": "completed",
        "end_reason": "final_marker",
        "request_id": "req_1",
    }
    client = make_client(capture(200, body, []))
    resp = client.get_avatar_realtime_session("rt_9f2c1a")
    assert resp.status == "completed"
    assert resp.end_reason == "final_marker"
    assert resp.hls_url is None


def test_get_avatar_realtime_session_not_found() -> None:
    err_body = {
        "error": {
            "message": "session rt_x not found",
            "type": "not_found",
            "code": "not_found",
        }
    }
    client = make_client(capture(404, err_body, []))
    with pytest.raises(APIError) as exc_info:
        client.get_avatar_realtime_session("rt_x")
    assert exc_info.value.status_code == 404
    assert exc_info.value.is_not_found()


# ---------------------------------------------------------------------------
# 3. POST /qai/v1/avatar/realtime/{id}/text — append text delta
# ---------------------------------------------------------------------------

TEXT_BODY = {
    "ok": True,
    "buffered_bytes": 512,
    "final": False,
    "request_id": "req_abc123def458",
}


def test_send_avatar_realtime_text_delta() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, TEXT_BODY, seen))
    resp = client.send_avatar_realtime_text("rt_9f2c1a", " and here is the rest.")
    req = seen[0]
    assert req.method == "POST"
    assert req.url.path == "/qai/v1/avatar/realtime/rt_9f2c1a/text"
    assert req.headers["Authorization"] == f"Bearer {API_KEY}"
    assert json.loads(req.content) == {"delta": " and here is the rest.", "final": False}
    assert resp.ok is True
    assert resp.buffered_bytes == 512
    assert resp.is_final is False
    assert resp.request_id == "req_abc123def458"


def test_send_avatar_realtime_text_final_marker_omits_empty_delta() -> None:
    seen: list[httpx.Request] = []
    body = {"ok": True, "buffered_bytes": 512, "final": True, "request_id": "req_1"}
    client = make_client(capture(200, body, seen))
    resp = client.send_avatar_realtime_text("rt_9f2c1a", final=True)
    assert json.loads(seen[0].content) == {"final": True}
    assert resp.is_final is True


def test_send_avatar_realtime_text_async() -> None:
    seen: list[httpx.Request] = []
    client = make_async_client(capture(200, TEXT_BODY, seen))
    resp = asyncio.run(client.send_avatar_realtime_text("rt_9f2c1a", "tok", final=False))
    assert seen[0].url.path == "/qai/v1/avatar/realtime/rt_9f2c1a/text"
    assert json.loads(seen[0].content) == {"delta": "tok", "final": False}
    assert resp.buffered_bytes == 512


def test_send_avatar_realtime_text_closed_stream_410() -> None:
    # Upstream 410 on a closed text stream passes through as provider_error.
    err_body = {
        "error": {
            "message": "text stream already closed",
            "type": "provider_error",
            "code": "provider_error",
        }
    }
    client = make_client(capture(410, err_body, []))
    with pytest.raises(APIError) as exc_info:
        client.send_avatar_realtime_text("rt_9f2c1a", "late")
    assert exc_info.value.status_code == 410
    assert exc_info.value.code == "provider_error"


# ---------------------------------------------------------------------------
# 4. POST /qai/v1/avatar/realtime/{id}/cancel — terminate early
# ---------------------------------------------------------------------------

CANCEL_BODY = {
    "stream_id": "rt_9f2c1a",
    "cancelled": True,
    "request_id": "req_abc123def459",
}


def test_cancel_avatar_realtime_session() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, CANCEL_BODY, seen))
    resp = client.cancel_avatar_realtime_session("rt_9f2c1a")
    req = seen[0]
    assert req.method == "POST"
    assert req.url.path == "/qai/v1/avatar/realtime/rt_9f2c1a/cancel"
    assert req.headers["Authorization"] == f"Bearer {API_KEY}"
    assert req.content == b""  # no request body
    assert resp.stream_id == "rt_9f2c1a"
    assert resp.cancelled is True
    assert resp.request_id == "req_abc123def459"


def test_cancel_avatar_realtime_session_async_idempotent() -> None:
    seen: list[httpx.Request] = []
    body = {"stream_id": "rt_9f2c1a", "cancelled": False, "request_id": "req_2"}
    client = make_async_client(capture(200, body, seen))
    resp = asyncio.run(client.cancel_avatar_realtime_session("rt_9f2c1a"))
    assert seen[0].url.path == "/qai/v1/avatar/realtime/rt_9f2c1a/cancel"
    assert seen[0].content == b""
    assert resp.cancelled is False  # already-terminal session


# ---------------------------------------------------------------------------
# 5. GET /qai/v1/audio/sounds — sounds catalog search
# ---------------------------------------------------------------------------

SOUNDS_BODY = {
    "sounds": [
        {
            "id": "trk_8842aa",
            "name": "Uplifting Corporate",
            "description": "Bright, optimistic corporate track with piano and strings",
            "audio_url": "https://resource.heygen.ai/sounds/trk_8842aa.wav?sig=abc",
            "duration": 94.5,
            "score": 0.91,
            "type": "music",
        }
    ],
    "has_more": True,
    "next_token": "eyJvZmZzZXQiOjEwfQ",
    "request_id": "req_abc123def45a",
}


def _assert_sounds_response(resp) -> None:
    assert len(resp.sounds) == 1
    track = resp.sounds[0]
    assert track.id == "trk_8842aa"
    assert track.name == "Uplifting Corporate"
    assert track.audio_url == "https://resource.heygen.ai/sounds/trk_8842aa.wav?sig=abc"
    assert track.duration == 94.5
    assert track.score == 0.91
    assert track.sound_type == "music"
    assert resp.has_more is True
    assert resp.next_token == "eyJvZmZzZXQiOjEwfQ"
    assert resp.request_id == "req_abc123def45a"


def test_search_audio_sounds() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, SOUNDS_BODY, seen))
    resp = client.search_audio_sounds(
        "calm piano", sound_type="music", limit=10, min_score=0.7, token="tok en",
    )
    req = seen[0]
    assert req.method == "GET"
    assert req.url.path == "/qai/v1/audio/sounds"
    assert req.headers["Authorization"] == f"Bearer {API_KEY}"
    # Query values are percent-encoded; wire param for the catalog is `type`.
    assert req.url.query == b"query=calm%20piano&type=music&limit=10&min_score=0.7&token=tok%20en"
    _assert_sounds_response(resp)


def test_search_audio_sounds_async_minimal_query() -> None:
    seen: list[httpx.Request] = []
    client = make_async_client(capture(200, SOUNDS_BODY, seen))
    resp = asyncio.run(client.search_audio_sounds("calm piano"))
    assert seen[0].url.query == b"query=calm%20piano"
    _assert_sounds_response(resp)


def test_search_audio_sounds_empty_page() -> None:
    body = {"sounds": [], "has_more": False, "next_token": "", "request_id": "req_1"}
    client = make_client(capture(200, body, []))
    resp = client.search_audio_sounds("nothing")
    assert resp.sounds == []
    assert resp.has_more is False
    assert resp.next_token == ""


# ---------------------------------------------------------------------------
# 6. GET /qai/v1/video/template/{id} — template variable schema
# ---------------------------------------------------------------------------

TEMPLATE_DETAIL_BODY = {
    "template": {
        "id": "tmpl_5f0a",
        "name": "Product Launch",
        "aspect_ratio": "16:9",
        "variables": {
            "headline": {"type": "text", "content": "Default headline"},
            "presenter": {
                "type": "character",
                "character_id": "Abigail_expressive_2024112501",
                "character_type": "avatar",
            },
        },
        "scenes": [
            {
                "scene_id": "scene_1",
                "script": "Introducing {{headline}}...",
                "variables": [{"name": "headline", "variable_type": "text"}],
            }
        ],
    },
    "request_id": "req_abc123def45b",
}


def _assert_template_detail(resp) -> None:
    tmpl = resp.template
    assert tmpl.id == "tmpl_5f0a"
    assert tmpl.name == "Product Launch"
    assert tmpl.aspect_ratio == "16:9"
    # Variable unions round-trip verbatim as raw dicts.
    assert tmpl.variables["headline"] == {"type": "text", "content": "Default headline"}
    assert tmpl.variables["presenter"]["character_id"] == "Abigail_expressive_2024112501"
    assert len(tmpl.scenes) == 1
    scene = tmpl.scenes[0]
    assert scene.scene_id == "scene_1"
    assert scene.script == "Introducing {{headline}}..."
    assert scene.variables[0].name == "headline"
    assert scene.variables[0].variable_type == "text"
    assert resp.request_id == "req_abc123def45b"


def test_video_template_detail() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, TEMPLATE_DETAIL_BODY, seen))
    resp = client.video_template_detail("tmpl_5f0a")
    req = seen[0]
    assert req.method == "GET"
    assert req.url.path == "/qai/v1/video/template/tmpl_5f0a"
    assert req.headers["Authorization"] == f"Bearer {API_KEY}"
    _assert_template_detail(resp)


def test_video_template_detail_async() -> None:
    seen: list[httpx.Request] = []
    client = make_async_client(capture(200, TEMPLATE_DETAIL_BODY, seen))
    resp = asyncio.run(client.video_template_detail("tmpl_5f0a"))
    assert seen[0].url.path == "/qai/v1/video/template/tmpl_5f0a"
    _assert_template_detail(resp)


def test_video_template_detail_unknown_id_provider_error() -> None:
    # Upstream 404 for an unknown template id passes through as provider_error.
    err_body = {
        "error": {
            "message": "template not found",
            "type": "provider_error",
            "code": "provider_error",
        }
    }
    client = make_client(capture(404, err_body, []))
    with pytest.raises(APIError) as exc_info:
        client.video_template_detail("tmpl_missing")
    assert exc_info.value.status_code == 404
    assert exc_info.value.code == "provider_error"


# ---------------------------------------------------------------------------
# 7. POST /qai/v1/video/template/{id} — render from template (async job)
# ---------------------------------------------------------------------------

TEMPLATE_ACCEPTED_BODY = {
    "job_id": "qai_job_3def45c00112",
    "status": "pending",
    "type": "video/template-v3",
    "request_id": "req_abc123def45c",
}


def _generate_request() -> VideoTemplateGenerateRequest:
    return VideoTemplateGenerateRequest(
        variables={"headline": {"type": "text", "content": "Big News"}},
        title="Launch video",
        scene_ids=["scene_1"],
        dimension=VideoTemplateDimension(width=1920, height=1080),
        fps=30,
        caption=True,
        subtitles=VideoTemplateSubtitles(preset_name="classic", alignment=2),
        reorder_music=False,
        include_gif=True,
    )


def _assert_generate_request(req: httpx.Request) -> None:
    assert req.method == "POST"
    assert req.url.path == "/qai/v1/video/template/tmpl_5f0a"
    assert req.headers["Authorization"] == f"Bearer {API_KEY}"
    sent = json.loads(req.content)
    assert sent == {
        "variables": {"headline": {"type": "text", "content": "Big News"}},
        "title": "Launch video",
        "scene_ids": ["scene_1"],
        "dimension": {"width": 1920, "height": 1080},
        "fps": 30,
        "caption": True,
        "subtitles": {"preset_name": "classic", "alignment": 2},
        "reorder_music": False,
        "include_gif": True,
    }


def _assert_accepted(resp) -> None:
    assert resp.job_id == "qai_job_3def45c00112"
    assert resp.status == "pending"
    assert resp.job_type == "video/template-v3"  # wire field: type
    assert resp.request_id == "req_abc123def45c"


def test_video_template_generate() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(202, TEMPLATE_ACCEPTED_BODY, seen))
    resp = client.video_template_generate("tmpl_5f0a", _generate_request())
    _assert_generate_request(seen[0])
    _assert_accepted(resp)


def test_video_template_generate_async() -> None:
    seen: list[httpx.Request] = []
    client = make_async_client(capture(202, TEMPLATE_ACCEPTED_BODY, seen))
    resp = asyncio.run(client.video_template_generate("tmpl_5f0a", _generate_request()))
    _assert_generate_request(seen[0])
    _assert_accepted(resp)


def test_video_template_generate_minimal_body_omits_optionals() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(202, TEMPLATE_ACCEPTED_BODY, seen))
    client.video_template_generate(
        "tmpl_5f0a",
        VideoTemplateGenerateRequest(variables={"headline": {"type": "text", "content": "x"}}),
    )
    assert json.loads(seen[0].content) == {
        "variables": {"headline": {"type": "text", "content": "x"}}
    }


def test_video_template_generate_empty_variables_400() -> None:
    err_body = {
        "error": {
            "message": "at least one variable is required",
            "type": "invalid_request",
            "code": "invalid_request",
        }
    }
    client = make_client(capture(400, err_body, []))
    with pytest.raises(APIError) as exc_info:
        client.video_template_generate("tmpl_5f0a", VideoTemplateGenerateRequest(variables={}))
    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "invalid_request"


# ---------------------------------------------------------------------------
# 8. POST /qai/v1/video/batch — submit batch
# ---------------------------------------------------------------------------

BATCH_SUBMIT_BODY = {
    "batch_id": "batch_66aa1c",
    "status": "processing",
    "total_items": 2,
    "request_id": "req_abc123def45d",
}

BATCH_VIDEOS: list[dict[str, Any]] = [
    {
        "type": "avatar",
        "avatar_id": "Abigail_expressive_2024112501",
        "voice_id": "73c0b6a2",
        "script": "Welcome to the team!",
    },
    {
        "type": "avatar",
        "avatar_id": "Abigail_expressive_2024112501",
        "voice_id": "73c0b6a2",
        "script": "Here is how billing works.",
    },
]


def _assert_batch_submit_request(req: httpx.Request) -> None:
    assert req.method == "POST"
    assert req.url.path == "/qai/v1/video/batch"
    assert req.headers["Authorization"] == f"Bearer {API_KEY}"
    sent = json.loads(req.content)
    # Raw HeyGen payloads pass through verbatim.
    assert sent == {"videos": BATCH_VIDEOS, "title": "Onboarding videos"}


def _assert_batch_submit_response(resp) -> None:
    assert resp.batch_id == "batch_66aa1c"
    assert resp.status == "processing"
    assert resp.total_items == 2
    assert resp.request_id == "req_abc123def45d"


def test_video_batch_submit() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(202, BATCH_SUBMIT_BODY, seen))
    resp = client.video_batch_submit(
        VideoBatchSubmitRequest(videos=BATCH_VIDEOS, title="Onboarding videos")
    )
    _assert_batch_submit_request(seen[0])
    _assert_batch_submit_response(resp)


def test_video_batch_submit_async() -> None:
    seen: list[httpx.Request] = []
    client = make_async_client(capture(202, BATCH_SUBMIT_BODY, seen))
    resp = asyncio.run(
        client.video_batch_submit(
            VideoBatchSubmitRequest(videos=BATCH_VIDEOS, title="Onboarding videos")
        )
    )
    _assert_batch_submit_request(seen[0])
    _assert_batch_submit_response(resp)


def test_video_batch_submit_without_title_omits_field() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(202, BATCH_SUBMIT_BODY, seen))
    client.video_batch_submit(VideoBatchSubmitRequest(videos=BATCH_VIDEOS[:1]))
    assert json.loads(seen[0].content) == {"videos": BATCH_VIDEOS[:1]}


# ---------------------------------------------------------------------------
# 9. GET /qai/v1/video/batch/{id} — batch status + settlement
# ---------------------------------------------------------------------------

BATCH_STATUS_SETTLED_BODY = {
    "batch_id": "batch_66aa1c",
    "title": "Onboarding videos",
    "status": "completed",
    "total_items": 3,
    "counts_by_status": {"completed": 2, "failed": 1},
    "created_at": 1752741600,
    "items": [
        {
            "item_index": 0,
            "status": "completed",
            "video_id": "vid_001",
            "video_url": "https://resource.heygen.ai/video/vid_001.mp4?sig=a",
        },
        {
            "item_index": 1,
            "status": "completed",
            "video_id": "vid_002",
            "video_url": "https://resource.heygen.ai/video/vid_002.mp4?sig=b",
        },
        {
            "item_index": 2,
            "status": "failed",
            "error": {"code": "avatar_not_found", "message": "avatar id not found"},
        },
    ],
    "has_more": False,
    "next_token": "",
    "billing_status": "settled",
    "cost_ticks": 46000000000,
    "request_id": "req_abc123def45e",
}


def _assert_batch_status_settled(resp) -> None:
    assert resp.batch_id == "batch_66aa1c"
    assert resp.title == "Onboarding videos"
    assert resp.status == "completed"
    assert resp.total_items == 3
    assert resp.counts_by_status == {"completed": 2, "failed": 1}
    assert resp.created_at == 1752741600  # unix seconds
    assert resp.billing_status == "settled"
    assert resp.cost_ticks == 46000000000
    assert resp.has_more is False
    assert resp.next_token == ""
    assert len(resp.items) == 3
    assert resp.items[0].item_index == 0
    assert resp.items[0].status == "completed"
    assert resp.items[0].video_id == "vid_001"
    assert resp.items[0].video_url == "https://resource.heygen.ai/video/vid_001.mp4?sig=a"
    assert resp.items[0].error is None
    failed = resp.items[2]
    assert failed.status == "failed"
    assert failed.video_id is None
    assert failed.video_url is None
    assert failed.error is not None
    assert failed.error.code == "avatar_not_found"
    assert failed.error.message == "avatar id not found"
    assert resp.request_id == "req_abc123def45e"


def test_video_batch_status() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, BATCH_STATUS_SETTLED_BODY, seen))
    resp = client.video_batch_status("batch_66aa1c", limit=50, token="cur sor")
    req = seen[0]
    assert req.method == "GET"
    assert req.url.path == "/qai/v1/video/batch/batch_66aa1c"
    assert req.url.query == b"limit=50&token=cur%20sor"
    assert req.headers["Authorization"] == f"Bearer {API_KEY}"
    _assert_batch_status_settled(resp)


def test_video_batch_status_async_no_query() -> None:
    seen: list[httpx.Request] = []
    client = make_async_client(capture(200, BATCH_STATUS_SETTLED_BODY, seen))
    resp = asyncio.run(client.video_batch_status("batch_66aa1c"))
    req = seen[0]
    assert req.url.path == "/qai/v1/video/batch/batch_66aa1c"
    assert req.url.query == b""
    _assert_batch_status_settled(resp)


def test_video_batch_status_pending_settlement_withholds_urls() -> None:
    body = {
        "batch_id": "batch_66aa1c",
        "title": "",
        "status": "completed",
        "total_items": 1,
        "counts_by_status": {"completed": 1},
        "created_at": 1752741600,
        "items": [{"item_index": 0, "status": "completed", "video_id": "vid_001"}],
        "has_more": False,
        "next_token": "",
        "billing_status": "settlement_pending",
        "cost_ticks": 0,
        "request_id": "req_1",
    }
    client = make_client(capture(200, body, []))
    resp = client.video_batch_status("batch_66aa1c")
    assert resp.billing_status == "settlement_pending"
    assert resp.cost_ticks == 0
    # URLs withheld until billing_status == "settled".
    assert resp.items[0].video_url is None
    assert resp.items[0].video_id == "vid_001"


def test_video_batch_status_not_found() -> None:
    err_body = {
        "error": {
            "message": "batch batch_x not found",
            "type": "not_found",
            "code": "not_found",
        }
    }
    client = make_client(capture(404, err_body, []))
    with pytest.raises(APIError) as exc_info:
        client.video_batch_status("batch_x")
    assert exc_info.value.is_not_found()
