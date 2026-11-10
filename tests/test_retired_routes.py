"""The gateway routes this SDK no longer wraps, and the photo-avatar body it sends.

The gateway answers 404 on the user batch, documents, search-answer and
video-studio routes, and on the scraper, compute-billing and
collection-documents paths, so no method or type for them may resurface. The
"screenshot" job type is rejected with a 400. Its
POST /qai/v1/video/photo-avatar takes only the talking-photo body
(photo_base64 + script); anything else is a 400.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from typing import Any, Callable

import httpx
import pytest

import quantum_sdk
from quantum_sdk import AsyncClient, Client, PhotoAvatarRequest

BASE_URL = "https://gw.test"
API_KEY = "qai_test_key"

JOB_ENVELOPE = {
    "job_id": "job_tp_1",
    "status": "pending",
    "type": "video/talking-photo",
    "request_id": "req_1",
}


def capture(seen: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(202, json=JOB_ENVELOPE)

    return handler


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> Client:
    return Client(API_KEY, base_url=BASE_URL, http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def make_async_client(handler: Callable[[httpx.Request], httpx.Response]) -> AsyncClient:
    return AsyncClient(
        API_KEY, base_url=BASE_URL, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


def sent_body(req: httpx.Request) -> Any:
    return json.loads(req.content)


# ---------------------------------------------------------------------------
# POST /qai/v1/video/photo-avatar — talking photo
# ---------------------------------------------------------------------------

def test_photo_avatar_sends_the_talking_photo_body() -> None:
    seen: list[httpx.Request] = []
    resp = make_client(capture(seen)).video_photo_avatar(
        PhotoAvatarRequest(photo_base64="aGVsbG8=", script="Hello there.")
    )
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/qai/v1/video/photo-avatar"
    assert sent_body(seen[0]) == {"photo_base64": "aGVsbG8=", "script": "Hello there."}
    assert resp.job_id == "job_tp_1"
    assert resp.status == "pending"


def test_photo_avatar_sends_optional_voice_and_aspect_ratio() -> None:
    seen: list[httpx.Request] = []
    make_client(capture(seen)).video_photo_avatar(
        PhotoAvatarRequest(photo_base64="aGVsbG8=", script="Hi.", voice_id="v_1", aspect_ratio="9:16")
    )
    assert sent_body(seen[0]) == {
        "photo_base64": "aGVsbG8=",
        "script": "Hi.",
        "voice_id": "v_1",
        "aspect_ratio": "9:16",
    }


def test_photo_avatar_async_sends_the_talking_photo_body() -> None:
    seen: list[httpx.Request] = []
    client = make_async_client(capture(seen))
    resp = asyncio.run(
        client.video_photo_avatar(PhotoAvatarRequest(photo_base64="aGVsbG8=", script="Hello there."))
    )
    assert seen[0].url.path == "/qai/v1/video/photo-avatar"
    assert sent_body(seen[0]) == {"photo_base64": "aGVsbG8=", "script": "Hello there."}
    assert resp.job_id == "job_tp_1"


def test_photo_avatar_never_sends_a_photo_url() -> None:
    """The gateway has no photo_url field; a body carrying one is a 400."""
    seen: list[httpx.Request] = []
    make_client(capture(seen)).video_photo_avatar(PhotoAvatarRequest(photo_base64="aGVsbG8=", script="x"))
    assert "photo_url" not in sent_body(seen[0])


# ---------------------------------------------------------------------------
# Routes the gateway no longer serves
# ---------------------------------------------------------------------------

RETIRED_METHODS = [
    # User batch (submit, JSONL submit, list, get)
    "batch_submit",
    "batch_submit_jsonl",
    "batch_jobs",
    "batch_job",
    # Documents (extract, chunk, process)
    "extract_document",
    "chunk_document",
    "process_document",
    # Search answer
    "search_answer",
    # Video studio
    "video_studio",
    # Scraper (scrape, screenshot) and the "screenshot" job type
    "scrape",
    "screenshot",
    "screenshot_job",
    # Compute billing
    "compute_billing",
    # Collection documents listing
    "collections_documents",
]

RETIRED_TYPES = [
    "BillingEntry",
    "BillingRequest",
    "BillingResponse",
    "CollectionDocument",
    "CollectionDocumentsResponse",
    "ScrapeRequest",
    "ScrapeResponse",
    "ScrapeTarget",
    "ScreenshotJobResponse",
    "ScreenshotRequest",
    "ScreenshotResponse",
    "ScreenshotResult",
    "ScreenshotURL",
    "BatchJob",
    "BatchJobInfo",
    "BatchJobInput",
    "BatchJobsResponse",
    "BatchJsonlResponse",
    "BatchSubmitRequest",
    "BatchSubmitResponse",
    "ChunkDocumentRequest",
    "ChunkDocumentResponse",
    "ChunkRequest",
    "ChunkResponse",
    "DocumentChunk",
    "DocumentRequest",
    "DocumentResponse",
    "ProcessDocumentRequest",
    "ProcessDocumentResponse",
    "ProcessRequest",
    "ProcessResponse",
    "SearchAnswerChoice",
    "SearchAnswerCitation",
    "SearchAnswerMessage",
    "SearchAnswerRequest",
    "SearchAnswerResponse",
    "StudioClip",
    "StudioVideoRequest",
    "VideoPhotoAvatarRequest",
    "VideoStudioRequest",
]


@pytest.mark.parametrize("cls", [Client, AsyncClient])
@pytest.mark.parametrize("name", RETIRED_METHODS)
def test_retired_method_is_gone(cls: type, name: str) -> None:
    assert not hasattr(cls, name), f"{cls.__name__}.{name} wraps a route the gateway removed"


@pytest.mark.parametrize("name", RETIRED_TYPES)
def test_retired_type_is_gone(name: str) -> None:
    import quantum_sdk.types as types
    import quantum_sdk.types_ext as types_ext

    for module in (quantum_sdk, types, types_ext):
        assert not hasattr(module, name), f"{module.__name__}.{name} belongs to a removed route"
    assert name not in quantum_sdk.__all__


@pytest.mark.parametrize("module", ["quantum_sdk.batch", "quantum_sdk.documents"])
def test_retired_module_is_gone(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)
