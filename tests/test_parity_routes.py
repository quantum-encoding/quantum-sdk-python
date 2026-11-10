"""Wire contract for the methods added to close the Rust-parity gap.

Mock-gateway only, no network. Each test pins the thing the Rust reference
(quantum-sdk 0.7.3) actually specifies: path, HTTP method, request body, and
response decode. Routes come from the matching src/<area>.rs impl block.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import httpx
import pytest

from quantum_sdk import (
    AsyncClient,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Client,
    CollectionSearchRequest,
    DialogueRequest,
    GoogleSearchRequest,
    MissionApproveRequest,
    MissionChatRequest,
    MissionCreateRequest,
    MissionPlanUpdate,
    SecurityScanHtmlRequest,
    VisionContext,
    VisionRequest,
    VoiceLibraryQuery,
)
from quantum_sdk.types_ext import DialogueTurn

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
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status_code, json=body)

    return handler


def sent_body(req: httpx.Request) -> Any:
    return json.loads(req.content) if req.content else None


# ---------------------------------------------------------------------------
# Missions
# ---------------------------------------------------------------------------

def test_mission_create_posts_the_plan() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"mission_id": "m_1", "status": "running"}, seen))

    resp = client.mission_create(MissionCreateRequest(goal="ship it", strategy="codegen"))

    assert seen[0].method == "POST"
    assert seen[0].url.path == "/qai/v1/missions/create"
    assert sent_body(seen[0]) == {"goal": "ship it", "strategy": "codegen"}
    assert resp.mission_id == "m_1"
    assert resp.status == "running"


def test_mission_list_encodes_the_status_filter() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"missions": [{"id": "m_1", "goal": "g"}]}, seen))

    resp = client.mission_list(status="running")

    assert seen[0].method == "GET"
    assert seen[0].url.path == "/qai/v1/missions/list"
    assert seen[0].url.params["status"] == "running"
    assert [m.id for m in resp.missions] == ["m_1"]


def test_mission_get_decodes_tasks() -> None:
    seen: list[httpx.Request] = []
    body = {
        "id": "m_1",
        "goal": "ship it",
        "status": "completed",
        "tasks": [{"id": "t_1", "name": "build", "status": "done", "step": 2}],
    }
    client = make_client(capture(200, body, seen))

    resp = client.mission_get("m_1")

    assert seen[0].url.path == "/qai/v1/missions/m_1"
    assert len(resp.tasks) == 1
    assert resp.tasks[0].name == "build"
    assert resp.tasks[0].step == 2


@pytest.mark.parametrize(
    ("call", "method", "path"),
    [
        (lambda c: c.mission_delete("m_1"), "DELETE", "/qai/v1/missions/m_1"),
        (lambda c: c.mission_cancel("m_1"), "POST", "/qai/v1/missions/m_1/cancel"),
        (lambda c: c.mission_pause("m_1"), "POST", "/qai/v1/missions/m_1/pause"),
        (lambda c: c.mission_resume("m_1"), "POST", "/qai/v1/missions/m_1/resume"),
        (lambda c: c.mission_retry_task("m_1", "t_2"), "POST", "/qai/v1/missions/m_1/retry/t_2"),
    ],
)
def test_mission_lifecycle_routes(call, method, path) -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"mission_id": "m_1", "status": "cancelled"}, seen))

    resp = call(client)

    assert seen[0].method == method
    assert seen[0].url.path == path
    assert resp.mission_id == "m_1"


def test_mission_update_plan_uses_put() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"mission_id": "m_1", "updated": True}, seen))

    resp = client.mission_update_plan("m_1", MissionPlanUpdate(max_steps=12))

    assert seen[0].method == "PUT"
    assert seen[0].url.path == "/qai/v1/missions/m_1/plan"
    assert sent_body(seen[0]) == {"max_steps": 12}
    assert resp.updated is True


def test_mission_chat_and_approve_and_checkpoints() -> None:
    seen: list[httpx.Request] = []
    client = make_client(
        capture(200, {"mission_id": "m_1", "content": "ok", "usage": {"input_tokens": 3}}, seen)
    )
    chat = client.mission_chat("m_1", MissionChatRequest(message="status?"))
    assert seen[-1].url.path == "/qai/v1/missions/m_1/chat"
    assert chat.usage is not None and chat.usage.input_tokens == 3

    client = make_client(capture(200, {"mission_id": "m_1", "approved": True}, seen))
    approved = client.mission_approve("m_1", MissionApproveRequest(commit_sha="abc123"))
    assert seen[-1].url.path == "/qai/v1/missions/m_1/approve"
    assert sent_body(seen[-1]) == {"commit_sha": "abc123"}
    assert approved.approved is True

    client = make_client(
        capture(200, {"mission_id": "m_1", "checkpoints": [{"commit_sha": "abc123"}]}, seen)
    )
    cps = client.mission_checkpoints("m_1")
    assert seen[-1].url.path == "/qai/v1/missions/m_1/checkpoints"
    assert cps.checkpoints[0].commit_sha == "abc123"


def test_mission_import_posts_tasks() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"mission_id": "m_2"}, seen))

    from quantum_sdk import MissionImportRequest

    client.mission_import(MissionImportRequest(goal="g", tasks=[{"name": "a"}], auto_execute=True))

    assert seen[0].url.path == "/qai/v1/missions/import"
    assert sent_body(seen[0]) == {"goal": "g", "tasks": [{"name": "a"}], "auto_execute": True}


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("method_name", "path"),
    [
        ("vision_analyze", "/qai/v1/vision/analyze"),
        ("vision_detect", "/qai/v1/vision/detect"),
        ("vision_describe", "/qai/v1/vision/describe"),
        ("vision_ocr", "/qai/v1/vision/ocr"),
        ("vision_quality", "/qai/v1/vision/quality"),
    ],
)
def test_vision_routes(method_name, path) -> None:
    seen: list[httpx.Request] = []
    body = {
        "caption": "a roof",
        "objects": [{"label": "panel", "confidence": 0.9}],
        "model": "gemini-flash-latest",
        "cost_ticks": 42,
    }
    client = make_client(capture(200, body, seen))

    req = VisionRequest(
        image_base64="aGk=",
        context=VisionContext(installation_type="solar"),
    )
    resp = getattr(client, method_name)(req)

    assert seen[0].method == "POST"
    assert seen[0].url.path == path
    assert sent_body(seen[0]) == {
        "image_base64": "aGk=",
        "context": {"installation_type": "solar"},
    }
    assert resp.caption == "a roof"
    assert resp.objects[0].label == "panel"
    assert resp.cost_ticks == 42


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

def test_security_scan_url_wraps_the_url_in_a_body() -> None:
    seen: list[httpx.Request] = []
    body = {"assessment": {"threat_level": "high", "threat_score": 88.0}, "request_id": "r1"}
    client = make_client(capture(200, body, seen))

    resp = client.security_scan_url("https://evil.example/page")

    assert seen[0].url.path == "/qai/v1/security/scan-url"
    assert sent_body(seen[0]) == {"url": "https://evil.example/page"}
    assert resp.assessment.threat_level == "high"
    assert resp.request_id == "r1"


def test_security_check_percent_encodes_the_url() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"url": "u", "blocked": True}, seen))

    resp = client.security_check("https://evil.example/a?b=c&d=e")

    assert seen[0].method == "GET"
    assert seen[0].url.path == "/qai/v1/security/check"
    # The whole URL must survive as ONE query value, not leak extra params.
    assert list(seen[0].url.params.keys()) == ["url"]
    assert seen[0].url.params["url"] == "https://evil.example/a?b=c&d=e"
    assert resp.blocked is True


def test_security_scan_html_and_blocklist_and_report() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"assessment": {"summary": "clean"}}, seen))
    client.security_scan_html(SecurityScanHtmlRequest(html="<p>hi</p>", url="https://a.example"))
    assert seen[-1].url.path == "/qai/v1/security/scan-html"
    assert sent_body(seen[-1]) == {"html": "<p>hi</p>", "url": "https://a.example"}

    client = make_client(capture(200, {"entries": [{"url": "u"}], "count": 1}, seen))
    feed = client.security_blocklist(status="confirmed")
    assert seen[-1].url.path == "/qai/v1/security/blocklist"
    assert seen[-1].url.params["status"] == "confirmed"
    assert feed.count == 1

    from quantum_sdk import SecurityReportRequest

    client = make_client(capture(200, {"url": "u", "status": "suspected"}, seen))
    rep = client.security_report(SecurityReportRequest(url="u", category="phishing"))
    assert seen[-1].url.path == "/qai/v1/security/report"
    assert rep.status == "suspected"


# ---------------------------------------------------------------------------
# Credits / batch / auth
# ---------------------------------------------------------------------------

def test_credit_routes() -> None:
    seen: list[httpx.Request] = []

    client = make_client(capture(200, {"balance_ticks": 500, "balance_usd": 5.0}, seen))
    bal = client.credit_balance()
    assert seen[-1].url.path == "/qai/v1/credits/balance"
    assert bal.balance_ticks == 500

    client = make_client(capture(200, {"packs": [{"id": "p1", "label": "$5", "ticks": 50}]}, seen))
    packs = client.credit_packs()
    assert seen[-1].url.path == "/qai/v1/credits/packs"
    # Field names follow the Rust reference (label/amount_usd/ticks).
    assert packs.packs[0].label == "$5"
    assert packs.packs[0].ticks == 50

    client = make_client(capture(200, {"checkout_url": "https://pay.example"}, seen))
    buy = client.credit_purchase("p1", success_url="https://ok.example")
    assert seen[-1].url.path == "/qai/v1/credits/purchase"
    assert sent_body(seen[-1]) == {"pack_id": "p1", "success_url": "https://ok.example"}
    assert buy.checkout_url == "https://pay.example"

    client = make_client(capture(200, {"tiers": [{"name": "gold", "bonus": 3}]}, seen))
    tiers = client.credit_tiers()
    assert seen[-1].url.path == "/qai/v1/credits/tiers"
    # Unknown keys land in `extra` rather than being dropped.
    assert tiers.tiers[0].extra == {"bonus": 3}

    client = make_client(capture(200, {"status": "submitted"}, seen))
    app = client.dev_program_apply("research", company="QE")
    assert seen[-1].url.path == "/qai/v1/credits/dev-program"
    assert sent_body(seen[-1]) == {"use_case": "research", "company": "QE"}
    assert app.status == "submitted"


def test_auth_apple_sends_name_only_on_first_sign_in() -> None:
    seen: list[httpx.Request] = []
    body = {"token": "qai_x", "user": {"id": "u1", "email": "a@b.c"}}

    client = make_client(capture(200, body, seen))
    resp = client.auth_apple("jwt.token.here", name="Ada")
    assert seen[-1].url.path == "/qai/v1/auth/apple"
    assert sent_body(seen[-1]) == {"id_token": "jwt.token.here", "name": "Ada"}
    assert resp.token == "qai_x"
    assert resp.user.email == "a@b.c"

    client = make_client(capture(200, body, seen))
    client.auth_apple("jwt.token.here")
    assert sent_body(seen[-1]) == {"id_token": "jwt.token.here"}


# ---------------------------------------------------------------------------
# RAG collections
# ---------------------------------------------------------------------------

# ragproxy.Collection as the gateway encodes it: provider_collection_id is
# always present, description is omitempty.
GATEWAY_COLLECTION = {
    "id": "c1",
    "owner": "u1",
    "provider": "xai",
    "name": "docs",
    "description": "product docs",
    "provider_collection_id": "xai_col_1",
    "document_count": 2,
    "created_at": "2026-11-10T12:00:00Z",
}

# ragproxy.CollectionDocument as the gateway encodes it.
GATEWAY_DOCUMENT = {
    "id": "d1",
    "collection_id": "c1",
    "file_id": "file_9",
    "filename": "a.md",
    "status": "indexed",
    "chunks": 0,
    "uploaded_at": "2026-11-10T12:01:00Z",
}


def test_collections_list_decodes_the_gateway_collection() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"collections": [GATEWAY_COLLECTION], "request_id": "r1"}, seen))

    cols = client.collections_list()

    assert seen[-1].method == "GET"
    assert seen[-1].url.path == "/qai/v1/rag/collections"
    assert cols[0].name == "docs"
    assert cols[0].provider_collection_id == "xai_col_1"
    assert (cols[0].owner, cols[0].document_count) == ("u1", 2)


def test_collections_list_null_is_empty() -> None:
    """The gateway encodes an empty slice as null."""
    client = make_client(capture(200, {"collections": None, "request_id": "r1"}, []))
    assert client.collections_list() == []


def test_collections_list_ignores_unknown_keys() -> None:
    col = {**GATEWAY_COLLECTION, "added_later": 1}
    client = make_client(capture(200, {"collections": [col]}, []))
    assert client.collections_list()[0].id == "c1"


def test_collections_create_decodes_the_gateway_collection() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(201, GATEWAY_COLLECTION, seen))

    created = client.collections_create("docs")

    assert seen[-1].method == "POST"
    assert seen[-1].url.path == "/qai/v1/rag/collections"
    assert sent_body(seen[-1]) == {"name": "docs"}
    assert created.id == "c1"
    assert created.provider_collection_id == "xai_col_1"


def test_collections_get_returns_the_collection_and_its_documents() -> None:
    seen: list[httpx.Request] = []
    body = {"collection": GATEWAY_COLLECTION, "documents": [GATEWAY_DOCUMENT]}
    client = make_client(capture(200, body, seen))

    detail = client.collections_get("c1")

    assert seen[-1].method == "GET"
    assert seen[-1].url.path == "/qai/v1/rag/collections/c1"
    assert detail.collection.name == "docs"
    assert detail.collection.document_count == 2
    doc = detail.documents[0]
    assert (doc.id, doc.collection_id, doc.file_id, doc.filename) == ("d1", "c1", "file_9", "a.md")
    assert (doc.status, doc.chunks, doc.uploaded_at) == ("indexed", 0, "2026-11-10T12:01:00Z")


def test_collections_get_null_documents_is_empty() -> None:
    client = make_client(capture(200, {"collection": GATEWAY_COLLECTION, "documents": None}, []))
    assert client.collections_get("c1").documents == []


def test_collections_delete_returns_deleted_and_id() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"deleted": True, "id": "c1"}, seen))

    resp = client.collections_delete("c1")

    assert seen[-1].method == "DELETE"
    assert seen[-1].url.path == "/qai/v1/rag/collections/c1"
    assert (resp.deleted, resp.id) == (True, "c1")


def test_collections_async_decodes_the_gateway_json() -> None:
    responses = iter([
        {"collections": None},
        GATEWAY_COLLECTION,
        {"collection": GATEWAY_COLLECTION, "documents": None},
        {"deleted": True, "id": "c1"},
        GATEWAY_DOCUMENT,
    ])
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=next(responses))

    client = make_async_client(handler)

    async def run() -> tuple:
        return (
            await client.collections_list(),
            await client.collections_create("docs"),
            await client.collections_get("c1"),
            await client.collections_delete("c1"),
            await client.collections_upload("c1", "a.md", b"# hi"),
        )

    listed, created, detail, deleted, uploaded = asyncio.run(run())
    assert listed == []
    assert created.provider_collection_id == "xai_col_1"
    assert (detail.collection.id, detail.documents) == ("c1", [])
    assert deleted.deleted is True
    assert uploaded.file_id == "file_9"
    assert [r.url.path for r in seen][-1] == "/qai/v1/rag/collections/c1/upload"


# The body handleRAGCollectionsSearch writes (routes_rag_collections.go):
# every searchResult key, with document_id/filename present (omitempty).
COLLECTIONS_SEARCH_BODY = {
    "results": [
        {
            "content": "hit",
            "score": 0.8,
            "collection": "docs",
            "collection_id": "c1",
            "document_id": "d1",
            "filename": "a.md",
            "is_shared": True,
        },
        {"content": "lower", "score": 0.2, "collection": "notes", "collection_id": "c2", "is_shared": False},
    ],
    "query": "q",
    "collections_searched": 2,
    "request_id": "req_1",
}


def test_collections_search_posts_the_handler_fields() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, COLLECTIONS_SEARCH_BODY, seen))

    results = client.collections_search(
        CollectionSearchRequest(query="q", collection_ids=["c1"], max_chunks=5)
    )

    assert seen[0].method == "POST"
    assert seen[0].url.path == "/qai/v1/rag/collections/search"
    # The handler decodes exactly these keys: query, collection_ids, max_chunks.
    assert sent_body(seen[0]) == {"query": "q", "collection_ids": ["c1"], "max_chunks": 5}

    assert [r.content for r in results] == ["hit", "lower"]
    top = results[0]
    assert (top.score, top.collection, top.collection_id) == (0.8, "docs", "c1")
    assert (top.document_id, top.filename, top.is_shared) == ("d1", "a.md", True)
    # document_id and filename are omitempty on the wire.
    assert (results[1].document_id, results[1].filename) == ("", "")


def test_collections_search_omits_unset_filters() -> None:
    """Empty collection_ids means "all accessible"; unset max_chunks means 10."""
    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"results": None, "query": "q", "collections_searched": 0}, seen))

    results = client.collections_search(CollectionSearchRequest(query="q"))

    assert sent_body(seen[0]) == {"query": "q"}
    assert results == []


def test_collections_search_async_posts_the_handler_fields() -> None:
    seen: list[httpx.Request] = []
    client = make_async_client(capture(200, COLLECTIONS_SEARCH_BODY, seen))

    results = asyncio.run(client.collections_search(CollectionSearchRequest(query="q", max_chunks=3)))

    assert seen[0].url.path == "/qai/v1/rag/collections/search"
    assert sent_body(seen[0]) == {"query": "q", "max_chunks": 3}
    assert results[0].filename == "a.md"


def test_collections_search_response_keeps_the_envelope() -> None:
    from quantum_sdk import CollectionSearchResponse

    resp = CollectionSearchResponse.from_dict(COLLECTIONS_SEARCH_BODY)
    assert (resp.query, resp.collections_searched, resp.request_id) == ("q", 2, "req_1")


def test_collections_upload_sends_multipart() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, GATEWAY_DOCUMENT, seen))

    resp = client.collections_upload("c1", "a.md", b"# hi")

    assert seen[0].url.path == "/qai/v1/rag/collections/c1/upload"
    assert seen[0].headers["Content-Type"].startswith("multipart/form-data")
    # The handler reads the multipart field "file".
    assert b'name="file"; filename="a.md"' in seen[0].content
    assert b"# hi" in seen[0].content
    assert (resp.id, resp.file_id, resp.filename, resp.status) == ("d1", "file_9", "a.md", "indexed")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def test_google_search_decodes_grounding_metadata() -> None:
    seen: list[httpx.Request] = []
    body = {
        "answer": "Yes.",
        "citations": [{"url": "https://a.example", "title": "A"}],
        "search_entry_point": "<div>chips</div>",
        "web_search_queries": ["is it so"],
        "supports": [
            {"start_index": 0, "end_index": 4, "text": "Yes.", "grounding_chunk_indices": [0]}
        ],
    }
    client = make_client(capture(200, body, seen))

    resp = client.google_search(GoogleSearchRequest(query="is it so?"))

    assert seen[0].url.path == "/qai/v1/search/google"
    assert sent_body(seen[0]) == {"query": "is it so?"}
    assert resp.citations[0].title == "A"
    # ToS-required widget must survive verbatim.
    assert resp.search_entry_point == "<div>chips</div>"
    assert resp.web_search_queries == ["is it so"]
    assert resp.supports[0].grounding_chunk_indices == [0]


# ---------------------------------------------------------------------------
# Voices / audio finetunes / compute
# ---------------------------------------------------------------------------

def test_voice_library_encodes_every_filter() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"voices": [], "has_more": False}, seen))

    client.voice_library(VoiceLibraryQuery(query="deep narrator", page_size=20, gender="female"))

    assert seen[0].url.path == "/qai/v1/voices/library"
    assert seen[0].url.params["query"] == "deep narrator"
    assert seen[0].url.params["page_size"] == "20"
    assert seen[0].url.params["gender"] == "female"
    assert "cursor" not in seen[0].url.params


def test_voice_library_without_filters_sends_no_query_string() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"voices": []}, seen))

    client.voice_library()

    assert str(seen[0].url) == f"{BASE_URL}/qai/v1/voices/library"


def test_add_voice_from_library() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"voice_id": "v9"}, seen))

    resp = client.add_voice_from_library("owner1", "v1", name="Narrator")

    assert seen[0].url.path == "/qai/v1/voices/library/add"
    assert sent_body(seen[0]) == {
        "public_owner_id": "owner1",
        "voice_id": "v1",
        "name": "Narrator",
    }
    assert resp.voice_id == "v9"


def test_finetune_lifecycle() -> None:
    seen: list[httpx.Request] = []

    client = make_client(capture(200, {"finetune_id": "ft1", "name": "jazz"}, seen))
    info = client.create_finetune("jazz", [("a.mp3", b"\x00\x01", "audio/mpeg")])
    assert seen[-1].url.path == "/qai/v1/audio/finetunes"
    assert seen[-1].headers["Content-Type"].startswith("multipart/form-data")
    assert info.finetune_id == "ft1"

    client = make_client(capture(200, {"finetunes": [{"finetune_id": "ft1", "name": "jazz"}]}, seen))
    lst = client.list_finetunes()
    assert seen[-1].method == "GET"
    assert lst.finetunes[0].finetune_id == "ft1"

    client = make_client(capture(200, {}, seen))
    client.delete_finetune("ft1")
    assert seen[-1].method == "DELETE"
    assert seen[-1].url.path == "/qai/v1/audio/finetunes/ft1"


# ---------------------------------------------------------------------------
# Chat estimate / jobs
# ---------------------------------------------------------------------------

def test_estimate_chat_drops_stream_from_the_payload() -> None:
    seen: list[httpx.Request] = []
    client = make_client(
        capture(200, {"estimated_cost_ticks": 1234, "estimated_cost_usd": 0.5, "model": "m"}, seen)
    )

    req = ChatRequest(model="m", messages=[ChatMessage.user("hi")], stream=True)
    resp = client.estimate_chat(req)

    assert seen[0].url.path == "/qai/v1/chat/estimate"
    # The output ceiling is the same either way; sending `stream` would make the
    # SDK's wire shape diverge from what the server sees.
    assert "stream" not in sent_body(seen[0])
    assert resp.estimated_cost_ticks == 1234
    assert resp.model == "m"


def test_chat_job_and_generate_3d_go_through_the_jobs_api() -> None:
    seen: list[httpx.Request] = []

    client = make_client(capture(200, {"job_id": "j1", "status": "queued"}, seen))
    client.chat_job(ChatRequest(model="m", messages=[ChatMessage.user("hi")], stream=True))
    body = sent_body(seen[-1])
    assert seen[-1].url.path == "/qai/v1/jobs"
    assert body["type"] == "chat"
    assert "stream" not in body["params"]

    client = make_client(capture(200, {"job_id": "j2"}, seen))
    client.generate_3d("meshy-6", prompt="a duck")
    assert sent_body(seen[-1]) == {
        "type": "3d/generate",
        "params": {"model": "meshy-6", "prompt": "a duck"},
    }


def test_stream_job_yields_progress_then_stops_at_complete() -> None:
    lines = [
        'data: {"type":"progress","status":"running"}',
        'data: {"type":"complete","result":{"ok":true}}',
        'data: {"type":"progress","status":"never seen"}',
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/qai/v1/jobs/j1/stream"
        assert request.headers["Accept"] == "text/event-stream"
        return httpx.Response(200, text="\n".join(lines))

    # stream_job opens its own httpx client, so patch the transport it builds.
    import quantum_sdk.client as client_mod

    real_client_cls = client_mod.httpx.Client

    def fake_client(**kwargs: Any) -> httpx.Client:
        return real_client_cls(transport=httpx.MockTransport(handler))

    client_mod.httpx.Client = fake_client  # type: ignore[assignment]
    try:
        client = Client(API_KEY, base_url=BASE_URL)
        events = list(client.stream_job("j1"))
    finally:
        client_mod.httpx.Client = real_client_cls  # type: ignore[assignment]

    assert [e.type for e in events] == ["progress", "complete"]


# ---------------------------------------------------------------------------
# Chat helpers added alongside the routes
# ---------------------------------------------------------------------------

def test_stop_reason_helpers() -> None:
    assert ChatResponse(stop_reason="tool_use").is_tool_use()
    assert ChatResponse(stop_reason="refusal").is_refusal()
    assert ChatResponse(stop_reason="max_tokens").is_max_tokens()
    assert not ChatResponse(stop_reason="end_turn").is_tool_use()


def test_tool_error_marks_the_result() -> None:
    msg = ChatMessage.tool_error("call_1", "boom")
    assert msg.to_dict() == {
        "role": "tool",
        "content": "boom",
        "tool_call_id": "call_1",
        "is_error": True,
    }


def test_stream_tool_use_triplet_is_parsed() -> None:
    from quantum_sdk.client import _parse_sse_event

    start = _parse_sse_event('{"type":"tool_use_start","id":"t1","name":"search"}')
    assert start.tool_use_start is not None
    assert start.tool_use_start.name == "search"

    delta = _parse_sse_event('{"type":"tool_use_input_delta","id":"t1","partial_json":"{\\"q\\":"}')
    assert delta.tool_use_input_delta is not None
    assert delta.tool_use_input_delta.partial_json == '{"q":'

    done = _parse_sse_event('{"type":"tool_use_complete","id":"t1","name":"search","input":{"q":"a"}}')
    assert done.tool_use_complete is not None
    assert done.tool_use_complete.input == {"q": "a"}

    # The legacy atomic event still lands on the old field.
    legacy = _parse_sse_event('{"type":"tool_use","id":"t1","name":"search","input":{"q":"a"}}')
    assert legacy.tool_use is not None and legacy.tool_use_complete is None


def test_dialogue_from_turns_builds_script_and_dedupes_voices() -> None:
    req = DialogueRequest.from_turns(
        [
            DialogueTurn(speaker="Ada", text="Hello.", voice="v1"),
            DialogueTurn(speaker="Bob", text="Hi.", voice="v2"),
            DialogueTurn(speaker="Ada", text="Bye.", voice="v9"),
            DialogueTurn(speaker="Narrator", text="They left."),
        ],
        model="eleven_v3",
    )

    assert req.text == "Ada: Hello.\nBob: Hi.\nAda: Bye.\nNarrator: They left."
    # First voice wins per speaker; a speaker with no voice gets no entry.
    assert [(v.name, v.voice_id) for v in req.voices] == [("Ada", "v1"), ("Bob", "v2")]
    assert req.model == "eleven_v3"


def test_avatar_text_request_constructors() -> None:
    from quantum_sdk import AvatarRealtimeTextRequest

    assert AvatarRealtimeTextRequest.delta_append("more").to_dict() == {
        "delta": "more",
        "final": False,
    }
    assert AvatarRealtimeTextRequest.final_marker().to_dict() == {"final": True}


# ---------------------------------------------------------------------------
# Async parity — the async client must speak the same wire
# ---------------------------------------------------------------------------

def test_async_client_matches_the_sync_routes() -> None:
    seen: list[httpx.Request] = []

    async def run() -> None:
        client = make_async_client(capture(200, {"mission_id": "m_1", "status": "running"}, seen))
        resp = await client.mission_create(MissionCreateRequest(goal="ship it"))
        assert resp.mission_id == "m_1"

        client = make_async_client(capture(200, {"caption": "x"}, seen))
        await client.vision_analyze(VisionRequest(image_url="https://i.example/a.png"))

        client = make_async_client(capture(200, {"balance_ticks": 7, "balance_usd": 0.1}, seen))
        bal = await client.credit_balance()
        assert bal.balance_ticks == 7

        client = make_async_client(capture(200, {"answer": "yes"}, seen))
        await client.google_search(GoogleSearchRequest(query="q"))

    asyncio.run(run())

    assert [r.url.path for r in seen] == [
        "/qai/v1/missions/create",
        "/qai/v1/vision/analyze",
        "/qai/v1/credits/balance",
        "/qai/v1/search/google",
    ]


# ---------------------------------------------------------------------------
# 3D mesh helpers (client_ext): submit a job, then poll it to a terminal state
# ---------------------------------------------------------------------------

def _job_gateway(seen: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"job_id": "j3d", "status": "pending"})
        return httpx.Response(200, json={"job_id": "j3d", "status": "completed", "result": {"ok": True}})

    return handler


@pytest.mark.parametrize(
    ("helper", "kwargs", "job_type"),
    [
        ("remesh", {"model_url": "https://m.example/a.glb"}, "3d/remesh"),
        ("rig", {"model_url": "https://m.example/a.glb"}, "3d/rig"),
        ("retexture", {"model_url": "https://m.example/a.glb"}, "3d/retexture"),
    ],
)
def test_mesh_helper_polls_the_created_job(monkeypatch, helper: str, kwargs: dict, job_type: str) -> None:
    import time

    monkeypatch.setattr(time, "sleep", lambda _s: None)
    seen: list[httpx.Request] = []
    client = make_client(_job_gateway(seen))

    status = getattr(client, helper)(**kwargs)

    assert seen[0].url.path == "/qai/v1/jobs"
    assert sent_body(seen[0]) == {"type": job_type, "params": {"model_url": "https://m.example/a.glb"}}
    assert seen[1].method == "GET"
    assert seen[1].url.path == "/qai/v1/jobs/j3d"
    assert status.status == "completed"


def test_mesh_animate_async_polls_the_created_job(monkeypatch) -> None:
    async def no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    seen: list[httpx.Request] = []
    client = make_async_client(_job_gateway(seen))

    status = asyncio.run(client.animate("rig_1", 42))

    assert sent_body(seen[0]) == {"type": "3d/animate", "params": {"rig_task_id": "rig_1", "action_id": 42}}
    assert seen[1].url.path == "/qai/v1/jobs/j3d"
    assert status.status == "completed"
