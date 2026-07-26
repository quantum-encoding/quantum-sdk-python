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
    ScrapeRequest,
    ScrapeTarget,
    ScreenshotRequest,
    ScreenshotURL,
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


def test_batch_routes() -> None:
    from quantum_sdk import BatchJobInput

    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"job_ids": ["j1"], "status": "queued"}, seen))
    resp = client.batch_submit([BatchJobInput(model="m", prompt="p")])
    assert seen[-1].url.path == "/qai/v1/batch"
    assert sent_body(seen[-1]) == {"jobs": [{"model": "m", "prompt": "p"}]}
    assert resp.job_ids == ["j1"]

    client = make_client(capture(200, {"job_ids": ["j2"]}, seen))
    client.batch_submit_jsonl('{"model":"m"}')
    assert seen[-1].url.path == "/qai/v1/batch/jsonl"
    assert sent_body(seen[-1]) == {"jsonl": '{"model":"m"}'}

    client = make_client(capture(200, {"jobs": [{"job_id": "j1", "status": "done"}]}, seen))
    jobs = client.batch_jobs()
    assert seen[-1].url.path == "/qai/v1/batch/jobs"
    assert jobs.jobs[0].job_id == "j1"


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

def test_collections_crud() -> None:
    seen: list[httpx.Request] = []

    client = make_client(capture(200, {"collections": [{"id": "c1", "name": "docs"}]}, seen))
    cols = client.collections_list()
    assert seen[-1].url.path == "/qai/v1/rag/collections"
    assert cols[0].name == "docs"

    client = make_client(capture(200, {"id": "c2", "name": "notes"}, seen))
    created = client.collections_create("notes")
    assert seen[-1].method == "POST"
    assert sent_body(seen[-1]) == {"name": "notes"}
    assert created.id == "c2"

    client = make_client(capture(200, {"id": "c2", "name": "notes", "document_count": 4}, seen))
    got = client.collections_get("c2")
    assert seen[-1].url.path == "/qai/v1/rag/collections/c2"
    assert got.document_count == 4

    client = make_client(capture(200, {"message": "deleted"}, seen))
    assert client.collections_delete("c2") == "deleted"
    assert seen[-1].method == "DELETE"

    client = make_client(capture(200, {"documents": [{"file_id": "f1", "name": "a.md"}]}, seen))
    docs = client.collections_documents("c2")
    assert seen[-1].url.path == "/qai/v1/rag/collections/c2/documents"
    assert docs[0].file_id == "f1"


def test_collections_search_posts_the_filter() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"results": [{"content": "hit", "score": 0.8}]}, seen))

    results = client.collections_search(
        CollectionSearchRequest(query="q", collection_ids=["c1"], mode="semantic")
    )

    assert seen[0].url.path == "/qai/v1/rag/search/collections"
    assert sent_body(seen[0]) == {
        "query": "q",
        "collection_ids": ["c1"],
        "mode": "semantic",
    }
    assert results[0].content == "hit"


def test_collections_upload_sends_multipart() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"file_id": "f9", "filename": "a.md"}, seen))

    resp = client.collections_upload("c1", "a.md", b"# hi")

    assert seen[0].url.path == "/qai/v1/rag/collections/c1/upload"
    assert seen[0].headers["Content-Type"].startswith("multipart/form-data")
    assert b"# hi" in seen[0].content
    assert resp.file_id == "f9"


# ---------------------------------------------------------------------------
# Search / scraper
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


def test_scrape_and_screenshot_routes() -> None:
    seen: list[httpx.Request] = []

    client = make_client(capture(200, {"job_id": "j1", "status": "queued", "targets": 1}, seen))
    resp = client.scrape(ScrapeRequest(targets=[ScrapeTarget(name="docs", url="https://d.example", recursive=True)]))
    assert seen[-1].url.path == "/qai/v1/scraper/scrape"
    assert sent_body(seen[-1]) == {
        "targets": [{"name": "docs", "url": "https://d.example", "recursive": True}]
    }
    assert resp.job_id == "j1"

    shots = {"screenshots": [{"url": "https://d.example", "base64": "aGk=", "format": "png"}], "count": 1}
    client = make_client(capture(200, shots, seen))
    resp2 = client.screenshot(ScreenshotRequest(urls=[ScreenshotURL(url="https://d.example", full_page=True)]))
    assert seen[-1].url.path == "/qai/v1/scraper/screenshot"
    assert sent_body(seen[-1]) == {"urls": [{"url": "https://d.example", "full_page": True}]}
    assert resp2.screenshots[0].format == "png"

    client = make_client(capture(200, {"job_id": "j2", "status": "queued"}, seen))
    job = client.screenshot_job(ScreenshotRequest(urls=[ScreenshotURL(url="https://d.example")]))
    assert seen[-1].url.path == "/qai/v1/jobs"
    assert sent_body(seen[-1]) == {
        "type": "screenshot",
        "params": {"urls": [{"url": "https://d.example"}]},
    }
    assert job.job_id == "j2"


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


def test_compute_billing_posts_the_window() -> None:
    seen: list[httpx.Request] = []
    client = make_client(capture(200, {"entries": [], "total_usd": 0.0}, seen))

    client.compute_billing(instance_id="i1", start_date="2026-07-01")

    assert seen[0].url.path == "/qai/v1/compute/billing"
    assert sent_body(seen[0]) == {"instance_id": "i1", "start_date": "2026-07-01"}


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
