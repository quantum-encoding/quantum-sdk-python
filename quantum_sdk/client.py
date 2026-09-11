"""Quantum AI API client — sync and async."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterator, AsyncIterator

import httpx

from urllib.parse import quote

from .errors import APIError, _typed_error_for
from . import auth as _auth
from . import batch as _batch
from . import credits as _credits
from .auth import AuthResponse
from .batch import (
    BatchJobInfo,
    BatchJobInput,
    BatchJobsResponse,
    BatchJsonlResponse,
    BatchSubmitResponse,
)
from .credits import (
    CreditBalanceResponse,
    CreditPacksResponse,
    CreditPurchaseResponse,
    CreditTiersResponse,
    DevProgramApplyResponse,
)
from .missions import (
    MissionApproveRequest,
    MissionCheckpointsResponse,
    MissionChatRequest,
    MissionChatResponse,
    MissionCreateRequest,
    MissionCreateResponse,
    MissionDetail,
    MissionImportRequest,
    MissionListResponse,
    MissionPlanUpdate,
    MissionStatusResponse,
)
from .security import (
    SecurityBlocklistResponse,
    SecurityCheckResponse,
    SecurityReportRequest,
    SecurityReportResponse,
    SecurityScanHtmlRequest,
    SecurityScanResponse,
    SecurityScanUrlRequest,
)
from .vision import VisionRequest, VisionResponse
from .types_ext import (
    AddVoiceFromLibraryResponse,
    AgentStreamEvent,
    AvatarRealtimeRequest,
    AvatarRealtimeCreateResponse,
    AvatarRealtimeStatusResponse,
    AvatarRealtimeTextResponse,
    AvatarRealtimeCancelResponse,
    AudioSoundsResponse,
    BillingResponse,
    Collection,
    CollectionDocument,
    CollectionDocumentsResponse,
    CollectionSearchRequest,
    CollectionSearchResponse,
    CollectionSearchResult,
    CollectionsListResponse,
    CollectionUploadResult,
    ElevenMusicRequest,
    ElevenMusicResponse,
    GoogleSearchRequest,
    GoogleSearchResponse,
    JobStreamEvent,
    MusicFinetuneInfo,
    MusicFinetuneListResponse,
    ScrapeRequest,
    ScrapeResponse,
    ScreenshotRequest,
    ScreenshotResponse,
    SharedVoicesResponse,
    VoiceLibraryQuery,
    VideoTemplateDetailResponse,
    VideoTemplateGenerateRequest,
    VideoBatchSubmitRequest,
    VideoBatchSubmitResponse,
    VideoBatchStatusResponse,
    JobAcceptedResponse,
)
from .types import (
    ChatRequest,
    ChatResponse,
    ChatUsage,
    ContentBlock,
    EstimateResponse,
    StreamDelta,
    StreamEvent,
    StreamToolUse,
    StreamToolUseComplete,
    StreamToolUseInputDelta,
    StreamToolUseStart,
    ImageRequest,
    ImageResponse,
    ImageEditRequest,
    ImageEditResponse,
    VideoRequest,
    VideoResponse,
    TTSRequest,
    TTSResponse,
    STTRequest,
    STTResponse,
    MusicRequest,
    MusicResponse,
    SoundEffectResponse,
    EmbedRequest,
    EmbedResponse,
    DocumentRequest,
    DocumentResponse,
    RAGSearchRequest,
    RAGSearchResponse,
    RAGCorpus,
    SurrealRAGSearchRequest,
    SurrealRAGSearchResponse,
    SurrealRAGProvidersResponse,
    ModelInfo,
    PricingInfo,
    BalanceResponse,
    UsageEntry,
    UsageResponse,
    UsageSummaryResponse,
    PricingEntry,
    PricingResponse,
    JobCreateResponse,
    JobStatusResponse,
    JobListResponse,
    SessionChatRequest,
    SessionChatResponse,
    ContextConfig,
    AgentRunRequest,
    AgentWorker,
    MissionRunRequest,
    MissionWorker,
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    APIKeyListResponse,
    APIKeyInfo,
    ComputeTemplate,
    ComputeProvisionRequest,
    ComputeProvisionResponse,
    ComputeInstance,
    VoiceListResponse,
    VoiceCloneResponse,
    VoiceInfo,
    DialogueRequest,
    DialogueVoice,
    AudioResponse,
    AlignmentResponse,
    VoiceDesignResponse,
    VideoStudioRequest,
    VideoTranslateRequest,
    VideoPhotoAvatarRequest,
    VideoDigitalTwinRequest,
    HeyGenAvatar,
    HeyGenVoice,
    HeyGenTemplate,
    ChunkDocumentRequest,
    ChunkDocumentResponse,
    ProcessDocumentRequest,
    ProcessDocumentResponse,
    ContactRequest,
    ContactResponse,
    ChatTool,
)

DEFAULT_BASE_URL = "https://api.quantumencoding.ai"
# A bare 60s applied to all httpx phases and aborted long buffered media
# generation (image/video return a single JSON blob only when the provider
# finishes — no bytes flow during generation). Use a generous read/write
# (600s, above the backend's 5-minute media deadline so the server errors
# first) with a short connect. Streaming paths use stream_timeout instead.
DEFAULT_TIMEOUT = httpx.Timeout(600.0, connect=15.0)
# Default timeout for streaming (SSE) connections. Long enough for a full
# mission orchestration or a slow provider stream, but finite so a dropped
# connection does not hang the client forever.
DEFAULT_STREAM_TIMEOUT = 600.0

# httpx multipart files: a field->file mapping, or a list of (field, file)
# pairs when the same field carries several parts (audio finetune samples).
_MultipartFiles = (
    dict[str, tuple[str, bytes, str]] | list[tuple[str, tuple[str, bytes, str]]]
)


class _ResponseMeta:
    """Parsed response metadata from HTTP headers."""

    __slots__ = ("cost_ticks", "request_id", "model", "balance_after")

    def __init__(self, headers: httpx.Headers) -> None:
        self.request_id = headers.get("x-qai-request-id", "")
        self.model = headers.get("x-qai-model", "")
        raw_ticks = headers.get("x-qai-cost-ticks", "")
        self.cost_ticks = int(raw_ticks) if raw_ticks else 0
        raw_balance = headers.get("x-qai-balance-after", "")
        self.balance_after = int(raw_balance) if raw_balance else None


def _parse_api_error(response: httpx.Response, request_id: str) -> APIError:
    """Parse an error response body into a typed APIError subclass when possible."""
    body = response.text
    code = str(response.status_code)
    message = body

    # /qai/v1/agent uses a flat error shape: {"error": <code>, "message": <msg>}.
    try:
        data = response.json()
        if isinstance(data, dict):
            if "error" in data and isinstance(data["error"], str):
                code = data["error"] or code
                if data.get("message"):
                    message = data["message"]
            else:
                err_obj = data.get("error", {})
                if isinstance(err_obj, dict):
                    if err_obj.get("message"):
                        message = err_obj["message"]
                    code = err_obj.get("code") or err_obj.get("type") or code
    except Exception:
        pass

    return _typed_error_for(
        status_code=response.status_code,
        code=code,
        message=message,
        request_id=request_id or None,
    )


def _parse_sse_event(payload: str) -> StreamEvent:
    """Parse a single SSE JSON payload into a StreamEvent (chat events only)."""
    raw: dict[str, Any] = json.loads(payload)
    event_type: str = raw.get("type", "")
    ev = StreamEvent(type=event_type)

    if event_type in ("content_delta", "thinking_delta"):
        delta_data = raw.get("delta")
        if isinstance(delta_data, dict):
            ev.delta = StreamDelta(text=delta_data.get("text", ""))
    elif event_type == "tool_use":
        # Legacy atomic event — kept for backends that have not shipped the
        # start/input_delta/complete triplet (v0.7+). Gemini rides its
        # signature here; the client echoes it on the tool_use block of the
        # next turn.
        ev.thought_signature = raw.get("thought_signature")
        ev.tool_use = StreamToolUse(
            id=raw.get("id", ""),
            name=raw.get("name", ""),
            input=raw.get("input", {}),
        )
    elif event_type == "tool_use_start":
        ev.tool_use_start = StreamToolUseStart(
            id=raw.get("id", ""),
            name=raw.get("name", ""),
        )
    elif event_type == "tool_use_input_delta":
        ev.tool_use_input_delta = StreamToolUseInputDelta(
            id=raw.get("id", ""),
            partial_json=raw.get("partial_json", ""),
        )
    elif event_type == "tool_use_complete":
        ev.tool_use_complete = StreamToolUseComplete(
            id=raw.get("id", ""),
            name=raw.get("name", ""),
            input=raw.get("input", {}),
        )
    elif event_type == "usage":
        ev.usage = ChatUsage(
            input_tokens=raw.get("input_tokens", 0),
            output_tokens=raw.get("output_tokens", 0),
            cached_tokens=raw.get("cached_tokens", 0),
            reasoning_tokens=raw.get("reasoning_tokens", 0),
            cost_ticks=raw.get("cost_ticks", 0),
        )
    elif event_type == "thought_signature":
        # Gemini 3 signs a turn that ended in TEXT; the gateway sends this
        # just before "done".
        ev.thought_signature = raw.get("thought_signature")
    elif event_type == "error":
        ev.error = raw.get("message", "")

    return ev


def _iter_sse_lines(lines_iter: Iterator[str]) -> Iterator[StreamEvent]:
    """Shared SSE line parser for sync iterators (chat events)."""
    for line in lines_iter:
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            yield StreamEvent(type="done", done=True)
            return
        try:
            yield _parse_sse_event(payload)
        except json.JSONDecodeError as exc:
            yield StreamEvent(type="error", error=f"parse SSE: {exc}")
            return


def _iter_job_sse_lines(lines_iter: Iterator[str]) -> Iterator[JobStreamEvent]:
    """Shared SSE line parser for job progress streams."""
    for line in lines_iter:
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            return
        try:
            event = JobStreamEvent.from_dict(json.loads(payload))
        except json.JSONDecodeError as exc:
            yield JobStreamEvent(type="error", error=f"parse SSE: {exc}")
            return

        yield event

        if event.type in ("complete", "error"):
            return


def _voice_library_params(query: VoiceLibraryQuery | None) -> str:
    """Encode a voice-library filter set as a query string (empty when unset)."""
    if query is None:
        return ""
    pairs = [
        ("query", query.query),
        ("page_size", query.page_size),
        ("cursor", query.cursor),
        ("gender", query.gender),
        ("language", query.language),
        ("use_case", query.use_case),
    ]
    return "&".join(
        f"{k}={quote(str(v), safe='')}" for k, v in pairs if v is not None and v != ""
    )


# Mission/agent lifecycle events (mission_started, task_started, wave_completed,
# mission_budget_exhausted, step_detail, mission_completed, mission_failed, ...)
# carry arbitrary payloads that the chat-only StreamEvent parser would drop.
# These iterators yield AgentStreamEvent preserving the full data dict.
def _iter_mission_sse_lines(lines_iter: Iterator[str]) -> Iterator[AgentStreamEvent]:
    for line in lines_iter:
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            yield AgentStreamEvent(event_type="done", data={"done": True})
            return
        try:
            raw = json.loads(payload)
            yield AgentStreamEvent.from_dict(raw)
        except json.JSONDecodeError as exc:
            yield AgentStreamEvent(event_type="error", data={"error": f"parse SSE: {exc}"})
            return


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class Client:
    """Synchronous Quantum AI API client."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        stream_timeout: float = DEFAULT_STREAM_TIMEOUT,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owns_client = http_client is None
        self.stream_timeout = stream_timeout

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _resolve_idempotency_key(idempotency_key: str | None) -> str | None:
        """Resolve an idempotency-key argument to a concrete header value.

        None / "" → omit the header. ``"auto"`` → generate a fresh UUID.
        Any other string → used verbatim.
        """
        if not idempotency_key:
            return None
        if idempotency_key == "auto":
            return str(uuid.uuid4())
        return idempotency_key

    def _do_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> tuple[Any, _ResponseMeta]:
        url = self.base_url + path
        headers = self._headers()
        if body is not None:
            headers["Content-Type"] = "application/json"
        ikey = self._resolve_idempotency_key(idempotency_key)
        if ikey is not None:
            headers["Idempotency-Key"] = ikey

        resp = self._client.request(
            method,
            url,
            headers=headers,
            content=json.dumps(body).encode() if body is not None else None,
        )
        meta = _ResponseMeta(resp.headers)

        if resp.status_code < 200 or resp.status_code >= 300:
            raise _parse_api_error(resp, meta.request_id)

        data = resp.json() if resp.content else None
        return data, meta

    def _do_json_no_auth(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[Any, _ResponseMeta]:
        """Like _do_json but without auth headers."""
        url = self.base_url + path
        headers: dict[str, str] = {}
        if body is not None:
            headers["Content-Type"] = "application/json"

        resp = self._client.request(
            method,
            url,
            headers=headers,
            content=json.dumps(body).encode() if body is not None else None,
        )
        meta = _ResponseMeta(resp.headers)

        if resp.status_code < 200 or resp.status_code >= 300:
            raise _parse_api_error(resp, meta.request_id)

        data = resp.json() if resp.content else None
        return data, meta

    def _do_multipart(
        self,
        path: str,
        fields: dict[str, Any],
        files: _MultipartFiles,
    ) -> tuple[Any, _ResponseMeta]:
        """POST multipart/form-data request."""
        url = self.base_url + path
        headers = self._headers()

        resp = self._client.post(
            url,
            headers=headers,
            data=fields,
            files=files,
        )
        meta = _ResponseMeta(resp.headers)

        if resp.status_code < 200 or resp.status_code >= 300:
            raise _parse_api_error(resp, meta.request_id)

        data = resp.json() if resp.content else None
        return data, meta

    def _stream_sse(
        self,
        path: str,
        body: dict[str, Any],
    ) -> Iterator[StreamEvent]:
        """POST and stream chat SSE events."""
        url = self.base_url + path
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"

        with httpx.Client(timeout=httpx.Timeout(self.stream_timeout, connect=15.0)) as stream_client:
            with stream_client.stream(
                "POST",
                url,
                headers=headers,
                content=json.dumps(body).encode(),
            ) as resp:
                if resp.status_code < 200 or resp.status_code >= 300:
                    resp.read()
                    meta = _ResponseMeta(resp.headers)
                    raise _parse_api_error(resp, meta.request_id)

                yield from _iter_sse_lines(resp.iter_lines())

    def _stream_mission_sse(
        self,
        path: str,
        body: dict[str, Any],
    ) -> Iterator[AgentStreamEvent]:
        """POST and stream mission/agent SSE events as AgentStreamEvent.

        Unlike _stream_sse (which parses chat events and drops mission
        lifecycle payloads), this preserves the full event dict so callers
        see mission_started / task_started / wave_completed / step_detail /
        mission_budget_exhausted / mission_completed / mission_failed / etc.
        """
        url = self.base_url + path
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"

        with httpx.Client(timeout=httpx.Timeout(self.stream_timeout, connect=15.0)) as stream_client:
            with stream_client.stream(
                "POST",
                url,
                headers=headers,
                content=json.dumps(body).encode(),
            ) as resp:
                if resp.status_code < 200 or resp.status_code >= 300:
                    resp.read()
                    meta = _ResponseMeta(resp.headers)
                    raise _parse_api_error(resp, meta.request_id)

                yield from _iter_mission_sse_lines(resp.iter_lines())

    # -- Chat ---------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any] | ChatRequest] | None = None,
        *,
        request: ChatRequest | None = None,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat request.

        Can be called with keyword convenience args:
            client.chat(model="...", messages=[{"role": "user", "content": "Hi"}])

        Or with a full ChatRequest:
            client.chat(request=ChatRequest(model="...", messages=[...]))

        ``idempotency_key`` sets the Idempotency-Key header; pass ``"auto"``
        to generate a fresh UUID for each call.
        """
        req = self._build_chat_request(model, messages, request, **kwargs)
        req.stream = False
        data, meta = self._do_json("POST", "/qai/v1/chat", req.to_dict(), idempotency_key=idempotency_key)
        resp = ChatResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if not resp.model:
            resp.model = meta.model
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any] | ChatRequest] | None = None,
        *,
        request: ChatRequest | None = None,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        """Send a streaming chat request. Yields StreamEvent objects."""
        req = self._build_chat_request(model, messages, request, **kwargs)
        req.stream = True

        url = self.base_url + "/qai/v1/chat"
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"
        ikey = self._resolve_idempotency_key(idempotency_key)
        if ikey is not None:
            headers["Idempotency-Key"] = ikey

        with httpx.Client(timeout=httpx.Timeout(self.stream_timeout, connect=15.0)) as stream_client:
            with stream_client.stream(
                "POST",
                url,
                headers=headers,
                content=json.dumps(req.to_dict()).encode(),
            ) as resp:
                if resp.status_code < 200 or resp.status_code >= 300:
                    resp.read()
                    meta = _ResponseMeta(resp.headers)
                    raise _parse_api_error(resp, meta.request_id)

                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        yield StreamEvent(type="done", done=True)
                        return
                    try:
                        yield _parse_sse_event(payload)
                    except json.JSONDecodeError as exc:
                        yield StreamEvent(type="error", error=f"parse SSE: {exc}")
                        return

    @staticmethod
    def _build_chat_request(
        model: str,
        messages: list[dict[str, Any] | ChatRequest] | None,
        request: ChatRequest | None,
        **kwargs: Any,
    ) -> ChatRequest:
        if request is not None:
            return request
        from .types import ChatMessage
        msg_list: list[ChatMessage] = []
        for m in (messages or []):
            if isinstance(m, dict):
                msg_list.append(ChatMessage(
                    role=m["role"],
                    content=m.get("content", ""),
                    tool_call_id=m.get("tool_call_id"),
                    is_error=m.get("is_error", False),
                ))
            elif isinstance(m, ChatMessage):
                msg_list.append(m)
            else:
                msg_list.append(m)  # type: ignore[arg-type]
        return ChatRequest(model=model, messages=msg_list, **kwargs)

    # -- Session Chat -------------------------------------------------------

    def chat_session(
        self,
        message: str,
        *,
        model: str | None = None,
        session_id: str | None = None,
        tools: list[ChatTool] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        stream: bool = False,
        system_prompt: str | None = None,
        context_config: ContextConfig | None = None,
        reasoning_effort: str | None = None,
        idempotency_key: str | None = None,
    ) -> SessionChatResponse | Iterator[StreamEvent]:
        """Send a session-based chat request.

        Returns SessionChatResponse for non-streaming, or an Iterator of
        StreamEvent for streaming.
        """
        req = SessionChatRequest(
            message=message,
            model=model,
            session_id=session_id,
            tools=tools,
            tool_results=tool_results,
            stream=stream,
            system_prompt=system_prompt,
            context_config=context_config,
            reasoning_effort=reasoning_effort,
        )
        if stream:
            return self._stream_sse("/qai/v1/chat/session", req.to_dict())
        data, meta = self._do_json("POST", "/qai/v1/chat/session", req.to_dict(), idempotency_key=idempotency_key)
        resp = SessionChatResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    # -- Agent / Missions ---------------------------------------------------
    #
    # agent_run is orchestration-shaped (conductor + workers + max_steps),
    # so it retargets to POST /qai/v1/missions with the real MissionRequest
    # wire shape: workers is a dict[str, MissionWorker], task is serialized
    # as ``goal``. The endpoint streams SSE; mission lifecycle events are
    # yielded as AgentStreamEvent (full payload preserved), not dropped.

    def agent_run(
        self,
        task: str,
        *,
        conductor_model: str | None = None,
        conductor_tier: str | None = None,
        workers: dict[str, MissionWorker] | None = None,
        max_steps: int | None = None,
        system_prompt: str | None = None,
        session_id: str | None = None,
        strategy: str | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Run an orchestrated agent task via /qai/v1/missions.

        ``workers`` must be a dict mapping worker name → MissionWorker
        (matching the backend's map[string]MissionWorkerConfig); a list
        is rejected by the server with 400. Yields AgentStreamEvent so
        mission lifecycle events keep their full payload.
        """
        req = AgentRunRequest(
            task=task,
            conductor_model=conductor_model,
            conductor_tier=conductor_tier,
            workers=workers,
            max_steps=max_steps,
            system_prompt=system_prompt,
            session_id=session_id,
            strategy=strategy,
        )
        return self._stream_mission_sse("/qai/v1/missions", req.to_dict())

    def mission_run(
        self,
        goal: str,
        *,
        strategy: str | None = None,
        conductor_model: str | None = None,
        conductor_tier: str | None = None,
        workers: dict[str, MissionWorker] | None = None,
        max_steps: int | None = None,
        system_prompt: str | None = None,
        session_id: str | None = None,
        auto_plan: bool | None = None,
        context_config: ContextConfig | None = None,
        deployment_id: str | None = None,
        build_command: str | None = None,
        workspace_path: str | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Run a mission via /qai/v1/missions.

        ``workers`` is a dict mapping worker name → MissionWorker. Yields
        AgentStreamEvent so mission lifecycle events keep their full payload.
        """
        req = MissionRunRequest(
            goal=goal,
            strategy=strategy,
            conductor_model=conductor_model,
            conductor_tier=conductor_tier,
            workers=workers,
            max_steps=max_steps,
            system_prompt=system_prompt,
            session_id=session_id,
            auto_plan=auto_plan,
            context_config=context_config,
            deployment_id=deployment_id,
            build_command=build_command,
            workspace_path=workspace_path,
        )
        return self._stream_mission_sse("/qai/v1/missions", req.to_dict())

    # -- API Keys -----------------------------------------------------------

    def create_key(
        self,
        name: str,
        *,
        endpoints: list[str] | None = None,
        spend_cap_usd: float | None = None,
        rate_limit: int | None = None,
    ) -> APIKeyCreateResponse:
        """Create a new API key."""
        req = APIKeyCreateRequest(
            name=name,
            endpoints=endpoints,
            spend_cap_usd=spend_cap_usd,
            rate_limit=rate_limit,
        )
        data, _ = self._do_json("POST", "/qai/v1/keys", req.to_dict())
        return APIKeyCreateResponse.from_dict(data)

    def list_keys(self) -> APIKeyListResponse:
        """List all API keys for the account."""
        data, _ = self._do_json("GET", "/qai/v1/keys")
        return APIKeyListResponse.from_dict(data)

    def revoke_key(self, key_id: str) -> None:
        """Revoke an API key."""
        self._do_json("DELETE", f"/qai/v1/keys/{key_id}")

    # -- Compute ------------------------------------------------------------

    def compute_templates(self) -> list[ComputeTemplate]:
        """List available compute templates."""
        data, _ = self._do_json("GET", "/qai/v1/compute/templates")
        return [
            ComputeTemplate.from_dict(t)
            for t in data.get("templates", [])
        ]

    def compute_provision(
        self,
        template: str,
        *,
        zone: str | None = None,
        spot: bool = False,
        auto_teardown_minutes: int = 30,
        ssh_public_key: str | None = None,
    ) -> ComputeProvisionResponse:
        """Provision a new compute instance."""
        req = ComputeProvisionRequest(
            template=template,
            zone=zone,
            spot=spot,
            auto_teardown_minutes=auto_teardown_minutes,
            ssh_public_key=ssh_public_key,
        )
        data, _ = self._do_json("POST", "/qai/v1/compute/provision", req.to_dict())
        return ComputeProvisionResponse.from_dict(data)

    def compute_instances(self) -> list[ComputeInstance]:
        """List all compute instances."""
        data, _ = self._do_json("GET", "/qai/v1/compute/instances")
        return [
            ComputeInstance.from_dict(i)
            for i in data.get("instances", [])
        ]

    def compute_instance(self, instance_id: str) -> ComputeInstance:
        """Get a specific compute instance."""
        data, _ = self._do_json("GET", f"/qai/v1/compute/instance/{instance_id}")
        return ComputeInstance.from_dict(data)

    def compute_delete(self, instance_id: str) -> None:
        """Delete a compute instance."""
        self._do_json("DELETE", f"/qai/v1/compute/instance/{instance_id}")

    def compute_ssh_key(
        self,
        instance_id: str,
        public_key: str,
        username: str = "cosmic",
    ) -> None:
        """Add an SSH key to a compute instance."""
        body: dict[str, Any] = {"public_key": public_key, "username": username}
        self._do_json("POST", f"/qai/v1/compute/instance/{instance_id}/ssh-key", body)

    def compute_keepalive(self, instance_id: str) -> None:
        """Send a keepalive to extend a compute instance's auto-teardown timer."""
        self._do_json("POST", f"/qai/v1/compute/instance/{instance_id}/keepalive")

    # -- Voices -------------------------------------------------------------

    def list_voices(self) -> VoiceListResponse:
        """List all available voices."""
        data, _ = self._do_json("GET", "/qai/v1/voices")
        return VoiceListResponse.from_dict(data)

    def clone_voice(
        self,
        name: str,
        audio_file: bytes,
        *,
        filename: str = "voice.mp3",
        content_type: str = "audio/mpeg",
    ) -> VoiceCloneResponse:
        """Clone a voice from an audio sample (multipart upload)."""
        data, _ = self._do_multipart(
            "/qai/v1/voices/clone",
            fields={"name": name},
            files={"audio_file": (filename, audio_file, content_type)},
        )
        return VoiceCloneResponse.from_dict(data)

    def delete_voice(self, voice_id: str) -> None:
        """Delete a cloned voice."""
        self._do_json("DELETE", f"/qai/v1/voices/{voice_id}")

    # -- Image --------------------------------------------------------------

    def generate_image(
        self,
        request: ImageRequest,
        *,
        idempotency_key: str | None = None,
    ) -> ImageResponse:
        """Generate images from a text prompt."""
        data, meta = self._do_json("POST", "/qai/v1/images/generate", request.to_dict(), idempotency_key=idempotency_key)
        resp = ImageResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    def edit_image(
        self,
        request: ImageEditRequest,
        *,
        idempotency_key: str | None = None,
    ) -> ImageEditResponse:
        """Edit images using an AI model."""
        data, meta = self._do_json("POST", "/qai/v1/images/edit", request.to_dict(), idempotency_key=idempotency_key)
        resp = ImageEditResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    # -- Video --------------------------------------------------------------

    def generate_video(
        self,
        request: VideoRequest,
        *,
        idempotency_key: str | None = None,
    ) -> VideoResponse:
        """Generate a video from a text prompt."""
        data, meta = self._do_json("POST", "/qai/v1/video/generate", request.to_dict(), idempotency_key=idempotency_key)
        resp = VideoResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    def video_studio(
        self,
        avatar_id: str,
        script: str,
        voice_id: str,
    ) -> JobStatusResponse:
        """Create a HeyGen studio video."""
        req = VideoStudioRequest(avatar_id=avatar_id, script=script, voice_id=voice_id)
        data, meta = self._do_json("POST", "/qai/v1/video/studio", req.to_dict())
        return JobStatusResponse.from_dict(data)

    def video_translate(
        self,
        video_url: str,
        target_language: str,
    ) -> JobStatusResponse:
        """Translate a video to another language."""
        req = VideoTranslateRequest(video_url=video_url, target_language=target_language)
        data, meta = self._do_json("POST", "/qai/v1/video/translate", req.to_dict())
        return JobStatusResponse.from_dict(data)

    def video_photo_avatar(
        self,
        photo_url: str,
        script: str,
    ) -> JobStatusResponse:
        """Generate a video from a photo avatar."""
        req = VideoPhotoAvatarRequest(photo_url=photo_url, script=script)
        data, meta = self._do_json("POST", "/qai/v1/video/photo-avatar", req.to_dict())
        return JobStatusResponse.from_dict(data)

    def video_digital_twin(
        self,
        avatar_id: str,
        script: str,
    ) -> JobStatusResponse:
        """Generate a video with a digital twin."""
        req = VideoDigitalTwinRequest(avatar_id=avatar_id, script=script)
        data, meta = self._do_json("POST", "/qai/v1/video/digital-twin", req.to_dict())
        return JobStatusResponse.from_dict(data)

    def video_avatars(self) -> list[HeyGenAvatar]:
        """List available HeyGen avatars."""
        data, _ = self._do_json("GET", "/qai/v1/video/avatars")
        return [HeyGenAvatar.from_dict(a) for a in data.get("avatars", [])]

    def video_templates(self) -> list[HeyGenTemplate]:
        """List available HeyGen video templates."""
        data, _ = self._do_json("GET", "/qai/v1/video/templates")
        return [HeyGenTemplate.from_dict(t) for t in data.get("templates", [])]

    def video_heygen_voices(self) -> list[HeyGenVoice]:
        """List available HeyGen voices."""
        data, _ = self._do_json("GET", "/qai/v1/video/heygen-voices")
        return [HeyGenVoice.from_dict(v) for v in data.get("voices", [])]

    def video_template_detail(self, template_id: str) -> VideoTemplateDetailResponse:
        """Inspect a HeyGen template's variable schema and scenes (unbilled).

        Only draft-v4 templates with variables are supported upstream; an
        unknown template id surfaces as a provider_error.
        """
        data, _ = self._do_json("GET", f"/qai/v1/video/template/{template_id}")
        return VideoTemplateDetailResponse.from_dict(data)

    def video_template_generate(
        self,
        template_id: str,
        request: VideoTemplateGenerateRequest,
    ) -> JobAcceptedResponse:
        """Render a video from a HeyGen template (async job "video/template-v3").

        Returns the accepted-job envelope — poll with get_job / poll_job (or
        SSE via stream_job) until "completed"/"failed", then read
        result["video_url"]. Deep validation happens at execution time, so
        violations surface as a failed job rather than a 4xx at submit.
        """
        data, _ = self._do_json("POST", f"/qai/v1/video/template/{template_id}", request.to_dict())
        return JobAcceptedResponse.from_dict(data)

    def video_batch_submit(self, request: VideoBatchSubmitRequest) -> VideoBatchSubmitResponse:
        """Submit 1-100 raw HeyGen video payloads as one batch (202 Accepted).

        Poll video_batch_status for progress and delivery.
        """
        data, _ = self._do_json("POST", "/qai/v1/video/batch", request.to_dict())
        return VideoBatchSubmitResponse.from_dict(data)

    def video_batch_status(
        self,
        batch_id: str,
        *,
        limit: int | None = None,
        token: str | None = None,
    ) -> VideoBatchStatusResponse:
        """Get a batch's status plus one cursor-paginated page of items.

        Poll (~5s) until ``status`` is terminal, then keep polling until
        ``billing_status == "settled"`` — per-item ``video_url`` values are
        withheld until settlement. Collect URLs across pages via ``next_token``.
        """
        params: list[str] = []
        if limit is not None:
            params.append(f"limit={limit}")
        if token is not None:
            params.append(f"token={quote(token, safe='')}")
        path = f"/qai/v1/video/batch/{batch_id}"
        if params:
            path += "?" + "&".join(params)
        data, _ = self._do_json("GET", path)
        return VideoBatchStatusResponse.from_dict(data)

    # -- Avatar Realtime (HeyGen Broadcast) ---------------------------------

    def create_avatar_realtime_session(
        self,
        request: AvatarRealtimeRequest,
    ) -> AvatarRealtimeCreateResponse:
        """Create a live avatar realtime session (HeyGen Broadcast).

        PREPAID: the entire ``max_duration_seconds`` block (1-3600 s) is
        charged at create time; cancelling early does NOT refund.
        """
        data, meta = self._do_json("POST", "/qai/v1/avatar/realtime", request.to_dict())
        resp = AvatarRealtimeCreateResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    def get_avatar_realtime_session(self, stream_id: str) -> AvatarRealtimeStatusResponse:
        """Get the live status of an avatar realtime session.

        Poll (~2s) until ``status == "streaming"``, then play ``hls_url``.
        "completed" and "error" are terminal.
        """
        data, _ = self._do_json("GET", f"/qai/v1/avatar/realtime/{stream_id}")
        return AvatarRealtimeStatusResponse.from_dict(data)

    def send_avatar_realtime_text(
        self,
        stream_id: str,
        delta: str = "",
        *,
        final: bool = False,
    ) -> AvatarRealtimeTextResponse:
        """Append a text delta to a ``text_stream`` session.

        ``delta`` is required unless ``final=True`` (which closes the input;
        appending afterwards fails upstream with a 410 provider_error).
        """
        body: dict[str, Any] = {}
        if delta:
            body["delta"] = delta
        body["final"] = final
        data, _ = self._do_json("POST", f"/qai/v1/avatar/realtime/{stream_id}/text", body)
        return AvatarRealtimeTextResponse.from_dict(data)

    def cancel_avatar_realtime_session(self, stream_id: str) -> AvatarRealtimeCancelResponse:
        """Terminate an avatar realtime session early (idempotent; no refund —
        this only stops HeyGen's upstream meter)."""
        data, _ = self._do_json("POST", f"/qai/v1/avatar/realtime/{stream_id}/cancel")
        return AvatarRealtimeCancelResponse.from_dict(data)

    # -- Audio --------------------------------------------------------------

    def speak(
        self,
        request: TTSRequest,
        *,
        idempotency_key: str | None = None,
    ) -> TTSResponse:
        """Generate speech from text."""
        data, meta = self._do_json("POST", "/qai/v1/audio/tts", request.to_dict(), idempotency_key=idempotency_key)
        resp = TTSResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    def transcribe(
        self,
        request: STTRequest,
        *,
        idempotency_key: str | None = None,
    ) -> STTResponse:
        """Convert speech to text."""
        data, meta = self._do_json("POST", "/qai/v1/audio/stt", request.to_dict(), idempotency_key=idempotency_key)
        resp = STTResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    def generate_music(
        self,
        request: MusicRequest,
        *,
        idempotency_key: str | None = None,
    ) -> MusicResponse:
        """Generate music from a text prompt."""
        data, meta = self._do_json("POST", "/qai/v1/audio/music", request.to_dict(), idempotency_key=idempotency_key)
        resp = MusicResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    def sound_effects(
        self,
        prompt: str,
        duration_seconds: float | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> SoundEffectResponse:
        """Generate sound effects from a text prompt."""
        body: dict[str, Any] = {"prompt": prompt}
        if duration_seconds is not None:
            body["duration_seconds"] = duration_seconds
        data, meta = self._do_json("POST", "/qai/v1/audio/sound-effects", body, idempotency_key=idempotency_key)
        resp = SoundEffectResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    def dialogue(
        self,
        text: str,
        voices: list[DialogueVoice],
        *,
        model: str | None = None,
        idempotency_key: str | None = None,
    ) -> AudioResponse:
        """Generate multi-voice dialogue audio."""
        req = DialogueRequest(text=text, voices=voices, model=model)
        data, meta = self._do_json("POST", "/qai/v1/audio/dialogue", req.to_dict(), idempotency_key=idempotency_key)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    def speech_to_speech(
        self,
        audio: str,
        voice_id: str,
    ) -> AudioResponse:
        """Convert speech audio to a different voice."""
        body: dict[str, Any] = {"audio": audio, "voice_id": voice_id}
        data, meta = self._do_json("POST", "/qai/v1/audio/speech-to-speech", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def isolate_voice(self, audio: str) -> AudioResponse:
        """Isolate voice from background noise in an audio clip."""
        body: dict[str, Any] = {"audio": audio}
        data, meta = self._do_json("POST", "/qai/v1/audio/isolate", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def remix_voice(self, audio: str, voice_id: str) -> AudioResponse:
        """Remix audio with a different voice."""
        body: dict[str, Any] = {"audio": audio, "voice_id": voice_id}
        data, meta = self._do_json("POST", "/qai/v1/audio/remix", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def dub(
        self,
        audio: str,
        target_lang: str,
        *,
        source_lang: str | None = None,
    ) -> AudioResponse:
        """Dub audio into a target language."""
        body: dict[str, Any] = {"audio": audio, "target_lang": target_lang}
        if source_lang is not None:
            body["source_lang"] = source_lang
        data, meta = self._do_json("POST", "/qai/v1/audio/dub", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def align(self, audio: str, text: str) -> AlignmentResponse:
        """Align audio with text to get word-level timing."""
        body: dict[str, Any] = {"audio": audio, "text": text}
        data, meta = self._do_json("POST", "/qai/v1/audio/align", body)
        resp = AlignmentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def voice_design(self, description: str) -> VoiceDesignResponse:
        """Design a new voice from a text description."""
        body: dict[str, Any] = {"description": description}
        data, meta = self._do_json("POST", "/qai/v1/audio/voice-design", body)
        resp = VoiceDesignResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def starfish_tts(self, text: str, voice_id: str) -> AudioResponse:
        """Generate speech using Starfish TTS engine."""
        body: dict[str, Any] = {"text": text, "voice_id": voice_id}
        data, meta = self._do_json("POST", "/qai/v1/audio/starfish-tts", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def search_audio_sounds(
        self,
        query: str,
        *,
        sound_type: str | None = None,
        limit: int | None = None,
        min_score: float | None = None,
        token: str | None = None,
    ) -> AudioSoundsResponse:
        """Search HeyGen's background-music and sound-effects catalogs
        (semantic ranking, best score first). Unbilled catalog route.

        ``sound_type`` maps to the wire param ``type``: "music" |
        "sound_effects" (API default "music"). ``limit`` is 1-50 (default 10),
        ``min_score`` 0-1 (default 0.7), ``token`` an opaque cursor from a
        previous response's ``next_token``.
        """
        params: list[str] = [f"query={quote(query, safe='')}"]
        if sound_type is not None:
            params.append(f"type={quote(sound_type, safe='')}")
        if limit is not None:
            params.append(f"limit={limit}")
        if min_score is not None:
            params.append(f"min_score={min_score}")
        if token is not None:
            params.append(f"token={quote(token, safe='')}")
        path = "/qai/v1/audio/sounds?" + "&".join(params)
        data, _ = self._do_json("GET", path)
        return AudioSoundsResponse.from_dict(data)

    # -- Embeddings ---------------------------------------------------------

    def embed(
        self,
        request: EmbedRequest,
        *,
        idempotency_key: str | None = None,
    ) -> EmbedResponse:
        """Generate text embeddings."""
        data, meta = self._do_json("POST", "/qai/v1/embeddings", request.to_dict(), idempotency_key=idempotency_key)
        resp = EmbedResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    # -- Documents ----------------------------------------------------------

    def extract_document(
        self,
        request: DocumentRequest,
        *,
        idempotency_key: str | None = None,
    ) -> DocumentResponse:
        """Extract text content from a document (PDF, image, etc.)."""
        data, meta = self._do_json("POST", "/qai/v1/documents/extract", request.to_dict(), idempotency_key=idempotency_key)
        resp = DocumentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def chunk_document(
        self,
        text: str,
        *,
        chunk_size: int | None = None,
    ) -> ChunkDocumentResponse:
        """Chunk a document into smaller pieces."""
        req = ChunkDocumentRequest(text=text, chunk_size=chunk_size)
        data, meta = self._do_json("POST", "/qai/v1/documents/chunk", req.to_dict())
        resp = ChunkDocumentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def process_document(self, text: str) -> ProcessDocumentResponse:
        """Process a document."""
        req = ProcessDocumentRequest(text=text)
        data, meta = self._do_json("POST", "/qai/v1/documents/process", req.to_dict())
        resp = ProcessDocumentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- RAG ----------------------------------------------------------------

    def rag_search(
        self,
        request: RAGSearchRequest,
        *,
        idempotency_key: str | None = None,
    ) -> RAGSearchResponse:
        """Search Vertex AI RAG corpora."""
        data, meta = self._do_json("POST", "/qai/v1/rag/search", request.to_dict(), idempotency_key=idempotency_key)
        resp = RAGSearchResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def rag_corpora(self) -> list[RAGCorpus]:
        """List available Vertex AI RAG corpora."""
        data, _ = self._do_json("GET", "/qai/v1/rag/corpora")
        return [
            RAGCorpus(
                name=c.get("name", ""),
                display_name=c.get("displayName", ""),
                description=c.get("description", ""),
                state=c.get("state", ""),
            )
            for c in data.get("corpora", [])
        ]

    def surreal_rag_search(self, request: SurrealRAGSearchRequest) -> SurrealRAGSearchResponse:
        """Search provider API docs via SurrealDB vector search."""
        data, meta = self._do_json("POST", "/qai/v1/rag/surreal/search", request.to_dict())
        resp = SurrealRAGSearchResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def surreal_rag_providers(self) -> SurrealRAGProvidersResponse:
        """List available SurrealDB RAG providers."""
        data, _ = self._do_json("GET", "/qai/v1/rag/surreal/providers")
        return SurrealRAGProvidersResponse.from_dict(data)

    # -- Models -------------------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        """Return all available models."""
        data, _ = self._do_json("GET", "/qai/v1/models")
        return [
            ModelInfo(
                id=m.get("id", ""),
                provider=m.get("provider", ""),
                display_name=m.get("display_name", ""),
                input_per_million=m.get("input_per_million", 0.0),
                output_per_million=m.get("output_per_million", 0.0),
            )
            for m in data.get("models", [])
        ]

    def get_pricing(self) -> list[PricingInfo]:
        """Return the complete pricing table."""
        data, _ = self._do_json("GET", "/qai/v1/pricing")
        return [
            PricingInfo(
                id=p.get("id", ""),
                provider=p.get("provider", ""),
                display_name=p.get("display_name", ""),
                input_per_million=p.get("input_per_million", 0.0),
                output_per_million=p.get("output_per_million", 0.0),
            )
            for p in data.get("pricing", [])
        ]

    # -- Account ------------------------------------------------------------

    def account_balance(self) -> BalanceResponse:
        """Get the account credit balance."""
        data, _ = self._do_json("GET", "/qai/v1/account/balance")
        return BalanceResponse.from_dict(data)

    def account_usage(
        self,
        limit: int | None = None,
        start_after: str | None = None,
    ) -> UsageResponse:
        """Get paginated usage history."""
        params: list[str] = []
        if limit is not None:
            params.append(f"limit={limit}")
        if start_after is not None:
            params.append(f"start_after={start_after}")
        path = "/qai/v1/account/usage"
        if params:
            path += "?" + "&".join(params)
        data, _ = self._do_json("GET", path)
        return UsageResponse.from_dict(data)

    def account_usage_summary(self, months: int | None = None) -> UsageSummaryResponse:
        """Get monthly usage summary."""
        path = "/qai/v1/account/usage/summary"
        if months is not None:
            path += f"?months={months}"
        data, _ = self._do_json("GET", path)
        return UsageSummaryResponse.from_dict(data)

    def account_pricing(self) -> PricingResponse:
        """Get the full pricing table (model to pricing entry map)."""
        data, _ = self._do_json("GET", "/qai/v1/pricing")
        return PricingResponse.from_dict(data)

    # -- Jobs ---------------------------------------------------------------

    def create_job(
        self,
        job_type: str,
        params: dict[str, Any],
    ) -> JobCreateResponse:
        """Create an async job. Returns the job ID for polling."""
        body: dict[str, Any] = {"type": job_type, "params": params}
        data, _ = self._do_json("POST", "/qai/v1/jobs", body)
        return JobCreateResponse.from_dict(data)

    def get_job(self, job_id: str) -> JobStatusResponse:
        """Check the status of an async job."""
        data, _ = self._do_json("GET", f"/qai/v1/jobs/{job_id}")
        return JobStatusResponse.from_dict(data)

    def list_jobs(self) -> JobListResponse:
        """List all jobs for the account."""
        data, _ = self._do_json("GET", "/qai/v1/jobs")
        return JobListResponse.from_dict(data)

    def poll_job(
        self,
        job_id: str,
        interval: float = 5.0,
        max_attempts: int = 120,
    ) -> JobStatusResponse:
        """Poll a job until completion or timeout."""
        for _ in range(max_attempts):
            time.sleep(interval)
            status = self.get_job(job_id)
            if status.status in ("completed", "failed"):
                return status
        return JobStatusResponse(
            job_id=job_id,
            status="timeout",
            error=f"Job polling timed out after {max_attempts} attempts",
        )

    # -- Contact ------------------------------------------------------------

    def contact(
        self,
        name: str,
        email: str,
        message: str,
    ) -> ContactResponse:
        """Submit a contact form (no auth required)."""
        req = ContactRequest(name=name, email=email, message=message)
        data, _ = self._do_json_no_auth("POST", "/qai/v1/contact", req.to_dict())
        return ContactResponse.from_dict(data)

    # -- Search (Brave) ---------------------------------------------------

    def web_search(self, query: str, *, count: int | None = None, offset: int | None = None,
                   country: str | None = None, language: str | None = None,
                   freshness: str | None = None, safesearch: str | None = None) -> dict:
        """Brave web search."""
        body: dict = {"query": query}
        if count is not None: body["count"] = count
        if offset is not None: body["offset"] = offset
        if country is not None: body["country"] = country
        if language is not None: body["language"] = language
        if freshness is not None: body["freshness"] = freshness
        if safesearch is not None: body["safesearch"] = safesearch
        data, _ = self._do_json("POST", "/qai/v1/search/web", body)
        return data

    def search_context(self, query: str, *, count: int | None = None,
                       country: str | None = None, language: str | None = None,
                       freshness: str | None = None) -> dict:
        """LLM-optimized content chunks for grounding."""
        body: dict = {"query": query}
        if count is not None: body["count"] = count
        if country is not None: body["country"] = country
        if language is not None: body["language"] = language
        if freshness is not None: body["freshness"] = freshness
        data, _ = self._do_json("POST", "/qai/v1/search/context", body)
        return data

    def search_answer(self, messages: list, *, model: str | None = None) -> dict:
        """Grounded AI answer with citations."""
        body: dict = {"messages": messages}
        if model is not None: body["model"] = model
        data, _ = self._do_json("POST", "/qai/v1/search/answer", body)
        return data

    def google_search(self, request: GoogleSearchRequest) -> GoogleSearchResponse:
        """Gemini-grounded Google search — an answer plus the sources behind it.

        Billed per Google query the model decides to run, which makes this
        materially more expensive than search_answer (Brave-backed). Reach for
        it when answer quality matters more than per-call cost. Render
        ``search_entry_point`` verbatim — Google's grounding terms require it.
        """
        data, _ = self._do_json("POST", "/qai/v1/search/google", request.to_dict())
        return GoogleSearchResponse.from_dict(data)

    # -- Missions -----------------------------------------------------------

    def mission_create(self, request: MissionCreateRequest) -> MissionCreateResponse:
        """Create a mission and start executing it asynchronously."""
        data, _ = self._do_json("POST", "/qai/v1/missions/create", request.to_dict())
        return MissionCreateResponse.from_dict(data)

    def mission_list(self, status: str | None = None) -> MissionListResponse:
        """List missions, optionally filtered by status."""
        path = "/qai/v1/missions/list"
        if status is not None:
            path += f"?status={quote(status, safe='')}"
        data, _ = self._do_json("GET", path)
        return MissionListResponse.from_dict(data)

    def mission_get(self, mission_id: str) -> MissionDetail:
        """Get a mission's details, including its tasks."""
        data, _ = self._do_json("GET", f"/qai/v1/missions/{mission_id}")
        return MissionDetail.from_dict(data)

    def mission_delete(self, mission_id: str) -> MissionStatusResponse:
        """Delete a mission."""
        data, _ = self._do_json("DELETE", f"/qai/v1/missions/{mission_id}")
        return MissionStatusResponse.from_dict(data)

    def mission_cancel(self, mission_id: str) -> MissionStatusResponse:
        """Cancel a running mission."""
        data, _ = self._do_json("POST", f"/qai/v1/missions/{mission_id}/cancel", {})
        return MissionStatusResponse.from_dict(data)

    def mission_pause(self, mission_id: str) -> MissionStatusResponse:
        """Pause a running mission."""
        data, _ = self._do_json("POST", f"/qai/v1/missions/{mission_id}/pause", {})
        return MissionStatusResponse.from_dict(data)

    def mission_resume(self, mission_id: str) -> MissionStatusResponse:
        """Resume a paused mission."""
        data, _ = self._do_json("POST", f"/qai/v1/missions/{mission_id}/resume", {})
        return MissionStatusResponse.from_dict(data)

    def mission_chat(self, mission_id: str, request: MissionChatRequest) -> MissionChatResponse:
        """Chat with the mission's architect."""
        data, _ = self._do_json("POST", f"/qai/v1/missions/{mission_id}/chat", request.to_dict())
        return MissionChatResponse.from_dict(data)

    def mission_retry_task(self, mission_id: str, task_id: str) -> MissionStatusResponse:
        """Retry a failed task within a mission."""
        data, _ = self._do_json("POST", f"/qai/v1/missions/{mission_id}/retry/{task_id}", {})
        return MissionStatusResponse.from_dict(data)

    def mission_approve(self, mission_id: str, request: MissionApproveRequest) -> MissionStatusResponse:
        """Approve a completed mission."""
        data, _ = self._do_json("POST", f"/qai/v1/missions/{mission_id}/approve", request.to_dict())
        return MissionStatusResponse.from_dict(data)

    def mission_update_plan(self, mission_id: str, request: MissionPlanUpdate) -> MissionStatusResponse:
        """Replace the mission's plan (tasks, workers, prompt)."""
        data, _ = self._do_json("PUT", f"/qai/v1/missions/{mission_id}/plan", request.to_dict())
        return MissionStatusResponse.from_dict(data)

    def mission_checkpoints(self, mission_id: str) -> MissionCheckpointsResponse:
        """List the git checkpoints a mission has committed."""
        data, _ = self._do_json("GET", f"/qai/v1/missions/{mission_id}/checkpoints")
        return MissionCheckpointsResponse.from_dict(data)

    def mission_import(self, request: MissionImportRequest) -> MissionCreateResponse:
        """Import an existing plan as a new mission."""
        data, _ = self._do_json("POST", "/qai/v1/missions/import", request.to_dict())
        return MissionCreateResponse.from_dict(data)

    # -- Vision -------------------------------------------------------------

    def vision_analyze(self, request: VisionRequest) -> VisionResponse:
        """Combined analysis: scene + objects + quality + OCR + relevance."""
        data, _ = self._do_json("POST", "/qai/v1/vision/analyze", request.to_dict())
        return VisionResponse.from_dict(data)

    def vision_detect(self, request: VisionRequest) -> VisionResponse:
        """Object detection with bounding boxes."""
        data, _ = self._do_json("POST", "/qai/v1/vision/detect", request.to_dict())
        return VisionResponse.from_dict(data)

    def vision_describe(self, request: VisionRequest) -> VisionResponse:
        """Scene description and tags."""
        data, _ = self._do_json("POST", "/qai/v1/vision/describe", request.to_dict())
        return VisionResponse.from_dict(data)

    def vision_ocr(self, request: VisionRequest) -> VisionResponse:
        """Text extraction and overlay metadata (OCR)."""
        data, _ = self._do_json("POST", "/qai/v1/vision/ocr", request.to_dict())
        return VisionResponse.from_dict(data)

    def vision_quality(self, request: VisionRequest) -> VisionResponse:
        """Image quality assessment (blur, exposure, resolution)."""
        data, _ = self._do_json("POST", "/qai/v1/vision/quality", request.to_dict())
        return VisionResponse.from_dict(data)

    # -- Security -----------------------------------------------------------

    def security_scan_url(self, url: str) -> SecurityScanResponse:
        """Fetch a URL and scan the page for prompt injection."""
        req = SecurityScanUrlRequest(url=url)
        data, _ = self._do_json("POST", "/qai/v1/security/scan-url", req.to_dict())
        return SecurityScanResponse.from_dict(data)

    def security_scan_html(self, request: SecurityScanHtmlRequest) -> SecurityScanResponse:
        """Scan already-fetched HTML for prompt injection."""
        data, _ = self._do_json("POST", "/qai/v1/security/scan-html", request.to_dict())
        return SecurityScanResponse.from_dict(data)

    def security_check(self, url: str) -> SecurityCheckResponse:
        """Check a URL against the injection registry without fetching it."""
        data, _ = self._do_json("GET", f"/qai/v1/security/check?url={quote(url, safe='')}")
        return SecurityCheckResponse.from_dict(data)

    def security_blocklist(self, status: str | None = None) -> SecurityBlocklistResponse:
        """Get the injection blocklist feed ("confirmed" or "suspected")."""
        path = "/qai/v1/security/blocklist"
        if status is not None:
            path += f"?status={quote(status, safe='')}"
        data, _ = self._do_json("GET", path)
        return SecurityBlocklistResponse.from_dict(data)

    def security_report(self, request: SecurityReportRequest) -> SecurityReportResponse:
        """Report a suspicious URL to the registry."""
        data, _ = self._do_json("POST", "/qai/v1/security/report", request.to_dict())
        return SecurityReportResponse.from_dict(data)

    # -- Credits ------------------------------------------------------------

    def credit_packs(self) -> CreditPacksResponse:
        """List the credit packs on sale. No authentication required."""
        return _credits.credit_packs_sync(self)

    def credit_purchase(
        self,
        pack_id: str,
        *,
        success_url: str | None = None,
        cancel_url: str | None = None,
    ) -> CreditPurchaseResponse:
        """Start a credit-pack purchase. Returns the checkout URL to open."""
        return _credits.credit_purchase_sync(self, pack_id, success_url, cancel_url)

    def credit_balance(self) -> CreditBalanceResponse:
        """Get the wallet balance in ticks and USD."""
        return _credits.credit_balance_sync(self)

    def credit_tiers(self) -> CreditTiersResponse:
        """List the volume-discount tiers. No authentication required."""
        return _credits.credit_tiers_sync(self)

    def dev_program_apply(
        self,
        use_case: str,
        *,
        company: str | None = None,
        expected_usd: float | None = None,
        website: str | None = None,
    ) -> DevProgramApplyResponse:
        """Apply for the developer program."""
        return _credits.dev_program_apply_sync(self, use_case, company, expected_usd, website)

    # -- Batch --------------------------------------------------------------

    def batch_submit(self, jobs: list[BatchJobInput]) -> BatchSubmitResponse:
        """Submit a batch of prompts. Each runs independently; poll via Jobs."""
        return _batch.batch_submit_sync(self, jobs)

    def batch_submit_jsonl(self, jsonl: str) -> BatchJsonlResponse:
        """Submit a batch as JSONL — one JSON job object per line."""
        return _batch.batch_submit_jsonl_sync(self, jsonl)

    def batch_jobs(self) -> BatchJobsResponse:
        """List the account's batch jobs."""
        return _batch.batch_jobs_sync(self)

    def batch_job(self, job_id: str) -> BatchJobInfo:
        """Get the status and result of a single batch job."""
        return _batch.batch_job_sync(self, job_id)

    # -- Auth ---------------------------------------------------------------

    def auth_apple(self, id_token: str, *, name: str | None = None) -> AuthResponse:
        """Exchange a Sign in with Apple identity token for an API token.

        Pass ``name`` on first sign-in — Apple only sends it once, so the
        account is created without a display name if you drop it.
        """
        return _auth.auth_apple_sync(self, id_token, name)

    # -- RAG collections (user-scoped xAI proxy) ----------------------------

    def collections_list(self) -> list[Collection]:
        """List the user's collections plus the shared ones."""
        data, _ = self._do_json("GET", "/qai/v1/rag/collections")
        return CollectionsListResponse.from_dict(data).collections

    def collections_create(self, name: str) -> Collection:
        """Create a user-owned collection."""
        data, _ = self._do_json("POST", "/qai/v1/rag/collections", {"name": name})
        return Collection(**data)

    def collections_get(self, collection_id: str) -> Collection:
        """Get one collection — must be owned by the caller or shared."""
        data, _ = self._do_json("GET", f"/qai/v1/rag/collections/{collection_id}")
        return Collection(**data)

    def collections_delete(self, collection_id: str) -> str:
        """Delete a collection (owner only). Returns the server's message."""
        data, _ = self._do_json("DELETE", f"/qai/v1/rag/collections/{collection_id}")
        return (data or {}).get("message", "")

    def collections_documents(self, collection_id: str) -> list[CollectionDocument]:
        """List the documents in a collection."""
        data, _ = self._do_json("GET", f"/qai/v1/rag/collections/{collection_id}/documents")
        return CollectionDocumentsResponse.from_dict(data).documents

    def collections_upload(
        self,
        collection_id: str,
        filename: str,
        content: bytes,
    ) -> CollectionUploadResult:
        """Upload a file to a collection.

        The server runs the two-step xAI upload (files API then management API)
        with the master key, so the caller only sends bytes.
        """
        data, _ = self._do_multipart(
            f"/qai/v1/rag/collections/{collection_id}/upload",
            fields={},
            files={"file": (filename, content, "application/octet-stream")},
        )
        return CollectionUploadResult(**data)

    def collections_search(self, request: CollectionSearchRequest) -> list[CollectionSearchResult]:
        """Search across collections (owned + shared), hybrid by default."""
        data, _ = self._do_json("POST", "/qai/v1/rag/search/collections", request.to_dict())
        return CollectionSearchResponse.from_dict(data).results

    # -- Scraper ------------------------------------------------------------

    def scrape(self, request: ScrapeRequest) -> ScrapeResponse:
        """Submit a doc-scraping job. Returns a job ID to poll."""
        data, _ = self._do_json("POST", "/qai/v1/scraper/scrape", request.to_dict())
        return ScrapeResponse.from_dict(data)

    def screenshot(self, request: ScreenshotRequest) -> ScreenshotResponse:
        """Screenshot URLs inline. Use screenshot_job() past ~5 URLs."""
        data, _ = self._do_json("POST", "/qai/v1/scraper/screenshot", request.to_dict())
        return ScreenshotResponse.from_dict(data)

    def screenshot_job(self, request: ScreenshotRequest) -> JobCreateResponse:
        """Submit a large screenshot batch as an async job."""
        return self.create_job("screenshot", request.to_dict())

    # -- Voice library ------------------------------------------------------

    def voice_library(self, query: VoiceLibraryQuery | None = None) -> SharedVoicesResponse:
        """Browse the shared voice library."""
        path = "/qai/v1/voices/library"
        params = _voice_library_params(query)
        if params:
            path += "?" + params
        data, _ = self._do_json("GET", path)
        return SharedVoicesResponse.from_dict(data)

    def add_voice_from_library(
        self,
        public_owner_id: str,
        voice_id: str,
        *,
        name: str | None = None,
    ) -> AddVoiceFromLibraryResponse:
        """Add a shared voice from the library to the account."""
        body: dict[str, Any] = {"public_owner_id": public_owner_id, "voice_id": voice_id}
        if name is not None:
            body["name"] = name
        data, _ = self._do_json("POST", "/qai/v1/voices/library/add", body)
        return AddVoiceFromLibraryResponse.from_dict(data)

    # -- Audio finetunes / advanced music -----------------------------------

    def create_finetune(
        self,
        name: str,
        files: list[tuple[str, bytes, str]],
        *,
        description: str | None = None,
    ) -> MusicFinetuneInfo:
        """Create a music finetune from audio samples.

        ``files`` are ``(filename, data, mime_type)`` triples, all sent under
        the ``files`` form field.
        """
        fields: dict[str, Any] = {"name": name}
        if description is not None:
            fields["description"] = description
        data, _ = self._do_multipart(
            "/qai/v1/audio/finetunes",
            fields=fields,
            files=[("files", f) for f in files],
        )
        return MusicFinetuneInfo.from_dict(data)

    def list_finetunes(self) -> MusicFinetuneListResponse:
        """List the account's music finetunes."""
        data, _ = self._do_json("GET", "/qai/v1/audio/finetunes")
        return MusicFinetuneListResponse.from_dict(data)

    def delete_finetune(self, finetune_id: str) -> None:
        """Delete a music finetune."""
        self._do_json("DELETE", f"/qai/v1/audio/finetunes/{finetune_id}")

    def generate_music_advanced(self, request: ElevenMusicRequest) -> ElevenMusicResponse:
        """Composition-plan music generation (sections, style, finetunes)."""
        data, meta = self._do_json("POST", "/qai/v1/audio/music/advanced", request.to_dict())
        resp = ElevenMusicResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Compute billing ----------------------------------------------------

    def compute_billing(
        self,
        *,
        instance_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> BillingResponse:
        """Query compute billing (BigQuery-backed) for an instance or range."""
        body: dict[str, Any] = {}
        if instance_id is not None:
            body["instance_id"] = instance_id
        if start_date is not None:
            body["start_date"] = start_date
        if end_date is not None:
            body["end_date"] = end_date
        data, _ = self._do_json("POST", "/qai/v1/compute/billing", body)
        return BillingResponse.from_dict(data)

    # -- Jobs (async submission) --------------------------------------------

    def chat_job(self, request: ChatRequest) -> JobCreateResponse:
        """Submit a chat completion as an async job.

        For long-running models (Opus and friends) where the synchronous
        /qai/v1/chat call would time out. Read the result with stream_job()
        or poll_job().
        """
        params = request.to_dict()
        params.pop("stream", None)
        return self.create_job("chat", params)

    def stream_job(self, job_id: str) -> Iterator[JobStreamEvent]:
        """Stream a job's progress over SSE.

        Yields "progress" events until a terminal "complete" or "error".
        """
        url = self.base_url + f"/qai/v1/jobs/{job_id}/stream"
        headers = self._headers()
        headers["Accept"] = "text/event-stream"

        with httpx.Client(timeout=httpx.Timeout(self.stream_timeout, connect=15.0)) as stream_client:
            with stream_client.stream("GET", url, headers=headers) as resp:
                if resp.status_code < 200 or resp.status_code >= 300:
                    resp.read()
                    meta = _ResponseMeta(resp.headers)
                    raise _parse_api_error(resp, meta.request_id)

                yield from _iter_job_sse_lines(resp.iter_lines())

    def generate_3d(
        self,
        model: str,
        *,
        prompt: str | None = None,
        image_url: str | None = None,
    ) -> JobCreateResponse:
        """Submit a text- or image-to-3D generation job. Poll with poll_job()."""
        params: dict[str, Any] = {"model": model}
        if prompt is not None:
            params["prompt"] = prompt
        if image_url is not None:
            params["image_url"] = image_url
        return self.create_job("3d/generate", params)

    # -- Chat cost estimate -------------------------------------------------

    def estimate_chat(self, request: ChatRequest) -> EstimateResponse:
        """Price a chat request without sending it.

        The number is the upfront reservation the real call would book — a
        ceiling, not a prediction. ``stream`` is dropped from the payload: the
        output ceiling is the same either way, and including it would make the
        SDK's wire shape diverge from what the server sees.
        """
        body = request.to_dict()
        body.pop("stream", None)
        data, _ = self._do_json("POST", "/qai/v1/chat/estimate", body)
        return EstimateResponse.from_dict(data)


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class AsyncClient:
    """Asynchronous Quantum AI API client."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        stream_timeout: float = DEFAULT_STREAM_TIMEOUT,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None
        self.stream_timeout = stream_timeout

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    _resolve_idempotency_key = staticmethod(Client._resolve_idempotency_key)

    async def _do_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> tuple[Any, _ResponseMeta]:
        url = self.base_url + path
        headers = self._headers()
        if body is not None:
            headers["Content-Type"] = "application/json"
        ikey = self._resolve_idempotency_key(idempotency_key)
        if ikey is not None:
            headers["Idempotency-Key"] = ikey

        resp = await self._client.request(
            method,
            url,
            headers=headers,
            content=json.dumps(body).encode() if body is not None else None,
        )
        meta = _ResponseMeta(resp.headers)

        if resp.status_code < 200 or resp.status_code >= 300:
            raise _parse_api_error(resp, meta.request_id)

        data = resp.json() if resp.content else None
        return data, meta

    async def _do_json_no_auth(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[Any, _ResponseMeta]:
        """Like _do_json but without auth headers."""
        url = self.base_url + path
        headers: dict[str, str] = {}
        if body is not None:
            headers["Content-Type"] = "application/json"

        resp = await self._client.request(
            method,
            url,
            headers=headers,
            content=json.dumps(body).encode() if body is not None else None,
        )
        meta = _ResponseMeta(resp.headers)

        if resp.status_code < 200 or resp.status_code >= 300:
            raise _parse_api_error(resp, meta.request_id)

        data = resp.json() if resp.content else None
        return data, meta

    async def _do_multipart(
        self,
        path: str,
        fields: dict[str, Any],
        files: _MultipartFiles,
    ) -> tuple[Any, _ResponseMeta]:
        """POST multipart/form-data request."""
        url = self.base_url + path
        headers = self._headers()

        resp = await self._client.post(
            url,
            headers=headers,
            data=fields,
            files=files,
        )
        meta = _ResponseMeta(resp.headers)

        if resp.status_code < 200 or resp.status_code >= 300:
            raise _parse_api_error(resp, meta.request_id)

        data = resp.json() if resp.content else None
        return data, meta

    async def _stream_sse(
        self,
        path: str,
        body: dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        """POST and stream chat SSE events."""
        url = self.base_url + path
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"

        stream_client = httpx.AsyncClient(timeout=httpx.Timeout(self.stream_timeout, connect=15.0))
        try:
            async with stream_client.stream(
                "POST",
                url,
                headers=headers,
                content=json.dumps(body).encode(),
            ) as resp:
                if resp.status_code < 200 or resp.status_code >= 300:
                    await resp.aread()
                    meta = _ResponseMeta(resp.headers)
                    raise _parse_api_error(resp, meta.request_id)

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        yield StreamEvent(type="done", done=True)
                        return
                    try:
                        yield _parse_sse_event(payload)
                    except json.JSONDecodeError as exc:
                        yield StreamEvent(type="error", error=f"parse SSE: {exc}")
                        return
        finally:
            await stream_client.aclose()

    async def _stream_mission_sse(
        self,
        path: str,
        body: dict[str, Any],
    ) -> AsyncIterator[AgentStreamEvent]:
        """POST and stream mission/agent SSE events as AgentStreamEvent.

        Preserves the full event dict so mission lifecycle events are not
        silently dropped (see _iter_mission_sse_lines).
        """
        url = self.base_url + path
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"

        stream_client = httpx.AsyncClient(timeout=httpx.Timeout(self.stream_timeout, connect=15.0))
        try:
            async with stream_client.stream(
                "POST",
                url,
                headers=headers,
                content=json.dumps(body).encode(),
            ) as resp:
                if resp.status_code < 200 or resp.status_code >= 300:
                    await resp.aread()
                    meta = _ResponseMeta(resp.headers)
                    raise _parse_api_error(resp, meta.request_id)

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        yield AgentStreamEvent(event_type="done", data={"done": True})
                        return
                    try:
                        raw = json.loads(payload)
                        yield AgentStreamEvent.from_dict(raw)
                    except json.JSONDecodeError as exc:
                        yield AgentStreamEvent(event_type="error", data={"error": f"parse SSE: {exc}"})
                        return
        finally:
            await stream_client.aclose()

    # -- Chat ---------------------------------------------------------------

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any] | ChatRequest] | None = None,
        *,
        request: ChatRequest | None = None,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat request."""
        req = Client._build_chat_request(model, messages, request, **kwargs)
        req.stream = False
        data, meta = await self._do_json("POST", "/qai/v1/chat", req.to_dict(), idempotency_key=idempotency_key)
        resp = ChatResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if not resp.model:
            resp.model = meta.model
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any] | ChatRequest] | None = None,
        *,
        request: ChatRequest | None = None,
        idempotency_key: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Send a streaming chat request. Yields StreamEvent objects."""
        req = Client._build_chat_request(model, messages, request, **kwargs)
        req.stream = True

        url = self.base_url + "/qai/v1/chat"
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"
        ikey = self._resolve_idempotency_key(idempotency_key)
        if ikey is not None:
            headers["Idempotency-Key"] = ikey

        stream_client = httpx.AsyncClient(timeout=httpx.Timeout(self.stream_timeout, connect=15.0))
        try:
            async with stream_client.stream(
                "POST",
                url,
                headers=headers,
                content=json.dumps(req.to_dict()).encode(),
            ) as resp:
                if resp.status_code < 200 or resp.status_code >= 300:
                    await resp.aread()
                    meta = _ResponseMeta(resp.headers)
                    raise _parse_api_error(resp, meta.request_id)

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        yield StreamEvent(type="done", done=True)
                        return
                    try:
                        yield _parse_sse_event(payload)
                    except json.JSONDecodeError as exc:
                        yield StreamEvent(type="error", error=f"parse SSE: {exc}")
                        return
        finally:
            await stream_client.aclose()

    # -- Session Chat -------------------------------------------------------

    async def chat_session(
        self,
        message: str,
        *,
        model: str | None = None,
        session_id: str | None = None,
        tools: list[ChatTool] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        stream: bool = False,
        system_prompt: str | None = None,
        context_config: ContextConfig | None = None,
        reasoning_effort: str | None = None,
        idempotency_key: str | None = None,
    ) -> SessionChatResponse | AsyncIterator[StreamEvent]:
        """Send a session-based chat request.

        Returns SessionChatResponse for non-streaming, or an AsyncIterator of
        StreamEvent for streaming.
        """
        req = SessionChatRequest(
            message=message,
            model=model,
            session_id=session_id,
            tools=tools,
            tool_results=tool_results,
            stream=stream,
            system_prompt=system_prompt,
            context_config=context_config,
            reasoning_effort=reasoning_effort,
        )
        if stream:
            return self._stream_sse("/qai/v1/chat/session", req.to_dict())
        data, meta = await self._do_json("POST", "/qai/v1/chat/session", req.to_dict(), idempotency_key=idempotency_key)
        resp = SessionChatResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    # -- Agent / Missions ---------------------------------------------------
    # See the sync Client.agent_run / mission_run docstrings for the
    # retargeting rationale (orchestration shape → /qai/v1/missions).

    async def agent_run(
        self,
        task: str,
        *,
        conductor_model: str | None = None,
        conductor_tier: str | None = None,
        workers: dict[str, MissionWorker] | None = None,
        max_steps: int | None = None,
        system_prompt: str | None = None,
        session_id: str | None = None,
        strategy: str | None = None,
    ) -> AsyncIterator[AgentStreamEvent]:
        """Run an orchestrated agent task via /qai/v1/missions (async)."""
        req = AgentRunRequest(
            task=task,
            conductor_model=conductor_model,
            conductor_tier=conductor_tier,
            workers=workers,
            max_steps=max_steps,
            system_prompt=system_prompt,
            session_id=session_id,
            strategy=strategy,
        )
        return self._stream_mission_sse("/qai/v1/missions", req.to_dict())

    async def mission_run(
        self,
        goal: str,
        *,
        strategy: str | None = None,
        conductor_model: str | None = None,
        conductor_tier: str | None = None,
        workers: dict[str, MissionWorker] | None = None,
        max_steps: int | None = None,
        system_prompt: str | None = None,
        session_id: str | None = None,
        auto_plan: bool | None = None,
        context_config: ContextConfig | None = None,
        deployment_id: str | None = None,
        build_command: str | None = None,
        workspace_path: str | None = None,
    ) -> AsyncIterator[AgentStreamEvent]:
        """Run a mission via /qai/v1/missions (async)."""
        req = MissionRunRequest(
            goal=goal,
            strategy=strategy,
            conductor_model=conductor_model,
            conductor_tier=conductor_tier,
            workers=workers,
            max_steps=max_steps,
            system_prompt=system_prompt,
            session_id=session_id,
            auto_plan=auto_plan,
            context_config=context_config,
            deployment_id=deployment_id,
            build_command=build_command,
            workspace_path=workspace_path,
        )
        return self._stream_mission_sse("/qai/v1/missions", req.to_dict())

    # -- API Keys -----------------------------------------------------------

    async def create_key(
        self,
        name: str,
        *,
        endpoints: list[str] | None = None,
        spend_cap_usd: float | None = None,
        rate_limit: int | None = None,
    ) -> APIKeyCreateResponse:
        """Create a new API key."""
        req = APIKeyCreateRequest(
            name=name,
            endpoints=endpoints,
            spend_cap_usd=spend_cap_usd,
            rate_limit=rate_limit,
        )
        data, _ = await self._do_json("POST", "/qai/v1/keys", req.to_dict())
        return APIKeyCreateResponse.from_dict(data)

    async def list_keys(self) -> APIKeyListResponse:
        """List all API keys for the account."""
        data, _ = await self._do_json("GET", "/qai/v1/keys")
        return APIKeyListResponse.from_dict(data)

    async def revoke_key(self, key_id: str) -> None:
        """Revoke an API key."""
        await self._do_json("DELETE", f"/qai/v1/keys/{key_id}")

    # -- Compute ------------------------------------------------------------

    async def compute_templates(self) -> list[ComputeTemplate]:
        """List available compute templates."""
        data, _ = await self._do_json("GET", "/qai/v1/compute/templates")
        return [
            ComputeTemplate.from_dict(t)
            for t in data.get("templates", [])
        ]

    async def compute_provision(
        self,
        template: str,
        *,
        zone: str | None = None,
        spot: bool = False,
        auto_teardown_minutes: int = 30,
        ssh_public_key: str | None = None,
    ) -> ComputeProvisionResponse:
        """Provision a new compute instance."""
        req = ComputeProvisionRequest(
            template=template,
            zone=zone,
            spot=spot,
            auto_teardown_minutes=auto_teardown_minutes,
            ssh_public_key=ssh_public_key,
        )
        data, _ = await self._do_json("POST", "/qai/v1/compute/provision", req.to_dict())
        return ComputeProvisionResponse.from_dict(data)

    async def compute_instances(self) -> list[ComputeInstance]:
        """List all compute instances."""
        data, _ = await self._do_json("GET", "/qai/v1/compute/instances")
        return [
            ComputeInstance.from_dict(i)
            for i in data.get("instances", [])
        ]

    async def compute_instance(self, instance_id: str) -> ComputeInstance:
        """Get a specific compute instance."""
        data, _ = await self._do_json("GET", f"/qai/v1/compute/instance/{instance_id}")
        return ComputeInstance.from_dict(data)

    async def compute_delete(self, instance_id: str) -> None:
        """Delete a compute instance."""
        await self._do_json("DELETE", f"/qai/v1/compute/instance/{instance_id}")

    async def compute_ssh_key(
        self,
        instance_id: str,
        public_key: str,
        username: str = "cosmic",
    ) -> None:
        """Add an SSH key to a compute instance."""
        body: dict[str, Any] = {"public_key": public_key, "username": username}
        await self._do_json("POST", f"/qai/v1/compute/instance/{instance_id}/ssh-key", body)

    async def compute_keepalive(self, instance_id: str) -> None:
        """Send a keepalive to extend a compute instance's auto-teardown timer."""
        await self._do_json("POST", f"/qai/v1/compute/instance/{instance_id}/keepalive")

    # -- Voices -------------------------------------------------------------

    async def list_voices(self) -> VoiceListResponse:
        """List all available voices."""
        data, _ = await self._do_json("GET", "/qai/v1/voices")
        return VoiceListResponse.from_dict(data)

    async def clone_voice(
        self,
        name: str,
        audio_file: bytes,
        *,
        filename: str = "voice.mp3",
        content_type: str = "audio/mpeg",
    ) -> VoiceCloneResponse:
        """Clone a voice from an audio sample (multipart upload)."""
        data, _ = await self._do_multipart(
            "/qai/v1/voices/clone",
            fields={"name": name},
            files={"audio_file": (filename, audio_file, content_type)},
        )
        return VoiceCloneResponse.from_dict(data)

    async def delete_voice(self, voice_id: str) -> None:
        """Delete a cloned voice."""
        await self._do_json("DELETE", f"/qai/v1/voices/{voice_id}")

    # -- Image --------------------------------------------------------------

    async def generate_image(
        self,
        request: ImageRequest,
        *,
        idempotency_key: str | None = None,
    ) -> ImageResponse:
        data, meta = await self._do_json("POST", "/qai/v1/images/generate", request.to_dict(), idempotency_key=idempotency_key)
        resp = ImageResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    async def edit_image(
        self,
        request: ImageEditRequest,
        *,
        idempotency_key: str | None = None,
    ) -> ImageEditResponse:
        data, meta = await self._do_json("POST", "/qai/v1/images/edit", request.to_dict(), idempotency_key=idempotency_key)
        resp = ImageEditResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    # -- Video --------------------------------------------------------------

    async def generate_video(
        self,
        request: VideoRequest,
        *,
        idempotency_key: str | None = None,
    ) -> VideoResponse:
        data, meta = await self._do_json("POST", "/qai/v1/video/generate", request.to_dict(), idempotency_key=idempotency_key)
        resp = VideoResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    async def video_studio(
        self,
        avatar_id: str,
        script: str,
        voice_id: str,
    ) -> JobStatusResponse:
        """Create a HeyGen studio video."""
        req = VideoStudioRequest(avatar_id=avatar_id, script=script, voice_id=voice_id)
        data, meta = await self._do_json("POST", "/qai/v1/video/studio", req.to_dict())
        return JobStatusResponse.from_dict(data)

    async def video_translate(
        self,
        video_url: str,
        target_language: str,
    ) -> JobStatusResponse:
        """Translate a video to another language."""
        req = VideoTranslateRequest(video_url=video_url, target_language=target_language)
        data, meta = await self._do_json("POST", "/qai/v1/video/translate", req.to_dict())
        return JobStatusResponse.from_dict(data)

    async def video_photo_avatar(
        self,
        photo_url: str,
        script: str,
    ) -> JobStatusResponse:
        """Generate a video from a photo avatar."""
        req = VideoPhotoAvatarRequest(photo_url=photo_url, script=script)
        data, meta = await self._do_json("POST", "/qai/v1/video/photo-avatar", req.to_dict())
        return JobStatusResponse.from_dict(data)

    async def video_digital_twin(
        self,
        avatar_id: str,
        script: str,
    ) -> JobStatusResponse:
        """Generate a video with a digital twin."""
        req = VideoDigitalTwinRequest(avatar_id=avatar_id, script=script)
        data, meta = await self._do_json("POST", "/qai/v1/video/digital-twin", req.to_dict())
        return JobStatusResponse.from_dict(data)

    async def video_avatars(self) -> list[HeyGenAvatar]:
        """List available HeyGen avatars."""
        data, _ = await self._do_json("GET", "/qai/v1/video/avatars")
        return [HeyGenAvatar.from_dict(a) for a in data.get("avatars", [])]

    async def video_templates(self) -> list[HeyGenTemplate]:
        """List available HeyGen video templates."""
        data, _ = await self._do_json("GET", "/qai/v1/video/templates")
        return [HeyGenTemplate.from_dict(t) for t in data.get("templates", [])]

    async def video_heygen_voices(self) -> list[HeyGenVoice]:
        """List available HeyGen voices."""
        data, _ = await self._do_json("GET", "/qai/v1/video/heygen-voices")
        return [HeyGenVoice.from_dict(v) for v in data.get("voices", [])]

    async def video_template_detail(self, template_id: str) -> VideoTemplateDetailResponse:
        """Inspect a HeyGen template's variable schema and scenes (unbilled).

        Only draft-v4 templates with variables are supported upstream; an
        unknown template id surfaces as a provider_error.
        """
        data, _ = await self._do_json("GET", f"/qai/v1/video/template/{template_id}")
        return VideoTemplateDetailResponse.from_dict(data)

    async def video_template_generate(
        self,
        template_id: str,
        request: VideoTemplateGenerateRequest,
    ) -> JobAcceptedResponse:
        """Render a video from a HeyGen template (async job "video/template-v3").

        Returns the accepted-job envelope — poll with get_job / poll_job (or
        SSE via stream_job) until "completed"/"failed", then read
        result["video_url"]. Deep validation happens at execution time, so
        violations surface as a failed job rather than a 4xx at submit.
        """
        data, _ = await self._do_json("POST", f"/qai/v1/video/template/{template_id}", request.to_dict())
        return JobAcceptedResponse.from_dict(data)

    async def video_batch_submit(self, request: VideoBatchSubmitRequest) -> VideoBatchSubmitResponse:
        """Submit 1-100 raw HeyGen video payloads as one batch (202 Accepted).

        Poll video_batch_status for progress and delivery.
        """
        data, _ = await self._do_json("POST", "/qai/v1/video/batch", request.to_dict())
        return VideoBatchSubmitResponse.from_dict(data)

    async def video_batch_status(
        self,
        batch_id: str,
        *,
        limit: int | None = None,
        token: str | None = None,
    ) -> VideoBatchStatusResponse:
        """Get a batch's status plus one cursor-paginated page of items.

        Poll (~5s) until ``status`` is terminal, then keep polling until
        ``billing_status == "settled"`` — per-item ``video_url`` values are
        withheld until settlement. Collect URLs across pages via ``next_token``.
        """
        params: list[str] = []
        if limit is not None:
            params.append(f"limit={limit}")
        if token is not None:
            params.append(f"token={quote(token, safe='')}")
        path = f"/qai/v1/video/batch/{batch_id}"
        if params:
            path += "?" + "&".join(params)
        data, _ = await self._do_json("GET", path)
        return VideoBatchStatusResponse.from_dict(data)

    # -- Avatar Realtime (HeyGen Broadcast) ---------------------------------

    async def create_avatar_realtime_session(
        self,
        request: AvatarRealtimeRequest,
    ) -> AvatarRealtimeCreateResponse:
        """Create a live avatar realtime session (HeyGen Broadcast).

        PREPAID: the entire ``max_duration_seconds`` block (1-3600 s) is
        charged at create time; cancelling early does NOT refund.
        """
        data, meta = await self._do_json("POST", "/qai/v1/avatar/realtime", request.to_dict())
        resp = AvatarRealtimeCreateResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    async def get_avatar_realtime_session(self, stream_id: str) -> AvatarRealtimeStatusResponse:
        """Get the live status of an avatar realtime session.

        Poll (~2s) until ``status == "streaming"``, then play ``hls_url``.
        "completed" and "error" are terminal.
        """
        data, _ = await self._do_json("GET", f"/qai/v1/avatar/realtime/{stream_id}")
        return AvatarRealtimeStatusResponse.from_dict(data)

    async def send_avatar_realtime_text(
        self,
        stream_id: str,
        delta: str = "",
        *,
        final: bool = False,
    ) -> AvatarRealtimeTextResponse:
        """Append a text delta to a ``text_stream`` session.

        ``delta`` is required unless ``final=True`` (which closes the input;
        appending afterwards fails upstream with a 410 provider_error).
        """
        body: dict[str, Any] = {}
        if delta:
            body["delta"] = delta
        body["final"] = final
        data, _ = await self._do_json("POST", f"/qai/v1/avatar/realtime/{stream_id}/text", body)
        return AvatarRealtimeTextResponse.from_dict(data)

    async def cancel_avatar_realtime_session(self, stream_id: str) -> AvatarRealtimeCancelResponse:
        """Terminate an avatar realtime session early (idempotent; no refund —
        this only stops HeyGen's upstream meter)."""
        data, _ = await self._do_json("POST", f"/qai/v1/avatar/realtime/{stream_id}/cancel")
        return AvatarRealtimeCancelResponse.from_dict(data)

    # -- Audio --------------------------------------------------------------

    async def speak(
        self,
        request: TTSRequest,
        *,
        idempotency_key: str | None = None,
    ) -> TTSResponse:
        data, meta = await self._do_json("POST", "/qai/v1/audio/tts", request.to_dict(), idempotency_key=idempotency_key)
        resp = TTSResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    async def transcribe(
        self,
        request: STTRequest,
        *,
        idempotency_key: str | None = None,
    ) -> STTResponse:
        data, meta = await self._do_json("POST", "/qai/v1/audio/stt", request.to_dict(), idempotency_key=idempotency_key)
        resp = STTResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    async def generate_music(
        self,
        request: MusicRequest,
        *,
        idempotency_key: str | None = None,
    ) -> MusicResponse:
        data, meta = await self._do_json("POST", "/qai/v1/audio/music", request.to_dict(), idempotency_key=idempotency_key)
        resp = MusicResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    async def sound_effects(
        self,
        prompt: str,
        duration_seconds: float | None = None,
        *,
        idempotency_key: str | None = None,
    ) -> SoundEffectResponse:
        """Generate sound effects from a text prompt."""
        body: dict[str, Any] = {"prompt": prompt}
        if duration_seconds is not None:
            body["duration_seconds"] = duration_seconds
        data, meta = await self._do_json("POST", "/qai/v1/audio/sound-effects", body, idempotency_key=idempotency_key)
        resp = SoundEffectResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    async def dialogue(
        self,
        text: str,
        voices: list[DialogueVoice],
        *,
        model: str | None = None,
        idempotency_key: str | None = None,
    ) -> AudioResponse:
        """Generate multi-voice dialogue audio."""
        req = DialogueRequest(text=text, voices=voices, model=model)
        data, meta = await self._do_json("POST", "/qai/v1/audio/dialogue", req.to_dict(), idempotency_key=idempotency_key)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    async def speech_to_speech(
        self,
        audio: str,
        voice_id: str,
    ) -> AudioResponse:
        """Convert speech audio to a different voice."""
        body: dict[str, Any] = {"audio": audio, "voice_id": voice_id}
        data, meta = await self._do_json("POST", "/qai/v1/audio/speech-to-speech", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def isolate_voice(self, audio: str) -> AudioResponse:
        """Isolate voice from background noise in an audio clip."""
        body: dict[str, Any] = {"audio": audio}
        data, meta = await self._do_json("POST", "/qai/v1/audio/isolate", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def remix_voice(self, audio: str, voice_id: str) -> AudioResponse:
        """Remix audio with a different voice."""
        body: dict[str, Any] = {"audio": audio, "voice_id": voice_id}
        data, meta = await self._do_json("POST", "/qai/v1/audio/remix", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def dub(
        self,
        audio: str,
        target_lang: str,
        *,
        source_lang: str | None = None,
    ) -> AudioResponse:
        """Dub audio into a target language."""
        body: dict[str, Any] = {"audio": audio, "target_lang": target_lang}
        if source_lang is not None:
            body["source_lang"] = source_lang
        data, meta = await self._do_json("POST", "/qai/v1/audio/dub", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def align(self, audio: str, text: str) -> AlignmentResponse:
        """Align audio with text to get word-level timing."""
        body: dict[str, Any] = {"audio": audio, "text": text}
        data, meta = await self._do_json("POST", "/qai/v1/audio/align", body)
        resp = AlignmentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def voice_design(self, description: str) -> VoiceDesignResponse:
        """Design a new voice from a text description."""
        body: dict[str, Any] = {"description": description}
        data, meta = await self._do_json("POST", "/qai/v1/audio/voice-design", body)
        resp = VoiceDesignResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def starfish_tts(self, text: str, voice_id: str) -> AudioResponse:
        """Generate speech using Starfish TTS engine."""
        body: dict[str, Any] = {"text": text, "voice_id": voice_id}
        data, meta = await self._do_json("POST", "/qai/v1/audio/starfish-tts", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def search_audio_sounds(
        self,
        query: str,
        *,
        sound_type: str | None = None,
        limit: int | None = None,
        min_score: float | None = None,
        token: str | None = None,
    ) -> AudioSoundsResponse:
        """Search HeyGen's background-music and sound-effects catalogs
        (semantic ranking, best score first). Unbilled catalog route.

        ``sound_type`` maps to the wire param ``type``: "music" |
        "sound_effects" (API default "music"). ``limit`` is 1-50 (default 10),
        ``min_score`` 0-1 (default 0.7), ``token`` an opaque cursor from a
        previous response's ``next_token``.
        """
        params: list[str] = [f"query={quote(query, safe='')}"]
        if sound_type is not None:
            params.append(f"type={quote(sound_type, safe='')}")
        if limit is not None:
            params.append(f"limit={limit}")
        if min_score is not None:
            params.append(f"min_score={min_score}")
        if token is not None:
            params.append(f"token={quote(token, safe='')}")
        path = "/qai/v1/audio/sounds?" + "&".join(params)
        data, _ = await self._do_json("GET", path)
        return AudioSoundsResponse.from_dict(data)

    # -- Embeddings ---------------------------------------------------------

    async def embed(
        self,
        request: EmbedRequest,
        *,
        idempotency_key: str | None = None,
    ) -> EmbedResponse:
        data, meta = await self._do_json("POST", "/qai/v1/embeddings", request.to_dict(), idempotency_key=idempotency_key)
        resp = EmbedResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if resp.balance_after is None:
            resp.balance_after = meta.balance_after
        return resp

    # -- Documents ----------------------------------------------------------

    async def extract_document(
        self,
        request: DocumentRequest,
        *,
        idempotency_key: str | None = None,
    ) -> DocumentResponse:
        data, meta = await self._do_json("POST", "/qai/v1/documents/extract", request.to_dict(), idempotency_key=idempotency_key)
        resp = DocumentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def chunk_document(
        self,
        text: str,
        *,
        chunk_size: int | None = None,
    ) -> ChunkDocumentResponse:
        """Chunk a document into smaller pieces."""
        req = ChunkDocumentRequest(text=text, chunk_size=chunk_size)
        data, meta = await self._do_json("POST", "/qai/v1/documents/chunk", req.to_dict())
        resp = ChunkDocumentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def process_document(self, text: str) -> ProcessDocumentResponse:
        """Process a document."""
        req = ProcessDocumentRequest(text=text)
        data, meta = await self._do_json("POST", "/qai/v1/documents/process", req.to_dict())
        resp = ProcessDocumentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- RAG ----------------------------------------------------------------

    async def rag_search(
        self,
        request: RAGSearchRequest,
        *,
        idempotency_key: str | None = None,
    ) -> RAGSearchResponse:
        data, meta = await self._do_json("POST", "/qai/v1/rag/search", request.to_dict(), idempotency_key=idempotency_key)
        resp = RAGSearchResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def rag_corpora(self) -> list[RAGCorpus]:
        data, _ = await self._do_json("GET", "/qai/v1/rag/corpora")
        return [
            RAGCorpus(
                name=c.get("name", ""),
                display_name=c.get("displayName", ""),
                description=c.get("description", ""),
                state=c.get("state", ""),
            )
            for c in data.get("corpora", [])
        ]

    async def surreal_rag_search(self, request: SurrealRAGSearchRequest) -> SurrealRAGSearchResponse:
        data, meta = await self._do_json("POST", "/qai/v1/rag/surreal/search", request.to_dict())
        resp = SurrealRAGSearchResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def surreal_rag_providers(self) -> SurrealRAGProvidersResponse:
        """List available SurrealDB RAG providers."""
        data, _ = await self._do_json("GET", "/qai/v1/rag/surreal/providers")
        return SurrealRAGProvidersResponse.from_dict(data)

    # -- Models -------------------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        data, _ = await self._do_json("GET", "/qai/v1/models")
        return [
            ModelInfo(
                id=m.get("id", ""),
                provider=m.get("provider", ""),
                display_name=m.get("display_name", ""),
                input_per_million=m.get("input_per_million", 0.0),
                output_per_million=m.get("output_per_million", 0.0),
            )
            for m in data.get("models", [])
        ]

    async def get_pricing(self) -> list[PricingInfo]:
        data, _ = await self._do_json("GET", "/qai/v1/pricing")
        return [
            PricingInfo(
                id=p.get("id", ""),
                provider=p.get("provider", ""),
                display_name=p.get("display_name", ""),
                input_per_million=p.get("input_per_million", 0.0),
                output_per_million=p.get("output_per_million", 0.0),
            )
            for p in data.get("pricing", [])
        ]

    # -- Account ------------------------------------------------------------

    async def account_balance(self) -> BalanceResponse:
        """Get the account credit balance."""
        data, _ = await self._do_json("GET", "/qai/v1/account/balance")
        return BalanceResponse.from_dict(data)

    async def account_usage(
        self,
        limit: int | None = None,
        start_after: str | None = None,
    ) -> UsageResponse:
        """Get paginated usage history."""
        params: list[str] = []
        if limit is not None:
            params.append(f"limit={limit}")
        if start_after is not None:
            params.append(f"start_after={start_after}")
        path = "/qai/v1/account/usage"
        if params:
            path += "?" + "&".join(params)
        data, _ = await self._do_json("GET", path)
        return UsageResponse.from_dict(data)

    async def account_usage_summary(self, months: int | None = None) -> UsageSummaryResponse:
        """Get monthly usage summary."""
        path = "/qai/v1/account/usage/summary"
        if months is not None:
            path += f"?months={months}"
        data, _ = await self._do_json("GET", path)
        return UsageSummaryResponse.from_dict(data)

    async def account_pricing(self) -> PricingResponse:
        """Get the full pricing table (model to pricing entry map)."""
        data, _ = await self._do_json("GET", "/qai/v1/pricing")
        return PricingResponse.from_dict(data)

    # -- Jobs ---------------------------------------------------------------

    async def create_job(
        self,
        job_type: str,
        params: dict[str, Any],
    ) -> JobCreateResponse:
        """Create an async job. Returns the job ID for polling."""
        body: dict[str, Any] = {"type": job_type, "params": params}
        data, _ = await self._do_json("POST", "/qai/v1/jobs", body)
        return JobCreateResponse.from_dict(data)

    async def get_job(self, job_id: str) -> JobStatusResponse:
        """Check the status of an async job."""
        data, _ = await self._do_json("GET", f"/qai/v1/jobs/{job_id}")
        return JobStatusResponse.from_dict(data)

    async def list_jobs(self) -> JobListResponse:
        """List all jobs for the account."""
        data, _ = await self._do_json("GET", "/qai/v1/jobs")
        return JobListResponse.from_dict(data)

    async def poll_job(
        self,
        job_id: str,
        interval: float = 5.0,
        max_attempts: int = 120,
    ) -> JobStatusResponse:
        """Poll a job until completion or timeout."""
        import asyncio
        for _ in range(max_attempts):
            await asyncio.sleep(interval)
            status = await self.get_job(job_id)
            if status.status in ("completed", "failed"):
                return status
        return JobStatusResponse(
            job_id=job_id,
            status="timeout",
            error=f"Job polling timed out after {max_attempts} attempts",
        )

    # -- Contact ------------------------------------------------------------

    async def contact(
        self,
        name: str,
        email: str,
        message: str,
    ) -> ContactResponse:
        """Submit a contact form (no auth required)."""
        req = ContactRequest(name=name, email=email, message=message)
        data, _ = await self._do_json_no_auth("POST", "/qai/v1/contact", req.to_dict())
        return ContactResponse.from_dict(data)

    # ── Search (Brave) ──────────────────────────────────────────────

    async def web_search(
        self,
        query: str,
        count: int | None = None,
        offset: int | None = None,
        country: str | None = None,
        language: str | None = None,
        freshness: str | None = None,
        safesearch: str | None = None,
    ) -> dict:
        """Perform a web search. Returns web results, news, videos, infobox, discussions."""
        body: dict = {"query": query}
        if count is not None: body["count"] = count
        if offset is not None: body["offset"] = offset
        if country is not None: body["country"] = country
        if language is not None: body["language"] = language
        if freshness is not None: body["freshness"] = freshness
        if safesearch is not None: body["safesearch"] = safesearch
        data, _ = await self._do_json("POST", "/qai/v1/search/web", body)
        return data

    async def search_context(
        self,
        query: str,
        count: int | None = None,
        country: str | None = None,
        language: str | None = None,
        freshness: str | None = None,
    ) -> dict:
        """Get LLM-optimized content chunks for grounding."""
        body: dict = {"query": query}
        if count is not None: body["count"] = count
        if country is not None: body["country"] = country
        if language is not None: body["language"] = language
        if freshness is not None: body["freshness"] = freshness
        data, _ = await self._do_json("POST", "/qai/v1/search/context", body)
        return data

    async def search_answer(
        self,
        messages: list[dict],
        model: str | None = None,
    ) -> dict:
        """Get a grounded AI answer with citations."""
        body: dict = {"messages": messages}
        if model is not None: body["model"] = model
        data, _ = await self._do_json("POST", "/qai/v1/search/answer", body)
        return data

    async def google_search(self, request: GoogleSearchRequest) -> GoogleSearchResponse:
        """Gemini-grounded Google search — an answer plus the sources behind it.

        Billed per Google query the model decides to run, which makes this
        materially more expensive than search_answer (Brave-backed). Reach for
        it when answer quality matters more than per-call cost. Render
        ``search_entry_point`` verbatim — Google's grounding terms require it.
        """
        data, _ = await self._do_json("POST", "/qai/v1/search/google", request.to_dict())
        return GoogleSearchResponse.from_dict(data)

    # -- Missions -----------------------------------------------------------

    async def mission_create(self, request: MissionCreateRequest) -> MissionCreateResponse:
        """Create a mission and start executing it asynchronously."""
        data, _ = await self._do_json("POST", "/qai/v1/missions/create", request.to_dict())
        return MissionCreateResponse.from_dict(data)

    async def mission_list(self, status: str | None = None) -> MissionListResponse:
        """List missions, optionally filtered by status."""
        path = "/qai/v1/missions/list"
        if status is not None:
            path += f"?status={quote(status, safe='')}"
        data, _ = await self._do_json("GET", path)
        return MissionListResponse.from_dict(data)

    async def mission_get(self, mission_id: str) -> MissionDetail:
        """Get a mission's details, including its tasks."""
        data, _ = await self._do_json("GET", f"/qai/v1/missions/{mission_id}")
        return MissionDetail.from_dict(data)

    async def mission_delete(self, mission_id: str) -> MissionStatusResponse:
        """Delete a mission."""
        data, _ = await self._do_json("DELETE", f"/qai/v1/missions/{mission_id}")
        return MissionStatusResponse.from_dict(data)

    async def mission_cancel(self, mission_id: str) -> MissionStatusResponse:
        """Cancel a running mission."""
        data, _ = await self._do_json("POST", f"/qai/v1/missions/{mission_id}/cancel", {})
        return MissionStatusResponse.from_dict(data)

    async def mission_pause(self, mission_id: str) -> MissionStatusResponse:
        """Pause a running mission."""
        data, _ = await self._do_json("POST", f"/qai/v1/missions/{mission_id}/pause", {})
        return MissionStatusResponse.from_dict(data)

    async def mission_resume(self, mission_id: str) -> MissionStatusResponse:
        """Resume a paused mission."""
        data, _ = await self._do_json("POST", f"/qai/v1/missions/{mission_id}/resume", {})
        return MissionStatusResponse.from_dict(data)

    async def mission_chat(
        self,
        mission_id: str,
        request: MissionChatRequest,
    ) -> MissionChatResponse:
        """Chat with the mission's architect."""
        data, _ = await self._do_json(
            "POST", f"/qai/v1/missions/{mission_id}/chat", request.to_dict()
        )
        return MissionChatResponse.from_dict(data)

    async def mission_retry_task(self, mission_id: str, task_id: str) -> MissionStatusResponse:
        """Retry a failed task within a mission."""
        data, _ = await self._do_json(
            "POST", f"/qai/v1/missions/{mission_id}/retry/{task_id}", {}
        )
        return MissionStatusResponse.from_dict(data)

    async def mission_approve(
        self,
        mission_id: str,
        request: MissionApproveRequest,
    ) -> MissionStatusResponse:
        """Approve a completed mission."""
        data, _ = await self._do_json(
            "POST", f"/qai/v1/missions/{mission_id}/approve", request.to_dict()
        )
        return MissionStatusResponse.from_dict(data)

    async def mission_update_plan(
        self,
        mission_id: str,
        request: MissionPlanUpdate,
    ) -> MissionStatusResponse:
        """Replace the mission's plan (tasks, workers, prompt)."""
        data, _ = await self._do_json(
            "PUT", f"/qai/v1/missions/{mission_id}/plan", request.to_dict()
        )
        return MissionStatusResponse.from_dict(data)

    async def mission_checkpoints(self, mission_id: str) -> MissionCheckpointsResponse:
        """List the git checkpoints a mission has committed."""
        data, _ = await self._do_json("GET", f"/qai/v1/missions/{mission_id}/checkpoints")
        return MissionCheckpointsResponse.from_dict(data)

    async def mission_import(self, request: MissionImportRequest) -> MissionCreateResponse:
        """Import an existing plan as a new mission."""
        data, _ = await self._do_json("POST", "/qai/v1/missions/import", request.to_dict())
        return MissionCreateResponse.from_dict(data)

    # -- Vision -------------------------------------------------------------

    async def vision_analyze(self, request: VisionRequest) -> VisionResponse:
        """Combined analysis: scene + objects + quality + OCR + relevance."""
        data, _ = await self._do_json("POST", "/qai/v1/vision/analyze", request.to_dict())
        return VisionResponse.from_dict(data)

    async def vision_detect(self, request: VisionRequest) -> VisionResponse:
        """Object detection with bounding boxes."""
        data, _ = await self._do_json("POST", "/qai/v1/vision/detect", request.to_dict())
        return VisionResponse.from_dict(data)

    async def vision_describe(self, request: VisionRequest) -> VisionResponse:
        """Scene description and tags."""
        data, _ = await self._do_json("POST", "/qai/v1/vision/describe", request.to_dict())
        return VisionResponse.from_dict(data)

    async def vision_ocr(self, request: VisionRequest) -> VisionResponse:
        """Text extraction and overlay metadata (OCR)."""
        data, _ = await self._do_json("POST", "/qai/v1/vision/ocr", request.to_dict())
        return VisionResponse.from_dict(data)

    async def vision_quality(self, request: VisionRequest) -> VisionResponse:
        """Image quality assessment (blur, exposure, resolution)."""
        data, _ = await self._do_json("POST", "/qai/v1/vision/quality", request.to_dict())
        return VisionResponse.from_dict(data)

    # -- Security -----------------------------------------------------------

    async def security_scan_url(self, url: str) -> SecurityScanResponse:
        """Fetch a URL and scan the page for prompt injection."""
        req = SecurityScanUrlRequest(url=url)
        data, _ = await self._do_json("POST", "/qai/v1/security/scan-url", req.to_dict())
        return SecurityScanResponse.from_dict(data)

    async def security_scan_html(self, request: SecurityScanHtmlRequest) -> SecurityScanResponse:
        """Scan already-fetched HTML for prompt injection."""
        data, _ = await self._do_json("POST", "/qai/v1/security/scan-html", request.to_dict())
        return SecurityScanResponse.from_dict(data)

    async def security_check(self, url: str) -> SecurityCheckResponse:
        """Check a URL against the injection registry without fetching it."""
        data, _ = await self._do_json("GET", f"/qai/v1/security/check?url={quote(url, safe='')}")
        return SecurityCheckResponse.from_dict(data)

    async def security_blocklist(self, status: str | None = None) -> SecurityBlocklistResponse:
        """Get the injection blocklist feed ("confirmed" or "suspected")."""
        path = "/qai/v1/security/blocklist"
        if status is not None:
            path += f"?status={quote(status, safe='')}"
        data, _ = await self._do_json("GET", path)
        return SecurityBlocklistResponse.from_dict(data)

    async def security_report(self, request: SecurityReportRequest) -> SecurityReportResponse:
        """Report a suspicious URL to the registry."""
        data, _ = await self._do_json("POST", "/qai/v1/security/report", request.to_dict())
        return SecurityReportResponse.from_dict(data)

    # -- Credits ------------------------------------------------------------

    async def credit_packs(self) -> CreditPacksResponse:
        """List the credit packs on sale. No authentication required."""
        return await _credits.credit_packs_async(self)

    async def credit_purchase(
        self,
        pack_id: str,
        *,
        success_url: str | None = None,
        cancel_url: str | None = None,
    ) -> CreditPurchaseResponse:
        """Start a credit-pack purchase. Returns the checkout URL to open."""
        return await _credits.credit_purchase_async(self, pack_id, success_url, cancel_url)

    async def credit_balance(self) -> CreditBalanceResponse:
        """Get the wallet balance in ticks and USD."""
        return await _credits.credit_balance_async(self)

    async def credit_tiers(self) -> CreditTiersResponse:
        """List the volume-discount tiers. No authentication required."""
        return await _credits.credit_tiers_async(self)

    async def dev_program_apply(
        self,
        use_case: str,
        *,
        company: str | None = None,
        expected_usd: float | None = None,
        website: str | None = None,
    ) -> DevProgramApplyResponse:
        """Apply for the developer program."""
        return await _credits.dev_program_apply_async(
            self, use_case, company, expected_usd, website
        )

    # -- Batch --------------------------------------------------------------

    async def batch_submit(self, jobs: list[BatchJobInput]) -> BatchSubmitResponse:
        """Submit a batch of prompts. Each runs independently; poll via Jobs."""
        return await _batch.batch_submit_async(self, jobs)

    async def batch_submit_jsonl(self, jsonl: str) -> BatchJsonlResponse:
        """Submit a batch as JSONL — one JSON job object per line."""
        return await _batch.batch_submit_jsonl_async(self, jsonl)

    async def batch_jobs(self) -> BatchJobsResponse:
        """List the account's batch jobs."""
        return await _batch.batch_jobs_async(self)

    async def batch_job(self, job_id: str) -> BatchJobInfo:
        """Get the status and result of a single batch job."""
        return await _batch.batch_job_async(self, job_id)

    # -- Auth ---------------------------------------------------------------

    async def auth_apple(self, id_token: str, *, name: str | None = None) -> AuthResponse:
        """Exchange a Sign in with Apple identity token for an API token.

        Pass ``name`` on first sign-in — Apple only sends it once, so the
        account is created without a display name if you drop it.
        """
        return await _auth.auth_apple_async(self, id_token, name)

    # -- RAG collections (user-scoped xAI proxy) ----------------------------

    async def collections_list(self) -> list[Collection]:
        """List the user's collections plus the shared ones."""
        data, _ = await self._do_json("GET", "/qai/v1/rag/collections")
        return CollectionsListResponse.from_dict(data).collections

    async def collections_create(self, name: str) -> Collection:
        """Create a user-owned collection."""
        data, _ = await self._do_json("POST", "/qai/v1/rag/collections", {"name": name})
        return Collection(**data)

    async def collections_get(self, collection_id: str) -> Collection:
        """Get one collection — must be owned by the caller or shared."""
        data, _ = await self._do_json("GET", f"/qai/v1/rag/collections/{collection_id}")
        return Collection(**data)

    async def collections_delete(self, collection_id: str) -> str:
        """Delete a collection (owner only). Returns the server's message."""
        data, _ = await self._do_json("DELETE", f"/qai/v1/rag/collections/{collection_id}")
        return (data or {}).get("message", "")

    async def collections_documents(self, collection_id: str) -> list[CollectionDocument]:
        """List the documents in a collection."""
        data, _ = await self._do_json(
            "GET", f"/qai/v1/rag/collections/{collection_id}/documents"
        )
        return CollectionDocumentsResponse.from_dict(data).documents

    async def collections_upload(
        self,
        collection_id: str,
        filename: str,
        content: bytes,
    ) -> CollectionUploadResult:
        """Upload a file to a collection.

        The server runs the two-step xAI upload (files API then management API)
        with the master key, so the caller only sends bytes.
        """
        data, _ = await self._do_multipart(
            f"/qai/v1/rag/collections/{collection_id}/upload",
            fields={},
            files={"file": (filename, content, "application/octet-stream")},
        )
        return CollectionUploadResult(**data)

    async def collections_search(
        self,
        request: CollectionSearchRequest,
    ) -> list[CollectionSearchResult]:
        """Search across collections (owned + shared), hybrid by default."""
        data, _ = await self._do_json(
            "POST", "/qai/v1/rag/search/collections", request.to_dict()
        )
        return CollectionSearchResponse.from_dict(data).results

    # -- Scraper ------------------------------------------------------------

    async def scrape(self, request: ScrapeRequest) -> ScrapeResponse:
        """Submit a doc-scraping job. Returns a job ID to poll."""
        data, _ = await self._do_json("POST", "/qai/v1/scraper/scrape", request.to_dict())
        return ScrapeResponse.from_dict(data)

    async def screenshot(self, request: ScreenshotRequest) -> ScreenshotResponse:
        """Screenshot URLs inline. Use screenshot_job() past ~5 URLs."""
        data, _ = await self._do_json("POST", "/qai/v1/scraper/screenshot", request.to_dict())
        return ScreenshotResponse.from_dict(data)

    async def screenshot_job(self, request: ScreenshotRequest) -> JobCreateResponse:
        """Submit a large screenshot batch as an async job."""
        return await self.create_job("screenshot", request.to_dict())

    # -- Voice library ------------------------------------------------------

    async def voice_library(self, query: VoiceLibraryQuery | None = None) -> SharedVoicesResponse:
        """Browse the shared voice library."""
        path = "/qai/v1/voices/library"
        params = _voice_library_params(query)
        if params:
            path += "?" + params
        data, _ = await self._do_json("GET", path)
        return SharedVoicesResponse.from_dict(data)

    async def add_voice_from_library(
        self,
        public_owner_id: str,
        voice_id: str,
        *,
        name: str | None = None,
    ) -> AddVoiceFromLibraryResponse:
        """Add a shared voice from the library to the account."""
        body: dict[str, Any] = {"public_owner_id": public_owner_id, "voice_id": voice_id}
        if name is not None:
            body["name"] = name
        data, _ = await self._do_json("POST", "/qai/v1/voices/library/add", body)
        return AddVoiceFromLibraryResponse.from_dict(data)

    # -- Audio finetunes / advanced music -----------------------------------

    async def create_finetune(
        self,
        name: str,
        files: list[tuple[str, bytes, str]],
        *,
        description: str | None = None,
    ) -> MusicFinetuneInfo:
        """Create a music finetune from audio samples.

        ``files`` are ``(filename, data, mime_type)`` triples, all sent under
        the ``files`` form field.
        """
        fields: dict[str, Any] = {"name": name}
        if description is not None:
            fields["description"] = description
        data, _ = await self._do_multipart(
            "/qai/v1/audio/finetunes",
            fields=fields,
            files=[("files", f) for f in files],
        )
        return MusicFinetuneInfo.from_dict(data)

    async def list_finetunes(self) -> MusicFinetuneListResponse:
        """List the account's music finetunes."""
        data, _ = await self._do_json("GET", "/qai/v1/audio/finetunes")
        return MusicFinetuneListResponse.from_dict(data)

    async def delete_finetune(self, finetune_id: str) -> None:
        """Delete a music finetune."""
        await self._do_json("DELETE", f"/qai/v1/audio/finetunes/{finetune_id}")

    async def generate_music_advanced(self, request: ElevenMusicRequest) -> ElevenMusicResponse:
        """Composition-plan music generation (sections, style, finetunes)."""
        data, meta = await self._do_json("POST", "/qai/v1/audio/music/advanced", request.to_dict())
        resp = ElevenMusicResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Compute billing ----------------------------------------------------

    async def compute_billing(
        self,
        *,
        instance_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> BillingResponse:
        """Query compute billing (BigQuery-backed) for an instance or range."""
        body: dict[str, Any] = {}
        if instance_id is not None:
            body["instance_id"] = instance_id
        if start_date is not None:
            body["start_date"] = start_date
        if end_date is not None:
            body["end_date"] = end_date
        data, _ = await self._do_json("POST", "/qai/v1/compute/billing", body)
        return BillingResponse.from_dict(data)

    # -- Jobs (async submission) --------------------------------------------

    async def chat_job(self, request: ChatRequest) -> JobCreateResponse:
        """Submit a chat completion as an async job.

        For long-running models (Opus and friends) where the synchronous
        /qai/v1/chat call would time out. Read the result with stream_job()
        or poll_job().
        """
        params = request.to_dict()
        params.pop("stream", None)
        return await self.create_job("chat", params)

    async def stream_job(self, job_id: str) -> AsyncIterator[JobStreamEvent]:
        """Stream a job's progress over SSE.

        Yields "progress" events until a terminal "complete" or "error".
        """
        url = self.base_url + f"/qai/v1/jobs/{job_id}/stream"
        headers = self._headers()
        headers["Accept"] = "text/event-stream"

        stream_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.stream_timeout, connect=15.0)
        )
        try:
            async with stream_client.stream("GET", url, headers=headers) as resp:
                if resp.status_code < 200 or resp.status_code >= 300:
                    await resp.aread()
                    meta = _ResponseMeta(resp.headers)
                    raise _parse_api_error(resp, meta.request_id)

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        return
                    try:
                        event = JobStreamEvent.from_dict(json.loads(payload))
                    except json.JSONDecodeError as exc:
                        yield JobStreamEvent(type="error", error=f"parse SSE: {exc}")
                        return

                    yield event

                    if event.type in ("complete", "error"):
                        return
        finally:
            await stream_client.aclose()

    async def generate_3d(
        self,
        model: str,
        *,
        prompt: str | None = None,
        image_url: str | None = None,
    ) -> JobCreateResponse:
        """Submit a text- or image-to-3D generation job. Poll with poll_job()."""
        params: dict[str, Any] = {"model": model}
        if prompt is not None:
            params["prompt"] = prompt
        if image_url is not None:
            params["image_url"] = image_url
        return await self.create_job("3d/generate", params)

    # -- Chat cost estimate -------------------------------------------------

    async def estimate_chat(self, request: ChatRequest) -> EstimateResponse:
        """Price a chat request without sending it.

        The number is the upfront reservation the real call would book — a
        ceiling, not a prediction. ``stream`` is dropped from the payload: the
        output ceiling is the same either way, and including it would make the
        SDK's wire shape diverge from what the server sees.
        """
        body = request.to_dict()
        body.pop("stream", None)
        data, _ = await self._do_json("POST", "/qai/v1/chat/estimate", body)
        return EstimateResponse.from_dict(data)
