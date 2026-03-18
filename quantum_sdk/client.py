"""Quantum AI API client — sync and async."""

from __future__ import annotations

import json
from typing import Any, Iterator, AsyncIterator

import httpx

from .errors import APIError
from .types import (
    ChatRequest,
    ChatResponse,
    ChatUsage,
    ContentBlock,
    StreamDelta,
    StreamEvent,
    StreamToolUse,
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
    EmbedRequest,
    EmbedResponse,
    DocumentRequest,
    DocumentResponse,
    RAGSearchRequest,
    RAGSearchResponse,
    RAGCorpus,
    SurrealRAGSearchRequest,
    SurrealRAGSearchResponse,
    ModelInfo,
    PricingInfo,
)

DEFAULT_BASE_URL = "https://api.quantumencoding.ai"
DEFAULT_TIMEOUT = 60.0


class _ResponseMeta:
    """Parsed response metadata from HTTP headers."""

    __slots__ = ("cost_ticks", "request_id", "model")

    def __init__(self, headers: httpx.Headers) -> None:
        self.request_id = headers.get("x-qai-request-id", "")
        self.model = headers.get("x-qai-model", "")
        raw_ticks = headers.get("x-qai-cost-ticks", "")
        self.cost_ticks = int(raw_ticks) if raw_ticks else 0


def _parse_api_error(response: httpx.Response, request_id: str) -> APIError:
    """Parse an error response body into an APIError."""
    body = response.text
    code = str(response.status_code)
    message = body

    try:
        data = response.json()
        err_obj = data.get("error", {})
        if isinstance(err_obj, dict):
            if err_obj.get("message"):
                message = err_obj["message"]
            code = err_obj.get("code") or err_obj.get("type") or code
    except Exception:
        pass

    return APIError(
        status_code=response.status_code,
        code=code,
        message=message,
        request_id=request_id or None,
    )


def _parse_sse_event(payload: str) -> StreamEvent:
    """Parse a single SSE JSON payload into a StreamEvent."""
    raw: dict[str, Any] = json.loads(payload)
    event_type: str = raw.get("type", "")
    ev = StreamEvent(type=event_type)

    if event_type in ("content_delta", "thinking_delta"):
        delta_data = raw.get("delta")
        if isinstance(delta_data, dict):
            ev.delta = StreamDelta(text=delta_data.get("text", ""))
    elif event_type == "tool_use":
        ev.tool_use = StreamToolUse(
            id=raw.get("id", ""),
            name=raw.get("name", ""),
            input=raw.get("input", {}),
        )
    elif event_type == "usage":
        ev.usage = ChatUsage(
            input_tokens=raw.get("input_tokens", 0),
            output_tokens=raw.get("output_tokens", 0),
            cost_ticks=raw.get("cost_ticks", 0),
        )
    elif event_type == "error":
        ev.error = raw.get("message", "")

    return ev


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
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owns_client = http_client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _do_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[Any, _ResponseMeta]:
        url = self.base_url + path
        headers = self._headers()
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

    # -- Chat ---------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any] | ChatRequest] | None = None,
        *,
        request: ChatRequest | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat request.

        Can be called with keyword convenience args:
            client.chat(model="...", messages=[{"role": "user", "content": "Hi"}])

        Or with a full ChatRequest:
            client.chat(request=ChatRequest(model="...", messages=[...]))
        """
        req = self._build_chat_request(model, messages, request, **kwargs)
        req.stream = False
        data, meta = self._do_json("POST", "/qai/v1/chat", req.to_dict())
        resp = ChatResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if not resp.model:
            resp.model = meta.model
        return resp

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any] | ChatRequest] | None = None,
        *,
        request: ChatRequest | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        """Send a streaming chat request. Yields StreamEvent objects."""
        req = self._build_chat_request(model, messages, request, **kwargs)
        req.stream = True

        url = self.base_url + "/qai/v1/chat"
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"

        # Use a separate client without timeout for streaming
        with httpx.Client() as stream_client:
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

    # -- Image --------------------------------------------------------------

    def generate_image(self, request: ImageRequest) -> ImageResponse:
        """Generate images from a text prompt."""
        data, meta = self._do_json("POST", "/qai/v1/images/generate", request.to_dict())
        resp = ImageResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def edit_image(self, request: ImageEditRequest) -> ImageEditResponse:
        """Edit images using an AI model."""
        data, meta = self._do_json("POST", "/qai/v1/images/edit", request.to_dict())
        resp = ImageEditResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Video --------------------------------------------------------------

    def generate_video(self, request: VideoRequest) -> VideoResponse:
        """Generate a video from a text prompt."""
        data, meta = self._do_json("POST", "/qai/v1/video/generate", request.to_dict())
        resp = VideoResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Audio --------------------------------------------------------------

    def speak(self, request: TTSRequest) -> TTSResponse:
        """Generate speech from text."""
        data, meta = self._do_json("POST", "/qai/v1/audio/tts", request.to_dict())
        resp = TTSResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def transcribe(self, request: STTRequest) -> STTResponse:
        """Convert speech to text."""
        data, meta = self._do_json("POST", "/qai/v1/audio/stt", request.to_dict())
        resp = STTResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def generate_music(self, request: MusicRequest) -> MusicResponse:
        """Generate music from a text prompt."""
        data, meta = self._do_json("POST", "/qai/v1/audio/music", request.to_dict())
        resp = MusicResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Embeddings ---------------------------------------------------------

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Generate text embeddings."""
        data, meta = self._do_json("POST", "/qai/v1/embeddings", request.to_dict())
        resp = EmbedResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Documents ----------------------------------------------------------

    def extract_document(self, request: DocumentRequest) -> DocumentResponse:
        """Extract text content from a document (PDF, image, etc.)."""
        data, meta = self._do_json("POST", "/qai/v1/documents/extract", request.to_dict())
        resp = DocumentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- RAG ----------------------------------------------------------------

    def rag_search(self, request: RAGSearchRequest) -> RAGSearchResponse:
        """Search Vertex AI RAG corpora."""
        data, meta = self._do_json("POST", "/qai/v1/rag/search", request.to_dict())
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
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def _do_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[Any, _ResponseMeta]:
        url = self.base_url + path
        headers = self._headers()
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

    # -- Chat ---------------------------------------------------------------

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any] | ChatRequest] | None = None,
        *,
        request: ChatRequest | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat request."""
        req = Client._build_chat_request(model, messages, request, **kwargs)
        req.stream = False
        data, meta = await self._do_json("POST", "/qai/v1/chat", req.to_dict())
        resp = ChatResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if not resp.model:
            resp.model = meta.model
        return resp

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any] | ChatRequest] | None = None,
        *,
        request: ChatRequest | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Send a streaming chat request. Yields StreamEvent objects."""
        req = Client._build_chat_request(model, messages, request, **kwargs)
        req.stream = True

        url = self.base_url + "/qai/v1/chat"
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"

        stream_client = httpx.AsyncClient()
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

    # -- Image --------------------------------------------------------------

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        data, meta = await self._do_json("POST", "/qai/v1/images/generate", request.to_dict())
        resp = ImageResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def edit_image(self, request: ImageEditRequest) -> ImageEditResponse:
        data, meta = await self._do_json("POST", "/qai/v1/images/edit", request.to_dict())
        resp = ImageEditResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Video --------------------------------------------------------------

    async def generate_video(self, request: VideoRequest) -> VideoResponse:
        data, meta = await self._do_json("POST", "/qai/v1/video/generate", request.to_dict())
        resp = VideoResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Audio --------------------------------------------------------------

    async def speak(self, request: TTSRequest) -> TTSResponse:
        data, meta = await self._do_json("POST", "/qai/v1/audio/tts", request.to_dict())
        resp = TTSResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def transcribe(self, request: STTRequest) -> STTResponse:
        data, meta = await self._do_json("POST", "/qai/v1/audio/stt", request.to_dict())
        resp = STTResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def generate_music(self, request: MusicRequest) -> MusicResponse:
        data, meta = await self._do_json("POST", "/qai/v1/audio/music", request.to_dict())
        resp = MusicResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Embeddings ---------------------------------------------------------

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        data, meta = await self._do_json("POST", "/qai/v1/embeddings", request.to_dict())
        resp = EmbedResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Documents ----------------------------------------------------------

    async def extract_document(self, request: DocumentRequest) -> DocumentResponse:
        data, meta = await self._do_json("POST", "/qai/v1/documents/extract", request.to_dict())
        resp = DocumentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- RAG ----------------------------------------------------------------

    async def rag_search(self, request: RAGSearchRequest) -> RAGSearchResponse:
        data, meta = await self._do_json("POST", "/qai/v1/rag/search", request.to_dict())
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
