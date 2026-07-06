"""Quantum AI API data types — all dataclasses with type hints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: str
    content: str = ""
    content_blocks: list[ContentBlock] | None = None
    tool_call_id: str | None = None
    is_error: bool = False
    # Phase echoes provider-side state (OpenAI Responses) so reasoning
    # models keep their chain of thought across multi-turn replay. Pass
    # back the phase received on the previous assistant ChatResponse.
    phase: str | None = None

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        return cls(role="assistant", content=content)

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role="system", content=content)

    @classmethod
    def tool(cls, tool_call_id: str, content: str, *, is_error: bool = False) -> ChatMessage:
        return cls(role="tool", content=content, tool_call_id=tool_call_id, is_error=is_error)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role}
        if self.content:
            d["content"] = self.content
        if self.content_blocks is not None:
            d["content_blocks"] = [b.to_dict() for b in self.content_blocks]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.is_error:
            d["is_error"] = True
        if self.phase is not None:
            d["phase"] = self.phase
        return d


@dataclass
class ContentBlock:
    """A single block in the response content array.

    Covers text, thinking, tool_use, image, file, and file_uri blocks.
    """

    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] | None = None
    # Gemini: must echo back with tool results for multi-turn thinking tool calls.
    thought_signature: str | None = None
    # base64-encoded payload for "image" and "file" blocks.
    data: str = ""
    # e.g. "image/png", "application/pdf", "video/mp4".
    mime_type: str = ""
    # for type "file".
    file_name: str = ""
    # for type "file_uri": remote resource URL (YouTube, gs://, etc.).
    file_uri: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type}
        if self.text:
            d["text"] = self.text
        if self.id:
            d["id"] = self.id
        if self.name:
            d["name"] = self.name
        if self.input is not None:
            d["input"] = self.input
        if self.thought_signature is not None:
            d["thought_signature"] = self.thought_signature
        if self.data:
            d["data"] = self.data
        if self.mime_type:
            d["mime_type"] = self.mime_type
        if self.file_name:
            d["file_name"] = self.file_name
        if self.file_uri:
            d["file_uri"] = self.file_uri
        return d

    @classmethod
    def from_dict(cls, b: dict[str, Any]) -> ContentBlock:
        return cls(
            type=b.get("type", ""),
            text=b.get("text", ""),
            id=b.get("id", ""),
            name=b.get("name", ""),
            input=b.get("input"),
            thought_signature=b.get("thought_signature"),
            data=b.get("data", ""),
            mime_type=b.get("mime_type", ""),
            file_name=b.get("file_name", ""),
            file_uri=b.get("file_uri", ""),
        )


@dataclass
class ChatTool:
    """Defines a function the model can call."""

    name: str
    description: str
    parameters: dict[str, Any] | None = None
    # Guaranteed schema validation (Anthropic, OpenAI). Serialized only when set.
    strict: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "description": self.description}
        if self.parameters is not None:
            d["parameters"] = self.parameters
        if self.strict is not None:
            d["strict"] = self.strict
        return d


@dataclass
class ChatUsage:
    """Token counts and cost for a chat response.

    output_tokens is the billable total (visible completion + reasoning).
    cached_tokens and reasoning_tokens are breakouts for transparency/audit.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost_ticks: int = 0


@dataclass
class ChatRequest:
    """Request body for text generation."""

    model: str
    messages: list[ChatMessage]
    tools: list[ChatTool] | None = None
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    # "auto" (default), "any" (force a tool), "none", or a specific tool name.
    tool_choice: str | None = None
    # JSON Schema for structured output — when set, the model is forced to
    # return JSON matching this schema.
    output_schema: dict[str, Any] | None = None
    # Chain-of-thought budget for reasoning models: "none"/"low"/"medium"/
    # "high"/"xhigh". Empty/None = provider default.
    reasoning_effort: str | None = None
    # Vertex context-cache resource name (e.g. "cachedContents/abc123"); Gemini only.
    cached_content: str | None = None
    provider_options: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
        }
        if self.tools:
            d["tools"] = [t.to_dict() for t in self.tools]
        if self.stream:
            d["stream"] = True
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.max_tokens is not None:
            d["max_tokens"] = self.max_tokens
        if self.tool_choice is not None:
            d["tool_choice"] = self.tool_choice
        if self.output_schema is not None:
            d["output_schema"] = self.output_schema
        if self.reasoning_effort is not None:
            d["reasoning_effort"] = self.reasoning_effort
        if self.cached_content is not None:
            d["cached_content"] = self.cached_content
        if self.provider_options is not None:
            d["provider_options"] = self.provider_options
        return d


@dataclass
class ChatResponse:
    """Response from a non-streaming chat request."""

    id: str = ""
    model: str = ""
    content: list[ContentBlock] = field(default_factory=list)
    usage: ChatUsage | None = None
    stop_reason: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    # True only when this response was served from the semantic cache.
    cached: bool = False
    # Provider-side state tag (OpenAI Responses) to echo back on the next turn.
    phase: str = ""
    cost_ticks: int = 0
    request_id: str = ""
    # Wallet balance after this request, from the X-QAI-Balance-After header.
    balance_after: int | None = None

    def text(self) -> str:
        """Concatenated text content, ignoring thinking and tool_use blocks."""
        return "".join(b.text for b in self.content if b.type == "text")

    def thinking(self) -> str:
        """Concatenated thinking content."""
        return "".join(b.text for b in self.content if b.type == "thinking")

    def tool_calls(self) -> list[ContentBlock]:
        """All tool_use blocks from the response."""
        return [b for b in self.content if b.type == "tool_use"]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatResponse:
        content = [ContentBlock.from_dict(b) for b in data.get("content", [])]
        usage_data = data.get("usage")
        usage = ChatUsage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cached_tokens=usage_data.get("cached_tokens", 0),
            reasoning_tokens=usage_data.get("reasoning_tokens", 0),
            cost_ticks=usage_data.get("cost_ticks", 0),
        ) if usage_data else None
        citations = data.get("citations") or []
        return cls(
            id=data.get("id", ""),
            model=data.get("model", ""),
            content=content,
            usage=usage,
            stop_reason=data.get("stop_reason", ""),
            citations=list(citations),
            cached=bool(data.get("cached", False)),
            phase=data.get("phase", ""),
        )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

@dataclass
class StreamDelta:
    """Incremental text in a streaming event."""

    text: str = ""


@dataclass
class StreamToolUse:
    """Tool call from a streaming event."""

    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamEvent:
    """A single event from an SSE chat stream."""

    type: str = ""
    event_type: str = ""
    delta: StreamDelta | None = None
    tool_use: StreamToolUse | None = None
    usage: ChatUsage | None = None
    error: str = ""
    done: bool = False


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

@dataclass
class ImageRequest:
    """Request body for image generation."""

    model: str
    prompt: str
    count: int | None = None
    size: str | None = None
    aspect_ratio: str | None = None
    quality: str | None = None
    output_format: str | None = None
    style: str | None = None
    background: str | None = None
    image_url: str | None = None
    topology: str | None = None
    target_polycount: int | None = None
    symmetry_mode: str | None = None
    pose_mode: str | None = None
    enable_pbr: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"model": self.model, "prompt": self.prompt}
        if self.count is not None:
            d["count"] = self.count
        if self.size is not None:
            d["size"] = self.size
        if self.aspect_ratio is not None:
            d["aspect_ratio"] = self.aspect_ratio
        if self.quality is not None:
            d["quality"] = self.quality
        if self.output_format is not None:
            d["output_format"] = self.output_format
        if self.style is not None:
            d["style"] = self.style
        if self.background is not None:
            d["background"] = self.background
        if self.image_url is not None:
            d["image_url"] = self.image_url
        if self.topology is not None:
            d["topology"] = self.topology
        if self.target_polycount is not None:
            d["target_polycount"] = self.target_polycount
        if self.symmetry_mode is not None:
            d["symmetry_mode"] = self.symmetry_mode
        if self.pose_mode is not None:
            d["pose_mode"] = self.pose_mode
        if self.enable_pbr is not None:
            d["enable_pbr"] = self.enable_pbr
        return d


@dataclass
class GeneratedImage:
    """A single generated image."""

    base64: str = ""
    format: str = ""
    index: int = 0


@dataclass
class ImageResponse:
    """Response from image generation."""

    images: list[GeneratedImage] = field(default_factory=list)
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""
    balance_after: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageResponse:
        images = [
            GeneratedImage(
                base64=img.get("base64", ""),
                format=img.get("format", ""),
                index=img.get("index", 0),
            )
            for img in data.get("images", [])
        ]
        return cls(
            images=images,
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class ImageEditRequest:
    """Request body for image editing."""

    model: str
    prompt: str
    input_images: list[str]
    count: int | None = None
    size: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model,
            "prompt": self.prompt,
            "input_images": self.input_images,
        }
        if self.count is not None:
            d["count"] = self.count
        if self.size is not None:
            d["size"] = self.size
        return d


# ImageEditResponse is the same shape as ImageResponse
ImageEditResponse = ImageResponse


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

@dataclass
class VideoRequest:
    """Request body for video generation."""

    model: str
    prompt: str
    duration_seconds: int | None = None
    aspect_ratio: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"model": self.model, "prompt": self.prompt}
        if self.duration_seconds is not None:
            d["duration_seconds"] = self.duration_seconds
        if self.aspect_ratio is not None:
            d["aspect_ratio"] = self.aspect_ratio
        return d


@dataclass
class GeneratedVideo:
    """A single generated video."""

    base64: str = ""
    format: str = ""
    size_bytes: int = 0
    index: int = 0


@dataclass
class VideoResponse:
    """Response from video generation."""

    videos: list[GeneratedVideo] = field(default_factory=list)
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""
    balance_after: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoResponse:
        videos = [
            GeneratedVideo(
                base64=v.get("base64", ""),
                format=v.get("format", ""),
                size_bytes=v.get("size_bytes", 0),
                index=v.get("index", 0),
            )
            for v in data.get("videos", [])
        ]
        return cls(
            videos=videos,
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

@dataclass
class TTSRequest:
    """Request body for text-to-speech."""

    model: str
    text: str
    voice: str | None = None
    output_format: str | None = None
    speed: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"model": self.model, "text": self.text}
        if self.voice is not None:
            d["voice"] = self.voice
        if self.output_format is not None:
            d["format"] = self.output_format
        if self.speed is not None:
            d["speed"] = self.speed
        return d


@dataclass
class TTSResponse:
    """Response from text-to-speech."""

    audio_base64: str = ""
    format: str = ""
    size_bytes: int = 0
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""
    balance_after: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TTSResponse:
        return cls(
            audio_base64=data.get("audio_base64", ""),
            format=data.get("format", ""),
            size_bytes=data.get("size_bytes", 0),
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class STTRequest:
    """Request body for speech-to-text."""

    model: str
    audio_base64: str
    filename: str | None = None
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"model": self.model, "audio_base64": self.audio_base64}
        if self.filename is not None:
            d["filename"] = self.filename
        if self.language is not None:
            d["language"] = self.language
        return d


@dataclass
class STTResponse:
    """Response from speech-to-text."""

    text: str = ""
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""
    balance_after: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> STTResponse:
        return cls(
            text=data.get("text", ""),
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class MusicRequest:
    """Request body for music generation."""

    model: str
    prompt: str
    duration_seconds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"model": self.model, "prompt": self.prompt}
        if self.duration_seconds is not None:
            d["duration_seconds"] = self.duration_seconds
        return d


@dataclass
class MusicClip:
    """A single generated music clip."""

    base64: str = ""
    format: str = ""
    size_bytes: int = 0
    index: int = 0


@dataclass
class MusicResponse:
    """Response from music generation."""

    audio_clips: list[MusicClip] = field(default_factory=list)
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""
    balance_after: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MusicResponse:
        clips = [
            MusicClip(
                base64=c.get("base64", ""),
                format=c.get("format", ""),
                size_bytes=c.get("size_bytes", 0),
                index=c.get("index", 0),
            )
            for c in data.get("audio_clips", [])
        ]
        return cls(
            audio_clips=clips,
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class SoundEffectResponse:
    """Response from sound effects generation."""

    audio_base64: str = ""
    format: str = ""
    size_bytes: int = 0
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""
    balance_after: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SoundEffectResponse:
        return cls(
            audio_base64=data.get("audio_base64", ""),
            format=data.get("format", ""),
            size_bytes=data.get("size_bytes", 0),
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

@dataclass
class EmbedRequest:
    """Request body for text embeddings."""

    model: str
    input: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "input": self.input}


@dataclass
class EmbedResponse:
    """Response from text embedding."""

    embeddings: list[list[float]] = field(default_factory=list)
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""
    balance_after: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbedResponse:
        return cls(
            embeddings=data.get("embeddings", []),
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@dataclass
class DocumentRequest:
    """Request body for document extraction."""

    file_base64: str
    filename: str
    output_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"file_base64": self.file_base64, "filename": self.filename}
        if self.output_format is not None:
            d["output_format"] = self.output_format
        return d


@dataclass
class DocumentResponse:
    """Response from document extraction."""

    content: str = ""
    format: str = ""
    meta: dict[str, Any] | None = None
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentResponse:
        return cls(
            content=data.get("content", ""),
            format=data.get("format", ""),
            meta=data.get("meta"),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

@dataclass
class RAGSearchRequest:
    """Request body for Vertex AI RAG search."""

    query: str
    corpus: str | None = None
    top_k: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"query": self.query}
        if self.corpus is not None:
            d["corpus"] = self.corpus
        if self.top_k is not None:
            d["top_k"] = self.top_k
        return d


@dataclass
class RAGResult:
    """A single result from RAG search."""

    source_uri: str = ""
    source_name: str = ""
    text: str = ""
    score: float = 0.0
    distance: float = 0.0


@dataclass
class RAGSearchResponse:
    """Response from RAG search."""

    results: list[RAGResult] = field(default_factory=list)
    query: str = ""
    corpora: list[str] | None = None
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RAGSearchResponse:
        results = [
            RAGResult(
                source_uri=r.get("source_uri", ""),
                source_name=r.get("source_name", ""),
                text=r.get("text", ""),
                score=r.get("score", 0.0),
                distance=r.get("distance", 0.0),
            )
            for r in data.get("results", [])
        ]
        return cls(
            results=results,
            query=data.get("query", ""),
            corpora=data.get("corpora"),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class RAGCorpus:
    """An available RAG corpus."""

    name: str = ""
    display_name: str = ""
    description: str = ""
    state: str = ""


@dataclass
class SurrealRAGSearchRequest:
    """Request body for SurrealDB-backed RAG search."""

    query: str
    provider: str | None = None
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"query": self.query}
        if self.provider is not None:
            d["provider"] = self.provider
        if self.limit is not None:
            d["limit"] = self.limit
        return d


@dataclass
class SurrealRAGResult:
    """A single result from SurrealDB RAG search."""

    provider: str = ""
    title: str = ""
    heading: str = ""
    source_file: str = ""
    content: str = ""
    score: float = 0.0


@dataclass
class SurrealRAGSearchResponse:
    """Response from SurrealDB RAG search."""

    results: list[SurrealRAGResult] = field(default_factory=list)
    query: str = ""
    provider: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurrealRAGSearchResponse:
        results = [
            SurrealRAGResult(
                provider=r.get("provider", ""),
                title=r.get("title", ""),
                heading=r.get("heading", ""),
                source_file=r.get("source_file", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
            )
            for r in data.get("results", [])
        ]
        return cls(
            results=results,
            query=data.get("query", ""),
            provider=data.get("provider", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """An available model."""

    id: str = ""
    provider: str = ""
    display_name: str = ""
    input_per_million: float = 0.0
    output_per_million: float = 0.0


@dataclass
class PricingInfo:
    """Pricing details for a model."""

    id: str = ""
    provider: str = ""
    display_name: str = ""
    input_per_million: float = 0.0
    output_per_million: float = 0.0


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

@dataclass
class BalanceResponse:
    """Response from account balance check."""

    user_id: str = ""
    credit_ticks: int = 0
    credit_usd: float = 0.0
    ticks_per_usd: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BalanceResponse:
        return cls(
            user_id=data.get("user_id", ""),
            credit_ticks=data.get("credit_ticks", 0),
            credit_usd=data.get("credit_usd", 0.0),
            ticks_per_usd=data.get("ticks_per_usd", 0),
        )


@dataclass
class UsageEntry:
    """A single usage entry from the ledger."""

    id: str = ""
    request_id: str | None = None
    model: str | None = None
    provider: str | None = None
    endpoint: str | None = None
    delta_ticks: int | None = None
    balance_after: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    created_at: str | None = None


@dataclass
class UsageResponse:
    """Paginated usage history response."""

    entries: list[UsageEntry] = field(default_factory=list)
    has_more: bool = False
    next_cursor: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageResponse:
        entries = [
            UsageEntry(
                id=e.get("id", ""),
                request_id=e.get("request_id"),
                model=e.get("model"),
                provider=e.get("provider"),
                endpoint=e.get("endpoint"),
                delta_ticks=e.get("delta_ticks"),
                balance_after=e.get("balance_after"),
                input_tokens=e.get("input_tokens"),
                output_tokens=e.get("output_tokens"),
                created_at=e.get("created_at"),
            )
            for e in data.get("entries", [])
        ]
        return cls(
            entries=entries,
            has_more=data.get("has_more", False),
            next_cursor=data.get("next_cursor"),
        )


@dataclass
class UsageSummaryMonth:
    """Monthly usage summary."""

    month: str = ""
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_margin_usd: float = 0.0
    by_provider: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class UsageSummaryResponse:
    """Response from usage summary."""

    months: list[UsageSummaryMonth] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UsageSummaryResponse:
        months = [
            UsageSummaryMonth(
                month=m.get("month", ""),
                total_requests=m.get("total_requests", 0),
                total_input_tokens=m.get("total_input_tokens", 0),
                total_output_tokens=m.get("total_output_tokens", 0),
                total_cost_usd=m.get("total_cost_usd", 0.0),
                total_margin_usd=m.get("total_margin_usd", 0.0),
                by_provider=m.get("by_provider", []),
            )
            for m in data.get("months", [])
        ]
        return cls(months=months)


@dataclass
class PricingEntry:
    """A single entry in the pricing table."""

    provider: str = ""
    model: str = ""
    display_name: str = ""
    input_per_million: float = 0.0
    output_per_million: float = 0.0
    cached_per_million: float = 0.0


@dataclass
class PricingResponse:
    """Response from the pricing endpoint (map of model_id to entry)."""

    pricing: dict[str, PricingEntry] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PricingResponse:
        pricing = {}
        for key, p in data.get("pricing", {}).items():
            pricing[key] = PricingEntry(
                provider=p.get("Provider", ""),
                model=p.get("Model", ""),
                display_name=p.get("DisplayName", ""),
                input_per_million=p.get("InputPerMillion", 0.0),
                output_per_million=p.get("OutputPerMillion", 0.0),
                cached_per_million=p.get("CachedPerMillion", 0.0),
            )
        return cls(pricing=pricing)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@dataclass
class JobCreateResponse:
    """Response from job creation."""

    job_id: str = ""
    status: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobCreateResponse:
        return cls(
            job_id=data.get("job_id", ""),
            status=data.get("status", ""),
        )


@dataclass
class JobStatusResponse:
    """Response from job status check."""

    job_id: str = ""
    status: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    cost_ticks: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobStatusResponse:
        return cls(
            job_id=data.get("job_id", ""),
            status=data.get("status", ""),
            result=data.get("result"),
            error=data.get("error"),
            cost_ticks=data.get("cost_ticks", 0),
        )


# ---------------------------------------------------------------------------
# Session Chat
# ---------------------------------------------------------------------------

@dataclass
class ContextConfig:
    """Configuration for session context management."""

    max_tokens: int | None = None
    auto_compact: bool | None = None
    strategy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.max_tokens is not None:
            d["max_tokens"] = self.max_tokens
        if self.auto_compact is not None:
            d["auto_compact"] = self.auto_compact
        if self.strategy is not None:
            d["strategy"] = self.strategy
        return d


@dataclass
class SessionChatRequest:
    """Request body for session-based chat."""

    message: str
    model: str | None = None
    session_id: str | None = None
    tools: list[ChatTool] | None = None
    tool_results: list[dict[str, Any]] | None = None
    stream: bool = False
    system_prompt: str | None = None
    context_config: ContextConfig | None = None
    # "none"/"low"/"medium"/"high"/"xhigh". Empty = provider default.
    reasoning_effort: str | None = None
    provider_options: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"message": self.message}
        if self.model is not None:
            d["model"] = self.model
        if self.session_id is not None:
            d["session_id"] = self.session_id
        if self.tools:
            d["tools"] = [t.to_dict() for t in self.tools]
        if self.tool_results is not None:
            d["tool_results"] = self.tool_results
        if self.stream:
            d["stream"] = True
        if self.system_prompt is not None:
            d["system_prompt"] = self.system_prompt
        if self.context_config is not None:
            d["context_config"] = self.context_config.to_dict()
        if self.reasoning_effort is not None:
            d["reasoning_effort"] = self.reasoning_effort
        if self.provider_options is not None:
            d["provider_options"] = self.provider_options
        return d


@dataclass
class SessionChatResponse:
    """Response from session chat.

    The backend wraps the full ChatResponse under a ``response`` field and
    surfaces ``session_id`` + ``context`` at the top level. The flat fields
    (model/content/usage/stop_reason/...) are mirrors of the inner response
    for convenience and back-compat.
    """

    session_id: str = ""
    model: str = ""
    content: list[ContentBlock] = field(default_factory=list)
    usage: ChatUsage | None = None
    stop_reason: str = ""
    cost_ticks: int = 0
    request_id: str = ""
    response: ChatResponse | None = None
    context: dict[str, Any] | None = None
    balance_after: int | None = None

    def text(self) -> str:
        """Concatenated text content."""
        return "".join(b.text for b in self.content if b.type == "text")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionChatResponse:
        inner_data = data.get("response") or {}
        inner = ChatResponse.from_dict(inner_data)
        return cls(
            session_id=data.get("session_id", ""),
            model=inner.model,
            content=inner.content,
            usage=inner.usage,
            stop_reason=inner.stop_reason,
            cost_ticks=inner.usage.cost_ticks if inner.usage else 0,
            request_id=inner.id,
            response=inner,
            context=data.get("context"),
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class MissionWorker:
    """A named worker in a mission's ``workers`` map.

    Mirrors the backend ``MissionWorkerConfig`` (routes_missions.go):
    model + tier + description + escalate_to + max_retries. The map key
    (worker name) is supplied by the caller's dict, not by this struct.
    """

    model: str = ""
    tier: str = ""  # "cheap", "mid", "expensive"
    description: str = ""
    # Worker to fall back to when this one fails.
    escalate_to: str = ""
    # Retries before escalating (default 1 on the backend).
    max_retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.model:
            d["model"] = self.model
        if self.tier:
            d["tier"] = self.tier
        if self.description:
            d["description"] = self.description
        if self.escalate_to:
            d["escalate_to"] = self.escalate_to
        if self.max_retries:
            d["max_retries"] = self.max_retries
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionWorker:
        return cls(
            model=data.get("model", ""),
            tier=data.get("tier", ""),
            description=data.get("description", ""),
            escalate_to=data.get("escalate_to", ""),
            max_retries=data.get("max_retries", 0),
        )


# Back-compat alias for callers that constructed workers positionally under
# the old name. New code should use MissionWorker.
AgentWorker = MissionWorker


@dataclass
class AgentRunRequest:
    """Request body for an agent run.

    Retargeted to POST /qai/v1/missions — the orchestration endpoint. The
    ``task`` field is serialized as ``goal`` on the wire; ``workers`` is a
    map of worker name → MissionWorker (matching the backend's
    MissionRequest.Workers), NOT a list.
    """

    task: str
    conductor_model: str | None = None
    conductor_tier: str | None = None
    workers: dict[str, MissionWorker] | None = None
    max_steps: int | None = None
    system_prompt: str | None = None
    session_id: str | None = None
    auto_plan: bool | None = None
    context_config: ContextConfig | None = None
    strategy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"goal": self.task}
        if self.conductor_model is not None:
            d["conductor_model"] = self.conductor_model
        if self.conductor_tier is not None:
            d["conductor_tier"] = self.conductor_tier
        if self.workers:
            d["workers"] = {k: v.to_dict() for k, v in self.workers.items()}
        if self.max_steps is not None:
            d["max_steps"] = self.max_steps
        if self.system_prompt is not None:
            d["system_prompt"] = self.system_prompt
        if self.session_id is not None:
            d["session_id"] = self.session_id
        if self.auto_plan is not None:
            d["auto_plan"] = self.auto_plan
        if self.context_config is not None:
            d["context_config"] = self.context_config.to_dict()
        if self.strategy is not None:
            d["strategy"] = self.strategy
        return d


@dataclass
class MissionRunRequest:
    """Request body for a mission run (POST /qai/v1/missions).

    ``workers`` is a map of worker name → MissionWorker, matching the
    backend's ``MissionRequest.Workers`` (map[string]MissionWorkerConfig).
    Sending a JSON array here returns 400.
    """

    goal: str
    strategy: str | None = None
    conductor_model: str | None = None
    conductor_tier: str | None = None
    workers: dict[str, MissionWorker] | None = None
    max_steps: int | None = None
    system_prompt: str | None = None
    session_id: str | None = None
    auto_plan: bool | None = None
    context_config: ContextConfig | None = None
    deployment_id: str | None = None
    build_command: str | None = None
    workspace_path: str | None = None
    context: str | None = None
    use_context: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"goal": self.goal}
        if self.strategy is not None:
            d["strategy"] = self.strategy
        if self.conductor_model is not None:
            d["conductor_model"] = self.conductor_model
        if self.conductor_tier is not None:
            d["conductor_tier"] = self.conductor_tier
        if self.workers:
            d["workers"] = {k: v.to_dict() for k, v in self.workers.items()}
        if self.max_steps is not None:
            d["max_steps"] = self.max_steps
        if self.system_prompt is not None:
            d["system_prompt"] = self.system_prompt
        if self.session_id is not None:
            d["session_id"] = self.session_id
        if self.auto_plan is not None:
            d["auto_plan"] = self.auto_plan
        if self.context_config is not None:
            d["context_config"] = self.context_config.to_dict()
        if self.deployment_id is not None:
            d["deployment_id"] = self.deployment_id
        if self.build_command is not None:
            d["build_command"] = self.build_command
        if self.workspace_path is not None:
            d["workspace_path"] = self.workspace_path
        if self.context is not None:
            d["context"] = self.context
        if self.use_context is not None:
            d["use_context"] = self.use_context
        return d


# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------

@dataclass
class APIKeyCreateRequest:
    """Request body for creating an API key."""

    name: str
    endpoints: list[str] | None = None
    spend_cap_usd: float | None = None
    rate_limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        if self.endpoints is not None:
            d["endpoints"] = self.endpoints
        if self.spend_cap_usd is not None:
            d["spend_cap_usd"] = self.spend_cap_usd
        if self.rate_limit is not None:
            d["rate_limit"] = self.rate_limit
        return d


@dataclass
class APIKeyInfo:
    """An API key entry."""

    id: str = ""
    name: str = ""
    prefix: str = ""
    endpoints: list[str] | None = None
    spend_cap_usd: float | None = None
    rate_limit: int | None = None
    created_at: str = ""
    last_used_at: str | None = None
    revoked: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> APIKeyInfo:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            prefix=data.get("prefix", ""),
            endpoints=data.get("endpoints"),
            spend_cap_usd=data.get("spend_cap_usd"),
            rate_limit=data.get("rate_limit"),
            created_at=data.get("created_at", ""),
            last_used_at=data.get("last_used_at"),
            revoked=data.get("revoked", False),
        )


@dataclass
class APIKeyCreateResponse:
    """Response from API key creation (includes the full key)."""

    id: str = ""
    key: str = ""
    name: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> APIKeyCreateResponse:
        return cls(
            id=data.get("id", ""),
            key=data.get("key", ""),
            name=data.get("name", ""),
        )


@dataclass
class APIKeyListResponse:
    """Response from listing API keys."""

    keys: list[APIKeyInfo] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> APIKeyListResponse:
        keys = [APIKeyInfo.from_dict(k) for k in data.get("keys", [])]
        return cls(keys=keys)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

@dataclass
class ComputeTemplate:
    """A compute instance template."""

    id: str = ""
    name: str = ""
    gpu: str = ""
    vcpus: int = 0
    memory_gb: int = 0
    gpu_count: int = 0
    price_per_hour: float = 0.0
    vram_gb: int | None = None
    ram_gb: int | None = None
    price_per_hour_usd: float | None = None
    zones: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComputeTemplate:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            gpu=data.get("gpu", ""),
            vcpus=data.get("vcpus", 0),
            memory_gb=data.get("memory_gb", 0),
            gpu_count=data.get("gpu_count", 0),
            price_per_hour=data.get("price_per_hour", 0.0),
            vram_gb=data.get("vram_gb"),
            ram_gb=data.get("ram_gb"),
            price_per_hour_usd=data.get("price_per_hour_usd"),
            zones=data.get("zones"),
        )


@dataclass
class ComputeProvisionRequest:
    """Request body for provisioning a compute instance."""

    template: str
    zone: str | None = None
    spot: bool = False
    auto_teardown_minutes: int = 30
    ssh_public_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "template": self.template,
            "spot": self.spot,
            "auto_teardown_minutes": self.auto_teardown_minutes,
        }
        if self.zone is not None:
            d["zone"] = self.zone
        if self.ssh_public_key is not None:
            d["ssh_public_key"] = self.ssh_public_key
        return d


@dataclass
class ComputeInstance:
    """A compute instance."""

    id: str = ""
    template: str = ""
    zone: str = ""
    status: str = ""
    ip_address: str = ""
    ssh_user: str = ""
    created_at: str = ""
    auto_teardown_at: str = ""
    spot: bool = False
    ssh_address: str | None = None
    price_per_hour_usd: float | None = None
    auto_teardown_minutes: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComputeInstance:
        return cls(
            id=data.get("id", ""),
            template=data.get("template", ""),
            zone=data.get("zone", ""),
            status=data.get("status", ""),
            ip_address=data.get("ip_address", ""),
            ssh_user=data.get("ssh_user", ""),
            created_at=data.get("created_at", ""),
            auto_teardown_at=data.get("auto_teardown_at", ""),
            spot=data.get("spot", False),
            ssh_address=data.get("ssh_address"),
            price_per_hour_usd=data.get("price_per_hour_usd"),
            auto_teardown_minutes=data.get("auto_teardown_minutes"),
        )


@dataclass
class ComputeProvisionResponse:
    """Response from provisioning a compute instance."""

    instance: ComputeInstance | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComputeProvisionResponse:
        inst_data = data.get("instance") or data
        return cls(instance=ComputeInstance.from_dict(inst_data))


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------

@dataclass
class VoiceInfo:
    """A voice entry."""

    voice_id: str = ""
    name: str = ""
    category: str = ""
    labels: dict[str, str] | None = None
    preview_url: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoiceInfo:
        return cls(
            voice_id=data.get("voice_id", ""),
            name=data.get("name", ""),
            category=data.get("category", ""),
            labels=data.get("labels"),
            preview_url=data.get("preview_url", ""),
        )


@dataclass
class VoiceListResponse:
    """Response from listing voices."""

    voices: list[VoiceInfo] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoiceListResponse:
        voices = [VoiceInfo.from_dict(v) for v in data.get("voices", [])]
        return cls(voices=voices)


@dataclass
class VoiceCloneResponse:
    """Response from cloning a voice."""

    voice_id: str = ""
    name: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoiceCloneResponse:
        return cls(
            voice_id=data.get("voice_id", ""),
            name=data.get("name", ""),
        )


# ---------------------------------------------------------------------------
# Advanced Audio
# ---------------------------------------------------------------------------

@dataclass
class DialogueVoice:
    """A voice assignment in a dialogue request."""

    voice_id: str
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"voice_id": self.voice_id}
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass
class DialogueRequest:
    """Request body for multi-voice dialogue generation."""

    text: str
    voices: list[DialogueVoice]
    model: str | None = None
    output_format: str | None = None
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "text": self.text,
            "voices": [v.to_dict() for v in self.voices],
        }
        if self.model is not None:
            d["model"] = self.model
        if self.output_format is not None:
            d["output_format"] = self.output_format
        if self.seed is not None:
            d["seed"] = self.seed
        return d


@dataclass
class AudioResponse:
    """Generic audio response used by multiple endpoints."""

    audio_base64: str = ""
    format: str = ""
    size_bytes: int = 0
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""
    balance_after: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioResponse:
        known = {"audio_base64", "format", "size_bytes", "model", "cost_ticks", "request_id"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            audio_base64=data.get("audio_base64", ""),
            format=data.get("format", ""),
            size_bytes=data.get("size_bytes", 0),
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
            extra=extra,
        )


@dataclass
class AlignmentWord:
    """A single aligned word with timing."""

    word: str = ""
    start: float = 0.0
    end: float = 0.0


@dataclass
class AlignmentResponse:
    """Response from audio-text alignment."""

    words: list[AlignmentWord] = field(default_factory=list)
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentResponse:
        words = [
            AlignmentWord(
                word=w.get("word", ""),
                start=w.get("start", 0.0),
                end=w.get("end", 0.0),
            )
            for w in data.get("words", [])
        ]
        return cls(
            words=words,
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class VoiceDesignResponse:
    """Response from voice design."""

    voice_id: str = ""
    audio_base64: str = ""
    format: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoiceDesignResponse:
        return cls(
            voice_id=data.get("voice_id", ""),
            audio_base64=data.get("audio_base64", ""),
            format=data.get("format", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


# ---------------------------------------------------------------------------
# HeyGen Video
# ---------------------------------------------------------------------------

@dataclass
class VideoStudioRequest:
    """Request body for HeyGen studio video generation."""

    avatar_id: str
    script: str
    voice_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "avatar_id": self.avatar_id,
            "script": self.script,
            "voice_id": self.voice_id,
        }


@dataclass
class VideoTranslateRequest:
    """Request body for video translation."""

    video_url: str
    target_language: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_url": self.video_url,
            "target_language": self.target_language,
        }


@dataclass
class VideoPhotoAvatarRequest:
    """Request body for photo avatar video generation."""

    photo_url: str
    script: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "photo_url": self.photo_url,
            "script": self.script,
        }


@dataclass
class VideoDigitalTwinRequest:
    """Request body for digital twin video generation."""

    avatar_id: str
    script: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "avatar_id": self.avatar_id,
            "script": self.script,
        }


@dataclass
class HeyGenAvatar:
    """A HeyGen avatar."""

    avatar_id: str = ""
    avatar_name: str = ""
    preview_url: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeyGenAvatar:
        return cls(
            avatar_id=data.get("avatar_id", ""),
            avatar_name=data.get("avatar_name", ""),
            preview_url=data.get("preview_url", ""),
        )


@dataclass
class HeyGenVoice:
    """A HeyGen voice."""

    voice_id: str = ""
    name: str = ""
    language: str = ""
    gender: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeyGenVoice:
        known = {"voice_id", "name", "language", "gender"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            voice_id=data.get("voice_id", ""),
            name=data.get("name", ""),
            language=data.get("language", ""),
            gender=data.get("gender", ""),
            extra=extra,
        )


@dataclass
class HeyGenTemplate:
    """A HeyGen video template."""

    template_id: str = ""
    name: str = ""
    preview_url: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeyGenTemplate:
        return cls(
            template_id=data.get("template_id", ""),
            name=data.get("name", ""),
            preview_url=data.get("preview_url", ""),
        )


# ---------------------------------------------------------------------------
# Documents (additional)
# ---------------------------------------------------------------------------

@dataclass
class ChunkDocumentRequest:
    """Request body for document chunking."""

    text: str
    chunk_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"text": self.text}
        if self.chunk_size is not None:
            d["chunk_size"] = self.chunk_size
        return d


@dataclass
class ChunkDocumentResponse:
    """Response from document chunking."""

    chunks: list[str] = field(default_factory=list)
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkDocumentResponse:
        return cls(
            chunks=data.get("chunks", []),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class ProcessDocumentRequest:
    """Request body for document processing."""

    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text}


@dataclass
class ProcessDocumentResponse:
    """Response from document processing."""

    result: str = ""
    meta: dict[str, Any] | None = None
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessDocumentResponse:
        return cls(
            result=data.get("result", ""),
            meta=data.get("meta"),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


# ---------------------------------------------------------------------------
# SurrealDB RAG Providers
# ---------------------------------------------------------------------------

@dataclass
class SurrealRAGProvider:
    """A SurrealDB RAG provider."""

    name: str = ""
    chunk_count: int = 0


@dataclass
class SurrealRAGProvidersResponse:
    """Response from listing SurrealDB RAG providers."""

    providers: list[SurrealRAGProvider] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurrealRAGProvidersResponse:
        providers = [
            SurrealRAGProvider(
                name=p.get("name", ""),
                chunk_count=p.get("chunk_count", 0),
            )
            for p in data.get("providers", [])
        ]
        return cls(providers=providers)


# ---------------------------------------------------------------------------
# Jobs (additional)
# ---------------------------------------------------------------------------

@dataclass
class JobListResponse:
    """Response from listing jobs."""

    jobs: list[JobStatusResponse] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobListResponse:
        jobs = [JobStatusResponse.from_dict(j) for j in data.get("jobs", [])]
        return cls(jobs=jobs)


# ---------------------------------------------------------------------------
# Contact
# ---------------------------------------------------------------------------

@dataclass
class ContactRequest:
    """Request body for contact form submission."""

    name: str
    email: str
    message: str
    subject: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "email": self.email,
            "message": self.message,
        }
        if self.subject is not None:
            d["subject"] = self.subject
        return d


@dataclass
class ContactResponse:
    """Response from contact form submission."""

    success: bool = False
    message: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContactResponse:
        return cls(
            success=data.get("success", False),
            message=data.get("message", ""),
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKS_PER_USD: int = 10_000_000_000
