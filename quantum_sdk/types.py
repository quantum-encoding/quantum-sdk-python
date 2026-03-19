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
        return d


@dataclass
class ContentBlock:
    """A single block in the response content array."""

    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] | None = None

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
        return d


@dataclass
class ChatTool:
    """Defines a function the model can call."""

    name: str
    description: str
    parameters: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "description": self.description}
        if self.parameters is not None:
            d["parameters"] = self.parameters
        return d


@dataclass
class ChatUsage:
    """Token counts and cost for a chat response."""

    input_tokens: int = 0
    output_tokens: int = 0
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
    cost_ticks: int = 0
    request_id: str = ""

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
        content = [
            ContentBlock(
                type=b.get("type", ""),
                text=b.get("text", ""),
                id=b.get("id", ""),
                name=b.get("name", ""),
                input=b.get("input"),
            )
            for b in data.get("content", [])
        ]
        usage_data = data.get("usage")
        usage = ChatUsage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cost_ticks=usage_data.get("cost_ticks", 0),
        ) if usage_data else None
        return cls(
            id=data.get("id", ""),
            model=data.get("model", ""),
            content=content,
            usage=usage,
            stop_reason=data.get("stop_reason", ""),
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
    strategy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.max_tokens is not None:
            d["max_tokens"] = self.max_tokens
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
        return d


@dataclass
class SessionChatResponse:
    """Response from session chat."""

    session_id: str = ""
    model: str = ""
    content: list[ContentBlock] = field(default_factory=list)
    usage: ChatUsage | None = None
    stop_reason: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    def text(self) -> str:
        """Concatenated text content."""
        return "".join(b.text for b in self.content if b.type == "text")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionChatResponse:
        content = [
            ContentBlock(
                type=b.get("type", ""),
                text=b.get("text", ""),
                id=b.get("id", ""),
                name=b.get("name", ""),
                input=b.get("input"),
            )
            for b in data.get("content", [])
        ]
        usage_data = data.get("usage")
        usage = ChatUsage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cost_ticks=usage_data.get("cost_ticks", 0),
        ) if usage_data else None
        return cls(
            session_id=data.get("session_id", ""),
            model=data.get("model", ""),
            content=content,
            usage=usage,
            stop_reason=data.get("stop_reason", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class AgentWorker:
    """A worker configuration for agent/mission runs."""

    model: str
    role: str | None = None
    instructions: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"model": self.model}
        if self.role is not None:
            d["role"] = self.role
        if self.instructions is not None:
            d["instructions"] = self.instructions
        return d


@dataclass
class AgentRunRequest:
    """Request body for an agent run."""

    task: str
    conductor_model: str | None = None
    workers: list[AgentWorker] | None = None
    max_steps: int | None = None
    system_prompt: str | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"task": self.task}
        if self.conductor_model is not None:
            d["conductor_model"] = self.conductor_model
        if self.workers:
            d["workers"] = [w.to_dict() for w in self.workers]
        if self.max_steps is not None:
            d["max_steps"] = self.max_steps
        if self.system_prompt is not None:
            d["system_prompt"] = self.system_prompt
        if self.session_id is not None:
            d["session_id"] = self.session_id
        return d


@dataclass
class MissionRunRequest:
    """Request body for a mission run."""

    goal: str
    strategy: str | None = None
    conductor_model: str | None = None
    workers: list[AgentWorker] | None = None
    max_steps: int | None = None
    system_prompt: str | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"goal": self.goal}
        if self.strategy is not None:
            d["strategy"] = self.strategy
        if self.conductor_model is not None:
            d["conductor_model"] = self.conductor_model
        if self.workers:
            d["workers"] = [w.to_dict() for w in self.workers]
        if self.max_steps is not None:
            d["max_steps"] = self.max_steps
        if self.system_prompt is not None:
            d["system_prompt"] = self.system_prompt
        if self.session_id is not None:
            d["session_id"] = self.session_id
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

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "text": self.text,
            "voices": [v.to_dict() for v in self.voices],
        }
        if self.model is not None:
            d["model"] = self.model
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioResponse:
        return cls(
            audio_base64=data.get("audio_base64", ""),
            format=data.get("format", ""),
            size_bytes=data.get("size_bytes", 0),
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeyGenVoice:
        return cls(
            voice_id=data.get("voice_id", ""),
            name=data.get("name", ""),
            language=data.get("language", ""),
            gender=data.get("gender", ""),
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "email": self.email,
            "message": self.message,
        }


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
